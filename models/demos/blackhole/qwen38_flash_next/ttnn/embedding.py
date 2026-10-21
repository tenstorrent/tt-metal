# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact TP4 token embedding and vocabulary-parallel LM head.

Qwen3.8-Flash-Next has *untied* ``[248320, 2560]`` input and output
tables.  Both tables are sharded by contiguous vocabulary rows over mesh axis
1; no coordinate stores a replicated table.

The token embedding uses the sentinel construction from MiniMax's parallel
embedding.  Each coordinate owns ``62080`` real vocabulary rows, padded with
one zero row on either side.  Range-checked host token IDs are converted into
the four exact local UINT32 indices and shard-uploaded; non-owning coordinates
select a sentinel.  A single-token decode is host-padded to one 32-index tile,
which selects TTNN's fused TILE embedding program; the 31 padding positions
select zero sentinel row 0.  The four full-width, mutually exclusive lookup
results are exposed through a metadata-only logical sequence-one view.  The
qualified single-token path all-gathers those four mutually exclusive
partials, selects the range-checked vocabulary owner's contribution, and
partitions hidden dimension 3 back across the line.  Multi-token embedding
retains the existing TT_CCL reduction.  Both paths return hidden-sharded
logical ``[1, 1, S, 640]`` while preserving their TILE backing allocations.

The LM head gathers that hidden shard, computes only its local ``62080``
vocabulary columns, and returns an object whose vocabulary range and TTNN
topology are explicit.  Greedy decode reduces each local shard to one
``(value, index)`` candidate and reads back only those eight scalars; it never
gathers the 248,320 logits.  Ties retain PyTorch's lowest-global-index rule.

There is intentionally no standalone final RMSNorm here.  The pinned
Qwen4Exp model applies the terminal zero-centred RMS normalization inside
``model.language_model.hyper_connection_mixer``.  The checkpoint contains no
``model.language_model.norm.weight`` tensor, and SGLang explicitly removes
the inherited norm.  :func:`validate_terminal_architecture` rejects a
checkpoint that would make a second final norm necessary instead of silently
inventing one.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    CHECKPOINT_FILE_MANIFEST_SHA256,
    CHECKPOINT_TENSOR_MANIFEST_SHA256,
    PINNED_CHECKPOINT_REVISION,
    Qwen38Checkpoint,
)
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256, Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROWS,
    MESH_SHAPE,
    Qwen38MeshContract,
    TensorPlacement,
    chunk_row_tiles,
    replicate_tensor_2d_mesh_mapper,
    require_lane_count,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import (
    dense_dtype_tag,
    dense_math_fidelity_name,
    dram_sharded_matmul_configs,
    dram_sharded_weight_memory_config,
    validate_decode_dram_workers,
    validate_dram_sharded_weight,
)
from models.tt_transformers.tt.ccl import tt_all_reduce

TP_AXIS = 1
TP_SIZE = 4
HIDDEN_SIZE = 2560
LOCAL_HIDDEN_SIZE = HIDDEN_SIZE // TP_SIZE
VOCAB_SIZE = 248_320
LOCAL_VOCAB_SIZE = VOCAB_SIZE // TP_SIZE
RMS_NORM_EPS = 1e-6
# Owner tie-break step of resolve_greedy_on_device: owner d's bf16 maximum is lowered
# by d * GREEDY_TIE_BREAK_EPS in FP32 before the device argmax, so equal maxima leave
# a unique maximum at the lowest owner (the torch.argmax rule of resolve_greedy).
# 2**-16 is the FP32 ulp of |v| in [128, 256): for every |v| < 256 the four shifts
# d * 2**-16 (d <= 3) are exact and strictly ordered, and the largest shift
# 3 * 2**-16 < 2**-14 is below the spacing of two distinct bf16 values whose larger
# magnitude is >= 2**-6, so distinct candidates keep their order.
GREEDY_TIE_BREAK_EPS = 2.0**-16

EMBEDDING_NAME = "model.language_model.embed_tokens.weight"
LM_HEAD_NAME = "lm_head.weight"
TERMINAL_MIXER_PREFIX = "model.language_model.hyper_connection_mixer."
TERMINAL_MIXER_NAMES = {
    TERMINAL_MIXER_PREFIX + "hc_norm.weight": (4 * HIDDEN_SIZE,),
    TERMINAL_MIXER_PREFIX + "input_mix_weight_down.weight": (320, 4 * HIDDEN_SIZE),
    TERMINAL_MIXER_PREFIX + "input_mix_weight_up.weight": (4 * HIDDEN_SIZE, 320),
}
FORBIDDEN_STANDALONE_NORMS = (
    "model.language_model.norm.weight",
    "model.norm.weight",
)


class Qwen38TTNNEmbeddingSyncPolicy(str, Enum):
    """Host-drain policy for the single-token owner-select chain."""

    CORRECTNESS_FENCED = "correctness-fenced"
    RESIDENT_ASYNC = "resident-async"


PINNED_TENSOR_MANIFEST_SHA256 = CHECKPOINT_TENSOR_MANIFEST_SHA256
IO_CACHE_FORMAT_VERSION = 1
IO_CACHE_LOCK_TIMEOUT_SECONDS = 60.0

# LM-head column chunks.  Each chunk stays under the wide-N L1 limits of the
# existing Blackhole Qwen path and spans a multiple of four tiles per DRAM
# bank, so the DRAM-sharded reader's two-tile K blocks page at the 16 KB NOC
# burst (must match dram_sharded_matmul_configs over 40 storage cores).
# 5 * 8192 + 3 * 7040 = 62080 pads twelve tiles per K row, the least of any
# such split.
LM_HEAD_CHUNK_COLUMNS = (8192,) * 5 + (7040,) * 3
TILE_SIZE = 32

# A "token row" is a replicated FP32 TILE [1,1,1,32] holding one global token id
# at column 0 and zeros elsewhere.  FP32 keeps the token localization exact on
# the SFPU (every id < 2**24) and a full tile width keeps the fused embedding
# lookup tile-aligned.
TOKEN_ROW_SHAPE = (1, 1, 1, TILE_SIZE)
# A token-row lane below every vocabulary range: each coordinate clamps it to its zero sentinel row, so the
# device-token embedding of that lane is exactly zero (the padding rows of an MTP v2 verify pass).
ZERO_EMBEDDING_TOKEN = -1

# The optional sampling epilogue of TAIL: per vocabulary shard the top-k logits and their
# global ids, packed per shard and all-gathered into one replicated FP32 ROW_MAJOR row
#   [shard 0: values(k) ids(k) | shard 1: values(k) ids(k) | shard 2 ... | shard 3 ...]
# i.e. ``row.reshape(TP_SIZE, 2, k)[:, 0]`` are the values and ``[:, 1]`` the ids.  k is a
# tile width so the per-shard pack is a tile-aligned concat; 32 and 64 cost the same on the
# device (measured, both 0.340 ms tails), 32 is enough for the card profiles' top_k 20 and
# halves the readback.  The host samples on this row; the greedy token row is untouched.
SAMPLING_CANDIDATES_PER_DEVICE = 32
SAMPLING_CANDIDATE_ROW_SHAPE = (1, 1, 1, 2 * TP_SIZE * SAMPLING_CANDIDATES_PER_DEVICE)


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(item) for item in tensor.shape)


def _padded_shape(tensor) -> tuple[int, ...]:
    return tuple(int(item) for item in tensor.padded_shape)


def _metadata(tensor) -> str:
    """The four fields every device-path contract checks, for actual-vs-expected error text."""

    return f"shape={_shape(tensor)} padded_shape={_padded_shape(tensor)} dtype={tensor.dtype} layout={tensor.layout}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _json_normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@contextmanager
def _exclusive_file_lock(path: Path, *, timeout_seconds: float = IO_CACHE_LOCK_TIMEOUT_SECONDS):
    if timeout_seconds <= 0:
        raise ValueError("I/O cache lock timeout must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring Qwen3.8 I/O cache lock {path}")
                time.sleep(0.25)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _deallocate(*tensors) -> None:
    for tensor in tensors:
        if tensor is not None:
            ttnn.deallocate(tensor)


def _diagnostic_stage(callback: Callable[[str], None] | None, event: str) -> None:
    """Emit an opt-in queue discriminator event without changing production scheduling."""

    if callback is not None:
        if not callable(callback):
            raise TypeError("embedding diagnostic stage callback must be callable")
        callback(event)


def _all_reduce_owner_select_hidden(
    local_partial, *, mesh_contract: Qwen38MeshContract, replicated_reference, collective_topology, rows: int = 1
):
    """Token-invariant owner selection for trace capture (``rows`` token rows, S=1 by default).

    The four devices hold mutually exclusive [1,1,rows,2560] partials; per row
    three are exactly zero (their sentinel row), so the TP sum reproduces the
    owner's row bitwise (x + 0 + 0 + 0 is exact at every hop) without the
    token-dependent slice offset that keeps
    :func:`_all_gather_owner_select_hidden_fallback` outside a trace.
    ``mesh_partition`` restores the hidden-sharded topology.
    """

    if collective_topology != ttnn.Topology.Linear:
        raise RuntimeError("device-token owner select requires Linear topology")
    mesh_contract.validate_tensor(local_partial, placement=TensorPlacement.LOCAL_PARTIAL)
    mesh_contract.validate_tensor(replicated_reference, placement=TensorPlacement.REPLICATED)
    reduced = ttnn.all_reduce(
        local_partial,
        cluster_axis=TP_AXIS,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=collective_topology,
    )
    _deallocate(local_partial)
    mesh_contract.validate_tensor(reduced, placement=TensorPlacement.REPLICATED)
    if _shape(reduced) != (1, 1, rows, HIDDEN_SIZE):
        raise RuntimeError(
            f"device-token owner all-reduce must return [1,1,{rows},{HIDDEN_SIZE}], got {_metadata(reduced)}"
        )
    hidden = ttnn.mesh_partition(reduced, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _deallocate(reduced)
    if (
        _shape(hidden) != (1, 1, rows, LOCAL_HIDDEN_SIZE)
        or hidden.dtype != ttnn.bfloat16
        or hidden.layout != ttnn.TILE_LAYOUT
    ):
        raise RuntimeError(
            f"device-token owner select must return BF16 TILE [1,1,{rows},{LOCAL_HIDDEN_SIZE}], got {_metadata(hidden)}"
        )
    mesh_contract.mark_collective_shard(
        hidden,
        replicated_reference=replicated_reference,
        shard_dim=3,
        expected_local_shape=(1, 1, rows, LOCAL_HIDDEN_SIZE),
    )
    return hidden


def _all_gather_owner_select_hidden_fallback(
    local_partial,
    *,
    active_vocab_shard: int,
    active_vocab_coordinate: tuple[int, int],
    mesh_device,
    mesh_contract: Qwen38MeshContract,
    replicated_reference,
    tt_ccl,
    collective_topology,
    synchronization_policy: Qwen38TTNNEmbeddingSyncPolicy = Qwen38TTNNEmbeddingSyncPolicy.CORRECTNESS_FENCED,
    _retain_async_failure_owners: Callable[..., None] | None = None,
    _diagnostic_stage_callback: Callable[[str], None] | None = None,
):
    """S=1 correctness fallback that selects the exact vocabulary owner.

    Host-token path for the eager and fixed-position lanes; the sequential
    trace chain embeds from a device token row through the token-invariant
    :func:`_all_reduce_owner_select_hidden`.  The slice offset here depends on
    the token, so this path is never trace-captured.

    This path is deliberately self-contained and is the qualified production
    S=1 path.  It uses the retained Linear
    all-gather-async configuration, selects the one contribution identified by
    the original range-checked CPU token, then mesh-partitions that exact
    hidden width over axis 1.  The three sentinel contributions are exactly
    zero, so owner selection is equivalent to their sum for one token without
    an add/reduce kernel or an activation host payload.

    The correctness-fenced policy drains each producer stage.  Resident decode
    instead relies on same-command-queue ordering and releases each producer
    only after its final consumer has been enqueued.  On an asynchronous
    failure, all wrappers and local handles are retained through teardown.
    """

    if type(synchronization_policy) is not Qwen38TTNNEmbeddingSyncPolicy:
        raise TypeError(
            "embedding synchronization_policy must be an exact Qwen38TTNNEmbeddingSyncPolicy, "
            f"got {synchronization_policy!r}"
        )
    resident_async = synchronization_policy is Qwen38TTNNEmbeddingSyncPolicy.RESIDENT_ASYNC
    if resident_async and not callable(_retain_async_failure_owners):
        raise TypeError("resident-async embedding requires an asynchronous failure owner sink")

    mesh_contract.validate_mesh(mesh_device)
    if tuple(int(value) for value in mesh_device.shape) != MESH_SHAPE:
        raise RuntimeError(f"embedding all-gather fallback requires mesh {MESH_SHAPE}")
    if collective_topology != ttnn.Topology.Linear:
        raise RuntimeError("embedding all-gather fallback requires Linear topology")
    if tt_ccl is None:
        raise RuntimeError("embedding all-gather fallback requires the builder-owned TT-CCL manager")
    if type(active_vocab_shard) is not int or not 0 <= active_vocab_shard < TP_SIZE:
        raise RuntimeError(f"S=1 embedding owner shard must be an exact integer in [0,{TP_SIZE})")
    expected_coordinate = (0, active_vocab_shard)
    if (
        type(active_vocab_coordinate) is not tuple
        or len(active_vocab_coordinate) != 2
        or any(type(value) is not int for value in active_vocab_coordinate)
        or active_vocab_coordinate != expected_coordinate
    ):
        raise RuntimeError(
            f"S=1 embedding owner coordinate must be logical {expected_coordinate}, got {active_vocab_coordinate!r}"
        )
    available_links = tt_ccl.get_num_links(cluster_axis=TP_AXIS)
    if type(available_links) is not int or available_links < 1:
        raise RuntimeError("embedding all-gather fallback requires at least one verified axis-1 link")
    if _shape(local_partial) != (1, 1, 1, HIDDEN_SIZE) or _padded_shape(local_partial) != (
        1,
        1,
        TILE_SIZE,
        HIDDEN_SIZE,
    ):
        raise RuntimeError("embedding all-gather fallback requires logical/padded [1,1,1,2560]/[1,1,32,2560]")
    if (
        local_partial.dtype != ttnn.bfloat16
        or local_partial.layout != ttnn.TILE_LAYOUT
        or local_partial.memory_config() != ttnn.DRAM_MEMORY_CONFIG
    ):
        raise RuntimeError("embedding all-gather fallback input must be BF16 TILE DRAM")
    mesh_contract.validate_tensor(local_partial, placement=TensorPlacement.LOCAL_PARTIAL)
    mesh_contract.validate_tensor(replicated_reference, placement=TensorPlacement.REPLICATED)

    gathered = selected = hidden = None
    retained_locals: tuple[Any, ...] = ()
    all_gather_drained = owner_slice_drained = partition_drained = False
    completed = False
    try:
        _diagnostic_stage(_diagnostic_stage_callback, "before-all-gather-async")
        gathered = ttnn.experimental.all_gather_async(
            local_partial,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=TP_AXIS),
            num_links=1,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            chunks_per_sync=1,
            num_workers_per_link=1,
            num_buffers_per_channel=2,
        )
        _diagnostic_stage(_diagnostic_stage_callback, "after-all-gather-async-enqueue")
        retained_locals += tuple(ttnn.get_device_tensors(local_partial))
        retained_locals += tuple(ttnn.get_device_tensors(gathered))
        _diagnostic_stage(_diagnostic_stage_callback, "before-all-gather-async-synchronize")
        if not resident_async:
            ttnn.synchronize_device(mesh_device)
        all_gather_drained = True
        _diagnostic_stage(_diagnostic_stage_callback, "after-all-gather-async-synchronize")
        if _shape(gathered) != (1, 1, 1, TP_SIZE * HIDDEN_SIZE) or _padded_shape(gathered) != (
            1,
            1,
            TILE_SIZE,
            TP_SIZE * HIDDEN_SIZE,
        ):
            raise RuntimeError("embedding fallback all-gather returned an unexpected logical/padded shape")
        reference_topology = replicated_reference.tensor_topology()
        gathered.update_tensor_topology(
            ttnn.TensorTopology(
                reference_topology.distribution_shape(),
                [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
                reference_topology.mesh_coords(),
            )
        )
        mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)

        # Axis-1 all-gather concatenates logical mesh columns along W.  The
        # validated owner is a logical column, never a physical device ID.
        owner_start = active_vocab_shard * HIDDEN_SIZE
        owner_end = owner_start + HIDDEN_SIZE
        _diagnostic_stage(_diagnostic_stage_callback, "before-owner-contribution-slice")
        selected = ttnn.slice(
            gathered,
            (0, 0, 0, owner_start),
            (1, 1, 1, owner_end),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        _diagnostic_stage(_diagnostic_stage_callback, "after-owner-contribution-slice-enqueue")
        retained_locals += tuple(ttnn.get_device_tensors(selected))
        _diagnostic_stage(_diagnostic_stage_callback, "before-owner-contribution-slice-synchronize")
        if not resident_async:
            ttnn.synchronize_device(mesh_device)
        owner_slice_drained = True
        _diagnostic_stage(_diagnostic_stage_callback, "after-owner-contribution-slice-synchronize")
        if (
            _shape(selected) != (1, 1, 1, HIDDEN_SIZE)
            or _padded_shape(selected) != (1, 1, TILE_SIZE, HIDDEN_SIZE)
            or selected.dtype != ttnn.bfloat16
            or selected.layout != ttnn.TILE_LAYOUT
            or selected.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise RuntimeError("embedding owner contribution must be BF16 TILE DRAM [1,1,1,2560]")
        mesh_contract.validate_tensor(selected, placement=TensorPlacement.REPLICATED)

        _diagnostic_stage(_diagnostic_stage_callback, "before-mesh-partition")
        hidden = ttnn.mesh_partition(
            selected,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _diagnostic_stage(_diagnostic_stage_callback, "after-mesh-partition-enqueue")
        retained_locals += tuple(ttnn.get_device_tensors(hidden))
        _diagnostic_stage(_diagnostic_stage_callback, "before-mesh-partition-synchronize")
        if not resident_async:
            ttnn.synchronize_device(mesh_device)
        partition_drained = True
        _diagnostic_stage(_diagnostic_stage_callback, "after-mesh-partition-synchronize")
        if _shape(hidden) != (1, 1, 1, LOCAL_HIDDEN_SIZE) or _padded_shape(hidden) != (
            1,
            1,
            TILE_SIZE,
            LOCAL_HIDDEN_SIZE,
        ):
            raise RuntimeError("embedding fallback mesh partition returned an unexpected shape")
        mesh_contract.mark_collective_shard(
            hidden,
            replicated_reference=replicated_reference,
            shard_dim=3,
            expected_local_shape=(1, 1, 1, LOCAL_HIDDEN_SIZE),
        )
        mesh_contract.validate_tensor(hidden, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if (
            hidden.dtype != ttnn.bfloat16
            or hidden.layout != ttnn.TILE_LAYOUT
            or hidden.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise RuntimeError("embedding all-gather fallback output must be BF16 TILE DRAM")
        if len(retained_locals) != TP_SIZE * 4:
            raise RuntimeError("embedding fallback did not retain every stage's four local handles")
        _deallocate(local_partial)
        completed = True
        return hidden
    except BaseException as error:
        if resident_async:
            _retain_async_failure_owners(
                error,
                local_partial,
                gathered,
                selected,
                hidden,
                *retained_locals,
            )
        raise
    finally:
        # Fenced mode has explicit completion proof.  Resident async success
        # has enqueued every final consumer on the same CQ.  Its failure path
        # retains all owners and submits no cleanup.
        if completed:
            _deallocate(gathered, selected)
        elif not resident_async:
            if partition_drained:
                _deallocate(hidden, selected)
            if owner_slice_drained:
                _deallocate(gathered)
            if all_gather_drained:
                _deallocate(local_partial)


def _localize_token_ids_on_host(input_ids: torch.Tensor) -> torch.Tensor:
    """Return exact sentinel indices for all four vocabulary shards.

    The caller still supplies global CPU token IDs.  Range checking and the
    vocabulary transform happen while those IDs are host resident, avoiding a
    device arithmetic chain for a value that is already known exactly.  Axis 1
    is laid out as the TP4 vocabulary axis so ``ShardTensor2dMesh`` gives every
    coordinate one local ``[1, 1, sequence]`` UINT32 index tensor.
    """

    if input_ids.device.type != "cpu" or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Qwen3.8 token IDs must be a global-B1 CPU tensor [1, sequence]")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"Qwen3.8 token IDs must be int32/int64, got {input_ids.dtype}")
    if input_ids.shape[1] <= 0:
        raise ValueError("Qwen3.8 token IDs must contain at least one position")

    global_ids = input_ids.to(torch.int64).reshape(1, 1, -1)
    if int(global_ids.min()) < 0 or int(global_ids.max()) >= VOCAB_SIZE:
        raise IndexError(f"token ID is outside the exact vocabulary [0, {VOCAB_SIZE})")

    # Padded local embedding row 0 is the below-range sentinel, rows 1..62080
    # are real vocabulary rows, and row 62081 is the above-range sentinel.
    starts_minus_one = torch.arange(TP_SIZE, dtype=torch.int64).reshape(1, TP_SIZE, 1) * LOCAL_VOCAB_SIZE - 1
    localized = torch.clamp(global_ids - starts_minus_one, min=0, max=LOCAL_VOCAB_SIZE + 1)
    real_rows = (localized > 0) & (localized < LOCAL_VOCAB_SIZE + 1)
    if not bool(torch.all(real_rows.sum(dim=1) == 1)):
        raise AssertionError("each range-checked token must select exactly one TP4 vocabulary shard")
    return localized.to(torch.uint32).contiguous()


def _single_token_vocab_owner(localized: torch.Tensor) -> tuple[int, tuple[int, int]]:
    """Return the one logical vocabulary owner encoded by localized S=1 IDs."""

    expected = (1, TP_SIZE, 1)
    if localized.device.type != "cpu" or localized.dtype != torch.uint32 or tuple(localized.shape) != expected:
        raise ValueError(f"single-token localized indices must be CPU UINT32 {expected}, got {localized.shape}")
    values = localized.to(torch.int64)
    real_rows = (values > 0) & (values < LOCAL_VOCAB_SIZE + 1)
    owners = torch.nonzero(real_rows[0, :, 0], as_tuple=False).flatten()
    if owners.numel() != 1:
        raise ValueError("single-token localized indices must contain exactly one real vocabulary owner")
    shard = int(owners.item())
    return shard, (0, shard)


def _pad_single_decode_local_indices(localized: torch.Tensor) -> torch.Tensor:
    """Pad one localized decode index to the fused embedding tile width.

    Row 0 of every local sentinel table is exactly zero.  Padding every shard
    with local index 0 therefore makes positions 1..31 additive zeros while
    preserving the one real owner at position 0.
    """

    expected = (1, TP_SIZE, 1)
    if localized.device.type != "cpu" or localized.dtype != torch.uint32 or tuple(localized.shape) != expected:
        raise ValueError(f"single-token localized indices must be CPU UINT32 {expected}, got {localized.shape}")
    padded = torch.zeros((1, TP_SIZE, TILE_SIZE), dtype=torch.uint32)
    padded[..., 0] = localized[..., 0]
    return padded.contiguous()


def _validate_token_row(token_row, *, mesh_contract: Qwen38MeshContract, label: str) -> None:
    if _shape(token_row) != TOKEN_ROW_SHAPE or token_row.dtype != ttnn.float32 or token_row.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{label} must be FP32 TILE {TOKEN_ROW_SHAPE}, got {_metadata(token_row)}")
    mesh_contract.validate_tensor(token_row, placement=TensorPlacement.REPLICATED)


@dataclass(frozen=True)
class Qwen38TTNNTokenRowConstants:
    """FP32 device constants for the token-row embedding and greedy-resolve paths.

    ``vocab_localize_row`` is a TILE constant for :meth:`embed_device_token`; the
    three resolve constants are ROW_MAJOR because :meth:`resolve_greedy_on_device`
    argmaxes a width-four gather that must stay padding-free (a sub-tile TILE
    reduction would fold tile padding into the result).
    """

    # Vocab-sharded [1,1,1,32] TILE: coordinate d holds ``d * LOCAL_VOCAB_SIZE - 1``
    # at column 0, the same per-shard offset ``_localize_token_ids_on_host``
    # subtracts so clamp([0, LOCAL_VOCAB_SIZE + 1]) selects the sentinel-padded row.
    vocab_localize_row: Any
    vocab_localize_lanes: Any  # the same offset in all 32 columns: localizes a 32-lane token row (prefill chunk)
    owner_tie_break: Any  # replicated [1,1,1,4] ROW_MAJOR: d * GREEDY_TIE_BREAK_EPS, subtracted before the argmax
    lm_head_vocab_starts: Any  # replicated [1,1,1,4] ROW_MAJOR: shard d's first global vocab id, d * LOCAL_VOCAB_SIZE
    unit_column: Any  # replicated [1,1,1,32] ROW_MAJOR: 1.0 at column 0, splats the resolved id into a token row

    @classmethod
    def build(cls, mesh_device, mesh_contract: Qwen38MeshContract) -> "Qwen38TTNNTokenRowConstants":
        replicate = replicate_tensor_2d_mesh_mapper(mesh_device)
        vocab_shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 1))

        def upload(host: torch.Tensor, mapper, layout):
            return ttnn.from_torch(
                host,
                dtype=ttnn.float32,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        vocab_localize_row = torch.zeros((1, TP_SIZE, 1, TILE_SIZE), dtype=torch.float32)
        vocab_localize_row[0, :, 0, 0] = torch.arange(TP_SIZE, dtype=torch.float32) * LOCAL_VOCAB_SIZE - 1.0
        owners = torch.arange(TP_SIZE, dtype=torch.float32).reshape(1, 1, 1, TP_SIZE)
        owner_tie_break = owners * GREEDY_TIE_BREAK_EPS
        lm_head_vocab_starts = owners * LOCAL_VOCAB_SIZE
        unit_column = torch.zeros(TOKEN_ROW_SHAPE, dtype=torch.float32)
        unit_column[..., 0] = 1.0
        result = cls(
            vocab_localize_row=upload(vocab_localize_row, vocab_shard, ttnn.TILE_LAYOUT),
            vocab_localize_lanes=upload(
                vocab_localize_row[..., :1].expand(1, TP_SIZE, 1, TILE_SIZE).contiguous(), vocab_shard, ttnn.TILE_LAYOUT
            ),
            owner_tie_break=upload(owner_tie_break, replicate, ttnn.ROW_MAJOR_LAYOUT),
            lm_head_vocab_starts=upload(lm_head_vocab_starts, replicate, ttnn.ROW_MAJOR_LAYOUT),
            unit_column=upload(unit_column, replicate, ttnn.ROW_MAJOR_LAYOUT),
        )
        result.validate(mesh_contract)
        return result

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        for name in ("vocab_localize_row", "vocab_localize_lanes"):
            localize = getattr(self, name)
            if (
                _shape(localize) != TOKEN_ROW_SHAPE
                or localize.dtype != ttnn.float32
                or localize.layout != ttnn.TILE_LAYOUT
            ):
                raise RuntimeError(f"{name} must be FP32 TILE {TOKEN_ROW_SHAPE}, got {_metadata(localize)}")
            mesh_contract.validate_tensor(localize, placement=TensorPlacement.VOCAB_SHARDED, shard_dim=1)
        for name, shape in (
            ("owner_tie_break", (1, 1, 1, TP_SIZE)),
            ("lm_head_vocab_starts", (1, 1, 1, TP_SIZE)),
            ("unit_column", TOKEN_ROW_SHAPE),
        ):
            tensor = getattr(self, name)
            if _shape(tensor) != shape or tensor.dtype != ttnn.float32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise RuntimeError(f"{name} must be FP32 ROW_MAJOR {shape}, got {_metadata(tensor)}")
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)


@dataclass(frozen=True)
class Qwen38TTNNSamplingCandidateConstants:
    """Device constants of :meth:`Qwen38TTNNLMHead.sampling_candidates`, built only by a sampling chain.

    ``shard_vocab_start`` is vocab-sharded FP32 TILE ``[1,1,1,1]``: coordinate d holds
    ``d * LOCAL_VOCAB_SIZE``, the scalar the shard's local top-k ids are rebased by
    before the gather.  ``readback_row`` is replicated FP32 ROW_MAJOR
    ``SAMPLING_CANDIDATE_ROW_SHAPE``, the persistent trace-stable target of the
    epilogue's final ``ttnn.copy`` and the one tensor the host reads on a sampled step.
    """

    shard_vocab_start: Any
    readback_row: Any

    @classmethod
    def build(cls, mesh_device, mesh_contract: Qwen38MeshContract) -> "Qwen38TTNNSamplingCandidateConstants":
        def upload(host: torch.Tensor, mapper, layout):
            return ttnn.from_torch(
                host,
                dtype=ttnn.float32,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        starts = torch.arange(TP_SIZE, dtype=torch.float32).reshape(1, TP_SIZE, 1, 1) * LOCAL_VOCAB_SIZE
        result = cls(
            shard_vocab_start=upload(
                starts, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 1)), ttnn.TILE_LAYOUT
            ),
            readback_row=upload(
                torch.zeros(SAMPLING_CANDIDATE_ROW_SHAPE, dtype=torch.float32),
                replicate_tensor_2d_mesh_mapper(mesh_device),
                ttnn.ROW_MAJOR_LAYOUT,
            ),
        )
        result.validate(mesh_contract)
        return result

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        start = self.shard_vocab_start
        if _shape(start) != (1, 1, 1, 1) or start.dtype != ttnn.float32 or start.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"shard_vocab_start must be FP32 TILE [1,1,1,1], got {_metadata(start)}")
        mesh_contract.validate_tensor(start, placement=TensorPlacement.VOCAB_SHARDED, shard_dim=1)
        row = self.readback_row
        if (
            _shape(row) != SAMPLING_CANDIDATE_ROW_SHAPE
            or row.dtype != ttnn.float32
            or row.layout != ttnn.ROW_MAJOR_LAYOUT
        ):
            raise RuntimeError(
                f"readback_row must be FP32 ROW_MAJOR {SAMPLING_CANDIDATE_ROW_SHAPE}, got {_metadata(row)}"
            )
        mesh_contract.validate_tensor(row, placement=TensorPlacement.REPLICATED)


def resolve_greedy_lanes_on_device(
    candidates: Qwen38GreedyCandidates,
    *,
    constants: Qwen38TTNNTokenRowConstants,
    mesh_contract: Qwen38MeshContract,
    collective_topology,
):
    """``Qwen38TTNNLMHead.resolve_greedy_on_device`` for ``rows`` = 1..32 candidate rows (the batched lanes).

    The same gather, tie-break, argmax, rebase and 32-bit select per row.  The two ``[1,1,1,4]`` constants are
    applied as TILE row broadcasts (exact SFPU fp32 elementwise ops on a tilized copy: no reduction touches the
    tile padding; ``ttnn.repeat`` of a 16-byte ROW_MAJOR row crashes the pinned runtime's repeat codegen); the
    argmax and the select stay ROW_MAJOR on the exact width-four rows.  The ``[1,1,rows,1]`` id column is padded
    to 32 rows with zeros (rows < 32), read as the ``[1,1,1,32]`` row (a ROW_MAJOR view of the contiguous
    column) and tilized: lane u = row u's global id, lanes ``rows..31`` zero.  Row u is bitwise ``resolve_greedy``
    row u; the 1-row method is untouched (its token row splats the id into every column, this one fills lane u).
    """

    if not isinstance(candidates, Qwen38GreedyCandidates):
        raise TypeError("lane greedy resolve requires Qwen38GreedyCandidates")
    rows = require_lane_count(candidates.rows, label="greedy candidate rows")
    if collective_topology != ttnn.Topology.Linear:
        raise RuntimeError("on-device greedy resolve requires Linear topology")
    mesh_contract.validate_tensor(candidates.local_indices, placement=TensorPlacement.LOCAL_PARTIAL)
    mesh_contract.validate_tensor(candidates.local_values, placement=TensorPlacement.LOCAL_PARTIAL)
    dram = ttnn.DRAM_MEMORY_CONFIG
    quad = (1, 1, rows, TP_SIZE)

    gathered_values = ttnn.all_gather(candidates.local_values, dim=3, cluster_axis=TP_AXIS, memory_config=dram)
    mesh_contract.validate_tensor(gathered_values, placement=TensorPlacement.REPLICATED)
    if _shape(gathered_values) != quad:
        raise RuntimeError(f"lane greedy value gather returned {_shape(gathered_values)}, expected {list(quad)}")
    values_fp32 = ttnn.typecast(gathered_values, ttnn.float32, memory_config=dram)
    _deallocate(gathered_values)
    tie_break = ttnn.to_layout(constants.owner_tie_break, ttnn.TILE_LAYOUT, memory_config=dram)
    ranked_tile = ttnn.subtract(values_fp32, tie_break, memory_config=dram)
    _deallocate(values_fp32, tie_break)
    ranked = ttnn.to_layout(ranked_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    _deallocate(ranked_tile)
    if _shape(ranked) != quad or ranked.dtype != ttnn.float32 or ranked.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError(f"lane greedy tie-break rows must be FP32 ROW_MAJOR {list(quad)}, got {_metadata(ranked)}")
    owner = ttnn.argmax(ranked, dim=-1, keepdim=True)
    _deallocate(ranked)
    if _shape(owner) != (1, 1, rows, 1) or owner.dtype != ttnn.uint32:
        raise RuntimeError(f"lane greedy owner argmax must be UINT32 [1,1,{rows},1], got {_metadata(owner)}")

    index_column = ttnn.reshape(candidates.local_indices, (1, 1, rows, 1))
    gathered_indices = ttnn.all_gather(index_column, dim=3, cluster_axis=TP_AXIS, memory_config=dram)
    mesh_contract.validate_tensor(gathered_indices, placement=TensorPlacement.REPLICATED)
    if _shape(gathered_indices) != quad:
        raise RuntimeError(f"lane greedy index gather returned {_shape(gathered_indices)}, expected {list(quad)}")
    index_tile = ttnn.to_layout(gathered_indices, ttnn.TILE_LAYOUT, memory_config=dram)
    _deallocate(gathered_indices)
    index_fp32 = ttnn.typecast(index_tile, ttnn.float32, memory_config=dram)
    _deallocate(index_tile)
    starts = ttnn.to_layout(constants.lm_head_vocab_starts, ttnn.TILE_LAYOUT, memory_config=dram)
    candidate_tile = ttnn.add(index_fp32, starts, memory_config=dram)
    _deallocate(index_fp32, starts)
    candidate_tokens = ttnn.to_layout(candidate_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    _deallocate(candidate_tile)
    if (
        _shape(candidate_tokens) != quad
        or candidate_tokens.dtype != ttnn.float32
        or candidate_tokens.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise RuntimeError(
            f"lane greedy candidate ids must be FP32 ROW_MAJOR {list(quad)}, got {_metadata(candidate_tokens)}"
        )

    token_column = ttnn.gather(candidate_tokens, 3, owner, memory_config=dram)
    _deallocate(candidate_tokens, owner)
    if (
        _shape(token_column) != (1, 1, rows, 1)
        or token_column.dtype != ttnn.float32
        or token_column.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise RuntimeError(
            f"lane greedy token select must be FP32 ROW_MAJOR [1,1,{rows},1], got {_metadata(token_column)}"
        )
    if rows < TILE_SIZE:
        padded = ttnn.pad(token_column, [(0, 0), (0, 0), (0, TILE_SIZE - rows), (0, 0)], value=0.0)
        _deallocate(token_column)
        token_column = padded
    # A ROW_MAJOR view of the contiguous 32-row column (fresh id, same buffer): released through the view only.
    token_lanes = ttnn.reshape(token_column, TOKEN_ROW_SHAPE)
    token_row = ttnn.to_layout(token_lanes, ttnn.TILE_LAYOUT, memory_config=dram)
    _deallocate(token_lanes)
    mesh_contract.validate_tensor(token_row, placement=TensorPlacement.REPLICATED)
    if _shape(token_row) != TOKEN_ROW_SHAPE or token_row.dtype != ttnn.float32 or token_row.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError(f"resolved lane token row must be FP32 TILE {TOKEN_ROW_SHAPE}, got {_metadata(token_row)}")
    return token_row


def _validate_exact_config(checkpoint: Qwen38Checkpoint, placement: Qwen38Placement) -> None:
    if checkpoint.config != placement.config:
        raise ValueError("checkpoint and placement configurations differ")
    config = checkpoint.config
    exact = {
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "rms_norm_eps": RMS_NORM_EPS,
    }
    for name, expected in exact.items():
        actual = getattr(config, name)
        if actual != expected:
            raise ValueError(f"Qwen3.8 model I/O requires {name}={expected}, got {actual}")
    if config.config_sha256 != CONFIG_SHA256:
        raise ValueError(f"Qwen3.8 model I/O requires config SHA-256 {CONFIG_SHA256}, got {config.config_sha256}")
    if tuple(placement.mesh_shape) != MESH_SHAPE or tuple(placement.vocab_ranges) != tuple(
        (device * LOCAL_VOCAB_SIZE, (device + 1) * LOCAL_VOCAB_SIZE) for device in range(TP_SIZE)
    ):
        raise ValueError("placement does not define the exact contiguous TP4 vocabulary ranges")


def validate_terminal_architecture(checkpoint: Qwen38Checkpoint) -> None:
    """Prove that terminal normalization belongs to the final GR mixer.

    An ordinary Qwen3/Qwen3.6-style ``model.norm`` would double-normalize this
    checkpoint.  Reject both that tensor and any missing/misshaped terminal
    mixer tensor.
    """

    present = checkpoint.weight_map
    unexpected = tuple(name for name in FORBIDDEN_STANDALONE_NORMS if name in present)
    if unexpected:
        raise ValueError(f"pinned Qwen4Exp must not have a standalone final norm: {unexpected}")
    for name, expected_shape in TERMINAL_MIXER_NAMES.items():
        metadata = checkpoint.metadata(name)
        if metadata.dtype != "BF16" or metadata.shape != expected_shape:
            raise ValueError(
                f"terminal mixer tensor {name} must be BF16 {expected_shape}, " f"got {metadata.dtype} {metadata.shape}"
            )


@dataclass(frozen=True)
class Qwen38IOCacheIdentity:
    """Pinned identity for the untied embedding and LM-head tensorbins."""

    checkpoint_revision: str
    checkpoint_config_sha256: str
    checkpoint_file_manifest_sha256: str
    checkpoint_hash_manifest_sha256: str
    tt_metal_revision: str
    ttnn_runtime_sha256: str
    mesh_shape: tuple[int, int]
    physical_ids: tuple[int, int, int, int]
    dtype: str = "BFLOAT16"
    format_version: int = IO_CACHE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.checkpoint_revision != PINNED_CHECKPOINT_REVISION:
            raise ValueError(
                f"model I/O cache requires checkpoint {PINNED_CHECKPOINT_REVISION}, got {self.checkpoint_revision}"
            )
        if self.checkpoint_config_sha256 != CONFIG_SHA256:
            raise ValueError("model I/O cache config hash is not the pinned Qwen3.8 config")
        if self.checkpoint_file_manifest_sha256 != CHECKPOINT_FILE_MANIFEST_SHA256:
            raise ValueError("model I/O cache file-manifest hash is not the pinned complete checkpoint")
        if self.checkpoint_hash_manifest_sha256 != PINNED_TENSOR_MANIFEST_SHA256:
            raise ValueError("model I/O cache tensor-manifest hash is not the pinned BF16 checkpoint")
        if not self.tt_metal_revision or len(self.tt_metal_revision) > 256:
            raise ValueError("model I/O cache requires a bounded, nonempty TT-Metal source revision")
        if not _is_sha256(self.ttnn_runtime_sha256):
            raise ValueError("model I/O cache requires the exact TTNN runtime SHA-256")
        if tuple(self.mesh_shape) != MESH_SHAPE:
            raise ValueError(f"model I/O cache requires mesh {MESH_SHAPE}, got {self.mesh_shape}")
        if (
            len(self.physical_ids) != TP_SIZE
            or len(set(self.physical_ids)) != TP_SIZE
            or any(not isinstance(device_id, int) or device_id < 0 for device_id in self.physical_ids)
        ):
            raise ValueError(f"model I/O cache requires four distinct physical IDs, got {self.physical_ids}")
        if self.dtype != "BFLOAT16" or self.format_version != IO_CACHE_FORMAT_VERSION:
            raise ValueError("model I/O cache dtype or format version is unsupported")

    @property
    def key(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _IOArtifact:
    name: str
    relative_path: str
    sha256: str
    bytes: int
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    layout: str
    placement: str
    shard_dim: int


class Qwen38IOCache:
    """Hash-verified, topology-qualified cache for model I/O weights."""

    def __init__(
        self,
        root: str | Path,
        identity: Qwen38IOCacheIdentity,
        mesh_contract: Qwen38MeshContract,
    ) -> None:
        if mesh_contract.physical_ids != identity.physical_ids:
            raise ValueError("mesh contract and model I/O cache physical ordering differ")
        self.identity = identity
        self.mesh_contract = mesh_contract
        self.root = Path(root).resolve() / identity.key
        self.manifest_path = self.root / "manifest.json"
        self.lock_path = self.root / ".conversion.lock"

    def _read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if document.get("format_version") != IO_CACHE_FORMAT_VERSION:
            raise RuntimeError("model I/O cache manifest format differs from this implementation")
        if document.get("identity") != _json_normalized(asdict(self.identity)):
            raise RuntimeError("model I/O cache manifest identity differs from this run")
        if document.get("identity_key") != self.identity.key:
            raise RuntimeError("model I/O cache identity digest is corrupt")
        artifacts = document.get("artifacts")
        if not isinstance(artifacts, dict):
            raise RuntimeError("model I/O cache manifest has no artifact map")
        return document

    def _artifact(self, name: str) -> _IOArtifact | None:
        document = self._read_manifest()
        if document is None:
            return None
        raw = document["artifacts"].get(name)
        if raw is None:
            return None
        values = dict(raw)
        values["global_shape"] = tuple(values["global_shape"])
        values["local_shape"] = tuple(values["local_shape"])
        artifact = _IOArtifact(**values)
        path = (self.root / artifact.relative_path).resolve()
        if path.parent != self.root or artifact.name != name:
            raise RuntimeError(f"model I/O cache artifact path/name is invalid for {name}")
        if not path.is_file() or path.stat().st_size != artifact.bytes or _sha256(path) != artifact.sha256:
            raise RuntimeError(f"model I/O cache artifact failed size/hash verification: {path}")
        return artifact

    def _record(self, artifact: _IOArtifact) -> None:
        document = self._read_manifest()
        if document is None:
            document = {
                "format_version": IO_CACHE_FORMAT_VERSION,
                "identity_key": self.identity.key,
                "identity": _json_normalized(asdict(self.identity)),
                "created_utc": _utc_now(),
                "artifacts": {},
            }
        serialized = _json_normalized(asdict(artifact))
        previous = document["artifacts"].get(artifact.name)
        if previous is not None and previous != serialized:
            raise RuntimeError(f"refusing to overwrite non-identical model I/O artifact {artifact.name}")
        document["artifacts"][artifact.name] = serialized
        document["updated_utc"] = _utc_now()
        _atomic_json(self.manifest_path, document)

    def load_or_create(
        self,
        *,
        name: str,
        mesh_device,
        host_factory: Callable[[], torch.Tensor],
        mapper,
        dtype,
        layout,
        memory_config,
        global_shape: tuple[int, ...],
        local_shape: tuple[int, ...],
        placement: TensorPlacement,
        shard_dim: int,
    ):
        """Load one immutable artifact or stage, hash, and publish it once."""

        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz-0123456789" for character in name):
            raise ValueError(f"invalid model I/O cache artifact name {name!r}")
        self.mesh_contract.validate_mesh(mesh_device)
        final_path = self.root / f"{name}.tensorbin"
        with _exclusive_file_lock(self.lock_path):
            artifact = self._artifact(name)
            if artifact is None and final_path.exists():
                raise RuntimeError(
                    "refusing to adopt an unmanifested model I/O artifact; classify and remove only the "
                    f"task-owned orphan before retrying: {final_path}"
                )
            if artifact is not None:
                expected = {
                    "relative_path": str(final_path.relative_to(self.root)),
                    "global_shape": global_shape,
                    "local_shape": local_shape,
                    "layout": str(layout),
                    "placement": placement.value,
                    "shard_dim": shard_dim,
                }
                actual = {
                    "relative_path": artifact.relative_path,
                    "global_shape": artifact.global_shape,
                    "local_shape": artifact.local_shape,
                    "layout": artifact.layout,
                    "placement": artifact.placement,
                    "shard_dim": artifact.shard_dim,
                }
                if actual != expected:
                    raise RuntimeError(f"model I/O artifact contract changed for {name}: {actual} != {expected}")
                tensor = ttnn.load_tensor(final_path, device=mesh_device)
            else:
                self.root.mkdir(parents=True, exist_ok=True)
                host = host_factory()
                if host.dtype != torch.bfloat16 or tuple(host.shape) != global_shape:
                    raise RuntimeError(
                        f"host model I/O tensor {name} must be BF16 {global_shape}, "
                        f"got {host.dtype} {tuple(host.shape)}"
                    )
                with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=self.root) as temporary_directory:
                    temporary_base = Path(temporary_directory) / "staged"
                    tensor = ttnn.as_tensor(
                        host.contiguous(),
                        dtype=dtype,
                        layout=layout,
                        device=mesh_device,
                        memory_config=memory_config,
                        mesh_mapper=mapper,
                        cache_file_name=temporary_base,
                    )
                    del host
                    candidates = tuple(Path(temporary_directory).glob("*.tensorbin"))
                    if len(candidates) != 1:
                        raise RuntimeError(
                            f"TTNN cache publication for {name} produced {len(candidates)} tensorbins, expected one"
                        )
                    if final_path.exists():
                        raise RuntimeError(f"model I/O destination appeared while its lock was held: {final_path}")
                    os.replace(candidates[0], final_path)
                if _shape(tensor) != local_shape:
                    raise RuntimeError(
                        f"new model I/O tensor {name} has local shape {_shape(tensor)}, expected {local_shape}"
                    )
                artifact = _IOArtifact(
                    name=name,
                    relative_path=str(final_path.relative_to(self.root)),
                    sha256=_sha256(final_path),
                    bytes=final_path.stat().st_size,
                    global_shape=global_shape,
                    local_shape=local_shape,
                    layout=str(layout),
                    placement=placement.value,
                    shard_dim=shard_dim,
                )
                self._record(artifact)

        if (
            _shape(tensor) != local_shape
            or tensor.dtype != dtype
            or tensor.layout != layout
            or tensor.memory_config() != memory_config
        ):
            raise RuntimeError(
                f"model I/O tensor {name} loaded with {_shape(tensor)} {tensor.dtype} {tensor.layout} "
                f"{tensor.memory_config()}, expected {local_shape} {dtype} {layout} {memory_config}"
            )
        self.mesh_contract.validate_tensor(tensor, placement=placement, shard_dim=shard_dim)
        return tensor


@dataclass(frozen=True)
class Qwen38ValidatedTokens:
    """Vocab-sharded local UINT32 indices derived from range-checked CPU IDs."""

    tensor: Any
    sequence_length: int
    active_vocab_shard: int | None
    active_vocab_coordinate: tuple[int, int] | None

    def __post_init__(self) -> None:
        if type(self.sequence_length) is not int or self.sequence_length <= 0:
            raise ValueError("validated token sequence length must be a positive exact integer")
        if self.sequence_length == 1:
            if type(self.active_vocab_shard) is not int or not 0 <= self.active_vocab_shard < TP_SIZE:
                raise ValueError(f"S=1 validated token owner shard must be an exact integer in [0,{TP_SIZE})")
            expected_coordinate = (0, self.active_vocab_shard)
            if (
                type(self.active_vocab_coordinate) is not tuple
                or len(self.active_vocab_coordinate) != 2
                or any(type(value) is not int for value in self.active_vocab_coordinate)
                or self.active_vocab_coordinate != expected_coordinate
            ):
                raise ValueError(
                    f"S=1 validated token owner coordinate must be logical {expected_coordinate}, "
                    f"got {self.active_vocab_coordinate!r}"
                )
        elif self.active_vocab_shard is not None or self.active_vocab_coordinate is not None:
            raise ValueError("multi-token validation cannot name one vocabulary owner")


@dataclass(frozen=True)
class Qwen38ShardedLogits:
    """Local logits plus the explicit global contiguous vocabulary mapping."""

    tensor: Any
    vocab_ranges: tuple[tuple[int, int], ...]
    global_shape: tuple[int, int, int, int]
    shard_dim: int = 3


@dataclass(frozen=True)
class Qwen38GreedyCandidates:
    """Per-coordinate local argmax and maximum; no full-vocabulary gather."""

    local_indices: Any
    local_values: Any
    rows: int
    vocab_ranges: tuple[tuple[int, int], ...]
    packed: Any = None  # fused greedy tail: the fp32 ROW_MAJOR [value | local id] row its resolve gathers
    shard_row: Any = (
        None  # fused candidate row: the shard's fp32 ROW_MAJOR [values | global ids] the sampler's gather takes
    )


@dataclass(frozen=True)
class Qwen38TTNNModelIOWeights:
    embedding: Any
    lm_head_chunks: tuple[Any, ...]
    lm_head_chunk_sizes: tuple[int, ...]
    replicated_anchor: Any
    vocab_start: Any
    vocab_ranges: tuple[tuple[int, int], ...]
    token_row: Qwen38TTNNTokenRowConstants
    # The LM-head chunks' dtype (QWEN38_DENSE_WEIGHT_DTYPE); the embedding gather table stays BF16.
    lm_head_dtype: Any = ttnn.bfloat16

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        cache: Qwen38IOCache,
        lm_head_dtype=None,
    ) -> "Qwen38TTNNModelIOWeights":
        mesh_contract.validate_mesh(mesh_device)
        if lm_head_dtype is None:
            lm_head_dtype = ttnn.bfloat16  # the production path
        lm_head_tag = dense_dtype_tag(lm_head_dtype)
        _validate_exact_config(checkpoint, placement)
        validate_terminal_architecture(checkpoint)
        if tuple(placement.physical_ids) != mesh_contract.physical_ids:
            raise ValueError("numerical placement and live mesh contract have different physical order")
        if cache.mesh_contract != mesh_contract:
            raise ValueError("model I/O cache was not constructed for this mesh contract")
        if cache.identity.checkpoint_config_sha256 != checkpoint.config.config_sha256:
            raise ValueError("model I/O cache and checkpoint config identities differ")

        for name in (EMBEDDING_NAME, LM_HEAD_NAME):
            metadata = checkpoint.metadata(name)
            if metadata.dtype != "BF16" or metadata.shape != (VOCAB_SIZE, HIDDEN_SIZE):
                raise ValueError(
                    f"checkpoint model I/O tensor {name} must be BF16 {(VOCAB_SIZE, HIDDEN_SIZE)}, "
                    f"got {metadata.dtype} {metadata.shape}"
                )
        if checkpoint.weight_map[EMBEDDING_NAME] == checkpoint.weight_map[LM_HEAD_NAME]:
            # The released tensors are untied and even reside in distinct source
            # shards.  This cheap invariant catches accidental aliasing before a
            # 1.27-GB tensor is read.
            raise ValueError("pinned input embedding and LM head must be distinct checkpoint artifacts")

        embedding_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 0))
        embedding = cache.load_or_create(
            name="embedding",
            mesh_device=mesh_device,
            host_factory=lambda: checkpoint.tensor(EMBEDDING_NAME),
            mapper=embedding_mapper,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            global_shape=(VOCAB_SIZE, HIDDEN_SIZE),
            local_shape=(LOCAL_VOCAB_SIZE, HIDDEN_SIZE),
            placement=TensorPlacement.VOCAB_SHARDED,
            shard_dim=0,
        )

        lm_head_chunks = []
        chunk_offset = 0
        for chunk_index, chunk_size in enumerate(LM_HEAD_CHUNK_COLUMNS):
            if chunk_size % TILE_SIZE:
                raise RuntimeError(f"LM-head chunk {chunk_size} is not tile aligned")

            def build_chunk(offset=chunk_offset, width=chunk_size):
                local_columns = []
                for start, _ in placement.vocab_ranges:
                    rows = checkpoint.tensor_slice(
                        LM_HEAD_NAME,
                        (slice(start + offset, start + offset + width), slice(None)),
                    )
                    if rows.dtype != torch.bfloat16 or tuple(rows.shape) != (width, HIDDEN_SIZE):
                        raise RuntimeError("LM-head checkpoint slice changed during conversion")
                    local_columns.append(rows.transpose(0, 1).contiguous())
                # Concatenating equal-width device slices before mesh sharding
                # gives coordinate d exactly local_columns[d].
                return torch.cat(local_columns, dim=1).reshape(1, 1, HIDDEN_SIZE, TP_SIZE * width)

            # DRAM width-sharded for the decode matmul program; the local column
            # range in the name orphans every chunk of a different plan.
            mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3))
            tensor = cache.load_or_create(
                # a converted chunk carries the dtype tag in its artifact name; the BF16 artifacts keep theirs
                name=f"lm-head-chunk-dram-sharded-{chunk_index:02d}-cols-{chunk_offset}-{chunk_offset + chunk_size}"
                if lm_head_dtype == ttnn.bfloat16
                else f"lm-head-chunk-dram-sharded-{lm_head_tag}-{chunk_index:02d}-cols-{chunk_offset}-{chunk_offset + chunk_size}",
                mesh_device=mesh_device,
                host_factory=build_chunk,
                mapper=mapper,
                dtype=lm_head_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, chunk_size),
                global_shape=(1, 1, HIDDEN_SIZE, TP_SIZE * chunk_size),
                local_shape=(1, 1, HIDDEN_SIZE, chunk_size),
                placement=TensorPlacement.VOCAB_SHARDED,
                shard_dim=3,
            )
            lm_head_chunks.append(tensor)
            chunk_offset += chunk_size
        if chunk_offset != LOCAL_VOCAB_SIZE:
            raise AssertionError("LM-head chunks do not cover one exact vocabulary shard")

        replicated_anchor = ttnn.from_torch(
            torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
        )
        starts = torch.arange(TP_SIZE, dtype=torch.float32).reshape(1, TP_SIZE, 1, 1) * LOCAL_VOCAB_SIZE - 1.0
        vocab_start = ttnn.from_torch(
            starts,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 1)),
        )
        mesh_contract.validate_tensor(replicated_anchor, placement=TensorPlacement.REPLICATED)
        mesh_contract.validate_tensor(vocab_start, placement=TensorPlacement.VOCAB_SHARDED, shard_dim=1)
        if _shape(replicated_anchor) != (1, 1, 1, 1) or _shape(vocab_start) != (1, 1, 1, 1):
            raise RuntimeError("model I/O anchor or vocabulary-start scalar has an unexpected local shape")

        return cls(
            embedding=embedding,
            lm_head_chunks=tuple(lm_head_chunks),
            lm_head_chunk_sizes=LM_HEAD_CHUNK_COLUMNS,
            replicated_anchor=replicated_anchor,
            vocab_start=vocab_start,
            vocab_ranges=tuple(tuple(pair) for pair in placement.vocab_ranges),
            token_row=Qwen38TTNNTokenRowConstants.build(mesh_device, mesh_contract),
            lm_head_dtype=lm_head_dtype,
        )


class Qwen38TTNNTokenEmbedding:
    """Vocab-row-sharded embedding yielding the persistent H/4 placement."""

    def __init__(
        self,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        weights: Qwen38TTNNModelIOWeights,
        *,
        tt_ccl,
        collective_topology=None,
        synchronization_policy: Qwen38TTNNEmbeddingSyncPolicy = Qwen38TTNNEmbeddingSyncPolicy.CORRECTNESS_FENCED,
    ) -> None:
        if type(synchronization_policy) is not Qwen38TTNNEmbeddingSyncPolicy:
            raise TypeError(
                "embedding synchronization_policy must be an exact Qwen38TTNNEmbeddingSyncPolicy, "
                f"got {synchronization_policy!r}"
            )
        mesh_contract.validate_mesh(mesh_device)
        if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
            raise ValueError("Qwen3.8 token embedding requires a Blackhole mesh")
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.weights = weights
        if tt_ccl is None:
            raise ValueError("Qwen3.8 token embedding requires the model-scoped TT-CCL manager")
        self.tt_ccl = tt_ccl
        self.collective_topology = collective_topology or ttnn.Topology.Linear
        self.synchronization_policy = synchronization_policy
        self._poisoned_error: BaseException | None = None
        self._poisoned_device_owners: list[Any] = []
        if self.collective_topology != ttnn.Topology.Linear:
            raise ValueError("Qwen3.8 token embedding requires Linear collective topology")
        mesh_contract.validate_tensor(weights.embedding, placement=TensorPlacement.VOCAB_SHARDED, shard_dim=0)
        if (
            _shape(weights.embedding) != (LOCAL_VOCAB_SIZE, HIDDEN_SIZE)
            or weights.embedding.dtype != ttnn.bfloat16
            or weights.embedding.layout != ttnn.ROW_MAJOR_LAYOUT
        ):
            raise ValueError("embedding weight is not an exact BF16 vocabulary-row shard")
        mesh_contract.validate_tensor(weights.replicated_anchor, placement=TensorPlacement.REPLICATED)
        mesh_contract.validate_tensor(weights.vocab_start, placement=TensorPlacement.VOCAB_SHARDED, shard_dim=1)

        # Pad every local vocabulary shard, not the global vocabulary.  The
        # inherited topology remains a dim-0 vocabulary shard.
        self.sentinel_weight = ttnn.pad(weights.embedding, [(1, 1), (0, 0)], value=0.0)
        if _shape(self.sentinel_weight) != (LOCAL_VOCAB_SIZE + 2, HIDDEN_SIZE):
            raise RuntimeError(
                f"sentinel embedding has local shape {_shape(self.sentinel_weight)}, "
                f"expected {(LOCAL_VOCAB_SIZE + 2, HIDDEN_SIZE)}"
            )
        mesh_contract.validate_tensor(
            self.sentinel_weight,
            placement=TensorPlacement.VOCAB_SHARDED,
            shard_dim=0,
        )

    @property
    def poisoned(self) -> bool:
        return getattr(self, "_poisoned_error", None) is not None

    @property
    def poisoned_device_owners(self) -> tuple[Any, ...]:
        return tuple(getattr(self, "_poisoned_device_owners", ()))

    def _require_healthy(self) -> None:
        poisoned_error = getattr(self, "_poisoned_error", None)
        if poisoned_error is not None:
            raise RuntimeError(
                "token embedding is poisoned after an asynchronous failure; tear down its owning process/mesh"
            ) from poisoned_error

    def _retain_async_failure_owners(self, error: BaseException, *owners: Any) -> None:
        if getattr(self, "_poisoned_error", None) is None:
            self._poisoned_error = error
        retained = getattr(self, "_poisoned_device_owners", None)
        if retained is None:
            retained = self._poisoned_device_owners = []
        retained_ids = {id(owner) for owner in retained}
        for owner in owners:
            if owner is None or id(owner) in retained_ids:
                continue
            retained.append(owner)
            retained_ids.add(id(owner))

    def upload_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        _diagnostic_stage_callback: Callable[[str], None] | None = None,
    ) -> Qwen38ValidatedTokens:
        """Range-check CPU IDs, localize all four shards, and upload once."""

        self._require_healthy()
        host = _localize_token_ids_on_host(input_ids)
        sequence = int(input_ids.shape[1])
        active_vocab_shard = active_vocab_coordinate = None
        physical_sequence = sequence
        if sequence == 1:
            active_vocab_shard, active_vocab_coordinate = _single_token_vocab_owner(host)
            host = _pad_single_decode_local_indices(host)
            physical_sequence = TILE_SIZE
        _diagnostic_stage(_diagnostic_stage_callback, "before-token-h2d")
        tokens = ttnn.from_torch(
            host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 1)),
        )
        # The diagnostic callback may fence here while ``host`` and ``tokens``
        # are both still strongly owned.  Production returns ownership of the
        # uploaded local indices to the model/session caller.
        _diagnostic_stage(_diagnostic_stage_callback, "after-token-h2d-enqueue")
        self.mesh_contract.validate_tensor(tokens, placement=TensorPlacement.VOCAB_SHARDED, shard_dim=1)
        expected_upload = (1, 1, physical_sequence)
        if _shape(tokens) != expected_upload:
            raise RuntimeError(f"localized token upload has local shape {_shape(tokens)}, expected {expected_upload}")
        return Qwen38ValidatedTokens(
            tensor=tokens,
            sequence_length=sequence,
            active_vocab_shard=active_vocab_shard,
            active_vocab_coordinate=active_vocab_coordinate,
        )

    def validate_token_row(self, token_row, *, label: str = "device token row") -> None:
        _validate_token_row(token_row, mesh_contract=self.mesh_contract, label=label)

    def upload_token_row(self, token_id: int):
        """Upload one range-checked token as a token row; seeds a device-token trace chain."""

        if isinstance(token_id, bool) or type(token_id) is not int or not 0 <= token_id < VOCAB_SIZE:
            raise ValueError(f"token row requires an exact integer token in [0,{VOCAB_SIZE}), got {token_id!r}")
        host = torch.zeros(TOKEN_ROW_SHAPE, dtype=torch.float32)
        host[..., 0] = float(token_id)
        token_row = ttnn.from_torch(
            host,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
        )
        self.validate_token_row(token_row, label="uploaded token row")
        return token_row

    def embed_device_token(self, token_row):
        """Hidden-sharded [1,1,1,640] embedding of a token row with token-invariant program identity.

        Every coordinate localizes the token against its own vocabulary range on
        device (out-of-range tokens clamp to the zero sentinel rows) and looks up
        its padded shard; the four mutually exclusive partials are summed instead
        of sliced.  No host upload or readback, so the whole chain is trace
        capturable.  The output matches ``__call__``'s owner-select result
        bitwise because three partials are exactly zero.
        """

        self._require_healthy()
        self.validate_token_row(token_row)
        constants = self.weights.token_row
        shifted = ttnn.subtract(token_row, constants.vocab_localize_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        localized = ttnn.clamp(shifted, min=0.0, max=float(LOCAL_VOCAB_SIZE + 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        localized_indices = ttnn.typecast(localized, ttnn.uint32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        row_major = ttnn.to_layout(localized_indices, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(shifted, localized, localized_indices)
        indices = ttnn.reshape(row_major, (1, 1, TILE_SIZE))
        expected_indices = (1, 1, TILE_SIZE)
        if (
            _shape(indices) != expected_indices
            or indices.dtype != ttnn.uint32
            or indices.layout != ttnn.ROW_MAJOR_LAYOUT
        ):
            raise RuntimeError(
                f"device-localized indices must be UINT32 ROW_MAJOR {expected_indices}, got {_metadata(indices)}"
            )
        embedded_base = ttnn.embedding(
            indices,
            self.sentinel_weight,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(row_major)
        # ttnn.embedding returns [batch, sentence, hidden] for indices of rank > 1
        # (ttnn/cpp/ttnn/operations/embedding/embedding.cpp:38-39,73); the fused
        # TILE program's 32-row allocation is the padded shape.  Same as __call__.
        embedded_padded = embedded_base
        if len(embedded_base.shape) == 3:
            embedded_padded = ttnn.unsqueeze_to_4D(embedded_base)
        expected_partial = (1, 1, TILE_SIZE, HIDDEN_SIZE)
        if (
            _shape(embedded_padded) != expected_partial
            or _padded_shape(embedded_padded) != expected_partial
            or embedded_padded.dtype != ttnn.bfloat16
            or embedded_padded.layout != ttnn.TILE_LAYOUT
        ):
            raise RuntimeError(
                f"device-token embedding partial must be BF16 TILE {expected_partial} backed by {expected_partial}, "
                f"got {_metadata(embedded_padded)} (ttnn.embedding returned {_metadata(embedded_base)})"
            )
        embedded = ttnn.reshape(embedded_padded, ttnn.Shape((1, 1, 1, HIDDEN_SIZE)), ttnn.Shape(expected_partial))
        if _shape(embedded) != (1, 1, 1, HIDDEN_SIZE) or _padded_shape(embedded) != expected_partial:
            raise RuntimeError(
                f"device-token embedding view must be [1,1,1,{HIDDEN_SIZE}] backed by {expected_partial}, "
                f"got {_metadata(embedded)}"
            )
        self.mesh_contract.mark_local_partial(
            embedded,
            replicated_reference=self.weights.replicated_anchor,
            expected_shape=(1, 1, 1, HIDDEN_SIZE),
        )
        hidden = _all_reduce_owner_select_hidden(
            embedded,
            mesh_contract=self.mesh_contract,
            replicated_reference=self.weights.replicated_anchor,
            collective_topology=self.collective_topology,
        )
        expected_hidden_padded = (1, 1, TILE_SIZE, LOCAL_HIDDEN_SIZE)
        if _padded_shape(hidden) != expected_hidden_padded:
            raise RuntimeError(
                f"device-token embedding must be backed by {expected_hidden_padded}, got {_metadata(hidden)}"
            )
        return hidden

    @staticmethod
    def host_token_rows(token_ids) -> torch.Tensor:
        """Host image of a token-rows tile (lane j = token j; ``[1,1,tiles,32]`` for 32 or 128 tokens) for
        ``copy_host_to_device_tensor`` into a chunk state's token rows."""

        token_ids = list(token_ids)
        tiles = chunk_row_tiles(len(token_ids))
        for token_id in token_ids:
            if isinstance(token_id, bool) or type(token_id) is not int or not 0 <= token_id < VOCAB_SIZE:
                raise ValueError(f"token rows require exact integer tokens in [0,{VOCAB_SIZE}), got {token_id!r}")
        return torch.tensor(token_ids, dtype=torch.float32).reshape(1, 1, tiles, TILE_SIZE)

    def validate_token_rows(self, token_rows, *, rows: int, label: str = "device token rows") -> None:
        expected = (1, 1, chunk_row_tiles(rows), TILE_SIZE)
        if _shape(token_rows) != expected or token_rows.dtype != ttnn.float32 or token_rows.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"{label} must be FP32 TILE {expected}, got {_metadata(token_rows)}")
        self.mesh_contract.validate_tensor(token_rows, placement=TensorPlacement.REPLICATED)

    def upload_token_rows(self, rows: int):
        """Upload the zero token-rows tile of a chunk state (``[1,1,tiles,32]``); the host rewrites it per chunk."""

        token_rows = ttnn.from_torch(
            torch.zeros((1, 1, chunk_row_tiles(rows), TILE_SIZE), dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
        )
        self.validate_token_rows(token_rows, rows=rows, label="uploaded token rows")
        return token_rows

    @staticmethod
    def host_verify_token_rows(token_ids) -> torch.Tensor:
        """Host image of a token row whose lanes past ``len(token_ids)`` embed to exact zeros (an MTP v2 verify pass).

        Lane j < R holds token j; the remaining lanes hold ``ZERO_EMBEDDING_TOKEN`` (-1), which every coordinate
        localizes below its range and clamps to its zero sentinel row, so :meth:`embed_device_token_rows`
        returns zero rows there.
        """

        token_ids = list(token_ids)
        if not 1 <= len(token_ids) <= CHUNK_ROWS:
            raise ValueError(f"verify token rows need 1..{CHUNK_ROWS} token ids, got {len(token_ids)}")
        for token_id in token_ids:
            if isinstance(token_id, bool) or type(token_id) is not int or not 0 <= token_id < VOCAB_SIZE:
                raise ValueError(
                    f"verify token rows require exact integer tokens in [0,{VOCAB_SIZE}), got {token_id!r}"
                )
        host = torch.full(TOKEN_ROW_SHAPE, float(ZERO_EMBEDDING_TOKEN), dtype=torch.float32)
        host[..., : len(token_ids)] = torch.tensor(token_ids, dtype=torch.float32)
        return host

    def embed_device_token_rows(self, token_row):
        """Hidden-sharded ``[1,1,rows,640]`` embedding of the lanes of a token-rows tile (one prefill chunk).

        :meth:`embed_device_token` without its one-row view: every lane is
        localized against ``vocab_localize_lanes`` (one row, broadcast over the
        tile's rows), the fused lookup's rows are the tokens, and the owner sum
        is per row.  Row j is bitwise the 1-row path's result for lane j.  No
        host upload or readback.
        """

        self._require_healthy()
        rows = _shape(token_row)[2] * TILE_SIZE
        self.validate_token_rows(token_row, rows=rows, label="device token rows")
        constants = self.weights.token_row
        shifted = ttnn.subtract(token_row, constants.vocab_localize_lanes, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        localized = ttnn.clamp(shifted, min=0.0, max=float(LOCAL_VOCAB_SIZE + 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        localized_indices = ttnn.typecast(localized, ttnn.uint32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        row_major = ttnn.to_layout(localized_indices, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(shifted, localized, localized_indices)
        indices = ttnn.reshape(row_major, (1, 1, rows))
        expected_indices = (1, 1, rows)
        if (
            _shape(indices) != expected_indices
            or indices.dtype != ttnn.uint32
            or indices.layout != ttnn.ROW_MAJOR_LAYOUT
        ):
            raise RuntimeError(
                f"device-localized index rows must be UINT32 ROW_MAJOR {expected_indices}, got {_metadata(indices)}"
            )
        embedded_base = ttnn.embedding(
            indices,
            self.sentinel_weight,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(row_major)
        embedded = ttnn.unsqueeze_to_4D(embedded_base) if len(embedded_base.shape) == 3 else embedded_base
        expected_partial = (1, 1, rows, HIDDEN_SIZE)
        if (
            _shape(embedded) != expected_partial
            or _padded_shape(embedded) != expected_partial
            or embedded.dtype != ttnn.bfloat16
            or embedded.layout != ttnn.TILE_LAYOUT
        ):
            raise RuntimeError(
                f"device-token embedding rows must be BF16 TILE {expected_partial} backed by {expected_partial}, "
                f"got {_metadata(embedded)} (ttnn.embedding returned {_metadata(embedded_base)})"
            )
        self.mesh_contract.mark_local_partial(
            embedded,
            replicated_reference=self.weights.replicated_anchor,
            expected_shape=expected_partial,
        )
        hidden = _all_reduce_owner_select_hidden(
            embedded,
            mesh_contract=self.mesh_contract,
            replicated_reference=self.weights.replicated_anchor,
            collective_topology=self.collective_topology,
            rows=rows,
        )
        expected_hidden = (1, 1, rows, LOCAL_HIDDEN_SIZE)
        if _padded_shape(hidden) != expected_hidden:
            raise RuntimeError(
                f"device-token embedding rows must be backed by {expected_hidden}, got {_metadata(hidden)}"
            )
        return hidden

    def __call__(
        self,
        validated: Qwen38ValidatedTokens,
        *,
        _diagnostic_stage_callback: Callable[[str], None] | None = None,
    ):
        self._require_healthy()
        if not isinstance(validated, Qwen38ValidatedTokens):
            raise TypeError("embedding requires Qwen38ValidatedTokens from upload_tokens")
        synchronization_policy = getattr(
            self,
            "synchronization_policy",
            Qwen38TTNNEmbeddingSyncPolicy.CORRECTNESS_FENCED,
        )
        local_indices = validated.tensor
        sequence = int(validated.sequence_length)
        physical_sequence = TILE_SIZE if sequence == 1 else sequence
        expected_indices = (1, 1, physical_sequence)
        if sequence <= 0 or _shape(local_indices) != expected_indices:
            raise ValueError(f"localized token tensor shape {_shape(local_indices)} disagrees with sequence={sequence}")
        if local_indices.dtype != ttnn.uint32 or local_indices.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("localized token tensor must be ROW_MAJOR UINT32")
        self.mesh_contract.validate_tensor(
            local_indices,
            placement=TensorPlacement.VOCAB_SHARDED,
            shard_dim=1,
        )

        expected_partial = (1, 1, physical_sequence, HIDDEN_SIZE)
        _diagnostic_stage(_diagnostic_stage_callback, "before-local-embedding")
        resident_async_result = False
        if sequence == 1:
            # The public embedding wrapper selects EmbeddingsFusedProgramFactory
            # exactly when the ROW_MAJOR index width and table width are tile
            # aligned and TILE output is requested.  Host padding establishes
            # width 32 without queuing the failing standalone RM->TILE program.
            if _padded_shape(local_indices) != expected_indices:
                raise RuntimeError(
                    f"padded decode indices have physical shape {_padded_shape(local_indices)}, "
                    f"expected {expected_indices}"
                )
            if _padded_shape(self.sentinel_weight)[-1] % TILE_SIZE:
                raise RuntimeError("sentinel embedding width is not TILE aligned for fused output")
            _diagnostic_stage(_diagnostic_stage_callback, "before-fused-padded-local-embedding")
            embedded_base = ttnn.embedding(
                local_indices,
                self.sentinel_weight,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _diagnostic_stage(_diagnostic_stage_callback, "after-fused-padded-local-embedding-enqueue")
        else:
            embedded_base = ttnn.embedding(
                local_indices,
                self.sentinel_weight,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        embedded = embedded_base
        if len(embedded_base.shape) == 3:
            embedded = ttnn.unsqueeze_to_4D(embedded_base)
        _diagnostic_stage(_diagnostic_stage_callback, "after-local-embedding-enqueue")
        if _shape(embedded) != expected_partial:
            raise RuntimeError(f"local embedding partial has shape {_shape(embedded)}, expected {expected_partial}")
        if embedded.dtype != ttnn.bfloat16 or embedded.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError("local embedding partial must be TILE BF16")
        embedded_padded = embedded
        if sequence == 1:
            # Give the collective the proven logical S=1 contract while
            # retaining the complete 32-row fused-embedding allocation.  Both
            # ``embedded_base`` and these metadata views stay strongly owned
            # until the diagnostic callback has drained the collective.
            embedded = ttnn.reshape(
                embedded_padded,
                ttnn.Shape((1, 1, 1, HIDDEN_SIZE)),
                ttnn.Shape(expected_partial),
            )
            if _shape(embedded) != (1, 1, 1, HIDDEN_SIZE) or _padded_shape(embedded) != expected_partial:
                raise RuntimeError(
                    "logical decode embedding view must be [1,1,1,2560] "
                    f"backed by {expected_partial}, got {_shape(embedded)}/{_padded_shape(embedded)}"
                )
            _diagnostic_stage(_diagnostic_stage_callback, "after-logical-sequence-view")
        expected_collective_input = (1, 1, sequence, HIDDEN_SIZE)
        self.mesh_contract.mark_local_partial(
            embedded,
            replicated_reference=self.weights.replicated_anchor,
            expected_shape=expected_collective_input,
        )
        if sequence == 1:
            # The qualified S=1 path uses the range-checked CPU token's exact
            # logical vocabulary owner.  All-gather preserves the four local
            # partials, one aligned slice selects the only non-sentinel row,
            # and mesh-partition restores the strict hidden-sharded topology.
            # The helper consumes ``embedded`` after producer-safe fences.
            _diagnostic_stage(_diagnostic_stage_callback, "before-owner-select-hidden-reducer")
            hidden_shard = _all_gather_owner_select_hidden_fallback(
                embedded,
                active_vocab_shard=validated.active_vocab_shard,
                active_vocab_coordinate=validated.active_vocab_coordinate,
                mesh_device=self.mesh_device,
                mesh_contract=self.mesh_contract,
                replicated_reference=self.weights.replicated_anchor,
                tt_ccl=self.tt_ccl,
                collective_topology=self.collective_topology,
                synchronization_policy=synchronization_policy,
                _retain_async_failure_owners=self._retain_async_failure_owners,
                _diagnostic_stage_callback=_diagnostic_stage_callback,
            )
            resident_async_result = synchronization_policy is Qwen38TTNNEmbeddingSyncPolicy.RESIDENT_ASYNC
        else:
            # Multi-token embedding retains the existing whole-line collective
            # path.  tt_all_reduce consumes its input tensor, including any
            # metadata view's backing allocation.
            _diagnostic_stage(_diagnostic_stage_callback, "before-tt-ccl-reduce-scatter")
            hidden_shard = tt_all_reduce(
                embedded,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=0,
                dim=3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.collective_topology,
            )
            _diagnostic_stage(_diagnostic_stage_callback, "after-tt-ccl-reduce-scatter-enqueue")
        try:
            if sequence == 1:
                _diagnostic_stage(_diagnostic_stage_callback, "after-owner-select-hidden-reducer-return")
            expected_output = (1, 1, sequence, LOCAL_HIDDEN_SIZE)
            if _shape(hidden_shard) != expected_output:
                raise RuntimeError(
                    f"embedding hidden reduction has shape {_shape(hidden_shard)}, expected {expected_output}"
                )
            if sequence == 1:
                expected_padded_output = (1, 1, TILE_SIZE, LOCAL_HIDDEN_SIZE)
                if _padded_shape(hidden_shard) != expected_padded_output:
                    raise RuntimeError(
                        f"single-token owner-select output has padded shape {_padded_shape(hidden_shard)}, "
                        f"expected {expected_padded_output}"
                    )
            self.mesh_contract.mark_collective_shard(
                hidden_shard,
                replicated_reference=self.weights.replicated_anchor,
                shard_dim=3,
                expected_local_shape=expected_output,
            )
            if (
                hidden_shard.dtype != ttnn.bfloat16
                or hidden_shard.layout != ttnn.TILE_LAYOUT
                or hidden_shard.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            ):
                raise RuntimeError("embedding hidden reduction must return BF16 TILE DRAM")
            return hidden_shard
        except BaseException as error:
            if resident_async_result:
                owners = [validated.tensor, hidden_shard]
                try:
                    owners.extend(ttnn.get_device_tensors(hidden_shard))
                except BaseException:
                    pass
                self._retain_async_failure_owners(error, *owners)
            raise


class Qwen38TTNNLMHead:
    """Exact untied BF16 head with local-vocabulary output and sparse greedy readback."""

    def __init__(
        self,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        weights: Qwen38TTNNModelIOWeights,
        *,
        collective_topology=None,
        decode_dram_workers_per_bank: int = 1,
    ) -> None:
        mesh_contract.validate_mesh(mesh_device)
        if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
            raise ValueError("Qwen3.8 LM head requires a Blackhole mesh")
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.weights = weights
        self.collective_topology = collective_topology or ttnn.Topology.Linear
        if sum(weights.lm_head_chunk_sizes) != LOCAL_VOCAB_SIZE:
            raise ValueError("LM-head chunks do not cover exactly one vocabulary shard")
        if len(weights.lm_head_chunks) != len(weights.lm_head_chunk_sizes):
            raise ValueError("LM-head chunk tensors and sizes differ")
        for tensor, width in zip(weights.lm_head_chunks, weights.lm_head_chunk_sizes):
            if (
                _shape(tensor) != (1, 1, HIDDEN_SIZE, width)
                or tensor.dtype != weights.lm_head_dtype
                or tensor.layout != ttnn.TILE_LAYOUT
            ):
                raise ValueError(f"LM-head chunk is not exact {weights.lm_head_dtype} [1,1,{HIDDEN_SIZE},{width}]")
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.VOCAB_SHARDED, shard_dim=3)
        mesh_contract.validate_tensor(weights.replicated_anchor, placement=TensorPlacement.REPLICATED)
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # The chunk linears run the fidelity of their weight format (decode_matmul: HiFi4 for bf16, HiFi2 for bf8,
        # LoFi for bf4); every other program keeps compute_config.
        self.weight_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, dense_math_fidelity_name(weights.lm_head_dtype)),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # One K-split storage grid serves every chunk; per-chunk program
        # configs only differ in per-core output width.
        workers = validate_decode_dram_workers(decode_dram_workers_per_bank)
        self.decode_dram_workers_per_bank = workers
        for tensor, width in zip(weights.lm_head_chunks, weights.lm_head_chunk_sizes):
            validate_dram_sharded_weight(tensor, mesh_device, HIDDEN_SIZE, width, num_workers_per_dram_bank=workers)
        chunk_configs = tuple(
            dram_sharded_matmul_configs(
                mesh_device, HIDDEN_SIZE, width, num_cores=40, num_workers_per_dram_bank=workers
            )
            for width in weights.lm_head_chunk_sizes
        )
        self.hidden_act_memory_config = chunk_configs[0][0]
        self.chunk_program_configs = tuple(program_config for _, program_config in chunk_configs)
        # QWEN38_FUSED=greedy_tail: candidates and the device resolve bind to ttnn/fused/greedy_tail at construction.
        from models.demos.blackhole.qwen38_flash_next.ttnn import fused as fused_kernels

        if fused_kernels.enabled("greedy_tail"):
            from models.demos.blackhole.qwen38_flash_next.ttnn.fused import greedy_tail as fused_greedy_tail

            self.greedy_candidates = functools.partial(fused_greedy_tail.greedy_candidates_fused, self)
            self.resolve_greedy_on_device = functools.partial(fused_greedy_tail.resolve_greedy_on_device_fused, self)
            # the lanes' epilogue: the same programs over the lane rows (the MTP rows path keeps the chain)
            self.greedy_candidates_lanes = functools.partial(fused_greedy_tail.greedy_candidates_lanes_fused, self)
            self.resolve_greedy_lanes_on_device = functools.partial(
                fused_greedy_tail.resolve_greedy_lanes_on_device_fused, self
            )
            # QWEN38_FUSED=candidate_row: the sampling server's candidate row from the scan's shard row
            if fused_kernels.enabled(fused_greedy_tail.CANDIDATE_ROW):
                self.sampling_candidates = functools.partial(fused_greedy_tail.sampling_candidates_fused, self)
        elif fused_kernels.enabled("candidate_row"):
            raise RuntimeError(
                "candidate_row folds into greedy_tail's scan: QWEN38_FUSED_OFF=greedy_tail must take it off too"
            )

    def _mark_vocab_shard(self, tensor, *, rows: int) -> None:
        expected = (1, 1, rows, LOCAL_VOCAB_SIZE)
        if _shape(tensor) != expected:
            raise RuntimeError(f"LM-head logits have local shape {_shape(tensor)}, expected {expected}")
        reference_topology = self.weights.lm_head_chunks[0].tensor_topology()
        tensor.update_tensor_topology(
            ttnn.TensorTopology(
                reference_topology.distribution_shape(),
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)],
                reference_topology.mesh_coords(),
            )
        )
        self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.VOCAB_SHARDED, shard_dim=3)

    def __call__(self, hidden_shard) -> Qwen38ShardedLogits:
        shape = _shape(hidden_shard)
        if len(shape) != 4 or shape[:2] != (1, 1) or shape[2] <= 0 or shape[3] != LOCAL_HIDDEN_SIZE:
            raise ValueError(f"LM-head input must be global-B1 [1,1,rows,{LOCAL_HIDDEN_SIZE}], got {shape}")
        if shape[2] > TILE_SIZE:
            raise ValueError(f"LM-head DRAM-sharded matmuls admit at most one tile row, got {shape[2]} rows")
        if hidden_shard.dtype != ttnn.bfloat16 or hidden_shard.layout != ttnn.TILE_LAYOUT:
            raise ValueError("LM-head input must be TILE BF16")
        self.mesh_contract.validate_tensor(
            hidden_shard,
            placement=TensorPlacement.HIDDEN_SHARDED,
            shard_dim=3,
        )
        rows = shape[2]
        full_hidden = ttnn.all_gather(
            hidden_shard,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.mesh_contract.validate_tensor(full_hidden, placement=TensorPlacement.REPLICATED)
        if _shape(full_hidden) != (1, 1, rows, HIDDEN_SIZE):
            raise RuntimeError(f"LM-head hidden gather has shape {_shape(full_hidden)}, expected [1,1,{rows},2560]")

        hidden_ws = ttnn.to_memory_config(full_hidden, self.hidden_act_memory_config)
        outputs = []
        for weight, program_config in zip(self.weights.lm_head_chunks, self.chunk_program_configs):
            chunk_ws = ttnn.linear(
                hidden_ws,
                weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=program_config,
                compute_kernel_config=self.weight_compute_config,
            )
            outputs.append(ttnn.to_memory_config(chunk_ws, ttnn.DRAM_MEMORY_CONFIG))
            _deallocate(chunk_ws)
        _deallocate(hidden_ws, full_hidden)
        logits = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if len(outputs) > 1:
            _deallocate(*outputs)
        self._mark_vocab_shard(logits, rows=rows)
        if logits.dtype != ttnn.bfloat16 or logits.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError("LM-head logits must be TILE BF16")
        return Qwen38ShardedLogits(
            tensor=logits,
            vocab_ranges=self.weights.vocab_ranges,
            global_shape=(1, 1, rows, VOCAB_SIZE),
        )

    def _validate_logits(self, logits: Qwen38ShardedLogits) -> int:
        if not isinstance(logits, Qwen38ShardedLogits):
            raise TypeError("expected Qwen38ShardedLogits with explicit vocabulary metadata")
        rows = int(logits.global_shape[2])
        if (
            logits.global_shape != (1, 1, rows, VOCAB_SIZE)
            or logits.shard_dim != 3
            or logits.vocab_ranges != self.weights.vocab_ranges
            or _shape(logits.tensor) != (1, 1, rows, LOCAL_VOCAB_SIZE)
        ):
            raise ValueError("sharded-logit shape or vocabulary ownership differs from the exact TP4 contract")
        self.mesh_contract.validate_tensor(
            logits.tensor,
            placement=TensorPlacement.VOCAB_SHARDED,
            shard_dim=3,
        )
        return rows

    def gather_full_logits(self, logits: Qwen38ShardedLogits):
        """Correctness-only full gather; ordinary greedy decode does not call it."""

        rows = self._validate_logits(logits)
        gathered = ttnn.all_gather(
            logits.tensor,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
        if _shape(gathered) != (1, 1, rows, VOCAB_SIZE):
            raise RuntimeError("full-logit correctness gather did not reconstruct the exact vocabulary")
        return gathered

    def greedy_candidates(
        self, logits: Qwen38ShardedLogits, *, values_by_gather: bool = False
    ) -> Qwen38GreedyCandidates:
        """The per-row local argmax and maximum of the vocabulary shard.

        ``values_by_gather``: the maximum of each row as a ROW_MAJOR ``ttnn.gather`` copy of the row's argmax element
        (a maximum is one of the row's own bf16 values, so this is the reduce's result bitwise), tilized for the
        resolve's all-gather: one data-movement op instead of the padded grid's pad, relayout and two reduces.  The
        default keeps the grid reduce (the 1-row production path's form).
        """

        rows = self._validate_logits(logits)
        row_major = ttnn.to_layout(logits.tensor, ttnn.ROW_MAJOR_LAYOUT)
        local_indices = ttnn.argmax(row_major, dim=-1, keepdim=False)
        if values_by_gather:
            index_column = ttnn.reshape(local_indices, (1, 1, rows, 1))
            picked = ttnn.gather(row_major, 3, index_column, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(row_major)
            if (
                _shape(picked) != (1, 1, rows, 1)
                or picked.dtype != ttnn.bfloat16
                or picked.layout != ttnn.ROW_MAJOR_LAYOUT
            ):
                raise RuntimeError(
                    f"gathered row maxima must be BF16 ROW_MAJOR [1,1,{rows},1], got {_metadata(picked)}"
                )
            local_values = ttnn.to_layout(picked, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(picked)
        else:
            _deallocate(row_major)

            # The reduction kernel parallelizes over the tall tile-row dimension.
            # Padding is only temporary and uses -inf semantics, so neither the
            # local candidate nor tie order can change.
            maxval_columns = 32
            maxval_rows = ((LOCAL_VOCAB_SIZE + maxval_columns - 1) // maxval_columns + 31) // 32 * 32
            padded_width = maxval_rows * maxval_columns
            padded = ttnn.pad(
                logits.tensor,
                [(0, 0), (0, 0), (0, 0), (0, padded_width - LOCAL_VOCAB_SIZE)],
                value=-1e30,
            )
            grid = ttnn.reshape(padded, (1, rows, maxval_rows, maxval_columns))
            partial = ttnn.max(grid, dim=-1)
            partial_rows = ttnn.reshape(partial, (1, 1, rows, maxval_rows))
            local_values = ttnn.max(partial_rows, dim=-1, keepdim=True)
            _deallocate(padded, grid, partial, partial_rows)

        self.mesh_contract.mark_local_partial(
            local_indices,
            replicated_reference=self.weights.replicated_anchor,
            expected_shape=(1, 1, rows),
        )
        self.mesh_contract.mark_local_partial(
            local_values,
            replicated_reference=self.weights.replicated_anchor,
            expected_shape=(1, 1, rows, 1),
        )
        return Qwen38GreedyCandidates(
            local_indices=local_indices,
            local_values=local_values,
            rows=rows,
            vocab_ranges=self.weights.vocab_ranges,
        )

    def resolve_greedy(self, candidates: Qwen38GreedyCandidates) -> torch.Tensor:
        """Read four local candidates and return exact global token IDs on CPU."""

        if not isinstance(candidates, Qwen38GreedyCandidates) or candidates.rows <= 0:
            raise TypeError("expected nonempty Qwen38GreedyCandidates")
        if candidates.vocab_ranges != self.weights.vocab_ranges:
            raise ValueError("greedy candidates have different vocabulary ownership")
        self.mesh_contract.validate_tensor(candidates.local_indices, placement=TensorPlacement.LOCAL_PARTIAL)
        self.mesh_contract.validate_tensor(candidates.local_values, placement=TensorPlacement.LOCAL_PARTIAL)
        composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        indices = ttnn.to_torch(candidates.local_indices, mesh_composer=composer).reshape(TP_SIZE, candidates.rows)
        values = ttnn.to_torch(candidates.local_values, mesh_composer=composer).reshape(TP_SIZE, candidates.rows)
        indices = indices.to(torch.int64)
        if torch.any(indices < 0) or torch.any(indices >= LOCAL_VOCAB_SIZE):
            raise RuntimeError("device local argmax returned an index outside its vocabulary shard")
        # torch.argmax selects the first owner on ties; contiguous ranges are
        # in mesh-coordinate order, hence this is the first global index too.
        owners = torch.argmax(values.float(), dim=0)
        starts = torch.tensor([start for start, _ in self.weights.vocab_ranges], dtype=torch.int64)
        tokens = starts[owners] + indices[owners, torch.arange(candidates.rows)]
        if torch.any(tokens < 0) or torch.any(tokens >= VOCAB_SIZE):
            raise RuntimeError("resolved greedy token is outside the pinned vocabulary")
        return tokens.reshape(1, 1, candidates.rows)

    def resolve_greedy_on_device(self, candidates: Qwen38GreedyCandidates, *, into=None):
        """Resolve the greedy token on device into one replicated token row; ``into`` (a resident TOKEN_ROW) receives a
        copy as the last op (the server's persistent token row; the fused greedy tail writes it from its program).

        Gathers the four (value, index) candidates, lowers owner ``d``'s value by
        ``d * GREEDY_TIE_BREAK_EPS`` in FP32 so equal maxima leave a unique maximum
        at the lowest owner (the lowest-index tie rule of ``torch.argmax`` in
        :meth:`resolve_greedy`, enforced by the data rather than by the device
        argmax kernel), argmaxes that row, rebases the four indices to global ids
        and copies the owner's id out of that row with ``ttnn.gather``.  The
        result is a ``TOKEN_ROW`` whose column 0 is the global id: the runner reads
        back this one tiny replicated tensor instead of two mesh-composer
        readbacks, and it is the exact input
        :meth:`Qwen38TTNNTokenEmbedding.embed_device_token` consumes.

        Exactness: the SFPU typecasts, add and multiply run in the fp32 DEST and
        are exact for every id below ``2**24``; the select is a 32-bit
        data-movement copy and the final tilize unpacks FP32 straight into DEST.
        No id may enter an FPU or reduce stage: Blackhole feeds those through the
        19-bit srcA/srcB (TF32), which keeps only 11 significant bits of an id.

        The width-four gather stays ROW_MAJOR through the tie-break, argmax and
        select so no sub-tile TILE reduction can fold tile padding into the
        result; only the final width-32 row is tilized.
        """

        if not isinstance(candidates, Qwen38GreedyCandidates) or candidates.rows != 1:
            raise TypeError("on-device greedy resolve requires single-token Qwen38GreedyCandidates")
        if candidates.vocab_ranges != self.weights.vocab_ranges:
            raise ValueError("greedy candidates have different vocabulary ownership")
        if self.collective_topology != ttnn.Topology.Linear:
            raise RuntimeError("on-device greedy resolve requires Linear topology")
        self.mesh_contract.validate_tensor(candidates.local_indices, placement=TensorPlacement.LOCAL_PARTIAL)
        self.mesh_contract.validate_tensor(candidates.local_values, placement=TensorPlacement.LOCAL_PARTIAL)
        constants = self.weights.token_row
        anchor = self.weights.replicated_anchor

        # Gather the four local maxima and pick the owner (lowest index on ties).
        # The bf16 maxima are widened to FP32 and owner d's is lowered by
        # d * GREEDY_TIE_BREAK_EPS: exact for |v| < 256 and below the spacing of
        # distinct bf16 values, so ties become a unique maximum at the lowest
        # owner and the argmax result no longer depends on the kernel's tie rule.
        # ROW_MAJOR argmax reduces the exact width-four logical row.
        gathered_values = ttnn.all_gather(
            candidates.local_values, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.mesh_contract.validate_tensor(gathered_values, placement=TensorPlacement.REPLICATED)
        if _shape(gathered_values) != (1, 1, 1, TP_SIZE):
            raise RuntimeError(f"greedy value gather returned {_shape(gathered_values)}, expected [1,1,1,{TP_SIZE}]")
        values_row_major = ttnn.to_layout(gathered_values, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(gathered_values)
        values_fp32 = ttnn.typecast(values_row_major, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(values_row_major)
        ranked = ttnn.subtract(values_fp32, constants.owner_tie_break, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(values_fp32)
        if (
            _shape(ranked) != (1, 1, 1, TP_SIZE)
            or ranked.dtype != ttnn.float32
            or ranked.layout != ttnn.ROW_MAJOR_LAYOUT
        ):
            raise RuntimeError(
                f"greedy tie-break row must be FP32 ROW_MAJOR [1,1,1,{TP_SIZE}], got {_metadata(ranked)}"
            )
        owner = ttnn.argmax(ranked, dim=-1, keepdim=True)
        _deallocate(ranked)
        if _shape(owner) != (1, 1, 1, 1) or owner.dtype != ttnn.uint32:
            raise RuntimeError(f"greedy owner argmax must be UINT32 [1,1,1,1], got {_metadata(owner)}")

        # Gather the four local indices and rebase each to its global vocabulary id.
        index_scalar = ttnn.reshape(candidates.local_indices, (1, 1, 1, 1))
        gathered_indices = ttnn.all_gather(
            index_scalar, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.mesh_contract.validate_tensor(gathered_indices, placement=TensorPlacement.REPLICATED)
        if _shape(gathered_indices) != (1, 1, 1, TP_SIZE):
            raise RuntimeError(f"greedy index gather returned {_shape(gathered_indices)}, expected [1,1,1,{TP_SIZE}]")
        index_fp32 = ttnn.typecast(gathered_indices, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(gathered_indices)
        candidate_tokens = ttnn.add(index_fp32, constants.lm_head_vocab_starts, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(index_fp32)

        # Copy the owner's id out of the row (a 32-bit data-movement gather: no FPU
        # or reduce stage, whose 19-bit srcA would drop the id's low mantissa bits)
        # and splat it into column 0.
        token = ttnn.gather(candidate_tokens, 3, owner, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(candidate_tokens, owner)
        if _shape(token) != (1, 1, 1, 1) or token.dtype != ttnn.float32 or token.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(f"greedy token select must be FP32 ROW_MAJOR [1,1,1,1], got {_metadata(token)}")
        token_wide = ttnn.multiply(token, constants.unit_column, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(token)
        token_row = ttnn.to_layout(token_wide, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(token_wide)
        self.mesh_contract.validate_tensor(token_row, placement=TensorPlacement.REPLICATED)
        if (
            _shape(token_row) != TOKEN_ROW_SHAPE
            or token_row.dtype != ttnn.float32
            or token_row.layout != ttnn.TILE_LAYOUT
        ):
            raise RuntimeError(f"resolved token row must be FP32 TILE {TOKEN_ROW_SHAPE}, got {_metadata(token_row)}")
        if into is not None:
            ttnn.copy(token_row, into)
        return token_row

    def sampling_candidates(
        self,
        logits: Qwen38ShardedLogits,
        constants: Qwen38TTNNSamplingCandidateConstants,
        *,
        into=None,
        candidates=None,
    ):
        """Optional TAIL epilogue after the greedy resolve: the host sampler's candidate row.

        ``candidates`` (the step's greedy candidates) is the fused candidate row's input (``QWEN38_FUSED=candidate_row``
        binds :func:`ttnn.fused.greedy_tail.sampling_candidates_fused` here); the chain does not read it.

        Per shard ``ttnn.topk`` over the TILE bf16 logits (k = ``SAMPLING_CANDIDATES_PER_DEVICE``,
        sorted descending: bf16 values and UINT16 local ids, the stock path without an
        ``indices_tensor``, whose label copy costs milliseconds), both widened to FP32 on
        the SFPU (exact: bf16 -> fp32, ids < 2**16) and the ids rebased by the shard's
        first global id; the pack ``[values | global ids]`` is untilized to a padding-free
        ROW_MAJOR row and the four shards' packs are all-gathered once.  The row is copied
        into ``into``, a trace-stable buffer the host reads: ``constants.readback_row`` by
        default (the 1-row TAIL's).  No id enters an FPU or reduce stage (the TF32 rule of
        :meth:`resolve_greedy_on_device`): there is no reduce at all, so every lane of the
        row is bitwise.  The softmax normalizer is not read back: with ``top_k`` at most k
        per shard the host's filters are exact over the row (``ttnn/sampling.py``), and the
        denominator lanes cost more than the whole tail is allowed to.

        ``rows`` in 1 .. 32 (the MTP verify head's k + 1 rows take a ``[1,1,rows,256]``
        ``into``): the same ops per row, row j of the result the 1-row row of logits row j.
        """

        rows = self._validate_logits(logits)
        if not 1 <= rows <= TILE_SIZE:
            raise TypeError(f"sampling candidates require 1..{TILE_SIZE}-row logits, got {rows} rows")
        if self.collective_topology != ttnn.Topology.Linear:
            raise RuntimeError("sampling candidates require Linear topology")
        constants.validate(self.mesh_contract)
        k = SAMPLING_CANDIDATES_PER_DEVICE
        row_shape = (1, 1, rows, SAMPLING_CANDIDATE_ROW_SHAPE[3])
        target = constants.readback_row if into is None else into

        def require(tensor, shape, dtype, layout, label) -> None:
            if _shape(tensor) != shape or tensor.dtype != dtype or tensor.layout != layout:
                raise RuntimeError(f"{label} must be {dtype} {layout} {shape}, got {_metadata(tensor)}")

        require(target, row_shape, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, "sampling candidates readback")
        local_values, local_ids = ttnn.topk(logits.tensor, k, dim=-1, largest=True, sorted=True)
        require(local_values, (1, 1, rows, k), ttnn.bfloat16, ttnn.TILE_LAYOUT, "sampling top-k values")
        require(local_ids, (1, 1, rows, k), ttnn.uint16, ttnn.TILE_LAYOUT, "sampling top-k ids")
        values = ttnn.typecast(local_values, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(local_values)
        ids_fp32 = ttnn.typecast(local_ids, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(local_ids)
        global_ids = ttnn.add(ids_fp32, constants.shard_vocab_start, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(ids_fp32)
        require(global_ids, (1, 1, rows, k), ttnn.float32, ttnn.TILE_LAYOUT, "sampling global ids")
        packed = ttnn.concat([values, global_ids], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(values, global_ids)
        pack = ttnn.to_layout(packed, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(packed)
        require(pack, (1, 1, rows, 2 * k), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, "sampling shard pack")
        row = ttnn.all_gather(pack, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(pack)
        require(row, row_shape, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, "sampling candidate row")
        self.mesh_contract.validate_tensor(row, placement=TensorPlacement.REPLICATED)
        ttnn.copy(row, target)
        return row

    def resolve_greedy_rows_on_device(self, candidates: Qwen38GreedyCandidates):
        """:meth:`resolve_greedy_on_device` for ``rows`` candidate rows (1 .. 32): one FP32 ROW_MAJOR ``[1,1,1,rows]``
        row whose lane j is the global greedy id of row j (the MTP v2 verify pass's per-row argmaxes).

        The same arithmetic per row (the owner tie-break in FP32, a width-four ROW_MAJOR argmax, the rebase and
        a 32-bit ``ttnn.gather`` copy of the owner's id), then one ROW_MAJOR transpose of the ``[1,1,rows,1]`` id
        column into the lane row the accept logic reads (a data-movement copy: no id enters an FPU or reduce
        stage).  Fewer rows are the 32-row form's first rows, bitwise.
        """

        if not isinstance(candidates, Qwen38GreedyCandidates) or not 1 <= candidates.rows <= TILE_SIZE:
            raise TypeError(f"on-device greedy row resolve requires 1..{TILE_SIZE}-row Qwen38GreedyCandidates")
        if candidates.vocab_ranges != self.weights.vocab_ranges:
            raise ValueError("greedy candidates have different vocabulary ownership")
        if self.collective_topology != ttnn.Topology.Linear:
            raise RuntimeError("on-device greedy resolve requires Linear topology")
        self.mesh_contract.validate_tensor(candidates.local_indices, placement=TensorPlacement.LOCAL_PARTIAL)
        self.mesh_contract.validate_tensor(candidates.local_values, placement=TensorPlacement.LOCAL_PARTIAL)
        constants = self.weights.token_row
        dram = ttnn.DRAM_MEMORY_CONFIG
        rows = candidates.rows

        gathered_values = ttnn.all_gather(candidates.local_values, dim=3, cluster_axis=TP_AXIS, memory_config=dram)
        self.mesh_contract.validate_tensor(gathered_values, placement=TensorPlacement.REPLICATED)
        if _shape(gathered_values) != (1, 1, rows, TP_SIZE):
            raise RuntimeError(
                f"greedy value gather returned {_shape(gathered_values)}, expected [1,1,{rows},{TP_SIZE}]"
            )
        values_row_major = ttnn.to_layout(gathered_values, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        _deallocate(gathered_values)
        values_fp32 = ttnn.typecast(values_row_major, ttnn.float32, memory_config=dram)
        _deallocate(values_row_major)
        ranked = ttnn.subtract(values_fp32, constants.owner_tie_break, memory_config=dram)
        _deallocate(values_fp32)
        if (
            _shape(ranked) != (1, 1, rows, TP_SIZE)
            or ranked.dtype != ttnn.float32
            or ranked.layout != ttnn.ROW_MAJOR_LAYOUT
        ):
            raise RuntimeError(
                f"greedy tie-break rows must be FP32 ROW_MAJOR [1,1,{rows},{TP_SIZE}], got {_metadata(ranked)}"
            )
        owner = ttnn.argmax(ranked, dim=-1, keepdim=True)
        _deallocate(ranked)
        if _shape(owner) != (1, 1, rows, 1) or owner.dtype != ttnn.uint32:
            raise RuntimeError(f"greedy owner argmax must be UINT32 [1,1,{rows},1], got {_metadata(owner)}")

        index_column = ttnn.reshape(candidates.local_indices, (1, 1, rows, 1))
        gathered_indices = ttnn.all_gather(index_column, dim=3, cluster_axis=TP_AXIS, memory_config=dram)
        self.mesh_contract.validate_tensor(gathered_indices, placement=TensorPlacement.REPLICATED)
        if _shape(gathered_indices) != (1, 1, rows, TP_SIZE):
            raise RuntimeError(
                f"greedy index gather returned {_shape(gathered_indices)}, expected [1,1,{rows},{TP_SIZE}]"
            )
        index_fp32 = ttnn.typecast(gathered_indices, ttnn.float32, memory_config=dram)
        _deallocate(gathered_indices)
        candidate_tokens = ttnn.add(index_fp32, constants.lm_head_vocab_starts, memory_config=dram)
        _deallocate(index_fp32)
        token_column = ttnn.gather(candidate_tokens, 3, owner, memory_config=dram)
        _deallocate(candidate_tokens, owner)
        if (
            _shape(token_column) != (1, 1, rows, 1)
            or token_column.dtype != ttnn.float32
            or token_column.layout != ttnn.ROW_MAJOR_LAYOUT
        ):
            raise RuntimeError(
                f"greedy token select must be FP32 ROW_MAJOR [1,1,{rows},1], got {_metadata(token_column)}"
            )
        token_lanes = ttnn.transpose(token_column, 2, 3, memory_config=dram)
        _deallocate(token_column)
        self.mesh_contract.validate_tensor(token_lanes, placement=TensorPlacement.REPLICATED)
        if (
            _shape(token_lanes) != (1, 1, 1, rows)
            or token_lanes.dtype != ttnn.float32
            or token_lanes.layout != ttnn.ROW_MAJOR_LAYOUT
        ):
            raise RuntimeError(
                f"resolved token lanes must be FP32 ROW_MAJOR [1,1,1,{rows}], got {_metadata(token_lanes)}"
            )
        return token_lanes

    def greedy_candidates_lanes(self, logits: Qwen38ShardedLogits) -> Qwen38GreedyCandidates:
        """The lanes' per-row candidates (rows = lanes, row u = lane u): the chain's :meth:`greedy_candidates`; the
        fused greedy tail binds its rows form here at construction (the 1-row and MTP rows callers are untouched)."""

        return self.greedy_candidates(logits)

    def resolve_greedy_lanes_on_device(self, candidates: Qwen38GreedyCandidates):
        """:func:`resolve_greedy_lanes_on_device` with this head's constants: lane u of the token row = row u's id."""

        if candidates.vocab_ranges != self.weights.vocab_ranges:
            raise ValueError("greedy candidates have different vocabulary ownership")
        return resolve_greedy_lanes_on_device(
            candidates,
            constants=self.weights.token_row,
            mesh_contract=self.mesh_contract,
            collective_topology=self.collective_topology,
        )

    def greedy_token(self, logits: Qwen38ShardedLogits) -> torch.Tensor:
        """Convenience path that transfers candidates, never full logits."""

        candidates = self.greedy_candidates(logits)
        try:
            return self.resolve_greedy(candidates)
        finally:
            _deallocate(candidates.local_indices, candidates.local_values)


class Qwen38TTNNModelIO:
    """Convenience owner for the exact embedding and LM-head modules."""

    def __init__(
        self,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        weights: Qwen38TTNNModelIOWeights,
        *,
        tt_ccl,
        collective_topology=None,
        synchronization_policy: Qwen38TTNNEmbeddingSyncPolicy = Qwen38TTNNEmbeddingSyncPolicy.CORRECTNESS_FENCED,
        decode_dram_workers_per_bank: int = 1,
    ) -> None:
        self.embedding = Qwen38TTNNTokenEmbedding(
            mesh_device,
            mesh_contract,
            weights,
            tt_ccl=tt_ccl,
            collective_topology=collective_topology,
            synchronization_policy=synchronization_policy,
        )
        self.lm_head = Qwen38TTNNLMHead(
            mesh_device,
            mesh_contract,
            weights,
            collective_topology=collective_topology,
            decode_dram_workers_per_bank=decode_dram_workers_per_bank,
        )
