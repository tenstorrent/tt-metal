# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fixed-row true-global Qwen4Exp MoE on a 1x4 Blackhole mesh.

The fused all-to-all dispatch path cannot represent one interactive token on
EP4.  This module instead replicates exactly one logical row set, runs the 128
resident BF4_B experts independently at each coordinate with
``local_combine=True``, applies the real normalized top-10 scores, adds the
separately and dynamically gated shared-expert partial, and reduce-scatters the
sum over the expert-parallel axis.  Ordinary decode uses one row.  The fixed
target verifier uses five rows (the current token plus four drafts).

The TTNN FullLocal source admits non-tile token counts, but the five-row path is
hardware-unproven on Blackhole while tt-metal issue #50038 remains relevant to
MoE numerics.  Static/source-contract validation is not a numerical claim.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import INDEX_SHA256, Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.moe import Qwen38MoEWeights
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROWS,
    LONG_CHUNK_ROWS,
    MAX_LANES,
    MESH_SHAPE,
    Qwen38MeshContract,
    TensorPlacement,
    is_slab_rows,
    replicate_tensor_2d_mesh_mapper,
    same_buffer,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import (
    dense_dtype_tag,
    dense_math_fidelity_name,
    dense_weight_name,
    dram_sharded_matmul_configs,
    dram_sharded_row_tiles,
    dram_sharded_weight_memory_config,
    prefill_matmul_program_config,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.prefill_dense import Qwen38TTNNPrefillDense, prefill_linear
from models.tt_transformers.tt.ccl import tt_all_reduce

HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 640
ROUTED_EXPERTS = 512
EXPERTS_PER_DEVICE = 128
TOP_K = 10
LOCAL_COMBINE_AXIS = 0
MOE_LOCAL_OUTPUT_ENV = "QWEN38_MOE_LOCAL_OUTPUT"
MOE_SLAB_ONE_CALL_ENV = "QWEN38_MOE_SLAB_ONE_CALL"
MOE_SLAB_RINGS_ENV = "QWEN38_MOE_SLAB_RINGS"
# The one-call slab's weight stream: 3 = three rings of cores splitting the chunks, each reading the slices of the
# experts it owns (the default since 2026-09-26: bitwise on the 4-chip line with the ring exchange's backpressure
# credit in place, twice, and about 42,500 device-checked calls bitwise on one die; 2.2 percent less TTFT than two
# rings at 32k); 2 = two rings (the default of 2026-09-25); 0 = the op's one-ring three-slot stream; 1 = the replay
# ring (one ring, each expert's slice read once per slab). Nothing is refused: the three-ring nondeterminism of
# 2026-09-25 was the ring exchange's missing backpressure, fixed the same day.
MOE_SLAB_RINGS_DEFAULT = 3
MOE_SLAB_RINGS_ADMITTED = (0, 1, 2, 3)
MOE_SLAB_RINGS_REFUSED: dict[int, str] = {}
# The one-tile rows forms' (rows <= 32: decode, the MTP verify rows, the lanes, the 32-row chunk) moe_compute launch
# form.  rings2 (the default since 2026-09-27; measured on the 1x4 p150 line: per-launch 42.0 + 10.87 n -> 36.1 + 7.39 n us
# over n distinct local experts, the combine pages bitwise): the local output path with two rings.  fulllocal (the
# form the pins were first taken with, the documented fallback): the streaming ring with the 4-core fused local combine.  localoutput:
# the op's local output path on the same streaming ring (dm1 writes each expert's rows straight into the
# [10, rows, 2560] buffer, no combine kernels: 12 cores).  replay / rings2 / rings3: the local output path with
# prefill_rings 1 / 2 / 3 -- the replay ring reads each expert's slice once; R rings are R x 8 ring cores, each ring
# reading the slices of the experts it owns (a 1-chunk expert goes whole to the least-loaded ring), so the DRAM
# banks have R readers and the compute R rings.  The 128-row chunk and the slab keep their own forms.
MOE_ROWS_FORM_ENV = "QWEN38_MOE_ROWS_FORM"
MOE_ROWS_FORM_DEFAULT = "rings2"
# form -> (local output path, prefill_rings passed to the op; None = the op's streaming ring)
MOE_ROWS_FORMS: dict[str, tuple[bool, int | None]] = {
    "fulllocal": (False, None),
    "localoutput": (True, None),
    "replay": (True, 1),
    "rings2": (True, 2),
    "rings3": (True, 3),
}
# The one-call slab's weighted reduce runs in blocks of this many rows: the fused reduce keeps one score table per
# row tile in L1 (512 rows admitted bitwise the 128-row form, 1024 refused) and the whole [10, rows, 2560] page set
# tilized at once would not fit L1.
SLAB_REDUCE_BLOCK_ROWS = 512
EP_AXIS = 1
TARGET_VERIFIER_ROWS = 5
PREFILL_CHUNK_ROWS = CHUNK_ROWS
LONG_PREFILL_CHUNK_ROWS = LONG_CHUNK_ROWS
SUPPORTED_ROWS = (
    *range(1, MAX_LANES + 1),
    LONG_PREFILL_CHUNK_ROWS,
)  # 1 = decode, 5 = MTP verify, 32 = chunk or B lanes, 128
# moe_compute processes the tokens of one expert in 32-token chunks and admits at most
# TOKEN_SIZE x num_data_parallel_cores x output_height_shard_dim tokens per call; num_data_parallel_cores is
# the largest d <= 4 dividing both the hidden tile count (80) and the live DRAM bank count (the matmul ring).
MOE_COMPUTE_TOKEN_SIZE = 32
MOE_COMPUTE_HIDDEN_TILES = HIDDEN_SIZE // 32


def moe_local_output_enabled() -> bool:
    """``QWEN38_MOE_LOCAL_OUTPUT=1``: ``moe_compute`` writes each expert's token rows straight into the ``[10, rows,
    2560]`` buffer (its LocalOutput path on the degenerate axis: 12 worker cores, no combine kernels, the rows of the
    experts this device does not hold zero-filled by the op); unset or 0, the fused local combine
    (``local_combine=True``) the pins were taken with."""

    value = os.environ.get(MOE_LOCAL_OUTPUT_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{MOE_LOCAL_OUTPUT_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


def moe_slab_one_call_enabled() -> bool:
    """``QWEN38_MOE_SLAB_ONE_CALL`` (unset = 1): a slab instance routes all of its rows through ONE ``moe_compute``
    call on the op's local output path (each expert's tokens packed into 32-row chunks once per slab instead of once
    per 128-row block: 223 chunks per layer per device against 745 on the captured natural-text routing; the rows
    nobody owns are not zero-filled, the buffer being zero at allocation and every slot it ever holds a finite expert
    output the weighted reduce multiplies by an exact 0) and reduces the ``[10, rows, 2560]`` pages in
    ``SLAB_REDUCE_BLOCK_ROWS``-row blocks. ``0`` restores the 16 x 128-row blocks the pins were taken with; the one
    call is bitwise those blocks on the 4-chip line (the acceptance records, the agreement rows and the long-prompt
    completions identical at three heads). Scoped to slab rows: decode and the 32-/128-row chunks keep their forms."""

    value = os.environ.get(MOE_SLAB_ONE_CALL_ENV, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{MOE_SLAB_ONE_CALL_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


def moe_slab_prefill_rings() -> int:
    """``QWEN38_MOE_SLAB_RINGS``: the one-call slab's ``prefill_rings`` (3 = three rings of cores splitting the
    chunks, each ring reading the slices of the experts it owns -- the default when unset; 2 = two rings; 0 = the
    op's one-ring 3-slot weight stream; 1 = the replay ring, one ring with each expert's weight slice read from DRAM
    once per slab; ``MOE_SLAB_RINGS_ADMITTED``). A count in ``MOE_SLAB_RINGS_REFUSED`` (none today) is refused with
    its reason. Only the one-call slab reads the switch; the chunk forms and decode never do."""

    value = os.environ.get(MOE_SLAB_RINGS_ENV, str(MOE_SLAB_RINGS_DEFAULT))
    for refused, reason in MOE_SLAB_RINGS_REFUSED.items():
        if value == str(refused):
            raise ValueError(f"{MOE_SLAB_RINGS_ENV}={value} is refused: {reason}")
    if value not in tuple(str(v) for v in MOE_SLAB_RINGS_ADMITTED):
        raise ValueError(f"{MOE_SLAB_RINGS_ENV} must be one of {MOE_SLAB_RINGS_ADMITTED}, got {value!r}")
    return int(value)


def moe_rows_form() -> str:
    """``QWEN38_MOE_ROWS_FORM`` (unset = rings2): the one-tile rows forms' moe_compute launch form
    (``MOE_ROWS_FORMS``).  Read once per instance at construction; a value outside the table ends the process there."""

    value = os.environ.get(MOE_ROWS_FORM_ENV, MOE_ROWS_FORM_DEFAULT)
    if value not in MOE_ROWS_FORMS:
        raise ValueError(f"{MOE_ROWS_FORM_ENV} must be one of {tuple(MOE_ROWS_FORMS)}, got {value!r}")
    return value


def admit_slab_moe_switches() -> tuple[bool, int]:
    """Read both slab MoE switches once, before any device is touched: the server calls this when ``--prefill-slab``
    is given (a refused ``QWEN38_MOE_SLAB_RINGS``, three rings today, ends the process at its start instead of at the
    first slab forward 79 s into the warm pass), and every slab MoE instance calls it at construction. Returns
    ``(one_call, rings)``."""

    one_call = moe_slab_one_call_enabled()
    rings = moe_slab_prefill_rings() if one_call else 0
    return one_call, rings


def moe_compute_output_height_shard_dim(rows: int, *, matmul_ring_size: int) -> int:
    """The smallest ``output_height_shard_dim`` under which ``moe_compute`` admits ``rows`` tokens (1 for every
    form up to 32 rows; the 128-row form needs 1 on an 8-bank part and 4 on a 7-bank part). A slab's rows only ever
    go through one call on the local output path, which stages nothing in the combine cores' L1 and admits any
    token count at 1."""

    if is_slab_rows(rows):
        return 1
    data_parallel_cores = max(d for d in range(1, 5) if MOE_COMPUTE_HIDDEN_TILES % d == 0 and matmul_ring_size % d == 0)
    return -(-rows // (MOE_COMPUTE_TOKEN_SIZE * data_parallel_cores))


# Both multi-row counts have run moe_compute on this model's silicon and matched the rows-1 path
# bitwise (the rows-1 path is the production decode; 5 = the MTP verifier, 32 = the prefill chunk share
# one code path past rows == 1): rows 32 vs 32 one-row calls on all 48 real layers, 1536/1536 rows
# (2026-09-03, prefill stage-1 chunk discriminator); rows 5 vs one-row in the MTP-v2 step-1
# discriminator and 245/245 rows in the MTP-v2 numerics gates (2026-09-03/04).  The 128-row form ran on
# silicon 2026-09-05/06: its per-tile routed stream (four moe_compute calls of 32 tokens) bitwise the 32-row
# chunk on 48 layers, and one moe_compute call of 128 tokens bitwise the per-tile calls page for page (the
# moe_compute 128 micro-test, 2026-09-06).
ROWS5_HARDWARE_PROVEN = True
ROWS32_HARDWARE_PROVEN = True
ROWS128_HARDWARE_PROVEN = True
# The 128-row chunk's routed stream: one moe_compute call of 128 tokens, so the union of the chunk's experts
# streams from DRAM once instead of once per row tile.  32 selects the per-tile form (four calls of the 32-row
# program; the micro-tests' oracle).  Until 2026-09-06 the 128-token call's rows past the first tile came back
# wrong: the fault was the weighted reduce (deepseek_moe_fast_reduce_nc_fused built one score tile for token
# rows 0..31 and scaled every row tile with it), fixed in this tree's tt-metal sources; moe_compute itself was right.
LONG_CHUNK_ROUTED_TOKENS_PER_CALL = LONG_PREFILL_CHUNK_ROWS


def routed_tokens_per_call_for(rows: int) -> int:
    """The token count of one ``moe_compute`` call of a ``rows``-row instance by default: the rows themselves up to
    32, for the 128-row chunk (``LONG_CHUNK_ROUTED_TOKENS_PER_CALL``), the whole slab for a slab (the one-call
    default) or 128-row blocks of it under ``QWEN38_MOE_SLAB_ONE_CALL=0``."""

    if is_slab_rows(rows) and moe_slab_one_call_enabled():
        return rows
    if rows == LONG_PREFILL_CHUNK_ROWS or is_slab_rows(rows):
        return LONG_CHUNK_ROUTED_TOKENS_PER_CALL
    return min(rows, CHUNK_ROWS)


BLACKHOLE_MOE_NUMERIC_ISSUE = "https://github.com/tenstorrent/tt-metal/issues/50038"
MOE_STAGE_FENCES = (
    "all-gather-hidden",
    "route",
    "shared-partial",
    "routed-partial",
    "local-add-mark",
    "final-all-reduce",
)
MOE_PHASE_STAGES = (
    "hidden-all-gather",
    "router-logits",
    "router-topk",
    "shared-partial",
    "routed-dispatch",
    "moe-compute-launch",
    "selective-reduce",
    "partial-combine",
    "output-reduce-scatter",
    "output-release",
)


class Qwen38TTNNMoESyncPolicy(str, Enum):
    """Host-drain policy for the six named MoE dependency boundaries.

    ``CORRECTNESS_FENCED`` is the fail-closed default for direct construction
    and the single-slot weight streamer.  ``RESIDENT_ASYNC`` is selected only
    by the resident builder: expert weights then remain owned for the complete
    graph lifetime, and every temporary is released only after its final
    same-command-queue consumer has been enqueued.  The latter matches the
    traced Qwen3.6 decode schedule without weakening exception ownership.
    """

    CORRECTNESS_FENCED = "correctness-fenced"
    RESIDENT_ASYNC = "resident-async"


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(item) for item in tensor.shape)


def _tensor_key(tensor) -> tuple[str, int]:
    return (str(tensor.device()), int(tensor.tensor_id))


def _deallocate(*tensors) -> None:
    for tensor in tensors:
        if tensor is not None:
            ttnn.deallocate(tensor)


def _ignore_phase(_phase: str) -> None:
    return None


@dataclass(frozen=True)
class Qwen38TTNNMoERowContract:
    """Exact logical shapes for ordinary decode (1 row), target verification (5) or a prefill chunk (32).

    ``admitted_rows`` is ``SUPPORTED_ROWS`` unless the caller admits another row count for this instance alone
    (an explicit diagnostic override, e.g. the k = 5 verifier's 6 rows); the module's admission and proof flags
    are untouched by it.
    """

    rows: int = 1
    admitted_rows: tuple[int, ...] = SUPPORTED_ROWS

    def __post_init__(self) -> None:
        if not isinstance(self.admitted_rows, tuple) or not self.admitted_rows:
            raise ValueError(f"MoE admitted rows must be a non-empty tuple, got {self.admitted_rows!r}")
        for count in self.admitted_rows:
            if (
                isinstance(count, bool)
                or type(count) is not int
                or not (1 <= count <= LONG_PREFILL_CHUNK_ROWS or is_slab_rows(count))
            ):
                raise ValueError(
                    f"MoE admitted rows must be ints in [1, {LONG_PREFILL_CHUNK_ROWS}] or a slab row count, "
                    f"got {self.admitted_rows!r}"
                )
        if type(self.rows) is not int or self.rows not in self.admitted_rows:
            raise ValueError(f"MoE rows must be exactly one of {self.admitted_rows}, got {self.rows!r}")

    @property
    def hidden_sharded(self) -> tuple[int, int, int, int]:
        return (1, 1, self.rows, HIDDEN_SIZE // 4)

    @property
    def full_hidden(self) -> tuple[int, int, int, int]:
        return (1, 1, self.rows, HIDDEN_SIZE)

    @property
    def routing(self) -> tuple[int, int, int, int]:
        return (1, 1, self.rows, TOP_K)

    @property
    def moe_sparse_input(self) -> tuple[int, ...]:
        # Preserve the existing B=1 API/byte path.  For several rows, expose the
        # row count in dimension 1 because moe_compute counts dims 0 * 1.
        if self.rows == 1:
            return self.full_hidden
        return (1, self.rows, HIDDEN_SIZE)

    @property
    def moe_routing(self) -> tuple[int, int, int]:
        return (1, self.rows, TOP_K)

    @property
    def routing_shard(self) -> tuple[int, int]:
        return (self.rows, TOP_K)

    @property
    def local_combine(self) -> tuple[int, int, int]:
        return (TOP_K, self.rows, HIDDEN_SIZE)

    @property
    def fast_reduce_input(self) -> tuple[int, int, int, int]:
        return (TOP_K, 1, self.rows, HIDDEN_SIZE)

    @property
    def fast_reduce_scores(self) -> tuple[int, int, int, int]:
        # deepseek_moe_fast_reduce_nc_fused reads score dim 0 as tokens.
        return (self.rows, 1, 1, TOP_K)

    @property
    def output_sharded(self) -> tuple[int, int, int, int]:
        return self.hidden_sharded

    @property
    def row_tiles(self) -> int:
        """Row tiles per DRAM-sharded linear call: one up to 32 rows, four for the long prefill chunk."""

        return -(-self.rows // CHUNK_ROWS)


def allocate_local_combine_output(mesh_device, mesh_contract: Qwen38MeshContract, rows: int):
    """The ``[10, rows, 2560]`` ROW_MAJOR BF16 local-combine buffer ``moe_compute`` writes (replicated: every
    device combines its own experts); one per row instance, one shared by the 48 long-chunk instances, or the
    one-call slab's whole page (``rows`` = the slab rows under ``QWEN38_MOE_SLAB_ONE_CALL=1``: admitted here the way
    the slab's layer instances admit them)."""

    contract = Qwen38TTNNMoERowContract(rows, admitted_rows=SUPPORTED_ROWS + ((rows,) if is_slab_rows(rows) else ()))
    tensor = ttnn.from_torch(
        torch.zeros(contract.local_combine, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
    )
    mesh_contract.validate_tensor(tensor, placement=TensorPlacement.LOCAL_PARTIAL)
    return tensor


def _expert_owner_mapping() -> torch.Tensor:
    """Return replicated routing metadata for four disjoint 128-expert shards."""

    owners = torch.arange(ROUTED_EXPERTS, dtype=torch.int32) // EXPERTS_PER_DEVICE
    if tuple(torch.bincount(owners, minlength=4).tolist()) != (128, 128, 128, 128):
        raise AssertionError("routed expert ownership must be exactly 128 experts per device")
    return owners.unsqueeze(0).repeat(4, 1).contiguous()


def _layer_cache_dir(
    root: str | Path,
    checkpoint: Qwen38Checkpoint,
    mesh_contract: Qwen38MeshContract,
    tt_metal_sha: str,
    namespace: str,
    layer_index: int,
) -> Path:
    if namespace not in {"backbone", "mtp"}:
        raise ValueError(f"unsupported MoE namespace {namespace!r}")
    if layer_index < 0:
        raise ValueError("layer index must be nonnegative")
    if len(tt_metal_sha) != 40 or any(character not in "0123456789abcdef" for character in tt_metal_sha):
        raise ValueError(f"tt_metal_sha must be a lowercase 40-hex revision, got {tt_metal_sha!r}")
    physical = "-".join(str(value) for value in mesh_contract.physical_ids)
    path = (
        Path(root).resolve()
        / "moe-small"
        / f"index-{INDEX_SHA256}"
        / f"config-{checkpoint.config.config_sha256}"
        / f"tt-metal-{tt_metal_sha}"
        / f"mesh-1x4-physical-{physical}"
        / namespace
        / f"layer-{layer_index:02d}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class Qwen38TTNNMoEWeights:
    """Small resident tensors; streamed routed BF4 tensors are separate."""

    router: Any
    shared_gate: Any
    shared_up: Any
    shared_down: Any
    shared_scalar_gate: Any
    shared_gate_up_scalar: Any = None  # [gate | up | scalar | 0] per device, only when the fused shared expert is on
    shared_dtype: Any = (
        ttnn.bfloat16
    )  # the shared expert weights' dtype (QWEN38_DENSE_WEIGHT_DTYPE); the router stays BF16

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        cache_root: str | Path,
        *,
        layer_index: int,
        namespace: str = "backbone",
        tt_metal_sha: str,
        shared_dtype=None,
    ) -> "Qwen38TTNNMoEWeights":
        mesh_contract.validate_mesh(mesh_device)
        if shared_dtype is None:
            shared_dtype = ttnn.bfloat16  # the production path
        dense_dtype_tag(shared_dtype)
        if namespace == "backbone":
            source = Qwen38MoEWeights(checkpoint, placement, layer_index=layer_index)
        elif namespace == "mtp":
            source = Qwen38MoEWeights(checkpoint, placement, mtp_layer_index=layer_index)
        else:
            raise ValueError(f"unsupported MoE namespace {namespace!r}")

        cache_dir = _layer_cache_dir(
            cache_root,
            checkpoint,
            mesh_contract,
            tt_metal_sha,
            namespace,
            layer_index,
        )
        output_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3))
        input_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 2))
        replicate_mapper = replicate_tensor_2d_mesh_mapper(mesh_device)

        def upload(value: torch.Tensor, name: str, mapper, memory_config, dtype=ttnn.bfloat16):
            # a matmul weight of another dtype packs on the host (bf16 -> bfp8 / bfp4) into a tagged tensorbin
            return ttnn.as_tensor(
                value.to(torch.bfloat16).contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=memory_config,
                mesh_mapper=mapper,
                cache_file_name=cache_dir / dense_weight_name(name, dtype),
            )

        # TTNN linear consumes [K,N]; checkpoint Linear weights are [N,K].
        # Weights are DRAM width-sharded for the decode matmul program; the
        # renamed tensorbins deliberately orphan interleaved caches.  The
        # 2.6 MB router is replicated so every device computes all 512 logits
        # locally; no logits all-gather.
        local_intermediate = INTERMEDIATE_SIZE // MESH_SHAPE[1]
        router = upload(
            source.router_weight.transpose(0, 1).reshape(1, 1, HIDDEN_SIZE, ROUTED_EXPERTS),
            "router_replicated_dram_sharded",
            replicate_mapper,
            dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, ROUTED_EXPERTS),
        )
        shared_gate = upload(
            source.shared_gate_proj.transpose(0, 1).reshape(1, 1, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            "shared_gate_dram_sharded",
            output_mapper,
            dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, local_intermediate),
            dtype=shared_dtype,
        )
        shared_up = upload(
            source.shared_up_proj.transpose(0, 1).reshape(1, 1, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            "shared_up_dram_sharded",
            output_mapper,
            dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, local_intermediate),
            dtype=shared_dtype,
        )
        shared_down = upload(
            source.shared_down_proj.transpose(0, 1).reshape(1, 1, INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "shared_down_dram_sharded",
            input_mapper,
            dram_sharded_weight_memory_config(mesh_device, local_intermediate, HIDDEN_SIZE),
            dtype=shared_dtype,
        )
        shared_scalar_gate = upload(
            source.shared_scalar_gate.transpose(0, 1).reshape(1, 1, HIDDEN_SIZE, 1),
            "shared_scalar_gate_replicated_dram_sharded",
            replicate_mapper,
            dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, 1),
            dtype=shared_dtype,
        )

        mesh_contract.validate_tensor(router, placement=TensorPlacement.REPLICATED)
        mesh_contract.validate_tensor(shared_gate, placement=TensorPlacement.INTERMEDIATE_SHARDED, shard_dim=3)
        mesh_contract.validate_tensor(shared_up, placement=TensorPlacement.INTERMEDIATE_SHARDED, shard_dim=3)
        mesh_contract.validate_tensor(shared_down, placement=TensorPlacement.INTERMEDIATE_SHARDED, shard_dim=2)
        mesh_contract.validate_tensor(shared_scalar_gate, placement=TensorPlacement.REPLICATED)
        shared_gate_up_scalar = None
        if fused.resolve("shared_expert") is fused.kernel("shared_expert").fused:
            # the fused shared expert's one linear (ttnn/fused/shared_expert): device d's [gate | up | scalar | 0]
            shared_gate_up_scalar = upload(
                fused.shared_expert.concat_shared_weights(
                    source.shared_gate_proj.transpose(0, 1),
                    source.shared_up_proj.transpose(0, 1),
                    source.shared_scalar_gate.transpose(0, 1),
                ).reshape(1, 1, HIDDEN_SIZE, MESH_SHAPE[1] * fused.shared_expert.CAT_WIDTH),
                "shared_gate_up_scalar_dram_sharded",
                output_mapper,
                dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, fused.shared_expert.CAT_WIDTH),
                dtype=shared_dtype,
            )
            mesh_contract.validate_tensor(
                shared_gate_up_scalar, placement=TensorPlacement.INTERMEDIATE_SHARDED, shard_dim=3
            )
        return cls(router, shared_gate, shared_up, shared_down, shared_scalar_gate, shared_gate_up_scalar, shared_dtype)


@dataclass(frozen=True)
class Qwen38TTNNRouting:
    """The ROW_MAJOR top-k scores (BF16) and indices (UINT16) ``[1,1,rows,10]``; ``tiles`` holds the same per
    32-row tile when the routed stream runs one ``moe_compute`` call per tile (the long chunk's oracle form)."""

    scores: Any
    indices: Any
    tiles: tuple["Qwen38TTNNRouting", ...] | None = None


@dataclass(frozen=True)
class Qwen38TTNNSharedPartial:
    """The fused shared expert's ungated down-linear shard and its column-broadcast sigmoid tile; ``moe_post``
    multiplies and adds them (ttnn/fused/shared_expert)."""

    partial: Any
    sigmoid: Any


@dataclass(frozen=True)
class Qwen38TTNNMoEResult:
    hidden_sharded: Any
    routing: Qwen38TTNNRouting | None = None


class Qwen38TTNNMoE:
    """One exact fixed-row MoE layer with streamed BF4_B routed weights."""

    moe_post_fused = False
    shared_expert_fused = False
    moe_dense_fused = False
    slab_combine_fused = False
    routing_in_l1 = False
    expert_owner = None
    # The prefill slab's dense-linear policy (ttnn/prefill_dense: the QWEN38_PREFILL_DENSE_* switches) for the shared
    # expert's slab linears; the router and every decode linear never read it.
    prefill_dense: Qwen38TTNNPrefillDense | None = None

    def __init__(
        self,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        weights: Qwen38TTNNMoEWeights,
        *,
        tt_ccl,
        collective_topology=None,
        rows: int = 1,
        synchronization_policy: Qwen38TTNNMoESyncPolicy = Qwen38TTNNMoESyncPolicy.CORRECTNESS_FENCED,
        local_combine_output=None,
        routed_tokens_per_call: int | None = None,
        admitted_rows: tuple[int, ...] = SUPPORTED_ROWS,
        prefill_dense: Qwen38TTNNPrefillDense | None = None,
    ) -> None:
        """``routed_tokens_per_call`` is the token count of one ``moe_compute`` call: the rows themselves up to 32,
        and for the 128-row form 128 (the default, ``routed_tokens_per_call_for``: one call, the chunk's expert
        union streamed once; bitwise the per-tile form page for page since the weighted reduce's one-tile score
        table was fixed) or 32 (four calls of the 32-row program, bitwise the 32-row chunk per tile; the
        micro-tests' oracle).  ``local_combine_output`` hands in a shared
        ``[10, routed_tokens_per_call, 2560]`` ROW_MAJOR BF16 combine buffer (the 48 long-chunk instances share one:
        the layers run one after another and every call fills it first); it is borrowed, so
        ``release_owned_buffers`` leaves it to its owner."""

        if type(synchronization_policy) is not Qwen38TTNNMoESyncPolicy:
            raise TypeError(
                "MoE synchronization_policy must be an exact Qwen38TTNNMoESyncPolicy, "
                f"got {synchronization_policy!r}"
            )
        row_contract = Qwen38TTNNMoERowContract(rows, admitted_rows)
        mesh_contract.validate_mesh(mesh_device)
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.tt_ccl = tt_ccl
        # Resident router/shared weights are borrowed.  A rows=5 verifier or a
        # rows=32 prefill instance can therefore be constructed from an ordinary
        # layer's exact same ``weights`` object without another copy.
        self.weights = weights
        self.collective_topology = collective_topology or ttnn.Topology.Linear
        self.prefill_dense = Qwen38TTNNPrefillDense.resolve(prefill_dense, mesh_device)
        self.row_contract = row_contract
        self.rows = self.row_contract.rows
        self.routed_tokens = (
            routed_tokens_per_call_for(self.rows) if routed_tokens_per_call is None else routed_tokens_per_call
        )
        if (
            self.routed_tokens not in (self.rows, CHUNK_ROWS, LONG_PREFILL_CHUNK_ROWS)
            or self.rows % self.routed_tokens
            or (
                self.routed_tokens == LONG_PREFILL_CHUNK_ROWS
                and self.rows != LONG_PREFILL_CHUNK_ROWS
                and not is_slab_rows(self.rows)
            )
        ):
            raise ValueError(
                f"routed tokens per call must be the rows ({self.rows}), {CHUNK_ROWS}, or {LONG_PREFILL_CHUNK_ROWS} "
                f"for a slab, got {routed_tokens_per_call!r}"
            )
        self.routed_calls = self.rows // self.routed_tokens
        if self.slab_one_call:
            admit_slab_moe_switches()  # a refused ring count ends the construction, not the first slab forward
        if self.slab_one_call and self.rows % SLAB_REDUCE_BLOCK_ROWS:
            raise ValueError(
                f"the one-call slab reduces in {SLAB_REDUCE_BLOCK_ROWS}-row blocks; {self.rows} rows is not a multiple"
            )
        # The one-tile router tail (rows <= 32): the fused program (ttnn/fused/router_tail) by default, the composed
        # chain under QWEN38_FUSED_OFF=router_tail; the long chunk and the slab keep their inline chain until the
        # four-tile form is proven in the model.
        self._route_tail = fused.resolve("router_tail") if self.row_contract.row_tiles == 1 else None
        if self._route_tail is fused.kernel("router_tail").fused:
            fused.router_tail.router_tail_prepare(mesh_device)  # the constant index tiles, before any trace capture
        # One-tile rows only: the MoE post program (ttnn/fused/moe_post: fill, tilize, weighted reduce and partial add
        # as one program) and the fused shared expert (ttnn/fused/shared_expert: one concatenated linear, one eltwise
        # program, the down linear); with the fused tail the routing rows land in moe_compute's drain-core shards.
        one_tile = self.row_contract.row_tiles == 1
        self.moe_post_fused = one_tile and fused.resolve("moe_post") is fused.kernel("moe_post").fused
        self.shared_expert_fused = one_tile and fused.resolve("shared_expert") is fused.kernel("shared_expert").fused
        self.routing_in_l1 = self.moe_post_fused and self._route_tail is fused.kernel("router_tail").fused
        # The one-call slab's combine as one program per layer (ttnn/fused/moe_combine: the page's owned rows read
        # straight into tiles, the fused reduce's MAC in its slot order; bitwise the 512-row blocks), the default;
        # QWEN38_FUSED_OFF=moe_combine restores the blocks.  It takes device d's owner row (below) and the routing rows.
        self.slab_combine_fused = (
            self.slab_one_call and fused.resolve("moe_combine") is fused.kernel("moe_combine").fused
        )
        if self.shared_expert_fused and weights.shared_gate_up_scalar is None:
            raise ValueError("the fused shared expert needs weights loaded with it on (shared_gate_up_scalar is None)")
        # The MoE dense composite (ttnn/fused/moe_dense): the fused top-k, the fused shared expert's eltwise and down
        # linear and the routed dispatch untilize as ONE program on disjoint cores (the shared work runs under the
        # top-k); it hosts those three fused forms, so it serves only with all of them on.
        self.moe_dense_fused = (
            self.routing_in_l1
            and self.shared_expert_fused
            and fused.resolve("moe_dense") is fused.kernel("moe_dense").fused
        )
        if self.moe_dense_fused:
            try:
                fused.moe_dense.plan_for(mesh_device)  # the core placement and the NoC map, before any trace capture
            except RuntimeError as error:
                # a die where the composite's groups do not fit beside the reserved cores, or whose devices map their
                # NoC differently, keeps the four programs (a production default never raises on such a mesh)
                warnings.warn(f"MoE dense composite not placed on this mesh; the four programs serve: {error}")
                self.moe_dense_fused = False
        # The slab's routed stream: the 128-row instance's call (moe_compute, tilize, weighted reduce) once per
        # 128-row block, through a worker instance sharing the combine buffer.
        self.block_instance: Qwen38TTNNMoE | None = None
        self.synchronization_policy = synchronization_policy
        self.expert_mapping = None
        self.local_combine_output = None
        self._owns_local_combine_output = local_combine_output is None
        self._owned_buffers_released = False
        self._poisoned_error: BaseException | None = None
        self._poisoned_device_owners: list[Any] = []
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # The shared-expert linears run the fidelity of their weight format (decode_matmul: HiFi4 for bf16, HiFi2 for
        # bf8, LoFi for bf4); the router and every other program keep compute_config.
        self.shared_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, dense_math_fidelity_name(weights.shared_dtype)),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        ring_size = effective_matmul_ring_size(mesh_device)
        output_width_shard_dim = auto_output_width_shard_dim(HIDDEN_SIZE, matmul_ring_size=ring_size)
        # The one-tile rows forms take QWEN38_MOE_ROWS_FORM; every other instance is fulllocal here (the slab's own
        # local output path is slab_one_call below, the 128-row chunk keeps the fused local combine).
        self.rows_form = moe_rows_form() if self.row_contract.row_tiles == 1 else "fulllocal"
        self.local_output = moe_local_output_enabled() or self.slab_one_call or MOE_ROWS_FORMS[self.rows_form][0]
        self.output_height_shard_dim = moe_compute_output_height_shard_dim(
            self.routed_tokens, matmul_ring_size=ring_size
        )
        drain = ttnn.experimental.get_moe_tilize_drain_core(
            mesh_device,
            1,
            output_width_shard_dim,
            HIDDEN_SIZE,
        )
        drain_cores = ttnn.CoreRangeSet({ttnn.CoreRange(drain, drain)})
        self.routing_l1_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(drain_cores, [self.routed_tokens, TOP_K], ttnn.ShardOrientation.ROW_MAJOR),
        )
        # DRAM-sharded decode matmul configs.  Every K=2560 linear (router,
        # shared gate/up/scalar) reads the gathered hidden, which the hidden
        # all-gather writes straight into the five-core width-sharded layout.  The
        # replicated router's 512 logits are 16 output tiles, four per core on
        # four of the five.  The shared chain stays on that grid: gate/up
        # outputs feed silu/mul/down without resharding, and the down input
        # splits its 160-column K into exactly one tile per core.
        local_intermediate = INTERMEDIATE_SIZE // MESH_SHAPE[1]
        self.hidden_act_memory_config, self.router_program_config = dram_sharded_matmul_configs(
            mesh_device, HIDDEN_SIZE, ROUTED_EXPERTS, num_cores=5
        )
        _, self.shared_gate_up_program_config = dram_sharded_matmul_configs(
            mesh_device, HIDDEN_SIZE, local_intermediate, num_cores=5
        )
        self.shared_intermediate_memory_config, self.shared_down_program_config = dram_sharded_matmul_configs(
            mesh_device, local_intermediate, HIDDEN_SIZE, num_cores=5
        )
        _, self.shared_scalar_program_config = dram_sharded_matmul_configs(mesh_device, HIDDEN_SIZE, 1, num_cores=5)
        _, self.shared_gate_up_scalar_program_config = dram_sharded_matmul_configs(
            mesh_device, HIDDEN_SIZE, fused.shared_expert.CAT_WIDTH, num_cores=5
        )
        # The gathered hidden lands in the five-core shard the dense linears read; the long chunk gathers its four
        # row tiles interleaved and moves each tile into that shard per linear.
        self.hidden_gather_memory_config = (
            self.hidden_act_memory_config if self.row_contract.row_tiles == 1 else ttnn.DRAM_MEMORY_CONFIG
        )

        try:
            mapping = _expert_owner_mapping()
            self.expert_mapping = ttnn.from_torch(
                mapping,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
            )
            if local_combine_output is None:
                local_combine_output = allocate_local_combine_output(mesh_device, mesh_contract, self.routed_tokens)
            self.local_combine_output = local_combine_output
            if not self._owns_local_combine_output and (
                _shape(self.local_combine_output) != (TOP_K, self.routed_tokens, HIDDEN_SIZE)
                or self.local_combine_output.dtype != ttnn.bfloat16
                or self.local_combine_output.layout != ttnn.ROW_MAJOR_LAYOUT
            ):
                raise RuntimeError(
                    f"MoE local combine buffer must be ROW_MAJOR BF16 {(TOP_K, self.routed_tokens, HIDDEN_SIZE)}, got "
                    f"{self.local_combine_output.layout} {self.local_combine_output.dtype} "
                    f"{_shape(self.local_combine_output)}"
                )
            mesh_contract.validate_tensor(self.expert_mapping, placement=TensorPlacement.REPLICATED)
            mesh_contract.validate_tensor(self.local_combine_output, placement=TensorPlacement.LOCAL_PARTIAL)
            if self.moe_post_fused or self.slab_combine_fused:
                # device d's [1, 512] expert owner row (the post and combine programs have no mesh coordinate)
                self.expert_owner = ttnn.from_torch(
                    fused.moe_post.owner_rows(MESH_SHAPE[1]),
                    device=mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.uint16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 0)),
                )
                mesh_contract.validate_tensor(self.expert_owner, placement=TensorPlacement.EXPERT_SHARDED, shard_dim=0)
            if self.slab and not self.slab_one_call:
                if self.routed_tokens != LONG_PREFILL_CHUNK_ROWS:
                    raise ValueError(f"a slab MoE instance routes {LONG_PREFILL_CHUNK_ROWS} tokens per call")
                self.block_instance = Qwen38TTNNMoE(
                    mesh_device,
                    mesh_contract,
                    weights,
                    tt_ccl=tt_ccl,
                    collective_topology=collective_topology,
                    rows=LONG_PREFILL_CHUNK_ROWS,
                    synchronization_policy=synchronization_policy,
                    local_combine_output=self.local_combine_output,
                    prefill_dense=self.prefill_dense,
                )
        except BaseException as initialization_error:
            try:
                self.release_owned_buffers()
            except BaseException as cleanup_error:
                raise RuntimeError(
                    f"MoE owned-buffer initialization failed and cleanup also failed: {cleanup_error}"
                ) from initialization_error
            raise

    def release_owned_buffers(self) -> None:
        """Retry-safe release of this row instance's two private buffers.

        Router/shared weights are borrowed and deliberately excluded.  Each
        successful release clears its attribute immediately; if another release
        fails, a later call retries only that still-owned tensor.
        """

        if self._poisoned_error is not None:
            raise RuntimeError(
                "cannot release MoE buffers after an asynchronous forward failure; "
                "live device owners are retained for process/mesh teardown"
            ) from self._poisoned_error

        failures: list[tuple[str, BaseException]] = []
        if getattr(self, "block_instance", None) is not None:
            try:
                self.block_instance.release_owned_buffers()
            except BaseException as error:
                failures.append(("block_instance", error))
            else:
                self.block_instance = None
        for name in ("expert_mapping", "local_combine_output", "expert_owner"):
            tensor = getattr(self, name, None)
            if tensor is None:
                continue
            try:
                if name != "local_combine_output" or self._owns_local_combine_output:
                    ttnn.deallocate(tensor)
            except BaseException as error:
                failures.append((name, error))
            else:
                setattr(self, name, None)
        self._owned_buffers_released = (
            self.expert_mapping is None and self.local_combine_output is None and self.expert_owner is None
        )
        if failures:
            names = ", ".join(name for name, _error in failures)
            raise RuntimeError(f"failed to release MoE instance-owned buffer(s): {names}") from failures[0][1]

    def _require_owned_buffers(self) -> None:
        if self._poisoned_error is not None:
            raise RuntimeError(
                "MoE instance is poisoned after an asynchronous forward failure"
            ) from self._poisoned_error
        names = ("expert_mapping", "local_combine_output") + (
            ("expert_owner",) if (self.moe_post_fused or self.slab_combine_fused) else ()
        )
        missing = tuple(name for name in names if getattr(self, name, None) is None)
        if self._owned_buffers_released or missing:
            raise RuntimeError(f"MoE instance-owned buffers are unavailable: {missing or 'released'}")

    def _synchronize_stage(self, stage: str) -> None:
        """Apply the configured policy at one named dependency boundary."""

        if stage not in MOE_STAGE_FENCES:
            raise ValueError(f"unknown production MoE stage fence {stage!r}")
        if self.synchronization_policy is Qwen38TTNNMoESyncPolicy.RESIDENT_ASYNC:
            return
        if self.synchronization_policy is Qwen38TTNNMoESyncPolicy.CORRECTNESS_FENCED:
            ttnn.synchronize_device(self.mesh_device)
            return
        raise RuntimeError(f"unknown MoE synchronization policy {self.synchronization_policy!r}")

    def _retain_async_failure_owners(self, error: BaseException, *owners: Any) -> None:
        """Poison the instance and retain every live wrapper through teardown."""

        if self._poisoned_error is None:
            self._poisoned_error = error
        retained_ids = {id(owner) for owner in self._poisoned_device_owners}
        for owner in owners:
            if owner is None or id(owner) in retained_ids:
                continue
            self._poisoned_device_owners.append(owner)
            retained_ids.add(id(owner))

    def _all_gather_hidden(self, hidden_sharded):
        if _shape(hidden_sharded) != self.row_contract.hidden_sharded:
            raise ValueError(
                f"hidden_sharded must have local shape {self.row_contract.hidden_sharded} on every coordinate; "
                f"got {_shape(hidden_sharded)}"
            )
        self.mesh_contract.validate_tensor(hidden_sharded, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        # Written in the dense linears' five-core layout; the routed untilize
        # reads the same shard, so the tensor stays allocated through the layer
        # (32 KB per core).  The long chunk's four row tiles are gathered
        # interleaved and moved into that layout one tile at a time per linear.
        full_hidden = ttnn.all_gather(
            hidden_sharded,
            dim=3,
            cluster_axis=EP_AXIS,
            memory_config=self.hidden_gather_memory_config,
        )
        self.mesh_contract.validate_tensor(full_hidden, placement=TensorPlacement.REPLICATED)
        if _shape(full_hidden) != self.row_contract.full_hidden:
            raise RuntimeError(
                f"hidden all-gather produced {_shape(full_hidden)}, expected {self.row_contract.full_hidden}"
            )
        return full_hidden

    def _route(self, full_hidden, *, hidden_tiles=None, phase_observer=None) -> Qwen38TTNNRouting:
        if phase_observer is None:
            phase_observer = _ignore_phase
        # The router is a DRAM-sharded decode linear: one row tile per call.  Up to 32 rows the gathered
        # shard is the call's input; the long chunk runs the router per row tile (``hidden_tiles``: the four
        # tiles in the activation shard, shared with the shared-expert chain), concatenates the four FP32
        # logits tiles and runs the softmax, the top-k and the normalization ONCE on the 128 rows: every op
        # of the tail is per row (the top-k's single-core factory hands each 32-row tile to its own core;
        # the multi-core factory needs a width of 1024, so the 512 logits never reach it), so row j sees the
        # same programs on the same values as in the 32-row form.
        if self.slab:
            hidden_tiles = []
        elif self.row_contract.row_tiles == 1:
            hidden_tiles = [full_hidden]
        elif hidden_tiles is None or len(hidden_tiles) != self.row_contract.row_tiles:
            raise ValueError(f"the long chunk's router needs its {self.row_contract.row_tiles} hidden row tiles")
        phase_observer("before-router-logits")
        logits_tiles = []
        for hidden_tile in hidden_tiles:
            logits_ws = ttnn.linear(
                hidden_tile,
                self.weights.router,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.router_program_config,
                compute_kernel_config=self.compute_config,
            )
            if self.routing_in_l1:
                return self._route_into_shards(logits_ws, phase_observer)
            # The pinned reference evaluates the router softmax in FP32 even though
            # the projection is BF16.  The move that drains the router's L1 shard
            # widens the logits on the way out (exact), so no separate typecast.
            # FP32 is kept through top-k renormalization; only the selected scores
            # are cast back to the BF16 format consumed by ``moe_compute``.
            logits_tiles.append(ttnn.to_memory_config(logits_ws, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32))
            _deallocate(logits_ws)
        expected_logits = (1, 1, self.rows, ROUTED_EXPERTS)
        logits = self._slab_router_logits(full_hidden) if self.slab else logits_tiles[0]
        if self.row_contract.row_tiles != 1:
            if not self.slab:  # the long chunk's four tiles; the slab's logits are one tensor already
                logits = ttnn.concat(logits_tiles, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                _deallocate(*logits_tiles)
        self.mesh_contract.validate_tensor(logits, placement=TensorPlacement.REPLICATED)
        if _shape(logits) != expected_logits or logits.dtype != ttnn.float32:
            raise RuntimeError(
                f"router produced {_shape(logits)} {logits.dtype}, expected {expected_logits} {ttnn.float32}"
            )
        phase_observer("after-router-logits")

        phase_observer("before-router-topk")
        if self._route_tail is not None:
            scores_rm, indices_rm = self._route_tail(logits, top_k=TOP_K, compute_kernel_config=self.compute_config)
            _deallocate(logits)
            if _shape(scores_rm) != self.row_contract.routing or _shape(indices_rm) != self.row_contract.routing:
                raise RuntimeError(
                    f"router tail shapes must be {self.row_contract.routing}, got scores={_shape(scores_rm)} "
                    f"indices={_shape(indices_rm)}"
                )
            self.mesh_contract.validate_tensor(scores_rm, placement=TensorPlacement.REPLICATED)
            self.mesh_contract.validate_tensor(indices_rm, placement=TensorPlacement.REPLICATED)
            phase_observer("after-router-topk")
            return Qwen38TTNNRouting(scores_rm, indices_rm, None)
        probabilities = ttnn.softmax(
            logits,
            dim=-1,
            numeric_stable=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(logits)
        scores, indices = ttnn.topk(
            probabilities,
            k=TOP_K,
            dim=-1,
            largest=True,
            sorted=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _deallocate(probabilities)
        denominator = ttnn.sum(
            scores,
            dim=-1,
            keepdim=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        normalized_fp32 = ttnn.div(scores, denominator, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(scores, denominator)
        normalized = ttnn.typecast(normalized_fp32, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(normalized_fp32)
        dram = ttnn.DRAM_MEMORY_CONFIG
        tiles = None
        if self.routed_calls != 1:
            # The per-tile routed stream reads the 32-row program's ROW_MAJOR routing per tile: whole 32-row
            # tiles sliced from the TILE results (the same values; literal bounds), converted as the 32-row form
            # converts them.  The slab slices its ROW_MAJOR routing rows per 128-row block instead.
            if not self.slab:
                tiles = tuple(
                    self._routing_rows(normalized_tile, indices_tile)
                    for normalized_tile, indices_tile in zip(
                        dram_sharded_row_tiles(normalized, None), dram_sharded_row_tiles(indices, None)
                    )
                )
        scores_rm = ttnn.to_layout(normalized, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        indices_rm = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        _deallocate(normalized, indices)
        if indices_rm.dtype != ttnn.uint16:
            converted = ttnn.typecast(indices_rm, ttnn.uint16, memory_config=dram)
            _deallocate(indices_rm)
            indices_rm = converted
        if _shape(scores_rm) != self.row_contract.routing or _shape(indices_rm) != self.row_contract.routing:
            raise RuntimeError(
                f"router top-k shapes must be {self.row_contract.routing}, got scores={_shape(scores_rm)} "
                f"indices={_shape(indices_rm)}"
            )
        self.mesh_contract.validate_tensor(scores_rm, placement=TensorPlacement.REPLICATED)
        self.mesh_contract.validate_tensor(indices_rm, placement=TensorPlacement.REPLICATED)
        phase_observer("after-router-topk")
        return Qwen38TTNNRouting(scores_rm, indices_rm, tiles)

    def _routed_partial(
        self,
        full_hidden,
        routing: Qwen38TTNNRouting,
        packed_w0_w1,
        packed_w2,
        *,
        phase_observer=None,
    ):
        if phase_observer is None:
            phase_observer = _ignore_phase
        if _shape(full_hidden) != self.row_contract.full_hidden:
            raise ValueError(f"routed MoE input must be {self.row_contract.full_hidden}, got {_shape(full_hidden)}")
        if _shape(routing.indices) != self.row_contract.routing or _shape(routing.scores) != self.row_contract.routing:
            raise ValueError(
                f"routing must retain external shape {self.row_contract.routing}, got "
                f"scores={_shape(routing.scores)} indices={_shape(routing.indices)}"
            )
        self.mesh_contract.validate_tensor(packed_w0_w1, placement=TensorPlacement.EXPERT_SHARDED, shard_dim=2)
        self.mesh_contract.validate_tensor(packed_w2, placement=TensorPlacement.EXPERT_SHARDED, shard_dim=2)
        for name, tensor in (("packed_w0_w1", packed_w0_w1), ("packed_w2", packed_w2)):
            shape = _shape(tensor)
            # A TTNN mesh tensor exposes its coordinate-local logical shape;
            # topology supplies the four-way global composition.  Requiring
            # 128 here plus PlacementShard(dim=2) proves 512/4 ownership and
            # rejects a replicated 512-expert cache.
            if len(shape) < 3 or shape[2] != EXPERTS_PER_DEVICE:
                raise RuntimeError(
                    f"{name} must expose exactly 128 local experts with dimension 2 sharded across EP4, got {shape}"
                )
        if packed_w0_w1.dtype != ttnn.bfloat4_b or packed_w2.dtype != ttnn.bfloat4_b:
            raise RuntimeError("routed expert tensors must be BFLOAT4_B")

        if self.slab and not self.slab_one_call:
            return self._routed_partial_blocks(full_hidden, routing, packed_w0_w1, packed_w2, phase_observer)
        if self.routed_calls != 1:
            return self._routed_partial_tiles(full_hidden, routing, packed_w0_w1, packed_w2, phase_observer)
        phase_observer("before-routed-dispatch")
        # Keep the ordinary one-row call byte/API-compatible with the existing
        # rank-four input.  A multi-row instance must expose its rows in dim 1:
        # moe_compute derives total_tokens from sparse_input.shape[0:2].  The
        # rows are untilized first (the one-row op on the same 32-row tile) and
        # reshaped ROW_MAJOR: the same bytes, no un-shard or relayout kernel.
        sparse_input = ttnn.to_layout(
            full_hidden,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.rows != 1:
            # ttnn.reshape of a ROW_MAJOR device tensor is a zero-cost view: a new tensor id over the SAME buffer, so a
            # tensor-id test says "different" and freeing the source frees the rows the view still reads (the
            # 2026-09-26 slab defect: the op's next DRAM allocation, its packed token-list page, landed on the input's
            # first rows; docs/NUMERICS.md).  The source is freed only when the reshape produced its own buffer; a
            # view's buffer is released once, through sparse_input, after the call.
            untilized_rows = sparse_input
            sparse_input = ttnn.reshape(untilized_rows, self.row_contract.moe_sparse_input)
            if not same_buffer(sparse_input, untilized_rows):
                _deallocate(untilized_rows)
        if _shape(sparse_input) != self.row_contract.moe_sparse_input:
            raise RuntimeError(
                f"moe_compute sparse input must be {self.row_contract.moe_sparse_input}, got {_shape(sparse_input)}"
            )
        self.mesh_contract.validate_tensor(sparse_input, placement=TensorPlacement.REPLICATED)

        indices_rank3 = ttnn.reshape(routing.indices, self.row_contract.moe_routing)
        scores_rank3 = ttnn.reshape(routing.scores, self.row_contract.moe_routing)
        indices_l1 = ttnn.to_memory_config(indices_rank3, self.routing_l1_memory_config)
        scores_l1 = ttnn.to_memory_config(scores_rank3, self.routing_l1_memory_config)
        if _shape(indices_l1) != self.row_contract.moe_routing or _shape(scores_l1) != self.row_contract.moe_routing:
            raise RuntimeError(
                f"moe_compute routing must be {self.row_contract.moe_routing}, got "
                f"scores={_shape(scores_l1)} indices={_shape(indices_l1)}"
            )
        if (
            indices_l1.layout != ttnn.ROW_MAJOR_LAYOUT
            or indices_l1.dtype != ttnn.uint16
            or indices_l1.memory_config() != self.routing_l1_memory_config
        ):
            raise RuntimeError("moe_compute indices are not RM UINT16 on the exact L1 drain-core shard")
        if (
            scores_l1.layout != ttnn.ROW_MAJOR_LAYOUT
            or scores_l1.dtype != ttnn.bfloat16
            or scores_l1.memory_config() != self.routing_l1_memory_config
        ):
            raise RuntimeError("moe_compute scores are not RM BF16 on the exact L1 drain-core shard")
        self.mesh_contract.validate_tensor(indices_l1, placement=TensorPlacement.REPLICATED)
        self.mesh_contract.validate_tensor(scores_l1, placement=TensorPlacement.REPLICATED)
        phase_observer("after-routed-dispatch")

        # Local combine writes only owned k slots.  Clear every invocation so a
        # masked 0*uninitialized-NaN cannot poison fast-reduce.  The long chunk's shared buffer is zero at
        # allocation and every slot it ever holds is a finite expert output (the weighted reduce multiplies
        # the slots this device does not own by an exact 0), so its 48 layers skip the 6.5 MB fill; the one-call slab's
        # 105 MB buffer likewise (and the op does not zero the unowned rows either: zero_fill_non_owned_rows below).
        if self.rows != LONG_PREFILL_CHUNK_ROWS and not self.slab_one_call:
            zeroed = ttnn.fill(self.local_combine_output, 0.0, output_tensor=self.local_combine_output)
            if zeroed.tensor_id != self.local_combine_output.tensor_id:
                raise RuntimeError("ttnn.fill did not update the persistent local-combine buffer in place")

        phase_observer("before-moe-compute-launch")
        outputs = ttnn.experimental.moe_compute(
            sparse_input,
            indices_l1,
            scores_l1,
            self.expert_mapping,
            packed_w0_w1,
            packed_w2,
            layer_id=0,
            output_height_shard_dim=self.output_height_shard_dim,
            intermediate_size=INTERMEDIATE_SIZE,
            has_bias=False,
            cluster_axis=LOCAL_COMBINE_AXIS,
            topology=None,
            num_links=None,
            mux_core_range_set=None,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            optional_output_tensor=self.local_combine_output,
            optional_cross_device_semaphore=None,
            activation_type=MoEActivationFunction.SILU,
            compute_only=False,
            local_combine=not self.local_output,
            num_shared_experts_per_device=0,
            zero_fill_non_owned_rows=self.zero_fill_non_owned_rows,
            prefill_rings=self.prefill_rings,
            enable_a2a_pipeline=not self.prefill_rings,
        )
        phase_observer("after-moe-compute-launch")
        if len(outputs) != 6 or outputs[5].tensor_id != self.local_combine_output.tensor_id:
            raise RuntimeError("local moe_compute did not return the persistent six-slot output")
        self.mesh_contract.validate_tensor(outputs[5], placement=TensorPlacement.LOCAL_PARTIAL)
        if _shape(outputs[5]) != self.row_contract.local_combine:
            raise RuntimeError(
                f"local combine produced {_shape(outputs[5])}, expected {self.row_contract.local_combine}"
            )

        # Slots 3/4 alias one L1 backing buffer; freeing slot 4 releases it.
        _deallocate(outputs[0], outputs[1], outputs[2], outputs[4], sparse_input, indices_l1, scores_l1)
        if self.slab_one_call:
            return self._weighted_reduce_slab_blocks(outputs[5], full_hidden, routing, phase_observer)
        phase_observer("before-selective-reduce")
        local_stack = ttnn.unsqueeze(outputs[5], dim=1)
        if _shape(local_stack) != self.row_contract.fast_reduce_input:
            raise RuntimeError(
                f"weighted-reduce input must be {self.row_contract.fast_reduce_input}, got {_shape(local_stack)}"
            )
        if self.rows == LONG_PREFILL_CHUNK_ROWS:
            # The 6.5 MB combine tilized as one [1280, 2560] tile grid (the tilize spreads its 40 tile rows over
            # the grid; the rank-4 form works one 4-tile-row batch at a time: 0.28 -> 0.14 ms) and viewed back
            # as the reduce's [10, 1, 128, 2560] (the same tile order; the view owns the buffer).
            combine_flat = ttnn.to_layout(
                ttnn.reshape(outputs[5], (TOP_K * self.rows, HIDDEN_SIZE)),
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                pad_value=0.0,
            )
            local_stack_tiled = ttnn.experimental.view(combine_flat, self.row_contract.fast_reduce_input)
        else:
            local_stack_tiled = ttnn.to_layout(
                local_stack, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, pad_value=0.0
            )
        # Preserve the externally visible [1,1,rows,10] routing tensor.  The
        # fused reducer independently treats score dim 0 as the token count.
        fast_reduce_scores = routing.scores
        if self.rows != 1:
            fast_reduce_scores = ttnn.reshape(routing.scores, self.row_contract.fast_reduce_scores)
        if _shape(fast_reduce_scores) != self.row_contract.fast_reduce_scores:
            raise RuntimeError(
                f"fast-reduce scores must be {self.row_contract.fast_reduce_scores}, "
                f"got {_shape(fast_reduce_scores)}"
            )
        fast_outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            local_stack_tiled,
            routing.indices,
            self.expert_mapping,
            reduce_dim=0,
            split_size=HIDDEN_SIZE,
            cluster_axis=LOCAL_COMBINE_AXIS,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            scores_tensor=fast_reduce_scores,
            num_shared_experts=0,
            shared_expert_scale=1.0,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(local_stack_tiled)
        if len(fast_outputs) != 1 or _shape(fast_outputs[0]) != self.row_contract.full_hidden:
            raise RuntimeError(
                f"weighted routed reduce must return one {self.row_contract.full_hidden} partial, got "
                f"{tuple(_shape(item) for item in fast_outputs)}"
            )
        self.mesh_contract.validate_tensor(fast_outputs[0], placement=TensorPlacement.LOCAL_PARTIAL)
        phase_observer("after-selective-reduce")
        return fast_outputs[0]

    def _weighted_reduce_slab_blocks(self, combine, full_hidden, routing: Qwen38TTNNRouting, phase_observer):
        """The one-call slab's weighted reduce, ``SLAB_REDUCE_BLOCK_ROWS`` rows at a time: each block's pages (a
        k-strided ROW_MAJOR slice of the ``[10, rows, 2560]`` buffer) tilized and viewed as the reduce's
        ``[10, 1, block, 2560]``, with the block's scores and indices; the partials concatenated."""

        phase_observer("before-selective-reduce")
        dram = ttnn.DRAM_MEMORY_CONFIG
        # getattr: the no-device aliasing test calls this method on a bare namespace without the flag
        if getattr(self, "slab_combine_fused", False) and fused.moe_combine.admits(
            combine, routing.scores, routing.indices, self.expert_owner
        ):
            # one program: the page's owned rows into tiles, the fused reduce's MAC over the ten slots, the tiled partial
            partial = fused.moe_combine.moe_combine(
                combine, routing.scores, routing.indices, self.expert_owner, memory_config=dram
            )
            self.mesh_contract.mark_local_partial(
                partial, replicated_reference=full_hidden, expected_shape=self.row_contract.full_hidden
            )
            phase_observer("after-selective-reduce")
            return partial
        block = SLAB_REDUCE_BLOCK_ROWS
        # A slab of exactly one block (512 rows) has no sub-range to slice: ``ttnn.slice`` over a tensor's full extent
        # returns its INPUT (an alias), and ``ttnn.concat`` of one tensor likewise, so the block's pages are the
        # combine buffer itself and the block's scores / indices are the routing's own tensors -- never freed here
        # (freeing "pages" would free the persistent combine page under the next layer). Only what this method
        # created is released.
        one_block = self.rows == block
        partials = []
        for start in range(0, self.rows, block):
            pages = (
                combine
                if one_block
                else ttnn.slice(combine, (0, start, 0), (TOP_K, start + block, HIDDEN_SIZE), memory_config=dram)
            )
            stack = ttnn.experimental.view(
                ttnn.to_layout(
                    ttnn.reshape(pages, (TOP_K * block, HIDDEN_SIZE)),
                    ttnn.TILE_LAYOUT,
                    memory_config=dram,
                    pad_value=0.0,
                ),
                (TOP_K, 1, block, HIDDEN_SIZE),
            )
            scores = (
                routing.scores
                if one_block
                else ttnn.slice(routing.scores, (0, 0, start, 0), (1, 1, start + block, TOP_K), memory_config=dram)
            )
            indices = (
                routing.indices
                if one_block
                else ttnn.slice(routing.indices, (0, 0, start, 0), (1, 1, start + block, TOP_K), memory_config=dram)
            )
            fast_outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                stack,
                indices,
                self.expert_mapping,
                reduce_dim=0,
                split_size=HIDDEN_SIZE,
                cluster_axis=LOCAL_COMBINE_AXIS,
                output_memory_config=dram,
                scores_tensor=ttnn.reshape(scores, (block, 1, 1, TOP_K)),
                num_shared_experts=0,
                shared_expert_scale=1.0,
                compute_kernel_config=self.compute_config,
            )
            _deallocate(stack, *(() if one_block else (pages, scores, indices)))
            if len(fast_outputs) != 1 or _shape(fast_outputs[0]) != (1, 1, block, HIDDEN_SIZE):
                raise RuntimeError(
                    f"weighted routed reduce of a {block}-row block must return one (1, 1, {block}, {HIDDEN_SIZE}) "
                    f"partial, got {tuple(_shape(item) for item in fast_outputs)}"
                )
            partials.append(fast_outputs[0])
        if one_block:
            partial = partials[0]  # ttnn.concat of one tensor would return it (an alias): the partial IS the result
        else:
            partial = ttnn.concat(partials, dim=2, memory_config=dram)
            _deallocate(*partials)
        self.mesh_contract.mark_local_partial(
            partial, replicated_reference=full_hidden, expected_shape=self.row_contract.full_hidden
        )
        phase_observer("after-selective-reduce")
        return partial

    def _routed_local_sum(
        self,
        full_hidden,
        routing: Qwen38TTNNRouting,
        packed_w0_w1,
        packed_w2,
        shared,
        *,
        phase_observer,
        sparse_rows=None,
    ):
        """The fused post form (``QWEN38_FUSED=moe_post``): ``moe_compute`` on the routing rows (in the drain-core
        shards already when the fused tail wrote them there), then one program for the weighted reduce of the owned
        slots, the shared partial's ``x sigmoid`` (a ``Qwen38TTNNSharedPartial``) and the partial add; no fill, no
        tilize, no copies.  Returns the layer's local sum ``[1, 1, rows, 2560]``."""

        if _shape(full_hidden) != self.row_contract.full_hidden:
            raise ValueError(f"routed MoE input must be {self.row_contract.full_hidden}, got {_shape(full_hidden)}")
        routing_shape = self.row_contract.moe_routing if self.routing_in_l1 else self.row_contract.routing
        if _shape(routing.indices) != routing_shape or _shape(routing.scores) != routing_shape:
            raise ValueError(
                f"routing must retain shape {routing_shape}, got scores={_shape(routing.scores)} "
                f"indices={_shape(routing.indices)}"
            )
        self.mesh_contract.validate_tensor(packed_w0_w1, placement=TensorPlacement.EXPERT_SHARDED, shard_dim=2)
        self.mesh_contract.validate_tensor(packed_w2, placement=TensorPlacement.EXPERT_SHARDED, shard_dim=2)
        for name, tensor in (("packed_w0_w1", packed_w0_w1), ("packed_w2", packed_w2)):
            shape = _shape(tensor)
            if len(shape) < 3 or shape[2] != EXPERTS_PER_DEVICE:
                raise RuntimeError(
                    f"{name} must expose exactly 128 local experts with dimension 2 sharded across EP4, got {shape}"
                )
        if packed_w0_w1.dtype != ttnn.bfloat4_b or packed_w2.dtype != ttnn.bfloat4_b:
            raise RuntimeError("routed expert tensors must be BFLOAT4_B")
        phase_observer("before-routed-dispatch")
        # The untilize reads the width-sharded hidden at any row count as it does at one row; several rows then take
        # the ROW_MAJOR view moe_compute counts (dims 0 x 1) -- no un-sharding of the tiles first (one program per
        # layer at B > 1 in the 2026-09-25 lane census).
        if sparse_rows is None:
            sparse_input = ttnn.to_layout(full_hidden, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            sparse_input = sparse_rows  # the MoE dense composite's untilize (the same rows, one program earlier)
        if self.rows != 1:
            sparse_input = ttnn.reshape(sparse_input, self.row_contract.moe_sparse_input)
        if _shape(sparse_input) != self.row_contract.moe_sparse_input:
            raise RuntimeError(
                f"moe_compute sparse input must be {self.row_contract.moe_sparse_input}, got {_shape(sparse_input)}"
            )
        self.mesh_contract.validate_tensor(sparse_input, placement=TensorPlacement.REPLICATED)
        if self.routing_in_l1:
            indices_l1, scores_l1 = routing.indices, routing.scores
        else:
            indices_l1 = ttnn.to_memory_config(
                ttnn.reshape(routing.indices, self.row_contract.moe_routing), self.routing_l1_memory_config
            )
            scores_l1 = ttnn.to_memory_config(
                ttnn.reshape(routing.scores, self.row_contract.moe_routing), self.routing_l1_memory_config
            )
        self._check_routing_shards(indices_l1, scores_l1)
        phase_observer("after-routed-dispatch")
        phase_observer("before-moe-compute-launch")
        outputs = ttnn.experimental.moe_compute(
            sparse_input,
            indices_l1,
            scores_l1,
            self.expert_mapping,
            packed_w0_w1,
            packed_w2,
            layer_id=0,
            output_height_shard_dim=self.output_height_shard_dim,
            intermediate_size=INTERMEDIATE_SIZE,
            has_bias=False,
            cluster_axis=LOCAL_COMBINE_AXIS,
            topology=None,
            num_links=None,
            mux_core_range_set=None,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            optional_output_tensor=self.local_combine_output,
            optional_cross_device_semaphore=None,
            activation_type=MoEActivationFunction.SILU,
            compute_only=False,
            local_combine=not self.local_output,
            num_shared_experts_per_device=0,
            zero_fill_non_owned_rows=self.zero_fill_non_owned_rows,
            prefill_rings=self.prefill_rings,
            enable_a2a_pipeline=not self.prefill_rings,
        )
        phase_observer("after-moe-compute-launch")
        if len(outputs) != 6 or outputs[5].tensor_id != self.local_combine_output.tensor_id:
            raise RuntimeError("local moe_compute did not return the persistent six-slot output")
        self.mesh_contract.validate_tensor(outputs[5], placement=TensorPlacement.LOCAL_PARTIAL)
        if _shape(outputs[5]) != self.row_contract.local_combine:
            raise RuntimeError(
                f"local combine produced {_shape(outputs[5])}, expected {self.row_contract.local_combine}"
            )
        _deallocate(outputs[0], outputs[1], outputs[2], outputs[4])
        if sparse_rows is None or not same_buffer(sparse_input, sparse_rows):
            # own rows, or a reshape that copied; a VIEW of the caller's rows shares their buffer and the caller frees it
            _deallocate(sparse_input)
        if not self.routing_in_l1:
            _deallocate(indices_l1, scores_l1)
        phase_observer("before-selective-reduce")
        if isinstance(shared, Qwen38TTNNSharedPartial):
            shared_partial, sigmoid = shared.partial, shared.sigmoid
        else:
            shared_partial, sigmoid = shared, None
        local_sum = fused.moe_post.moe_post(
            outputs[5],
            routing.scores,
            routing.indices,
            self.expert_owner,
            shared_partial,
            sigmoid=sigmoid,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if _shape(local_sum) != self.row_contract.full_hidden:
            raise RuntimeError(
                f"MoE post program produced {_shape(local_sum)}, expected {self.row_contract.full_hidden}"
            )
        phase_observer("after-selective-reduce")
        return local_sum

    def _dense_composite(self, full_hidden, phase_observer):
        """The MoE dense composite (``moe_dense`` in the registry): the router linear and its fp32 drain, the shared
        ``[gate | up | scalar]`` linear, then ONE program for the top-k (into the drain-core shards), the shared
        eltwise + down linear and the routed dispatch untilize.  Returns the routing, the shared partial (ungated, with
        its sigmoid tile, for ``moe_post``) and ``moe_compute``'s ROW_MAJOR rows."""

        l1_ws = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        phase_observer("before-router-logits")
        logits_ws = ttnn.linear(
            full_hidden,
            self.weights.router,
            memory_config=l1_ws,
            program_config=self.router_program_config,
            compute_kernel_config=self.compute_config,
        )
        logits = ttnn.to_memory_config(logits_ws, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)  # the proven drain
        _deallocate(logits_ws)
        phase_observer("after-router-logits")
        phase_observer("before-shared-partial")
        gate_up_scalar_ws = ttnn.linear(
            full_hidden,
            self.weights.shared_gate_up_scalar,
            memory_config=l1_ws,
            program_config=self.shared_gate_up_scalar_program_config,
            compute_kernel_config=self.shared_compute_config,
        )
        if _shape(gate_up_scalar_ws) != (1, 1, self.rows, fused.shared_expert.CAT_WIDTH):
            raise RuntimeError(
                f"concatenated shared linear produced {_shape(gate_up_scalar_ws)}, expected "
                f"[1, 1, {self.rows}, {fused.shared_expert.CAT_WIDTH}]"
            )
        phase_observer("before-router-topk")
        routing_shape = ttnn.Shape(list(self.row_contract.moe_routing))
        scores_l1 = ttnn.allocate_tensor_on_device(
            routing_shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device, self.routing_l1_memory_config
        )
        indices_l1 = ttnn.allocate_tensor_on_device(
            routing_shape, ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device, self.routing_l1_memory_config
        )
        fused.program.stamp_topology(scores_l1, full_hidden)
        fused.program.stamp_topology(indices_l1, full_hidden)
        try:
            scores_l1, indices_l1, partial, sigmoid, sparse_rows = fused.moe_dense.moe_dense(
                logits,
                gate_up_scalar_ws,
                self.weights.shared_down,
                full_hidden,
                scores=scores_l1,
                indices=indices_l1,
                top_k=TOP_K,
                compute_kernel_config=self.shared_compute_config,
                partial_memory_config=self.hidden_act_memory_config,
            )
        except BaseException:
            _deallocate(logits, gate_up_scalar_ws, scores_l1, indices_l1)
            raise
        _deallocate(logits, gate_up_scalar_ws)
        self._check_routing_shards(indices_l1, scores_l1)
        self.mesh_contract.validate_tensor(scores_l1, placement=TensorPlacement.REPLICATED)
        self.mesh_contract.validate_tensor(indices_l1, placement=TensorPlacement.REPLICATED)
        phase_observer("after-router-topk")
        self.mesh_contract.mark_local_partial(
            partial, replicated_reference=full_hidden, expected_shape=self.row_contract.full_hidden
        )
        phase_observer("after-shared-partial")
        return Qwen38TTNNRouting(scores_l1, indices_l1, None), Qwen38TTNNSharedPartial(partial, sigmoid), sparse_rows

    def _check_routing_shards(self, indices_l1, scores_l1) -> None:
        if _shape(indices_l1) != self.row_contract.moe_routing or _shape(scores_l1) != self.row_contract.moe_routing:
            raise RuntimeError(
                f"moe_compute routing must be {self.row_contract.moe_routing}, got "
                f"scores={_shape(scores_l1)} indices={_shape(indices_l1)}"
            )
        if (
            indices_l1.layout != ttnn.ROW_MAJOR_LAYOUT
            or indices_l1.dtype != ttnn.uint16
            or indices_l1.memory_config() != self.routing_l1_memory_config
        ):
            raise RuntimeError("moe_compute indices are not RM UINT16 on the exact L1 drain-core shard")
        if (
            scores_l1.layout != ttnn.ROW_MAJOR_LAYOUT
            or scores_l1.dtype != ttnn.bfloat16
            or scores_l1.memory_config() != self.routing_l1_memory_config
        ):
            raise RuntimeError("moe_compute scores are not RM BF16 on the exact L1 drain-core shard")

    def _route_into_shards(self, logits_ws, phase_observer) -> Qwen38TTNNRouting:
        """The fused tail on the drained fp32 logits (the chain's own drain: the TensorAccessor misreads the router's
        bf16 L1 width shard on the upper tiles, measured 2026-09-15), writing its rows into the drain-core shards
        moe_compute reads: no routing copies."""

        logits = ttnn.to_memory_config(logits_ws, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
        _deallocate(logits_ws)
        phase_observer("after-router-logits")
        phase_observer("before-router-topk")
        routing_shape = ttnn.Shape(list(self.row_contract.moe_routing))
        scores_l1 = ttnn.allocate_tensor_on_device(
            routing_shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device, self.routing_l1_memory_config
        )
        indices_l1 = ttnn.allocate_tensor_on_device(
            routing_shape, ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device, self.routing_l1_memory_config
        )
        fused.program.stamp_topology(scores_l1, logits_ws)
        fused.program.stamp_topology(indices_l1, logits_ws)
        fused.router_tail.router_tail_into(logits, scores_l1, indices_l1, top_k=TOP_K)
        _deallocate(logits)
        self._check_routing_shards(indices_l1, scores_l1)
        self.mesh_contract.validate_tensor(scores_l1, placement=TensorPlacement.REPLICATED)
        self.mesh_contract.validate_tensor(indices_l1, placement=TensorPlacement.REPLICATED)
        phase_observer("after-router-topk")
        return Qwen38TTNNRouting(scores_l1, indices_l1, None)

    def _routing_rows(self, normalized, indices) -> Qwen38TTNNRouting:
        """One row tile's top-k as the 32-row program's ROW_MAJOR BF16 scores and UINT16 indices ``[1,1,32,10]``."""

        scores_rm = ttnn.to_layout(normalized, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        indices_rm = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(normalized, indices)
        if indices_rm.dtype != ttnn.uint16:
            converted = ttnn.typecast(indices_rm, ttnn.uint16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(indices_rm)
            indices_rm = converted
        expected = (1, 1, CHUNK_ROWS, TOP_K)
        if _shape(scores_rm) != expected or _shape(indices_rm) != expected:
            raise RuntimeError(
                f"router top-k tile shapes must be {expected}, got scores={_shape(scores_rm)} indices={_shape(indices_rm)}"
            )
        self.mesh_contract.validate_tensor(scores_rm, placement=TensorPlacement.REPLICATED)
        self.mesh_contract.validate_tensor(indices_rm, placement=TensorPlacement.REPLICATED)
        return Qwen38TTNNRouting(scores_rm, indices_rm)

    @property
    def slab_one_call(self) -> bool:
        """The one-call slab: the op's local output path, no zero fill of the rows this device's experts do not own
        (the buffer is zero at allocation and only ever holds finite expert outputs that the weighted reduce
        multiplies by an exact 0 where unowned), the reduce in 512-row blocks. Derived from the row contract (a slab
        instance whose one call covers its rows); ``QWEN38_MOE_SLAB_ONE_CALL`` (unset = 1) only sets that default."""

        return self.slab and self.routed_calls == 1

    @property
    def zero_fill_non_owned_rows(self) -> bool:
        # The one-call slab skips the fill (its buffer is zero at allocation); a rows form with several rings must
        # (the op refuses the fill with R >= 2: one ring's zero writes would race the others' rows) -- its buffer is
        # cleared by the fill before every call above, so the unowned rows are zero either way.
        if self.slab_one_call:
            return False
        return (MOE_ROWS_FORMS[getattr(self, "rows_form", "fulllocal")][1] or 0) < 2

    @property
    def prefill_rings(self) -> int | None:
        """The one-call slab's ring mode (``QWEN38_MOE_SLAB_RINGS``); ``None`` = the op's default for every other
        instance (the kwarg is not passed as a value, so the chunk forms' program hashes are untouched)."""

        if self.slab_one_call:
            rings = moe_slab_prefill_rings()
            return rings if rings else None
        return MOE_ROWS_FORMS[getattr(self, "rows_form", "fulllocal")][1]

    @property
    def slab(self) -> bool:
        """A prefill-slab instance (256..4096 rows): its dense linears are 2D-multicast matmuls, its routed stream
        the 128-row worker's call per block."""

        return is_slab_rows(self.rows)

    def _slab_router_logits(self, full_hidden):
        """The slab's router logits: one 2D-multicast matmul over every row (bf16 out, as the decode linear), widened
        to fp32 for the softmax exactly as the drain of the L1 shard widens them."""

        # EXACT: the logits pick the ten experts, so the router never takes the prefill dense policy (today's
        # config, the module's HiFi4 + fp32 config, the per-slab weight copy).
        router_bf16 = prefill_linear(
            full_hidden,
            self.weights.router,
            self._slab_program_config(HIDDEN_SIZE, ROUTED_EXPERTS, exact=True),
            compute_kernel_config=self.compute_config,
        )
        widened = ttnn.typecast(router_bf16, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(router_bf16)
        return widened

    def _slab_program_config(self, k: int, n: int, *, exact: bool = False):
        """The slab's 2D-multicast matmul config for one dense linear, built once per (k, n): today's config for the
        exact set (``exact=True``: the router) and under the default policy, else the prefill dense policy's wide grid
        (QWEN38_PREFILL_DENSE_GRID=wide)."""

        configs = self.__dict__.setdefault("_slab_program_configs", {})
        key = (k, n, exact)
        if key not in configs:
            if exact or self.prefill_dense.policy.grid == "today":
                configs[key] = prefill_matmul_program_config(self.mesh_device, self.rows, k, n)
            else:
                configs[key] = self.prefill_dense.program_config(self.rows, k, n)
        return configs[key]

    def _routed_partial_blocks(self, full_hidden, routing: Qwen38TTNNRouting, packed_w0_w1, packed_w2, phase_observer):
        """The slab's routed stream: the 128-row instance's call (``moe_compute`` on 128 tokens, the tilize, the
        weighted reduce) once per 128-row block of the slab, on the block's rows (a whole-tile slice) and its
        ROW_MAJOR routing rows, through the worker sharing the combine buffer; the partials concatenated."""

        if getattr(self, "block_instance", None) is None:
            raise RuntimeError("the slab MoE instance has no 128-row worker")
        phase_observer("before-routed-dispatch")
        dram = ttnn.DRAM_MEMORY_CONFIG
        block = LONG_PREFILL_CHUNK_ROWS
        partials = []
        for start in range(0, self.rows, block):
            rows = ttnn.slice(full_hidden, (0, 0, start, 0), (1, 1, start + block, HIDDEN_SIZE), memory_config=dram)
            scores = ttnn.slice(routing.scores, (0, 0, start, 0), (1, 1, start + block, TOP_K), memory_config=dram)
            indices = ttnn.slice(routing.indices, (0, 0, start, 0), (1, 1, start + block, TOP_K), memory_config=dram)
            self.mesh_contract.validate_tensor(rows, placement=TensorPlacement.REPLICATED)
            self.mesh_contract.validate_tensor(scores, placement=TensorPlacement.REPLICATED)
            self.mesh_contract.validate_tensor(indices, placement=TensorPlacement.REPLICATED)
            partials.append(
                self.block_instance._routed_partial(rows, Qwen38TTNNRouting(scores, indices), packed_w0_w1, packed_w2)
            )
            _deallocate(rows, scores, indices)
        phase_observer("after-routed-dispatch")
        partial = ttnn.concat(partials, dim=2, memory_config=dram)
        _deallocate(*partials)
        self.mesh_contract.mark_local_partial(
            partial, replicated_reference=full_hidden, expected_shape=self.row_contract.full_hidden
        )
        phase_observer("before-selective-reduce")
        phase_observer("after-selective-reduce")
        return partial

    def _routed_partial_tiles(self, full_hidden, routing: Qwen38TTNNRouting, packed_w0_w1, packed_w2, phase_observer):
        """The routed stream one 32-row tile at a time: the 32-row instance's ``moe_compute`` and weighted reduce
        on each tile (its rows and its ROW_MAJOR routing), the partials concatenated; bitwise the 32-row chunk."""

        if routing.tiles is None or len(routing.tiles) != self.routed_calls:
            raise ValueError(f"the per-tile routed stream needs {self.routed_calls} routing tiles")
        phase_observer("before-routed-dispatch")
        tile_rows = (1, 1, self.routed_tokens, HIDDEN_SIZE)
        sparse_shape = (1, self.routed_tokens, HIDDEN_SIZE)
        routing_shape = (1, self.routed_tokens, TOP_K)
        combine_shape = (TOP_K, self.routed_tokens, HIDDEN_SIZE)
        partials = []
        for rows, tile_routing in zip(dram_sharded_row_tiles(full_hidden, None), routing.tiles):
            if _shape(rows) != tile_rows:
                raise RuntimeError(f"routed tile rows are {_shape(rows)}, expected {tile_rows}")
            sparse_input = ttnn.to_layout(
                ttnn.reshape(rows, sparse_shape), ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            _deallocate(rows)
            if _shape(sparse_input) != sparse_shape:
                raise RuntimeError(f"moe_compute sparse input must be {sparse_shape}, got {_shape(sparse_input)}")
            indices_l1 = ttnn.to_memory_config(
                ttnn.reshape(tile_routing.indices, routing_shape), self.routing_l1_memory_config
            )
            scores_l1 = ttnn.to_memory_config(
                ttnn.reshape(tile_routing.scores, routing_shape), self.routing_l1_memory_config
            )
            if (
                _shape(indices_l1) != routing_shape
                or indices_l1.layout != ttnn.ROW_MAJOR_LAYOUT
                or indices_l1.dtype != ttnn.uint16
                or _shape(scores_l1) != routing_shape
                or scores_l1.layout != ttnn.ROW_MAJOR_LAYOUT
                or scores_l1.dtype != ttnn.bfloat16
            ):
                raise RuntimeError("moe_compute tile routing is not RM UINT16 / BF16 on the exact L1 drain-core shard")
            zeroed = ttnn.fill(self.local_combine_output, 0.0, output_tensor=self.local_combine_output)
            if zeroed.tensor_id != self.local_combine_output.tensor_id:
                raise RuntimeError("ttnn.fill did not update the persistent local-combine buffer in place")
            phase_observer("before-moe-compute-launch")
            outputs = ttnn.experimental.moe_compute(
                sparse_input,
                indices_l1,
                scores_l1,
                self.expert_mapping,
                packed_w0_w1,
                packed_w2,
                layer_id=0,
                output_height_shard_dim=self.output_height_shard_dim,
                intermediate_size=INTERMEDIATE_SIZE,
                has_bias=False,
                cluster_axis=LOCAL_COMBINE_AXIS,
                topology=None,
                num_links=None,
                mux_core_range_set=None,
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                optional_output_tensor=self.local_combine_output,
                optional_cross_device_semaphore=None,
                activation_type=MoEActivationFunction.SILU,
                compute_only=False,
                local_combine=not self.local_output,
                num_shared_experts_per_device=0,
                zero_fill_non_owned_rows=self.zero_fill_non_owned_rows,
                prefill_rings=self.prefill_rings,
                enable_a2a_pipeline=not self.prefill_rings,
            )
            phase_observer("after-moe-compute-launch")
            if len(outputs) != 6 or outputs[5].tensor_id != self.local_combine_output.tensor_id:
                raise RuntimeError("local moe_compute did not return the persistent six-slot output")
            if _shape(outputs[5]) != combine_shape:
                raise RuntimeError(f"local combine produced {_shape(outputs[5])}, expected {combine_shape}")
            _deallocate(outputs[0], outputs[1], outputs[2], outputs[4], sparse_input, indices_l1, scores_l1)
            local_stack_tiled = ttnn.to_layout(
                ttnn.unsqueeze(outputs[5], dim=1), ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, pad_value=0.0
            )
            fast_outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                local_stack_tiled,
                tile_routing.indices,
                self.expert_mapping,
                reduce_dim=0,
                split_size=HIDDEN_SIZE,
                cluster_axis=LOCAL_COMBINE_AXIS,
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                scores_tensor=ttnn.reshape(tile_routing.scores, (self.routed_tokens, 1, 1, TOP_K)),
                num_shared_experts=0,
                shared_expert_scale=1.0,
                compute_kernel_config=self.compute_config,
            )
            _deallocate(local_stack_tiled)
            if len(fast_outputs) != 1 or _shape(fast_outputs[0]) != tile_rows:
                raise RuntimeError(
                    f"weighted routed reduce must return one {tile_rows} partial, got "
                    f"{tuple(_shape(item) for item in fast_outputs)}"
                )
            partials.append(fast_outputs[0])
        phase_observer("after-routed-dispatch")
        partial = ttnn.concat(partials, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(*partials)
        self.mesh_contract.mark_local_partial(
            partial, replicated_reference=full_hidden, expected_shape=self.row_contract.full_hidden
        )
        phase_observer("before-selective-reduce")
        phase_observer("after-selective-reduce")
        return partial

    def _shared_partial_slab(self, full_hidden):
        """The shared-expert chain over every slab row: the four linears as 2D-multicast matmuls on interleaved
        weight copies, the activations interleaved (the same element arithmetic as the per-tile form).  Under the
        prefill dense policy's switches the linears read their resident prefill copies and the policy's fidelity, and
        gate and up run as one [gate | up] linear cut into two whole-tile slices (the prefill dense design note)."""

        dram = ttnn.DRAM_MEMORY_CONFIG
        local_intermediate = INTERMEDIATE_SIZE // MESH_SHAPE[1]
        dense = self.prefill_dense
        if dense.resident("shared_gate_up") is not None:
            # QWEN38_PREFILL_DENSE_GRID=wide: gate and up as one [gate | up] linear on the resident prefill weight, cut
            # into two whole-tile column slices with literal bounds (the captured body admits no host-integer shape op).
            fused = dense.fused_linear(
                full_hidden, "shared_gate_up", self.rows, compute_kernel_config=self.shared_compute_config
            )
            gate = ttnn.slice(fused, (0, 0, 0, 0), (1, 1, self.rows, local_intermediate), memory_config=dram)
            up = ttnn.slice(
                fused, (0, 0, 0, local_intermediate), (1, 1, self.rows, 2 * local_intermediate), memory_config=dram
            )
            _deallocate(fused)
            dense.retag_sharded(gate, up, reference=full_hidden, shard_dim=3)
        else:
            gate = prefill_linear(
                full_hidden,
                self.weights.shared_gate,
                self._slab_program_config(HIDDEN_SIZE, local_intermediate),
                compute_kernel_config=self.prefill_dense.compute_config(self.shared_compute_config),
                resident_weight=dense.resident("shared_gate"),
            )
            up = prefill_linear(
                full_hidden,
                self.weights.shared_up,
                self._slab_program_config(HIDDEN_SIZE, local_intermediate),
                compute_kernel_config=self.prefill_dense.compute_config(self.shared_compute_config),
                resident_weight=dense.resident("shared_up"),
            )
        self.mesh_contract.validate_tensor(gate, placement=TensorPlacement.INTERMEDIATE_SHARDED, shard_dim=3)
        self.mesh_contract.validate_tensor(up, placement=TensorPlacement.INTERMEDIATE_SHARDED, shard_dim=3)
        gate_activated = ttnn.silu(gate, memory_config=dram)
        intermediate = ttnn.mul(gate_activated, up, memory_config=dram)
        _deallocate(gate, gate_activated, up)
        partial = prefill_linear(
            intermediate,
            self.weights.shared_down,
            self._slab_program_config(local_intermediate, HIDDEN_SIZE),
            compute_kernel_config=self.prefill_dense.compute_config(self.shared_compute_config),
            resident_weight=dense.resident("shared_down"),
        )
        _deallocate(intermediate)
        self.mesh_contract.mark_local_partial(
            partial, replicated_reference=full_hidden, expected_shape=self.row_contract.full_hidden
        )
        scalar = prefill_linear(
            full_hidden,
            self.weights.shared_scalar_gate,
            self._slab_program_config(HIDDEN_SIZE, 1),
            compute_kernel_config=self.prefill_dense.compute_config(self.shared_compute_config),
            resident_weight=dense.resident("shared_scalar_gate"),
        )
        self.mesh_contract.validate_tensor(scalar, placement=TensorPlacement.REPLICATED)
        scalar_gate = ttnn.sigmoid(scalar, memory_config=dram)
        gated_partial = ttnn.mul(partial, scalar_gate, memory_config=dram)
        _deallocate(partial, scalar, scalar_gate)
        self.mesh_contract.mark_local_partial(
            gated_partial, replicated_reference=full_hidden, expected_shape=self.row_contract.full_hidden
        )
        return gated_partial

    def _shared_partial(self, hidden_sharded, full_hidden, hidden_tiles=None):
        if _shape(hidden_sharded) != self.row_contract.hidden_sharded:
            raise ValueError(
                f"shared-expert hidden shard must be {self.row_contract.hidden_sharded}, got {_shape(hidden_sharded)}"
            )
        if _shape(full_hidden) != self.row_contract.full_hidden:
            raise ValueError(
                f"shared-expert full hidden must be {self.row_contract.full_hidden}, got {_shape(full_hidden)}"
            )
        # The four shared-expert linears are DRAM-sharded decode linears (one row tile per call): the long
        # chunk runs the whole shared chain per row tile (``hidden_tiles``: the router's tiles, moved into the
        # activation shard once per layer) and concatenates the gated partials.
        tile_shape = self.row_contract.full_hidden
        if self.slab:
            return self._shared_partial_slab(full_hidden)
        if self.shared_expert_fused:
            partial, sigmoid = fused.shared_expert.shared_expert(
                full_hidden,
                self.weights.shared_gate_up_scalar,
                self.weights.shared_down,
                gate_up_scalar_program_config=self.shared_gate_up_scalar_program_config,
                down_program_config=self.shared_down_program_config,
                intermediate_memory_config=self.shared_intermediate_memory_config,
                compute_kernel_config=self.shared_compute_config,
            )
            self.mesh_contract.mark_local_partial(
                partial, replicated_reference=full_hidden, expected_shape=self.row_contract.full_hidden
            )
            return Qwen38TTNNSharedPartial(partial, sigmoid)
        if self.row_contract.row_tiles == 1:
            hidden_tiles = [full_hidden]
        elif hidden_tiles is None or len(hidden_tiles) != self.row_contract.row_tiles:
            raise ValueError(f"the long chunk's shared experts need its {self.row_contract.row_tiles} hidden row tiles")
        else:
            tile_shape = (1, 1, CHUNK_ROWS, HIDDEN_SIZE)
        gated_partials = []
        for hidden_tile in hidden_tiles:
            gate = ttnn.linear(
                hidden_tile,
                self.weights.shared_gate,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.shared_gate_up_program_config,
                compute_kernel_config=self.shared_compute_config,
            )
            up = ttnn.linear(
                hidden_tile,
                self.weights.shared_up,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.shared_gate_up_program_config,
                compute_kernel_config=self.shared_compute_config,
            )
            self.mesh_contract.validate_tensor(gate, placement=TensorPlacement.INTERMEDIATE_SHARDED, shard_dim=3)
            self.mesh_contract.validate_tensor(up, placement=TensorPlacement.INTERMEDIATE_SHARDED, shard_dim=3)
            gate_activated = ttnn.silu(gate, memory_config=self.shared_intermediate_memory_config)
            intermediate = ttnn.mul(gate_activated, up, memory_config=self.shared_intermediate_memory_config)
            _deallocate(gate, gate_activated, up)
            partial = ttnn.linear(
                intermediate,
                self.weights.shared_down,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.shared_down_program_config,
                compute_kernel_config=self.shared_compute_config,
            )
            _deallocate(intermediate)
            self.mesh_contract.mark_local_partial(
                partial,
                replicated_reference=full_hidden,
                expected_shape=tile_shape,
            )

            scalar_ws = ttnn.linear(
                hidden_tile,
                self.weights.shared_scalar_gate,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.shared_scalar_program_config,
                compute_kernel_config=self.shared_compute_config,
            )
            scalar = ttnn.to_memory_config(scalar_ws, ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(scalar_ws)
            self.mesh_contract.validate_tensor(scalar, placement=TensorPlacement.REPLICATED)
            scalar_gate = ttnn.sigmoid(scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # The five-core partial is read in place; the scalar gate broadcasts
            # from DRAM and the product lands interleaved for the branch add.
            gated_partials.append(ttnn.mul(partial, scalar_gate, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            _deallocate(partial, scalar, scalar_gate)
        gated_partial = gated_partials[0]
        if self.row_contract.row_tiles != 1:
            gated_partial = ttnn.concat(gated_partials, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(*gated_partials)
        self.mesh_contract.mark_local_partial(
            gated_partial,
            replicated_reference=full_hidden,
            expected_shape=self.row_contract.full_hidden,
        )
        return gated_partial

    def forward(
        self,
        hidden_sharded,
        packed_w0_w1,
        packed_w2,
        *,
        return_routing: bool = False,
        phase_observer=None,
    ) -> Qwen38TTNNMoEResult:
        """Execute the exact MoE at this instance's row count (1-row decode, 5-row verifier, 32-row prefill chunk)."""

        if phase_observer is not None and not callable(phase_observer):
            raise TypeError("MoE phase observer must be callable")
        self._require_owned_buffers()
        deferred_phase_errors: list[BaseException] = []

        def observe(phase: str) -> None:
            boundary, separator, stage = phase.partition("-")
            if separator != "-" or boundary not in {"before", "after"} or stage not in MOE_PHASE_STAGES:
                raise RuntimeError(f"invalid internal MoE phase {phase!r}")
            if phase_observer is None:
                return
            try:
                phase_observer(phase)
            except BaseException as error:
                deferred_phase_errors.append(error)

        def raise_deferred_phase_error() -> None:
            if deferred_phase_errors:
                raise deferred_phase_errors[0]

        routing = None
        temporaries = {
            "full_hidden": None,
            "hidden_tiles": None,
            "routing_scores": None,
            "routing_indices": None,
            "routing_tiles": None,
            "shared_partial": None,
            "sparse_rows": None,
            "routed_partial": None,
            "local_sum": None,
            "output": None,
        }

        def release(name: str) -> None:
            tensor = temporaries[name]
            if tensor is None:
                return
            if name == "routing_tiles":
                for tile in tensor:
                    ttnn.deallocate(tile.scores)
                    ttnn.deallocate(tile.indices)
            elif name == "hidden_tiles":
                for tile in tensor:
                    ttnn.deallocate(tile)
            elif isinstance(tensor, Qwen38TTNNSharedPartial):
                ttnn.deallocate(tensor.partial)
                ttnn.deallocate(tensor.sigmoid)
            else:
                ttnn.deallocate(tensor)
            # Clear immediately after each successful release so a later
            # release failure never retries an already-freed device owner.
            temporaries[name] = None

        def release_many(*names: str) -> None:
            for name in names:
                release(name)

        try:
            # Keep the six proven dependency boundaries explicit.  Direct or
            # streamed construction drains at each boundary.  A resident
            # graph visits the same boundaries without a host call: each
            # producer remains owned until its final same-CQ consumer enqueue
            # returns, matching the Qwen3.6 traced schedule.
            observe("before-hidden-all-gather")
            raise_deferred_phase_error()
            temporaries["full_hidden"] = self._all_gather_hidden(hidden_sharded)
            self._synchronize_stage("all-gather-hidden")
            observe("after-hidden-all-gather")
            raise_deferred_phase_error()

            # The gathered width-sharded hidden feeds the router, every
            # shared-expert linear and the routed untilize; it is owned until
            # the reduce-scatter enqueue returns.  The long chunk moves its four
            # row tiles into the dense linears' shard once for the router and
            # the shared chain and lets them go before the expert stream.
            if self.row_contract.row_tiles != 1:
                if not self.slab:  # the slab's linears take the interleaved rows whole
                    temporaries["hidden_tiles"] = dram_sharded_row_tiles(
                        temporaries["full_hidden"], self.hidden_act_memory_config
                    )
            if self.moe_dense_fused:
                # the MoE dense composite: the router linear and drain, the shared [gate | up | scalar] linear, then
                # one program for the top-k, the shared eltwise + down linear and the routed dispatch untilize
                routing, shared, sparse_rows = self._dense_composite(temporaries["full_hidden"], observe)
                temporaries["shared_partial"], temporaries["sparse_rows"] = shared, sparse_rows
            else:
                routing = self._route(
                    temporaries["full_hidden"], hidden_tiles=temporaries["hidden_tiles"], phase_observer=observe
                )
            temporaries["routing_scores"] = routing.scores
            temporaries["routing_indices"] = routing.indices
            temporaries["routing_tiles"] = routing.tiles
            self._synchronize_stage("route")
            raise_deferred_phase_error()

            if not self.moe_dense_fused:
                observe("before-shared-partial")
                raise_deferred_phase_error()
                temporaries["shared_partial"] = self._shared_partial(
                    hidden_sharded, temporaries["full_hidden"], temporaries["hidden_tiles"]
                )
            release("hidden_tiles")
            self._synchronize_stage("shared-partial")
            if not self.moe_dense_fused:
                observe("after-shared-partial")
            raise_deferred_phase_error()

            if self.moe_post_fused:
                # the fused post program returns the local sum (routed + shared) itself
                temporaries["local_sum"] = self._routed_local_sum(
                    temporaries["full_hidden"],
                    routing,
                    packed_w0_w1,
                    packed_w2,
                    temporaries["shared_partial"],
                    phase_observer=observe,
                    sparse_rows=temporaries[
                        "sparse_rows"
                    ],  # the composite's rows: owned here until moe_compute has them
                )
                release("sparse_rows")
            else:
                temporaries["routed_partial"] = self._routed_partial(
                    temporaries["full_hidden"],
                    routing,
                    packed_w0_w1,
                    packed_w2,
                    phase_observer=observe,
                )
            self._synchronize_stage("routed-partial")
            raise_deferred_phase_error()

            observe("before-partial-combine")
            raise_deferred_phase_error()
            if self.moe_post_fused:
                release("shared_partial")
            else:
                temporaries["local_sum"] = ttnn.add(
                    temporaries["routed_partial"],
                    temporaries["shared_partial"],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                release_many("routed_partial", "shared_partial")
            self.mesh_contract.mark_local_partial(
                temporaries["local_sum"],
                replicated_reference=temporaries["full_hidden"],
                expected_shape=self.row_contract.full_hidden,
            )
            self._synchronize_stage("local-add-mark")
            observe("after-partial-combine")
            raise_deferred_phase_error()

            # Match the Qwen3.6 TP4 line-mesh path: the helper deliberately
            # uses cluster_axis=0 on (1,4) to select whole-line minimal
            # reduce-scatter with the model-scoped, cyclic TT_CCL semaphore
            # pool.
            observe("before-output-reduce-scatter")
            raise_deferred_phase_error()
            temporaries["output"] = tt_all_reduce(
                temporaries["local_sum"],
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=0,
                dim=3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.collective_topology,
            )
            self.mesh_contract.mark_collective_shard(
                temporaries["output"],
                replicated_reference=temporaries["full_hidden"],
                shard_dim=3,
                expected_local_shape=self.row_contract.output_sharded,
            )
            if _shape(temporaries["output"]) != self.row_contract.output_sharded:
                raise RuntimeError(
                    f"MoE reduce-scatter produced {_shape(temporaries['output'])}, "
                    f"expected {self.row_contract.output_sharded}"
                )
            self._synchronize_stage("final-all-reduce")
            observe("after-output-reduce-scatter")
            raise_deferred_phase_error()

            # The reduce-scatter is asynchronous.  Retain its direct input and
            # replicated reference through the final boundary.  In resident
            # async mode, successful reduce-scatter enqueue is the legal
            # same-CQ ownership boundary; fenced mode additionally drains.
            observe("before-output-release")
            raise_deferred_phase_error()
            release_many("local_sum", "full_hidden", "routing_tiles")

            if return_routing:
                # The routing tiles (the long chunk's per-tile inputs) are released above; the caller reads scores/indices.
                result = Qwen38TTNNMoEResult(temporaries["output"], routing)
                temporaries["output"] = None
                temporaries["routing_scores"] = None
                temporaries["routing_indices"] = None
                observe("after-output-release")
                if deferred_phase_errors:
                    temporaries["output"] = result.hidden_sharded
                    temporaries["routing_scores"] = routing.scores
                    temporaries["routing_indices"] = routing.indices
                    raise_deferred_phase_error()
                return result
            release_many("routing_scores", "routing_indices")
            routing = None
            result = Qwen38TTNNMoEResult(temporaries["output"])
            temporaries["output"] = None
            observe("after-output-release")
            if deferred_phase_errors:
                temporaries["output"] = result.hidden_sharded
                raise_deferred_phase_error()
            return result
        except BaseException as forward_error:
            # A diagnostic may install this private failure-only hook to drain
            # asynchronous work before exception cleanup releases its owners.
            # Without that proof, resident async retains every live wrapper and
            # poisons the instance instead of enqueueing uncertain cleanup.
            diagnostic_before_cleanup = getattr(self, "_diagnostic_before_exception_cleanup", None)
            if diagnostic_before_cleanup is not None:
                if not callable(diagnostic_before_cleanup):
                    raise RuntimeError("MoE diagnostic exception-cleanup hook is not callable") from forward_error
                try:
                    diagnostic_before_cleanup(forward_error, temporaries)
                except BaseException as diagnostic_drain_error:
                    # Do not release potentially in-flight owners after a drain
                    # failure.  Mesh teardown is the only safe recovery.
                    self._retain_async_failure_owners(
                        diagnostic_drain_error,
                        hidden_sharded,
                        packed_w0_w1,
                        packed_w2,
                        *temporaries.values(),
                    )
                    raise RuntimeError(
                        "MoE forward failed and its diagnostic pre-cleanup drain also failed; cleanup skipped"
                    ) from diagnostic_drain_error
            elif self.synchronization_policy is Qwen38TTNNMoESyncPolicy.RESIDENT_ASYNC:
                # No host drain has established completion.  Retain borrowed
                # producers and every still-live result instead of submitting
                # cleanup to a queue whose ownership is now ambiguous.  The
                # enclosing model also poisons itself; recovery is teardown.
                self._retain_async_failure_owners(
                    forward_error,
                    hidden_sharded,
                    packed_w0_w1,
                    packed_w2,
                    *temporaries.values(),
                )
                raise RuntimeError(
                    "asynchronous MoE forward failed; live device owners retained and cleanup skipped"
                ) from forward_error
            cleanup_failures: list[tuple[str, BaseException]] = []
            for name in (
                "output",
                "local_sum",
                "routed_partial",
                "shared_partial",
                "sparse_rows",
                "hidden_tiles",
                "full_hidden",
                "routing_scores",
                "routing_indices",
                "routing_tiles",
            ):
                try:
                    release(name)
                except BaseException as cleanup_error:
                    cleanup_failures.append((name, cleanup_error))
            if cleanup_failures:
                failed = ", ".join(name for name, _error in cleanup_failures)
                raise RuntimeError(f"MoE forward failed and temporary cleanup also failed: {failed}") from forward_error
            raise
