# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact four-P150 owner for ordinary Qwen3.8-Flash-Next TTNN decode.

This module composes already-created, provenance-bound device modules.  It
does not hide conversion or construct weights: callers must provide the exact
vocabulary-sharded model I/O, all 48 target layers, and the backbone terminal
hyper-connection mixer.  Construction validates their live 1x4 physical mesh
identity and rejects replicated or numerically similar substitutes.

One ordinary decode step is deliberately serialized:

* validate and upload one true-global-B1 token;
* look up its vocab-row-sharded embedding and reduce-scatter to H/4;
* repeat that H/4 shard across the checkpoint's four residual branches;
* run all 48 target layers in checkpoint order, streaming exactly one BF4_B
  routed-expert layer at a time through their shared streamer;
* apply the terminal hyper-connection read and the untied vocab-sharded head.

QSA RoPE is generated for the exact absolute position on the host and uploaded
once per model step as a replicated read-only input.  Positions which close a
four-token index block additionally receive that block's first-position RoPE.
GDN layers never receive QSA inputs.

State is explicit and single-sequence.  Model-wide one-shot transactions
compose the per-layer GDN copy snapshots, QSA immutable cache views, and PLE
history snapshots.  The first speculative step after :meth:`snapshot_state`
must set ``retain_input_state=True``; the QSA owner independently fail-closes
if that rule is violated.  Any exception after a model-level mutation boundary
permanently poisons this owner; no state or snapshot can then be reused.  The
caller must close the task-owned process/mesh without resetting hardware.

No operation here opens a device or weakens lease requirements.  The supplied
mesh must already have been opened under the task's full-lifetime exact-device
locks and enforcing broker lease.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, NoReturn

import torch
from ttnn.tools.trace_allocation_tracker import corruptible_allocation_scope

import ttnn
from models.demos.blackhole.qwen38_flash_next import mrope, vision_splice
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256, LAYER_PATTERN, Qwen38Config
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import moe as moe_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROW_COUNTS,
    CHUNK_ROWS,
    LONG_CHUNK_ROWS,
    MESH_SHAPE,
    Qwen38MeshContract,
    Qwen38TTNNDevicePosition,
    Qwen38TTNNDevicePositionRow,
    TensorPlacement,
    chunk_row_tiles,
    is_slab_rows,
    replicate_tensor_2d_mesh_mapper,
    require_lane_count,
    require_rope_shift,
    tensor_metadata,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    Qwen38ShardedLogits,
    Qwen38TTNNEmbeddingSyncPolicy,
    Qwen38TTNNModelIO,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.final_mixer import Qwen38TTNNFinalMixer
from models.demos.blackhole.qwen38_flash_next.ttnn.gr import block_rows_shape, residual_rows_shape
from models.demos.blackhole.qwen38_flash_next.ttnn.lanes import Qwen38LaneLayout
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import (
    BACKBONE_LAYERS,
    BLOCK_LOCAL_SHAPE,
    PLE_CHECKPOINT_LAYER,
    RESIDUAL_LOCAL_SHAPE,
    Qwen38TTNNDecoderLayer,
    Qwen38TTNNDecoderLayerAux,
    Qwen38TTNNDecoderLayerChunkState,
    Qwen38TTNNDecoderLayerGenericState,
    Qwen38TTNNDecoderLayerLaneState,
    Qwen38TTNNDecoderLayerSnapshot,
    Qwen38TTNNDecoderLayerState,
    Qwen38TTNNLayerCleanupError,
    Qwen38TTNNLayerNamespace,
    Qwen38TTNNLayerType,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import (
    Qwen38TTNNPLELanesPreparedInput,
    Qwen38TTNNPLEPreparedInput,
    Qwen38TTNNPLERowsPreparedInput,
)

TP_SIZE = 4
HIDDEN_SIZE = 2560
LOCAL_HIDDEN_SIZE = HIDDEN_SIZE // TP_SIZE
RESIDUAL_BRANCHES = 4
VOCAB_SIZE = 248320
LOCAL_VOCAB_SIZE = VOCAB_SIZE // TP_SIZE
QSA_ROPE_DIM = 64
QSA_COMPRESS_RATIO = 4
ROPE_THETA = 10_000_000
MAX_CONTEXT = 262144
EXPECTED_CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
EXPECTED_LAYER_PATTERN = ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 12
EXPECTED_VOCAB_RANGES = tuple(
    (coordinate * LOCAL_VOCAB_SIZE, (coordinate + 1) * LOCAL_VOCAB_SIZE) for coordinate in range(TP_SIZE)
)
# HEAD/TAIL split of the position-generic body.  HEAD = device-token embedding +
# layers[:GENERIC_HEAD_LAYERS]; TAIL = the device position derivation,
# layers[GENERIC_HEAD_LAYERS:], final mixer, LM head and the position advance.
# The first TAIL layer is the PLE consumer (PLE_CHECKPOINT_LAYER), so HEAD reads
# neither the host-written PLE row nor the position and the retained handoff is
# released by that layer's release_input keyword alone.
GENERIC_HEAD_LAYERS = 1
GENERIC_TRACE_PARTS_SINGLE = ("body",)
GENERIC_TRACE_PARTS_SPLIT = ("head", "tail")


class Qwen38TTNNCleanupError(RuntimeError):
    """One or more independent task-owned tensor releases failed."""

    def __init__(self, label: str, errors: list[BaseException], *, primary: BaseException | None = None) -> None:
        self.label = label
        self.errors = tuple(errors)
        self.primary = primary
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(f"{label} cleanup failed for {len(errors)} resource(s): {detail}")


class Qwen38TTNNModelPoisonedError(RuntimeError):
    """The owner observed an exception after state may have mutated."""

    def __init__(self, operation: str, processed_layers: int, cause: BaseException) -> None:
        self.operation = operation
        self.processed_layers = processed_layers
        self.original_cause = cause
        super().__init__(
            f"TTNN text model is poisoned after {operation}; {processed_layers} layer result(s) completed. "
            "Do not reuse this owner or any of its state; close the task-owned mesh/process without reset. "
            f"Cause: {type(cause).__name__}: {cause}"
        )


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _padded_shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.padded_shape)


def _tensor_key(tensor) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


def _deallocate_unique(*tensors) -> None:
    slots = list(tensors)
    _release_tensor_slots(slots, label="TTNN tensor")


def _copy_inplace(source, target, *, label: str) -> None:
    """Copy into a persistent target and reject an address-changing result (gdn.py's rule)."""

    target_id = _tensor_key(target)
    copied = ttnn.copy(source, target)
    if _tensor_key(target) != target_id:
        raise RuntimeError(f"{label} target address changed during ttnn.copy")
    if copied is not None and _tensor_key(copied) != target_id:
        raise RuntimeError(f"{label} ttnn.copy returned a different tensor")


def _release_tensor_slot_indices(
    slots: list[Any | None],
    indices: Sequence[int],
    *,
    label: str,
    primary: BaseException | None = None,
) -> None:
    """Release selected alias groups and retain failed groups for one retry."""

    selected = tuple(dict.fromkeys(indices))
    if any(index < 0 or index >= len(slots) for index in selected):
        raise IndexError(f"{label} cleanup selected an out-of-range tensor slot")
    groups: dict[tuple[str, int], list[int]] = {}
    for index in selected:
        tensor = slots[index]
        if tensor is not None:
            groups.setdefault(_tensor_key(tensor), []).append(index)
    errors = []
    for key, selected_indices in groups.items():
        tensor = slots[selected_indices[0]]
        # Clear every alias in the complete owner, not only selected aliases.
        # Otherwise a later whole-owner release could free the same allocation.
        alias_indices = [
            index for index, candidate in enumerate(slots) if candidate is not None and _tensor_key(candidate) == key
        ]
        try:
            ttnn.deallocate(tensor)
        except BaseException as error:
            errors.append(error)
        else:
            for index in alias_indices:
                slots[index] = None
    if errors:
        raise Qwen38TTNNCleanupError(label, errors, primary=primary) from primary


def _release_tensor_slots(
    slots: list[Any | None],
    *,
    label: str,
    primary: BaseException | None = None,
) -> None:
    """Release every unique tensor and clear only successfully released slots.

    The caller retains the mutable ``slots`` list.  If one release fails, a
    retry visits only the failed ownership groups and cannot double-free a
    tensor which a previous attempt released successfully.
    """

    _release_tensor_slot_indices(slots, range(len(slots)), label=label, primary=primary)


def _run_cleanup_actions(
    label: str,
    actions: Sequence[tuple[str, Callable[[], None]]],
    *,
    primary: BaseException | None = None,
) -> None:
    """Attempt every independent cleanup action before reporting failures."""

    errors = []
    for resource, action in actions:
        try:
            action()
        except BaseException as error:
            wrapped = RuntimeError(f"{resource}: {type(error).__name__}: {error}")
            wrapped.__cause__ = error
            errors.append(wrapped)
    if errors:
        raise Qwen38TTNNCleanupError(label, errors, primary=primary) from primary


def _cleanup_failure_cause(
    label: str,
    actions: Sequence[tuple[str, Callable[[], None]]],
    primary: BaseException,
) -> BaseException:
    """Return ``primary`` unless cleanup itself adds actionable uncertainty."""

    try:
        _run_cleanup_actions(label, actions, primary=primary)
    except BaseException as cleanup_error:
        return cleanup_error
    return primary


def _validate_exact_config(config: Qwen38Config) -> None:
    expected = {
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "hidden_size": HIDDEN_SIZE,
        "residual_branches": RESIDUAL_BRANCHES,
        "vocab_size": VOCAB_SIZE,
        "num_hidden_layers": BACKBONE_LAYERS,
        "layer_types": EXPECTED_LAYER_PATTERN,
        "max_position_embeddings": MAX_CONTEXT,
        "qsa_rope_dim": QSA_ROPE_DIM,
        "index_compress_ratio": QSA_COMPRESS_RATIO,
        "rope_theta": ROPE_THETA,
    }
    for name, required in expected.items():
        actual = getattr(config, name)
        if actual != required:
            raise ValueError(f"pinned Qwen3.8 model requires {name}={required!r}, got {actual!r}")
    if CONFIG_SHA256 != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("compiled Qwen3.8 config identity drifted")
    if tuple(LAYER_PATTERN) != EXPECTED_LAYER_PATTERN:
        raise RuntimeError("compiled Qwen3.8 layer pattern drifted")


def _normalize_token_id(token_id: int | torch.Tensor) -> tuple[int, torch.Tensor]:
    if isinstance(token_id, bool):
        raise TypeError("token ID cannot be bool")
    if isinstance(token_id, int):
        value = token_id
    elif isinstance(token_id, torch.Tensor):
        if token_id.device.type != "cpu":
            raise ValueError("ordinary decode token ID must be host resident")
        if token_id.dtype not in (torch.int32, torch.int64) or tuple(token_id.shape) != (1, 1):
            raise ValueError("ordinary decode token tensor must be CPU int32/int64 [1,1]")
        value = int(token_id.item())
    else:
        raise TypeError(f"ordinary decode requires one integer token ID, got {type(token_id).__name__}")
    if not 0 <= value < VOCAB_SIZE:
        raise IndexError(f"token ID must be in [0,{VOCAB_SIZE}), got {value}")
    return value, torch.tensor([[value]], dtype=torch.long)


def _inverse_frequency(config: Qwen38Config) -> torch.Tensor:
    return mrope.rope_inverse_frequency(float(config.rope_theta), QSA_ROPE_DIM)


def _host_rope(position: int, inverse_frequency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The plain RoPE row of one position (``mrope.rope_row``: the scalar path every table row is built from)."""

    if not 0 <= position < MAX_CONTEXT:
        raise IndexError(f"RoPE position must be in [0,{MAX_CONTEXT}), got {position}")
    return mrope.rope_row(position, inverse_frequency)


@dataclass
class Qwen38TTNNRoPEInputs:
    """Task-owned replicated device RoPE tensors for one absolute position.

    ``position`` is the host position of a per-position upload and ``None`` for
    rows selected on device by :class:`Qwen38TTNNRoPETable`.
    """

    position: int | None
    cos: Any
    sin: Any
    block_start_cos: Any | None
    block_start_sin: Any | None
    active: bool = True
    _owned_tensors: list[Any | None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._owned_tensors = [self.cos, self.sin, self.block_start_cos, self.block_start_sin]

    def deallocate(self) -> None:
        if not self.active:
            raise RuntimeError("QSA RoPE inputs were already deallocated")
        _release_tensor_slots(self._owned_tensors, label="QSA RoPE inputs")
        self.active = not all(tensor is None for tensor in self._owned_tensors)


class Qwen38TTNNRoPE:
    """Exact text-only Qwen4Exp RoPE uploader for serialized decode."""

    def __init__(self, mesh_device, mesh_contract: Qwen38MeshContract, config: Qwen38Config) -> None:
        mesh_contract.validate_mesh(mesh_device)
        _validate_exact_config(config)
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.inverse_frequency = _inverse_frequency(config)

    def _upload(self, host: torch.Tensor):
        if host.dtype != torch.bfloat16 or tuple(host.shape) != (1, 1, 1, QSA_ROPE_DIM):
            raise RuntimeError("host QSA RoPE row must be BF16 [1,1,1,64]")
        result = ttnn.from_torch(
            host.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
        )
        try:
            self.mesh_contract.validate_tensor(result, placement=TensorPlacement.REPLICATED)
            if _shape(result) != (1, 1, 1, QSA_ROPE_DIM):
                raise RuntimeError(f"replicated QSA RoPE has local shape {_shape(result)}, expected [1,1,1,64]")
        except BaseException as error:
            slots = [result]
            _release_tensor_slots(slots, label="invalid QSA RoPE upload", primary=error)
            raise
        return result

    def for_position(self, position: int) -> Qwen38TTNNRoPEInputs:
        host_cos, host_sin = _host_rope(position, self.inverse_frequency)
        cos = self._upload(host_cos)
        try:
            sin = self._upload(host_sin)
        except BaseException as error:
            slots = [cos]
            _release_tensor_slots(slots, label="partial QSA RoPE inputs", primary=error)
            raise

        block_start_cos = None
        block_start_sin = None
        if position % QSA_COMPRESS_RATIO == QSA_COMPRESS_RATIO - 1:
            block_start = position - (QSA_COMPRESS_RATIO - 1)
            host_block_cos, host_block_sin = _host_rope(block_start, self.inverse_frequency)
            try:
                block_start_cos = self._upload(host_block_cos)
                block_start_sin = self._upload(host_block_sin)
            except BaseException as error:
                slots = [cos, sin, block_start_cos, block_start_sin]
                _release_tensor_slots(slots, label="partial block-closing QSA RoPE inputs", primary=error)
                raise
        return Qwen38TTNNRoPEInputs(position, cos, sin, block_start_cos, block_start_sin)


@dataclass(frozen=True)
class Qwen38TTNNRoPETable:
    """Resident cos/sin rows for every allocated position, selected on device by ``P``.

    Row ``p`` of each table is exactly ``_host_rope(p)``, so an embedding lookup
    returns the same bits a per-position upload would.  One table pair per
    model; the position-generic body reads it with the device position and
    never uploads RoPE per token.
    """

    cos_table: Any
    sin_table: Any
    allocated_context: int
    inverse_frequency: torch.Tensor = field(repr=False, compare=False)
    mesh_contract: Qwen38MeshContract = field(repr=False, compare=False)
    # The host image of the two tables, BF16 ``[allocated_context, 64]``: the prefill chunks' host-written rows are
    # gathered from it (``host_rows_for``), so a text token's row is bitwise the device table's row.
    host_cos: torch.Tensor = field(repr=False, compare=False, default=None)
    host_sin: torch.Tensor = field(repr=False, compare=False, default=None)

    @classmethod
    def build(
        cls, mesh_device, mesh_contract: Qwen38MeshContract, config: Qwen38Config, allocated_context: int
    ) -> Qwen38TTNNRoPETable:
        mesh_contract.validate_mesh(mesh_device)
        _validate_exact_config(config)
        if isinstance(allocated_context, bool) or not isinstance(allocated_context, int):
            raise TypeError(f"RoPE table context must be an integer, got {allocated_context!r}")
        if not 0 < allocated_context <= MAX_CONTEXT or allocated_context % ttnn.TILE_SIZE:
            raise ValueError(
                f"RoPE table context must be a tile multiple in (0,{MAX_CONTEXT}], got {allocated_context}"
            )
        inverse_frequency = _inverse_frequency(config)
        # One _host_rope call per row: the batched torch.outer(arange, ...) path
        # may differ from the per-position scalar path in the last bit.
        rows = [_host_rope(position, inverse_frequency) for position in range(allocated_context)]
        hosts = (torch.cat([cos for cos, _ in rows], dim=2), torch.cat([sin for _, sin in rows], dim=2))
        tables: list[Any] = []
        try:
            for host in hosts:
                table = ttnn.from_torch(
                    host.contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
                )
                tables.append(table)
                expected = (1, 1, allocated_context, QSA_ROPE_DIM)
                if _shape(table) != expected or table.dtype != ttnn.bfloat16 or table.layout != ttnn.ROW_MAJOR_LAYOUT:
                    raise RuntimeError(
                        f"RoPE table must be BF16 ROW_MAJOR {list(expected)}, got {tensor_metadata(table)}"
                    )
                mesh_contract.validate_tensor(table, placement=TensorPlacement.REPLICATED)
        except BaseException:
            _deallocate_unique(*tables)
            raise
        return cls(
            tables[0],
            tables[1],
            allocated_context,
            inverse_frequency,
            mesh_contract,
            hosts[0].reshape(allocated_context, QSA_ROPE_DIM),
            hosts[1].reshape(allocated_context, QSA_ROPE_DIM),
        )

    def host_rows_for(self, axes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The cos / sin rows ``[R, 64]`` BF16 of tokens with (t, h, w) positions ``axes`` (int64 ``[3, R]``,
        ``mrope``): column c reads the host table row of axis (c % 32) % 3, so a token whose three positions agree
        (every text token) reads the table row bitwise and an image token reads its interleaved mix."""

        if self.host_cos is None or self.host_sin is None:
            raise RuntimeError("the RoPE table was built without its host image")
        return mrope.mrope_rows(axes, self.host_cos, self.host_sin)

    def _lookup_tile(self, indices, table, *, label: str, row_count: int = ttnn.TILE_SIZE):
        """The fused lookup of one ``[1,1,row_count]`` index row as its full tile rows ``[1,1,row_count,64]``."""

        looked_up = ttnn.embedding(
            indices, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # ttnn.embedding returns [batch, sentence, hidden] for indices of rank > 1
        # (ttnn/cpp/ttnn/operations/embedding/embedding.cpp:38-39,73), so the
        # [1,1,32] index row comes back as [1,32,64]; the fused TILE program's
        # 32-row tile is its padded shape.  Same rule as embed_device_token.
        rows = ttnn.unsqueeze_to_4D(looked_up) if len(looked_up.shape) == 3 else looked_up
        padded = (1, 1, row_count, QSA_ROPE_DIM)
        if (
            _shape(rows) != padded
            or _padded_shape(rows) != padded
            or rows.dtype != ttnn.bfloat16
            or rows.layout != ttnn.TILE_LAYOUT
        ):
            raise RuntimeError(
                f"{label} lookup must be BF16 TILE {list(padded)} backed by {list(padded)}, got "
                f"{tensor_metadata(rows)} (ttnn.embedding returned {tensor_metadata(looked_up)})"
            )
        return rows

    def _lookup(self, indices, table, *, label: str):
        rows = self._lookup_tile(indices, table, label=label)
        padded = (1, 1, ttnn.TILE_SIZE, QSA_ROPE_DIM)
        # Same logical/padded view as the device-token embedding: 32 identical
        # rows read as the one [1,1,1,64] row the QSA layers expect.
        row = ttnn.reshape(rows, ttnn.Shape((1, 1, 1, QSA_ROPE_DIM)), ttnn.Shape(padded))
        if (
            _shape(row) != (1, 1, 1, QSA_ROPE_DIM)
            or _padded_shape(row) != padded
            or row.dtype != ttnn.bfloat16
            or row.layout != ttnn.TILE_LAYOUT
        ):
            raise RuntimeError(
                f"{label} row must be BF16 TILE [1,1,1,{QSA_ROPE_DIM}] backed by {list(padded)}, got "
                f"{tensor_metadata(row)}"
            )
        self.mesh_contract.validate_tensor(row, placement=TensorPlacement.REPLICATED)
        return row

    def rows(self, index_row, block_start_row) -> Qwen38TTNNRoPEInputs:
        """In-trace rows at ``P`` and ``P & ~3`` from two ``[1,1,1,32]`` UINT32 index rows."""

        for name, row in (("index_row", index_row), ("block_start_row", block_start_row)):
            if (
                _shape(row) != (1, 1, 1, ttnn.TILE_SIZE)
                or row.dtype != ttnn.uint32
                or row.layout != ttnn.ROW_MAJOR_LAYOUT
            ):
                raise RuntimeError(
                    f"RoPE table {name} must be UINT32 ROW_MAJOR [1,1,1,{ttnn.TILE_SIZE}], got {tensor_metadata(row)}"
                )
            self.mesh_contract.validate_tensor(row, placement=TensorPlacement.REPLICATED)
        indices = ttnn.reshape(index_row, (1, 1, ttnn.TILE_SIZE))
        block_indices = ttnn.reshape(block_start_row, (1, 1, ttnn.TILE_SIZE))
        looked_up: list[Any] = []
        try:
            for label, table_indices, table in (
                ("QSA RoPE cos", indices, self.cos_table),
                ("QSA RoPE sin", indices, self.sin_table),
                ("QSA block-start RoPE cos", block_indices, self.cos_table),
                ("QSA block-start RoPE sin", block_indices, self.sin_table),
            ):
                looked_up.append(self._lookup(table_indices, table, label=label))
        except BaseException:
            _deallocate_unique(*looked_up)
            raise
        return Qwen38TTNNRoPEInputs(None, *looked_up)

    def rows_chunk(self, index_rows, block_start_rows) -> Qwen38TTNNRoPEInputs:
        """In-trace rows for one prefill chunk from the UINT32 index rows ``[1,1,tiles,32]`` and ``[1,1,1,32]``.

        ``index_rows`` lane j = P + j gives cos/sin ``[1,1,rows,64]`` (row j at P + j; rows = 32 tiles);
        ``block_start_rows`` lane i = P + 4i (i < rows / 4) gives the block-start rows, whose first rows / 4
        rows the compressed keys use.  The fused lookup's rows are all kept (no one-row view).
        """

        tiles = _shape(index_rows)[2]
        block_tiles = _shape(block_start_rows)[2]  # one for the chunk forms, rows / 128 for a slab
        for name, row, expected in (
            ("index_rows", index_rows, (1, 1, tiles, ttnn.TILE_SIZE)),
            ("block_start_rows", block_start_rows, (1, 1, block_tiles, ttnn.TILE_SIZE)),
        ):
            if _shape(row) != expected or row.dtype != ttnn.uint32 or row.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise RuntimeError(
                    f"RoPE table {name} must be UINT32 ROW_MAJOR {list(expected)}, got {tensor_metadata(row)}"
                )
            self.mesh_contract.validate_tensor(row, placement=TensorPlacement.REPLICATED)
        indices = ttnn.reshape(index_rows, (1, 1, tiles * ttnn.TILE_SIZE))
        block_indices = ttnn.reshape(block_start_rows, (1, 1, block_tiles * ttnn.TILE_SIZE))
        looked_up: list[Any] = []
        try:
            for label, table_indices, table, row_count in (
                ("QSA chunk RoPE cos", indices, self.cos_table, tiles * ttnn.TILE_SIZE),
                ("QSA chunk RoPE sin", indices, self.sin_table, tiles * ttnn.TILE_SIZE),
                ("QSA chunk block-start RoPE cos", block_indices, self.cos_table, block_tiles * ttnn.TILE_SIZE),
                ("QSA chunk block-start RoPE sin", block_indices, self.sin_table, block_tiles * ttnn.TILE_SIZE),
            ):
                rows = self._lookup_tile(table_indices, table, label=label, row_count=row_count)
                looked_up.append(rows)
                self.mesh_contract.validate_tensor(rows, placement=TensorPlacement.REPLICATED)
        except BaseException:
            _deallocate_unique(*looked_up)
            raise
        return Qwen38TTNNRoPEInputs(None, *looked_up)

    def host_rows(self, position: int) -> dict[str, torch.Tensor]:
        """Host image of :meth:`rows` at ``position`` (verification only)."""

        cos, sin = _host_rope(position, self.inverse_frequency)
        block_start_cos, block_start_sin = _host_rope(position - position % QSA_COMPRESS_RATIO, self.inverse_frequency)
        return {"cos": cos, "sin": sin, "block_start_cos": block_start_cos, "block_start_sin": block_start_sin}

    def deallocate(self) -> None:
        _deallocate_unique(self.cos_table, self.sin_table)


@dataclass
class Qwen38TTNNPreparedDecodeInputs:
    """Persistent host-prepared inputs consumed only by a captured device body.

    ``position`` and ``rope`` are ``None`` for the position-generic body, which
    reads its position and RoPE rows on device.
    """

    input_token_id: int
    position: int | None
    host_token: torch.Tensor
    residual_sharded: Any
    rope: Qwen38TTNNRoPEInputs | None
    ple: Qwen38TTNNPLEPreparedInput
    active: bool = True
    # Caller-owned token row (see embedding.TOKEN_ROW_SHAPE); when set, the body
    # embeds it in place of residual_sharded and release() leaves it alone.
    device_token: Any | None = None
    _owned_residual: list[Any | None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._owned_residual = [self.residual_sharded]

    def release(self) -> None:
        if not self.active:
            raise RuntimeError("prepared decode inputs were already released")
        actions = []
        if self.ple.active:
            actions.append(("prepared PLE input", self.ple.release))
        if self.rope is not None and self.rope.active:
            actions.append(("prepared QSA RoPE inputs", self.rope.deallocate))
        if self._owned_residual[0] is not None:
            actions.append(
                (
                    "prepared initial residual",
                    lambda: _release_tensor_slots(self._owned_residual, label="prepared residual"),
                )
            )
        try:
            _run_cleanup_actions("prepared decode inputs", actions)
        finally:
            self.residual_sharded = self._owned_residual[0]
            self.active = bool(
                self.ple.active or (self.rope is not None and self.rope.active) or self._owned_residual[0] is not None
            )


@dataclass(frozen=True)
class Qwen38TTNNTextModelState:
    """All 48 target-layer states for one global-B1 sequence."""

    position: int
    layers: tuple[Qwen38TTNNDecoderLayerState, ...]
    _owner: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class Qwen38TTNNTextModelGenericState:
    """Fixed-address 48-layer state plus the device position for the position-generic body."""

    position: Qwen38TTNNDevicePosition
    layers: tuple[Qwen38TTNNDecoderLayerGenericState, ...]
    _owner: object = field(repr=False, compare=False)


@dataclass
class Qwen38TTNNChunkFeatureRows:
    """A chunk's persistent vision feature rows: hidden-sharded BF16 TILE ``[1,1,rows,640]`` per device, added to the
    device embedding of the chunk's token rows (whose image lanes hold the zero sentinel).  ``clean`` records that
    every element is -0.0 (the add's identity: ``x + -0.0 == x`` for every x), so a text chunk after a text chunk
    skips the copy and a text chunk after an image chunk restores the clean image first."""

    tensor: Any
    rows: int
    clean: bool = True


@dataclass(frozen=True)
class Qwen38TTNNTextModelChunkState:
    """Fixed-address buffers of the prefill chunk body, allocated beside the generic state before any capture.

    ``token_row`` (replicated FP32 TILE ``[1,1,tiles,32]``, lane j = token j), ``ple_rows`` (the persistent
    ``[1,1,rows,640]`` upload of the n-gram rows), ``rope_rows`` (the rows' RoPE cos / sin ``[1,1,rows,64]`` and
    block-start rows, replicated BF16 TILE: written per chunk from the rows' rotary positions, so image tokens read
    their three-axis rows and text tokens the table's) and ``feature_rows`` (the vision splice) are the host-written
    inputs of a chunk; ``accepted`` (FP32
    ``[1,1,1,1]``) is 31 for a full chunk and r - 1 for the padded last chunk with r real rows, so one trace
    serves both (the 128-row form has none: it always commits every row).  ``rows_constants`` /
    ``qsa_chunk_constants`` are the model-lifetime chunk constants of this row count; ``local_combine_output``
    is the MoE combine buffer the 128-row layer instances share.  A 128-row state is allocated beside the
    32-row state whose GDN and PLE histories it shares.
    """

    rows_constants: gdn_module.Qwen38TTNNGDNRowsConstants
    qsa_chunk_constants: qsa_module.Qwen38TTNNQSAChunkConstants
    layers: tuple[Qwen38TTNNDecoderLayerChunkState, ...]
    token_row: Any
    ple_rows: Qwen38TTNNPLERowsPreparedInput
    accepted: Any | None
    rope_rows: Qwen38TTNNRoPEInputs
    feature_rows: Qwen38TTNNChunkFeatureRows
    _owner: object = field(repr=False, compare=False)
    rows: int = CHUNK_ROWS
    local_combine_output: Any | None = None
    # The generic state's device position: a caller that writes a chunk's inputs without the rows' positions (the
    # eager tools) gets the plain rows from a readback of ``P``; the prefill driver passes the positions instead.
    position: Any = None


@dataclass(frozen=True)
class Qwen38TTNNChunkHostInputs:
    """One chunk's inputs prepared on the host (:meth:`Qwen38TTNNTextModel.prepare_chunk_inputs`, no device call):
    the token-rows image and the n-gram rows as host tensors for :meth:`Qwen38TTNNTextModel.upload_chunk_inputs`,
    the rows + 1 contexts of :meth:`Qwen38TTNNPLE.host_rows` (``contexts[r]`` after committing r rows), the four
    RoPE row tensors (cos, sin, block-start cos, block-start sin) and the feature rows (None: a chunk without image
    pads; ``image_lanes`` counts the pads)."""

    token_rows: Any
    embedding_rows: Any
    contexts: tuple[tuple[int, int] | None, ...]
    rope_rows: tuple[Any, Any, Any, Any] = ()
    feature_rows: Any | None = None
    image_lanes: int = 0


@dataclass
class Qwen38TTNNTextModelGenericSnapshot:
    """Device-resident copy of the generic state's recurrent buffers at one position (the chat server's prompt-end
    snapshot).

    ``pairs`` is ``(label, state tensor, snapshot tensor)`` for every GDN recurrent state and conv ring slot, every
    PLE conv slot and every QSA staging tile and raw-key ring of the state's layers (and of ``extra_layers``: an
    MTP alignment layer's generic state).  The QSA KV and compressed caches are positional (rows and blocks past the
    position are rewritten before they are read) and are not copied.  ``gdn_states`` take ``position mod 4`` as
    their conv phase on a restore; ``ple_states`` drop their host context (the generic body's caller owns it).
    """

    pairs: tuple[tuple[str, Any, Any], ...]
    gdn_states: tuple[Any, ...]
    ple_states: tuple[Any, ...]
    _owner: object = field(repr=False, compare=False)
    position: int = 0
    captured: bool = False
    rope_shift: int = 0  # the sequence's rotary shift at the snapshot (restored with the position)


@dataclass(frozen=True)
class Qwen38TTNNTextModelLaneState:
    """Fixed-address 48-layer state of B batched decode lanes, the per-lane position row and the lane body's
    host-written inputs, all allocated before any capture (the lane traces bake every address in).

    ``token_row`` (replicated FP32 TILE ``[1,1,1,32]``, lane u = the token lane u consumes this step, lanes B..31
    zero) and ``ple_rows`` (the persistent ROW_MAJOR ``[1,1,B,640]`` upload of lane u's n-gram row) are the
    host-written inputs; ``embedding_rows`` the persistent ``[1,1,B,640]`` the 32-lane embedding is sliced into
    (None at B = 32); ``qsa_chunk_constants`` / ``qsa_lane_constants`` the lane count's constants of the QSA lane
    derive and body.  All resident lanes share ``P_u mod 4`` with every GDN ring phase (the residue rule).
    """

    lanes: int
    position: Qwen38TTNNDevicePositionRow
    layers: tuple[Qwen38TTNNDecoderLayerLaneState, ...]
    token_row: Any
    ple_rows: Qwen38TTNNPLELanesPreparedInput
    embedding_rows: Any | None
    qsa_chunk_constants: qsa_module.Qwen38TTNNQSAChunkConstants
    qsa_lane_constants: qsa_module.Qwen38TTNNQSALaneConstants
    _owner: object = field(repr=False, compare=False)


@dataclass
class Qwen38TTNNGenericDecodeOutput:
    """Logits of one position-generic step; every other tensor is in-place state or released.

    ``retain_mtp_inputs`` (the MTP-drafting chat server) keeps the layer-47 residual (the MTP alignment root at
    this position) and the step's RoPE rows and QSA position inputs alive for the epilogue's MTP-layer row;
    they are released with the output.
    """

    logits: Qwen38ShardedLogits | None
    active: bool = True
    residual: Any = None
    rope: Any = None
    qsa_position: Any = None

    def release_tensors(self) -> None:
        if not self.active:
            raise RuntimeError("generic decode output tensors were already released")
        if self.logits is not None:
            ttnn.deallocate(self.logits.tensor)
        if self.residual is not None:
            ttnn.deallocate(self.residual)
        if self.rope is not None:
            self.rope.deallocate()
        if self.qsa_position is not None:
            self.qsa_position.deallocate()
        self.active = False


@dataclass
class Qwen38TTNNGenericHead:
    """HEAD's output, the tensors TAIL reads (GENERIC_HEAD_HANDOFF).

    Owned until TAIL consumes them (``release_head=True``) or the holder of a
    retained handoff calls :meth:`release_tensors`.
    """

    residual: Any
    _owner: object = field(repr=False, compare=False)
    active: bool = True

    def release_tensors(self) -> None:
        if not self.active:
            raise RuntimeError("generic head tensors were already released")
        ttnn.deallocate(self.residual)
        self.active = False


class _TTNNConstantField:
    """Preserve native-valued metadata without retaining module constants."""

    def __init__(self, constant_name: str) -> None:
        self.constant_name = constant_name

    def __set_name__(self, owner, name: str) -> None:
        self.storage_name = f"_{name}"

    def __get__(self, instance, owner=None):
        if instance is None:
            raise AttributeError(self.storage_name)
        if getattr(instance, self.storage_name + "_is_constant"):
            return getattr(ttnn, self.constant_name)
        return getattr(instance, self.storage_name)

    def __set__(self, instance, value) -> None:
        is_constant = value is getattr(ttnn, self.constant_name)
        object.__setattr__(instance, self.storage_name, None if is_constant else value)
        object.__setattr__(instance, self.storage_name + "_is_constant", is_constant)


@dataclass(frozen=True)
class Qwen38TTNNGenericHandoff:
    """Metadata contract of one tensor HEAD writes and TAIL reads."""

    name: str
    attribute: str
    shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    dtype: Any = _TTNNConstantField("bfloat16")
    layout: Any = _TTNNConstantField("TILE_LAYOUT")
    memory_config: Any = _TTNNConstantField("DRAM_MEMORY_CONFIG")

    def describe(self, tensor=None) -> str:
        """The contract's fields, of ``tensor`` when given, for actual-vs-expected error text."""

        shape, padded_shape, dtype, layout, memory_config = (
            (self.shape, self.padded_shape, self.dtype, self.layout, self.memory_config)
            if tensor is None
            else (_shape(tensor), _padded_shape(tensor), tensor.dtype, tensor.layout, tensor.memory_config())
        )
        return (
            f"shape={list(shape)} padded_shape={list(padded_shape)} dtype={dtype} layout={layout} "
            f"memory_config={memory_config}"
        )


# Every tensor crossing the HEAD | TAIL split, checked at capture and in eager
# mode by _validate_generic_head: the layer-0 output residual only.  RoPE rows
# and QSA position inputs are derived at the top of TAIL from the position scalar.
GENERIC_HEAD_HANDOFF = (
    Qwen38TTNNGenericHandoff(
        name="layer_0_output_residual",
        attribute="residual",
        shape=RESIDUAL_LOCAL_SHAPE,
        padded_shape=(1, RESIDUAL_BRANCHES, 32, LOCAL_HIDDEN_SIZE),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ),
)


@dataclass(frozen=True)
class Qwen38TTNNGenericTraceKey:
    """Index of one captured trace: body part, GDN ring residue class it binds, indexer regime."""

    part: str
    residue: int
    regime: int = 0


@dataclass
class Qwen38TTNNGenericDecodeCapture:
    """Traces of one (residue, regime) of the generic body and the tensors those traces own.

    ``parts`` is ``("body",)`` for one single-body trace or ``("head", "tail")``
    for the split.  ``head`` is the retained handoff of the split: both traces
    bake its buffer address in, so it lives until :meth:`release_tensors`.
    ``epilogue`` is whatever the caller's epilogue returned inside the last
    part's capture; the caller owns those tensors.
    """

    residue: int
    regime: int
    parts: tuple[str, ...]
    trace_ids: dict[Qwen38TTNNGenericTraceKey, int]
    head: Qwen38TTNNGenericHead | None
    output: Qwen38TTNNGenericDecodeOutput
    epilogue: Any
    capture_ns: dict[str, int]
    guard_attempts: tuple[str, ...]
    active: bool = True

    def release_tensors(self) -> None:
        if not self.active:
            raise RuntimeError("generic decode capture tensors were already released")
        actions = []
        if self.output.active:
            actions.append(("captured logits", self.output.release_tensors))
        if self.head is not None and self.head.active:
            actions.append(("retained head handoff", self.head.release_tensors))
        try:
            _run_cleanup_actions("generic decode capture", actions)
        finally:
            self.active = self.output.active or (self.head is not None and self.head.active)


@dataclass
class Qwen38TTNNTextModelSnapshot:
    """One-shot model-wide state transaction for target verification."""

    position: int
    layers: tuple[Qwen38TTNNDecoderLayerSnapshot, ...]
    _owner: object = field(repr=False, compare=False)
    active: bool = True


@dataclass
class Qwen38TTNNTextModelOutput:
    """One ordinary token result and its explicitly owned device tensors.

    ``hyper_residual_sharded`` is the four-branch target row required by the
    released MTP input mixer.  ``hidden_sharded`` is the terminal-mixer result.
    Calling :meth:`release_tensors` never releases layer/cache state or QSA
    selections, which remain owned by ``state``.
    """

    input_token_id: int
    position: int
    hyper_residual_sharded: Any | None
    hidden_sharded: Any | None
    logits: Qwen38ShardedLogits | None
    greedy_token: torch.Tensor | None
    state: Qwen38TTNNTextModelState
    layer_aux: tuple[Qwen38TTNNDecoderLayerAux, ...]
    active: bool = True
    _owned_tensors: list[Any | None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        routed = []
        for aux in self.layer_aux:
            if aux.routing is not None:
                routed.extend((aux.routing.scores, aux.routing.indices))
        self._owned_tensors = [
            self.hyper_residual_sharded,
            self.hidden_sharded,
            None if self.logits is None else self.logits.tensor,
            *routed,
        ]

    def _take_unique_owned_tensor(self, index: int, *, label: str) -> Any:
        """Transfer one non-aliased tensor out of this output owner.

        A hybrid target/MTP owner must keep a target hyper-residual alive after
        releasing the other output tensors.  Clearing the public field alone
        is insufficient because ``_owned_tensors`` is the authoritative
        cleanup ledger.  This helper updates that ledger before returning the
        tensor and refuses ambiguous alias ownership.
        """

        if not self.active:
            raise RuntimeError("model-output device tensors are no longer owned")
        tensor = self._owned_tensors[index]
        if tensor is None:
            raise RuntimeError(f"model output no longer owns {label}")
        key = _tensor_key(tensor)
        aliases = [
            candidate_index
            for candidate_index, candidate in enumerate(self._owned_tensors)
            if candidate is not None and _tensor_key(candidate) == key
        ]
        if aliases != [index]:
            raise RuntimeError(f"cannot transfer aliased {label}; owned slots are {aliases}")
        self._owned_tensors[index] = None
        self.active = any(candidate is not None for candidate in self._owned_tensors)
        return tensor

    def take_hyper_residual(self) -> Any:
        """Transfer the exact four-branch target root to an MTP owner."""

        tensor = self.hyper_residual_sharded
        if tensor is None or self._owned_tensors[0] is not tensor:
            raise RuntimeError("model output does not own its advertised hyper-residual")
        result = self._take_unique_owned_tensor(0, label="hyper-residual")
        self.hyper_residual_sharded = None
        return result

    def take_logits(self) -> Qwen38ShardedLogits:
        """Transfer the vocab-sharded logits wrapper without reading values."""

        logits = self.logits
        if logits is None or self._owned_tensors[2] is not logits.tensor:
            raise RuntimeError("model output does not own its advertised logits")
        tensor = self._take_unique_owned_tensor(2, label="vocab-sharded logits")
        if tensor is not logits.tensor:
            raise AssertionError("model-output logit ownership changed during transfer")
        self.logits = None
        return logits

    def release_tensors(self) -> None:
        if not self.active:
            raise RuntimeError("model-output device tensors were already released")
        _release_tensor_slots(self._owned_tensors, label="text-model output")
        self.active = not all(tensor is None for tensor in self._owned_tensors)


@dataclass(frozen=True)
class Qwen38TTNNGreedyStep:
    input_token_id: int
    next_token_id: int
    state: Qwen38TTNNTextModelState
    layer_aux: tuple[Qwen38TTNNDecoderLayerAux, ...]


LayerObserver = Callable[
    [int, Any, Qwen38TTNNDecoderLayerState, Qwen38TTNNDecoderLayerAux],
    None,
]
DecodePhaseObserver = Callable[[str], None]


QSA_LANE_FUSED_KERNELS = (  # the six programs the fused QSA lane body needs together (ttnn/qsa.py binds it on all six)
    "qsa_index_tail",
    "qsa_main_tail",
    "qsa_post_attention",
    "qsa_widen_partial",
    "qsa_selection_row",
    "qsa_score_merge",
)


class Qwen38TTNNTextModel:
    """Exact 48-layer ordinary-decode owner on one physical 1x4 mesh."""

    allocated_context = MAX_CONTEXT
    semantic_max_context = MAX_CONTEXT

    _position_derive = None  # the fused tail derive when QWEN38_FUSED names position_derive (see __init__)
    _position_derive_lanes = None  # its lane form, when the six QSA programs also serve the lane body (see __init__)

    def __init__(
        self,
        *,
        config: Qwen38Config,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        model_io: Qwen38TTNNModelIO,
        layers: Sequence[Qwen38TTNNDecoderLayer],
        final_mixer: Qwen38TTNNFinalMixer,
    ) -> None:
        _validate_exact_config(config)
        mesh_contract.validate_mesh(mesh_device)
        if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
            raise ValueError("Qwen3.8 ordinary decode requires a Blackhole mesh")
        if not isinstance(model_io, Qwen38TTNNModelIO):
            raise TypeError("model_io must be the exact vocab-sharded Qwen38TTNNModelIO")
        if not isinstance(final_mixer, Qwen38TTNNFinalMixer):
            raise TypeError("final_mixer must be the exact backbone Qwen38TTNNFinalMixer")

        layers = tuple(layers)
        if len(layers) != BACKBONE_LAYERS:
            raise ValueError(f"ordinary target requires exactly {BACKBONE_LAYERS} layers, got {len(layers)}")
        if not all(isinstance(layer, Qwen38TTNNDecoderLayer) for layer in layers):
            raise TypeError("all target layers must be Qwen38TTNNDecoderLayer instances")

        io_components = (model_io.embedding, model_io.lm_head)
        for name, component in zip(("embedding", "LM head"), io_components):
            if component.mesh_contract != mesh_contract or component.mesh_device is not mesh_device:
                raise ValueError(f"{name} belongs to a different live physical mesh")
        if model_io.embedding.weights is not model_io.lm_head.weights:
            raise ValueError("embedding and LM head must share one provenance-bound model-I/O weight owner")
        if tuple(model_io.embedding.weights.vocab_ranges) != EXPECTED_VOCAB_RANGES:
            raise ValueError("model-I/O vocabulary ranges are not the exact contiguous TP4 placement")
        if final_mixer.mesh_contract != mesh_contract or final_mixer.mesh_device is not mesh_device:
            raise ValueError("backbone final mixer belongs to a different live physical mesh")
        if final_mixer.weights.namespace != Qwen38TTNNLayerNamespace.BACKBONE.value:
            raise ValueError("ordinary target requires the backbone terminal hyper-connection mixer")

        shared_streamer = layers[0].expert_streamer
        for layer_index, layer in enumerate(layers):
            if layer.namespace is not Qwen38TTNNLayerNamespace.BACKBONE:
                raise ValueError(f"target layer {layer_index} is not in the backbone namespace")
            if layer.layer_index != layer_index:
                raise ValueError(f"target layer slot {layer_index} contains checkpoint layer {layer.layer_index}")
            expected_type = (
                Qwen38TTNNLayerType.QSA
                if EXPECTED_LAYER_PATTERN[layer_index] == "full_attention"
                else Qwen38TTNNLayerType.GDN
            )
            if layer.layer_type is not expected_type:
                raise ValueError(
                    f"target layer {layer_index} has type {layer.layer_type.value}, expected {expected_type.value}"
                )
            if layer.mesh_contract != mesh_contract:
                raise ValueError(f"target layer {layer_index} belongs to a different physical mesh contract")
            components = (
                ("attention", layer.attention),
                ("attention GR", layer.attention_gr),
                ("MoE", layer.mlp),
                ("MLP GR", layer.mlp_gr),
            )
            if layer.ple is not None:
                components += (("PLE", layer.ple),)
            for component_name, component in components:
                if component.mesh_device is not mesh_device or component.mesh_contract != mesh_contract:
                    raise ValueError(
                        f"target layer {layer_index} {component_name} belongs to a different live mesh object"
                    )
            if layer.expert_streamer is not shared_streamer:
                raise ValueError("all target layers must share one BF4_B streamer and its single active slot")
        if shared_streamer.mesh_device is not mesh_device or shared_streamer.cache.mesh_contract != mesh_contract:
            raise ValueError("BF4_B streamer belongs to a different live physical mesh")

        qsa_capacities = tuple(
            layer.attention.allocated_context for layer in layers if layer.layer_type is Qwen38TTNNLayerType.QSA
        )
        if not qsa_capacities or len(set(qsa_capacities)) != 1:
            raise ValueError(f"all ordinary QSA layers must share one physical cache capacity, got {qsa_capacities}")

        self.config = config
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.model_io = model_io
        self.layers = layers
        self.final_mixer = final_mixer
        self.allocated_context = qsa_capacities[0]
        self.semantic_max_context = MAX_CONTEXT
        self.rope = Qwen38TTNNRoPE(mesh_device, mesh_contract, config)
        # Position-generic constants (resident RoPE tables, QSA position
        # constants): built on the first allocate_generic_state, kept for the
        # model lifetime so every captured address stays valid.
        self.rope_table: Qwen38TTNNRoPETable | None = None
        self.qsa_position_constants: Any | None = None
        from models.demos.blackhole.qwen38_flash_next.ttnn import fused as fused_kernels

        if fused_kernels.enabled("position_derive"):
            self._position_derive = fused_kernels.kernel("position_derive").fused
            # the lane form derives only the fused QSA lane body's inputs: it serves when that body serves
            if all(fused_kernels.enabled(name) for name in QSA_LANE_FUSED_KERNELS):
                self._position_derive_lanes = fused_kernels.position_derive.derive_lanes_fused
        self._clean_feature_rows_hosts: dict[int, Any] = {}  # the -0.0 feature rows per chunk row count (host)
        self._state_owner = object()
        self._poisoned_error: Qwen38TTNNModelPoisonedError | None = None
        self._poisoned_device_owners: list[Any] = []
        self._active_snapshot: Qwen38TTNNTextModelSnapshot | None = None
        self._runtime_owner: object | None = None

    @property
    def poisoned(self) -> bool:
        return self._poisoned_error is not None

    @property
    def poisoned_error(self) -> Qwen38TTNNModelPoisonedError | None:
        return self._poisoned_error

    def _require_healthy(self) -> None:
        if self._poisoned_error is not None:
            raise self._poisoned_error from self._poisoned_error.original_cause

    def _mark_poisoned(self, operation: str, processed_layers: int, cause: BaseException) -> NoReturn:
        if self._active_snapshot is not None:
            self._active_snapshot.active = False
            self._active_snapshot = None
        if self._poisoned_error is None:
            self._poisoned_error = Qwen38TTNNModelPoisonedError(operation, processed_layers, cause)
        raise self._poisoned_error from cause

    def _require_no_active_snapshot(self, operation: str) -> None:
        if self._active_snapshot is not None:
            raise RuntimeError(
                f"cannot {operation} while a text-model snapshot is active; "
                "restore or commit that exact transaction first"
            )

    def claim_runtime_owner(self, owner: object) -> None:
        """Exclusively bind one ordinary/MTP controller to this live model."""

        self._require_healthy()
        if owner is None:
            raise TypeError("runtime owner token cannot be None")
        if self._runtime_owner is not None:
            raise RuntimeError("the TTNN text model already has a live runtime owner")
        self._runtime_owner = owner

    def release_runtime_owner(self, owner: object) -> None:
        """Release a healthy controller claim after its state is gone."""

        self._require_healthy()
        self._require_no_active_snapshot("release the runtime owner")
        if self._runtime_owner is not owner:
            raise ValueError("runtime owner token does not own this TTNN text model")
        self._runtime_owner = None

    def transfer_runtime_owner(self, current_owner: object, next_owner: object) -> None:
        """Explicitly hand the live model to a combined ordinary/MTP owner."""

        self._require_healthy()
        self._require_no_active_snapshot("transfer the runtime owner")
        if next_owner is None:
            raise TypeError("next runtime owner token cannot be None")
        if self._runtime_owner is not current_owner:
            raise ValueError("current runtime owner token does not own this TTNN text model")
        if next_owner is current_owner:
            raise ValueError("runtime owner transfer requires a distinct next owner token")
        self._runtime_owner = next_owner

    def _validate_hidden(self, tensor, *, label: str) -> None:
        if _shape(tensor) != BLOCK_LOCAL_SHAPE or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"{label} must be BF16 TILE {list(BLOCK_LOCAL_SHAPE)}, got {tensor_metadata(tensor)}")
        self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _validate_residual(self, tensor, *, label: str) -> None:
        if _shape(tensor) != RESIDUAL_LOCAL_SHAPE or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"{label} must be BF16 TILE {list(RESIDUAL_LOCAL_SHAPE)}, got {tensor_metadata(tensor)}")
        self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _validate_state(self, state: Qwen38TTNNTextModelState) -> None:
        self._require_healthy()
        if not isinstance(state, Qwen38TTNNTextModelState) or state._owner is not self._state_owner:
            raise ValueError("text-model state was not allocated by this model owner")
        if not 0 <= state.position <= self.allocated_context:
            raise ValueError(
                f"text-model position must be in [0,{self.allocated_context}], got {state.position}; "
                f"semantic model limit remains {MAX_CONTEXT}"
            )
        if len(state.layers) != BACKBONE_LAYERS:
            raise ValueError(f"text-model state must contain {BACKBONE_LAYERS} target layers")
        for layer_index, layer_state in enumerate(state.layers):
            if (
                layer_state.namespace is not Qwen38TTNNLayerNamespace.BACKBONE
                or layer_state.layer_index != layer_index
                or layer_state.position != state.position
            ):
                raise ValueError(
                    f"target state {layer_index} identity/position does not match model position {state.position}"
                )
            self.layers[layer_index].validate_state(layer_state)

    def _validate_snapshot(self, snapshot: Qwen38TTNNTextModelSnapshot) -> None:
        self._require_healthy()
        if not isinstance(snapshot, Qwen38TTNNTextModelSnapshot) or snapshot._owner is not self._state_owner:
            raise ValueError("text-model snapshot was not allocated by this model owner")
        if self._active_snapshot is not snapshot:
            raise ValueError("text-model snapshot is not the active transaction owned by this model")
        if not snapshot.active:
            raise RuntimeError("text-model snapshot was already restored or committed")
        if not 0 <= snapshot.position <= self.allocated_context:
            raise ValueError(
                f"text-model snapshot position must be in [0,{self.allocated_context}], got {snapshot.position}; "
                f"semantic model limit remains {MAX_CONTEXT}"
            )
        if len(snapshot.layers) != BACKBONE_LAYERS:
            raise ValueError(f"text-model snapshot must contain {BACKBONE_LAYERS} target layers")
        for layer_index, (layer, layer_snapshot) in enumerate(zip(self.layers, snapshot.layers)):
            layer.validate_snapshot(layer_snapshot)
            if layer_snapshot.position != snapshot.position:
                raise ValueError(
                    f"target snapshot {layer_index} position {layer_snapshot.position} "
                    f"does not match model snapshot position {snapshot.position}"
                )

    def _preflight_transaction(
        self,
        current: Qwen38TTNNTextModelState,
        snapshot: Qwen38TTNNTextModelSnapshot,
        *,
        operation: str,
    ) -> None:
        """Validate all 48 pairs before the first one can be consumed."""

        if operation not in {"restore", "commit"}:
            raise ValueError(f"unsupported text-model transaction operation {operation!r}")
        self._validate_state(current)
        self._validate_snapshot(snapshot)
        if current.position < snapshot.position:
            raise ValueError(f"cannot {operation} text-model state preceding its snapshot")
        validator_name = f"validate_{operation}_pair"
        for layer, layer_state, layer_snapshot in zip(self.layers, current.layers, snapshot.layers):
            getattr(layer, validator_name)(layer_state, layer_snapshot)

    def _preflight_forward_transaction(
        self,
        state: Qwen38TTNNTextModelState,
        *,
        retain_input_state: bool,
    ) -> Qwen38TTNNTextModelSnapshot | None:
        snapshot = self._active_snapshot
        if snapshot is None:
            if retain_input_state:
                raise RuntimeError("retain_input_state requires an active text-model snapshot")
            return None
        self._preflight_transaction(state, snapshot, operation="commit")
        first_branch_step = state.position == snapshot.position
        if retain_input_state != first_branch_step:
            required = "true" if first_branch_step else "false"
            raise RuntimeError(
                f"retain_input_state must be {required} at branch position {state.position} "
                f"for snapshot position {snapshot.position}"
            )
        return snapshot

    def allocate_state(self) -> Qwen38TTNNTextModelState:
        self._require_healthy()
        self._require_no_active_snapshot("allocate another state")
        allocated: list[Qwen38TTNNDecoderLayerState] = []
        try:
            for layer in self.layers:
                allocated.append(layer.allocate_state())
        except BaseException as error:
            actions = [
                (
                    f"target layer {index} state",
                    lambda layer=layer, layer_state=layer_state: layer.release_state(layer_state),
                )
                for index, (layer, layer_state) in reversed(
                    tuple(enumerate(zip(self.layers[: len(allocated)], allocated)))
                )
            ]
            cause = _cleanup_failure_cause("text-model state allocation", actions, error)
            if cause is not error or isinstance(error, Qwen38TTNNLayerCleanupError):
                self._mark_poisoned("allocate_state cleanup", 0, cause)
            raise
        result = Qwen38TTNNTextModelState(0, tuple(allocated), self._state_owner)
        try:
            self._validate_state(result)
        except BaseException as error:
            actions = [
                (
                    f"target layer {index} state",
                    lambda layer=layer, layer_state=layer_state: layer.release_state(layer_state),
                )
                for index, (layer, layer_state) in reversed(tuple(enumerate(zip(self.layers, allocated))))
            ]
            cause = _cleanup_failure_cause("invalid allocated text-model state", actions, error)
            if cause is not error or isinstance(error, Qwen38TTNNLayerCleanupError):
                self._mark_poisoned("allocate_state validation cleanup", 0, cause)
            raise
        return result

    def reset_state(self, state: Qwen38TTNNTextModelState) -> Qwen38TTNNTextModelState:
        self._validate_state(state)
        self._require_no_active_snapshot("reset state")
        reset = []
        try:
            for layer, layer_state in zip(self.layers, state.layers):
                reset.append(layer.reset_state(layer_state))
            result = Qwen38TTNNTextModelState(0, tuple(reset), self._state_owner)
            self._validate_state(result)
            return result
        except BaseException as error:
            self._mark_poisoned("reset_state", len(reset), error)

    def release_state(self, state: Qwen38TTNNTextModelState) -> None:
        self._validate_state(state)
        self._require_no_active_snapshot("release state")
        errors = []
        released = 0
        for index, (layer, layer_state) in reversed(tuple(enumerate(zip(self.layers, state.layers)))):
            try:
                layer.release_state(layer_state)
            except BaseException as error:
                wrapped = RuntimeError(f"target layer {index}: {type(error).__name__}: {error}")
                wrapped.__cause__ = error
                errors.append(wrapped)
            else:
                released += 1
        if errors:
            self._mark_poisoned(
                "release_state",
                released,
                Qwen38TTNNCleanupError("text-model state", errors),
            )

    def snapshot_state(self, state: Qwen38TTNNTextModelState) -> Qwen38TTNNTextModelSnapshot:
        self._validate_state(state)
        self._require_no_active_snapshot("create another snapshot")
        snapshots: list[Qwen38TTNNDecoderLayerSnapshot] = []
        try:
            for layer, layer_state in zip(self.layers, state.layers):
                snapshots.append(layer.snapshot_state(layer_state))
        except BaseException as error:
            actions = [
                (
                    f"target layer {index} snapshot",
                    lambda index=index: self.layers[index].commit_state(state.layers[index], snapshots[index]),
                )
                for index in reversed(range(len(snapshots)))
            ]
            cause = _cleanup_failure_cause("text-model snapshot creation", actions, error)
            if cause is not error or isinstance(error, Qwen38TTNNLayerCleanupError):
                self._mark_poisoned("snapshot_state cleanup", len(snapshots), cause)
            raise
        result = Qwen38TTNNTextModelSnapshot(state.position, tuple(snapshots), self._state_owner)
        self._active_snapshot = result
        return result

    def restore_state(
        self,
        current: Qwen38TTNNTextModelState,
        snapshot: Qwen38TTNNTextModelSnapshot,
    ) -> Qwen38TTNNTextModelState:
        self._preflight_transaction(current, snapshot, operation="restore")
        restored = []
        try:
            for layer, layer_state, layer_snapshot in zip(self.layers, current.layers, snapshot.layers):
                restored.append(layer.restore_state(layer_state, layer_snapshot))
            snapshot.active = False
            self._active_snapshot = None
            result = Qwen38TTNNTextModelState(snapshot.position, tuple(restored), self._state_owner)
            self._validate_state(result)
            return result
        except BaseException as error:
            # Some inner snapshots/views may already be consumed.  Prevent an
            # apparently live outer transaction from ever being retried.
            snapshot.active = False
            self._active_snapshot = None
            self._mark_poisoned("restore_state", len(restored), error)

    def commit_state(
        self,
        current: Qwen38TTNNTextModelState,
        snapshot: Qwen38TTNNTextModelSnapshot,
    ) -> Qwen38TTNNTextModelState:
        self._preflight_transaction(current, snapshot, operation="commit")
        committed = []
        try:
            for layer, layer_state, layer_snapshot in zip(self.layers, current.layers, snapshot.layers):
                committed.append(layer.commit_state(layer_state, layer_snapshot))
            snapshot.active = False
            self._active_snapshot = None
            result = Qwen38TTNNTextModelState(current.position, tuple(committed), self._state_owner)
            self._validate_state(result)
            return result
        except BaseException as error:
            snapshot.active = False
            self._active_snapshot = None
            self._mark_poisoned("commit_state", len(committed), error)

    def _embed_residual_from_device_token(self, token_row):
        """Trace-capturable branch-major four-branch residual from a device token row (no host token upload)."""

        owned: list[Any | None] = [self.model_io.embedding.embed_device_token(token_row), None]
        try:
            self._validate_hidden(owned[0], label="device-token embedding")
            # Same branch-major [1,4,1,640] construction as _embed_residual.
            owned[1] = ttnn.repeat_interleave(
                owned[0],
                repeats=RESIDUAL_BRANCHES,
                dim=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._validate_residual(owned[1], label="device-token four-branch residual")
        except BaseException as error:
            # Producer/output ownership is ambiguous once repeat_interleave may
            # have been enqueued; retain both through teardown.
            self._poisoned_device_owners.extend(tensor for tensor in owned if tensor is not None)
            self._mark_poisoned("device-token residual", 0, error)
        _release_tensor_slot_indices(owned, (0,), label="device-token embedding transient")
        return owned[1]

    def _embed_residual_rows_from_device_token(self, token_row, feature_rows=None):
        """Trace-capturable branch-major ``[1,4,rows,640]`` residual rows from a token-rows tile (prefill chunk).

        ``feature_rows`` (the chunk state's vision feature rows, hidden-sharded ``[1,1,rows,640]``) is added to the
        embedding: the image lanes embed the zero sentinel and the text lanes of the feature rows hold -0.0, so the
        add puts the tower's rows in place and leaves every text row bitwise (``x + -0.0 == x``)."""

        rows = _shape(token_row)[2] * ttnn.TILE_SIZE
        owned: list[Any | None] = [self.model_io.embedding.embed_device_token_rows(token_row), None]
        try:
            hidden = owned[0]
            if (
                _shape(hidden) != block_rows_shape(rows)
                or hidden.dtype != ttnn.bfloat16
                or hidden.layout != ttnn.TILE_LAYOUT
            ):
                raise RuntimeError(
                    f"device-token embedding rows must be BF16 TILE {list(block_rows_shape(rows))}, "
                    f"got {tensor_metadata(hidden)}"
                )
            self.mesh_contract.validate_tensor(hidden, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
            if feature_rows is not None:
                if _shape(feature_rows) != block_rows_shape(rows) or feature_rows.dtype != ttnn.bfloat16:
                    raise RuntimeError(
                        f"chunk feature rows must be BF16 {list(block_rows_shape(rows))}, got {tensor_metadata(feature_rows)}"
                    )
                spliced = ttnn.add(hidden, feature_rows, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                _release_tensor_slot_indices(owned, (0,), label="pre-splice embedding rows")
                owned[0] = hidden = spliced
                if (
                    _shape(hidden) != block_rows_shape(rows)
                    or hidden.dtype != ttnn.bfloat16
                    or hidden.layout != ttnn.TILE_LAYOUT
                ):
                    raise RuntimeError(
                        f"spliced embedding rows must be BF16 TILE {list(block_rows_shape(rows))}, got {tensor_metadata(hidden)}"
                    )
                self.mesh_contract.validate_tensor(hidden, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
            # Same branch-major construction as _embed_residual_from_device_token, over the rows.
            owned[1] = ttnn.repeat_interleave(
                hidden, repeats=RESIDUAL_BRANCHES, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            residual = owned[1]
            if (
                _shape(residual) != residual_rows_shape(rows)
                or residual.dtype != ttnn.bfloat16
                or residual.layout != ttnn.TILE_LAYOUT
            ):
                raise RuntimeError(
                    f"device-token residual rows must be BF16 TILE {list(residual_rows_shape(rows))}, "
                    f"got {tensor_metadata(residual)}"
                )
            self.mesh_contract.validate_tensor(residual, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        except BaseException as error:
            self._poisoned_device_owners.extend(tensor for tensor in owned if tensor is not None)
            self._mark_poisoned("device-token residual rows", 0, error)
        _release_tensor_slot_indices(owned, (0,), label="device-token embedding rows transient")
        return owned[1]

    def _embed_residual(self, host_token: torch.Tensor):
        # Host-token path; the trace-capturable variant is _embed_residual_from_device_token.
        validated = self.model_io.embedding.upload_tokens(host_token)
        owned = [validated.tensor, None, None]
        try:
            hidden_sharded = self.model_io.embedding(validated)
            owned[1] = hidden_sharded
            self._validate_hidden(hidden_sharded, label="token embedding")
            repeat_enqueue_owners: list[Any] = []
            try:
                for tensor in owned[:2]:
                    repeat_enqueue_owners.append(tensor)
                    repeat_enqueue_owners.extend(ttnn.get_device_tensors(tensor))
                residual = ttnn.repeat_interleave(
                    hidden_sharded,
                    repeats=RESIDUAL_BRANCHES,
                    dim=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                owned[2] = residual
                repeat_enqueue_owners.append(residual)
                repeat_enqueue_owners.extend(ttnn.get_device_tensors(residual))
                self._validate_residual(residual, label="initial four-branch residual")
            except BaseException as error:
                # Once repeat_interleave is attempted, producer/output ownership
                # is ambiguous.  Keep every live owner through process/mesh
                # teardown and make this model terminally unusable rather than
                # scheduling cleanup onto the same queue.
                for tensor in owned:
                    if tensor is not None and not any(tensor is retained for retained in repeat_enqueue_owners):
                        repeat_enqueue_owners.append(tensor)
                self._poisoned_device_owners.extend(repeat_enqueue_owners)
                owned[:] = [None] * len(owned)
                self._mark_poisoned("initial residual repeat-interleave", 0, error)
            # repeat_interleave is the final same-CQ consumer of the embedding.
            # Once its enqueue returns, producer deallocation is trace-safe and
            # does not require a host completion fence.
            repeat_enqueue_owners.clear()
            _release_tensor_slot_indices(owned, (0, 1), label="embedding transients")
            return owned[2]
        except Qwen38TTNNModelPoisonedError:
            raise
        except BaseException as error:
            embedding = self.model_io.embedding
            embedding_async = (
                getattr(embedding, "synchronization_policy", None) is Qwen38TTNNEmbeddingSyncPolicy.RESIDENT_ASYNC
            )
            if getattr(embedding, "poisoned", False) or (embedding_async and owned[1] is not None):
                retained = list(getattr(embedding, "poisoned_device_owners", ()))
                retained_ids = {id(owner) for owner in retained}
                for tensor in owned:
                    if tensor is None:
                        continue
                    if id(tensor) not in retained_ids:
                        retained.append(tensor)
                        retained_ids.add(id(tensor))
                    try:
                        local_tensors = tuple(ttnn.get_device_tensors(tensor))
                    except BaseException:
                        local_tensors = ()
                    for local_tensor in local_tensors:
                        if id(local_tensor) not in retained_ids:
                            retained.append(local_tensor)
                            retained_ids.add(id(local_tensor))
                self._poisoned_device_owners.extend(retained)
                owned[:] = [None] * len(owned)
                self._mark_poisoned("token embedding asynchronous chain", 0, error)
            _release_tensor_slots(owned, label="failed embedding step", primary=error)
            raise

    def prepare_decode_inputs(
        self,
        token_id: int | torch.Tensor,
        state: Qwen38TTNNTextModelState,
        *,
        device_token=None,
    ) -> Qwen38TTNNPreparedDecodeInputs:
        """Prepare token, RoPE, and host PLE data before a trace capture/replay body.

        With ``device_token`` (a caller-owned token row holding ``token_id``) the
        initial residual is not uploaded here; the captured body embeds it.
        """

        self._validate_state(state)
        if state.position >= self.allocated_context:
            raise ValueError(
                f"ordinary decode exceeds allocated cache capacity {self.allocated_context}; "
                f"semantic model limit remains {MAX_CONTEXT}"
            )
        value, host_token = _normalize_token_id(token_id)
        residual = rope_inputs = prepared_ple = None
        try:
            if device_token is None:
                residual = self._embed_residual(host_token)
            else:
                self.model_io.embedding.validate_token_row(device_token, label="prepared device token row")
            rope_inputs = self.rope.for_position(state.position)
            ple_layer = self.layers[1]
            ple_state = state.layers[1].ple
            if ple_layer.ple is None or ple_state is None:
                raise RuntimeError("checkpoint layer 1 PLE owner/state is unavailable")
            prepared_ple = ple_layer.ple.prepare_decode_input(host_token, ple_state)
            return Qwen38TTNNPreparedDecodeInputs(
                input_token_id=value,
                position=state.position,
                host_token=host_token,
                residual_sharded=residual,
                rope=rope_inputs,
                ple=prepared_ple,
                device_token=device_token,
            )
        except BaseException as error:
            actions = []
            if prepared_ple is not None and prepared_ple.active:
                actions.append(("prepared PLE input", prepared_ple.release))
            if rope_inputs is not None and rope_inputs.active:
                actions.append(("prepared QSA RoPE inputs", rope_inputs.deallocate))
            if residual is not None:
                residual_slot = [residual]
                actions.append(
                    (
                        "prepared initial residual",
                        lambda: _release_tensor_slots(residual_slot, label="failed prepared residual"),
                    )
                )
            cause = _cleanup_failure_cause("prepare decode inputs", actions, error)
            if cause is not error:
                self._mark_poisoned("prepare_decode_inputs cleanup", 0, cause)
            raise

    def forward_decode_prepared(
        self,
        prepared: Qwen38TTNNPreparedDecodeInputs,
        state: Qwen38TTNNTextModelState,
        *,
        return_logits: bool = True,
        resolve_greedy: bool = False,
        retain_hidden: bool = True,
        retain_hyper_residual: bool = True,
        return_routing: bool = False,
    ) -> Qwen38TTNNTextModelOutput:
        """Execute only queueable device work using persistent prepared inputs."""

        if not isinstance(prepared, Qwen38TTNNPreparedDecodeInputs) or not prepared.active:
            raise TypeError("prepared decode requires live Qwen38TTNNPreparedDecodeInputs")
        self._validate_state(state)
        if prepared.position != state.position:
            raise ValueError(
                f"prepared decode position {prepared.position} does not match state position {state.position}"
            )
        return self.forward_decode(
            prepared.input_token_id,
            state,
            return_logits=return_logits,
            resolve_greedy=resolve_greedy,
            retain_hidden=retain_hidden,
            retain_hyper_residual=retain_hyper_residual,
            retain_input_state=False,
            return_routing=return_routing,
            _prepared_inputs=prepared,
        )

    def forward_decode(
        self,
        token_id: int | torch.Tensor,
        state: Qwen38TTNNTextModelState,
        *,
        return_logits: bool = True,
        resolve_greedy: bool = False,
        retain_hidden: bool = True,
        retain_hyper_residual: bool = True,
        retain_input_state: bool = False,
        return_routing: bool = False,
        layer_observer: LayerObserver | None = None,
        phase_observer: DecodePhaseObserver | None = None,
        _prepared_inputs: Qwen38TTNNPreparedDecodeInputs | None = None,
    ) -> Qwen38TTNNTextModelOutput:
        """Advance one token through all 48 target layers.

        The input state is consumed into the returned state.  Activations are
        returned only when their ``retain_*`` flag is true.  ``resolve_greedy``
        transfers only four local candidates to the host; it never gathers the
        full vocabulary.  Set ``return_logits`` independently when a later
        verifier needs device-resident sharded logits.

        An exception after any decoder layer may have mutated state and
        permanently poisons this owner.  This method never guesses that a
        partial layer is rollback-safe.
        """

        if phase_observer is not None and not callable(phase_observer):
            raise TypeError("decode phase observer must be callable")
        self._validate_state(state)
        active_snapshot = self._preflight_forward_transaction(
            state,
            retain_input_state=retain_input_state,
        )
        if state.position >= self.allocated_context:
            raise ValueError(
                f"ordinary decode exceeds allocated cache capacity {self.allocated_context}; "
                f"semantic model limit remains {MAX_CONTEXT}"
            )
        if _prepared_inputs is None:
            value, host_token = _normalize_token_id(token_id)
        else:
            if (
                not isinstance(_prepared_inputs, Qwen38TTNNPreparedDecodeInputs)
                or not _prepared_inputs.active
                or _prepared_inputs.input_token_id != token_id
                or _prepared_inputs.position != state.position
            ):
                raise ValueError("prepared decode inputs do not match the token/state invocation")
            value = _prepared_inputs.input_token_id
            host_token = _prepared_inputs.host_token
        owns_rope_inputs = _prepared_inputs is None
        try:
            if _prepared_inputs is None:
                residual = self._embed_residual(host_token)
            elif _prepared_inputs.device_token is not None:
                residual = self._embed_residual_from_device_token(_prepared_inputs.device_token)
            else:
                residual = ttnn.clone(
                    _prepared_inputs.residual_sharded,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            self._validate_residual(residual, label="prepared initial four-branch residual")
        except Qwen38TTNNCleanupError as error:
            self._mark_poisoned("embedding cleanup", 0, error)
        except BaseException as error:
            if active_snapshot is not None:
                self._mark_poisoned("speculative decode setup", 0, error)
            raise
        if _prepared_inputs is None:
            try:
                rope_inputs = self.rope.for_position(state.position)
            except BaseException as error:
                residual_slots = [residual]
                cause = _cleanup_failure_cause(
                    "decode setup",
                    [("initial residual", lambda: _release_tensor_slots(residual_slots, label="initial residual"))],
                    error,
                )
                if active_snapshot is not None or cause is not error or isinstance(error, Qwen38TTNNCleanupError):
                    self._mark_poisoned("decode setup cleanup", 0, cause)
                raise
        else:
            rope_inputs = _prepared_inputs.rope
            if not rope_inputs.active or rope_inputs.position != state.position:
                residual_slots = [residual]
                _release_tensor_slots(residual_slots, label="invalid prepared initial residual")
                raise ValueError("prepared QSA RoPE inputs do not match the state position")

        next_layers: list[Qwen38TTNNDecoderLayerState] = []
        aux_outputs: list[Qwen38TTNNDecoderLayerAux] = []
        processed_layers = 0
        inside_layer = False
        owned: list[Any | None] | None = None
        hidden = None
        logits = None
        greedy_token = None
        try:
            for layer_index, (layer, layer_state) in enumerate(zip(self.layers, state.layers)):
                qsa = layer.layer_type is Qwen38TTNNLayerType.QSA
                prepared_ple = _prepared_inputs.ple if _prepared_inputs is not None and layer_index == 1 else None
                deferred_phase_errors: list[BaseException] = []
                layer_phase_observer = None
                if phase_observer is not None:
                    phase_observer(f"before-layer-{layer_index}")

                    def observe_layer_phase(phase: str, *, index: int = layer_index) -> None:
                        if deferred_phase_errors:
                            return
                        try:
                            if type(phase) is not str:
                                raise TypeError("decoder-layer phase observer payload must be str")
                            phase_observer(f"layer-{index}-{phase}")
                        except BaseException as error:
                            deferred_phase_errors.append(error)

                    layer_phase_observer = observe_layer_phase
                inside_layer = True
                if layer_phase_observer is None:
                    result = layer.forward_decode(
                        residual,
                        layer_state,
                        token_id=host_token,
                        prepared_ple=prepared_ple,
                        cos=rope_inputs.cos if qsa else None,
                        sin=rope_inputs.sin if qsa else None,
                        block_start_cos=rope_inputs.block_start_cos if qsa else None,
                        block_start_sin=rope_inputs.block_start_sin if qsa else None,
                        retain_input_state=retain_input_state,
                        return_routing=return_routing,
                    )
                else:
                    result = layer.forward_decode(
                        residual,
                        layer_state,
                        token_id=host_token,
                        prepared_ple=prepared_ple,
                        cos=rope_inputs.cos if qsa else None,
                        sin=rope_inputs.sin if qsa else None,
                        block_start_cos=rope_inputs.block_start_cos if qsa else None,
                        block_start_sin=rope_inputs.block_start_sin if qsa else None,
                        retain_input_state=retain_input_state,
                        return_routing=return_routing,
                        phase_observer=layer_phase_observer,
                    )
                next_residual = result.residual_sharded
                inside_layer = False
                residual = next_residual
                next_layers.append(result.state)
                aux_outputs.append(result.aux)
                processed_layers += 1
                if result.state.position != state.position + 1:
                    raise RuntimeError(
                        f"target layer {layer_index} advanced to {result.state.position}, "
                        f"expected {state.position + 1}"
                    )
                self._validate_residual(residual, label=f"target layer {layer_index} output")
                if deferred_phase_errors:
                    raise deferred_phase_errors[0]
                if layer_observer is not None:
                    layer_observer(layer_index, residual, result.state, result.aux)
                if phase_observer is not None:
                    phase_observer(f"after-layer-{layer_index}")

            if owns_rope_inputs:
                rope_inputs.deallocate()

            next_state = Qwen38TTNNTextModelState(state.position + 1, tuple(next_layers), self._state_owner)
            self._validate_state(next_state)
            routing_tensors = []
            for aux in aux_outputs:
                if aux.routing is not None:
                    routing_tensors.extend((aux.routing.scores, aux.routing.indices))
            # Slots remain authoritative until ownership moves into the output.
            # Successful partial releases are cleared, so exception cleanup can
            # retry only failed groups without double-freeing earlier groups.
            owned = [residual, None, None, *routing_tensors]

            if phase_observer is not None:
                phase_observer("before-final-mixer")
            hidden = self.final_mixer(residual)
            owned[1] = hidden
            if phase_observer is not None:
                phase_observer("after-final-mixer")
            self._validate_hidden(hidden, label="terminal hyper-connection output")

            if return_logits or resolve_greedy:
                if phase_observer is not None:
                    phase_observer("before-lm-head")
                logits = self.model_io.lm_head(hidden)
                owned[2] = logits.tensor
                if phase_observer is not None:
                    phase_observer("after-lm-head")
                if resolve_greedy:
                    if phase_observer is None:
                        greedy_token = self.model_io.lm_head.greedy_token(logits)
                    else:
                        phase_observer("before-greedy-resolve")
                        greedy_token = self.model_io.lm_head.greedy_token(logits)
                        phase_observer("after-greedy-resolve")
                    if greedy_token.device.type != "cpu" or tuple(greedy_token.shape) != (1, 1, 1):
                        raise RuntimeError("greedy decode must return one CPU token as [1,1,1]")

            release_indices = []
            if not return_logits and owned[2] is not None:
                release_indices.append(2)
            if not retain_hidden:
                release_indices.append(1)
            if not retain_hyper_residual:
                release_indices.append(0)
            _release_tensor_slot_indices(owned, release_indices, label="unretained model outputs")

            output = Qwen38TTNNTextModelOutput(
                input_token_id=value,
                position=state.position,
                hyper_residual_sharded=owned[0],
                hidden_sharded=owned[1],
                logits=logits if owned[2] is not None else None,
                greedy_token=greedy_token,
                state=next_state,
                layer_aux=tuple(aux_outputs),
            )
            # ``output`` now owns exactly these still-live tensor groups.
            owned = None
            return output
        except BaseException as error:
            routing_tensors = []
            for aux in aux_outputs:
                if aux.routing is not None:
                    routing_tensors.extend((aux.routing.scores, aux.routing.indices))
            if owned is None:
                # A layer which raised may already have consumed its input.
                # Only a residual returned from a completed layer is safe for
                # this owner to release.
                owned = [None if inside_layer else residual, hidden, None if logits is None else logits.tensor]
                owned.extend(routing_tensors)
            actions = []
            if owns_rope_inputs and rope_inputs.active:
                actions.append(("QSA RoPE inputs", rope_inputs.deallocate))
            actions.append(
                (
                    "known decode outputs",
                    lambda: _release_tensor_slots(owned, label="failed decode outputs"),
                )
            )
            cause = _cleanup_failure_cause("failed ordinary decode", actions, error)
            self._mark_poisoned("forward_decode", processed_layers, cause)

    def greedy_step(
        self,
        token_id: int | torch.Tensor,
        state: Qwen38TTNNTextModelState,
        *,
        retain_input_state: bool = False,
        layer_observer: LayerObserver | None = None,
        phase_observer: DecodePhaseObserver | None = None,
    ) -> Qwen38TTNNGreedyStep:
        """Run one ordinary target step and retain only the resolved CPU token."""

        if phase_observer is None:
            output = self.forward_decode(
                token_id,
                state,
                return_logits=False,
                resolve_greedy=True,
                retain_hidden=False,
                retain_hyper_residual=False,
                retain_input_state=retain_input_state,
                layer_observer=layer_observer,
            )
        else:
            output = self.forward_decode(
                token_id,
                state,
                return_logits=False,
                resolve_greedy=True,
                retain_hidden=False,
                retain_hyper_residual=False,
                retain_input_state=retain_input_state,
                layer_observer=layer_observer,
                phase_observer=phase_observer,
            )
        try:
            if output.greedy_token is None:
                raise AssertionError("greedy target step did not resolve a token")
            next_token = int(output.greedy_token.item())
            result = Qwen38TTNNGreedyStep(
                input_token_id=output.input_token_id,
                next_token_id=next_token,
                state=output.state,
                layer_aux=output.layer_aux,
            )
            output.release_tensors()
            return result
        except BaseException as error:
            actions = [("greedy output", output.release_tensors)] if output.active else []
            cause = _cleanup_failure_cause("failed greedy step", actions, error)
            self._mark_poisoned("greedy_step", BACKBONE_LAYERS, cause)

    def prefill_serial(
        self,
        input_ids: torch.Tensor,
        *,
        state: Qwen38TTNNTextModelState | None = None,
        return_logits: bool = True,
        resolve_greedy: bool = False,
        retain_hidden: bool = True,
        retain_hyper_residual: bool = True,
        layer_observer: LayerObserver | None = None,
    ) -> Qwen38TTNNTextModelOutput:
        """Correctness-first prefill by exact one-token device decode.

        This is intentionally not a parallel prefill performance claim.  Only
        the final token's activations/logits are retained; earlier rows are
        released after their state has been committed.
        """

        self._require_healthy()
        self._require_no_active_snapshot("run serial prefill")
        if input_ids.device.type != "cpu" or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("serial prefill IDs must be CPU int32/int64")
        if input_ids.ndim != 2 or tuple(input_ids.shape[:1]) != (1,) or input_ids.shape[1] <= 0:
            raise ValueError("serial prefill IDs must be nonempty true-global-B1 [1,sequence]")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= VOCAB_SIZE:
            raise IndexError(f"serial prefill token is outside [0,{VOCAB_SIZE})")
        prompt_length = int(input_ids.shape[1])
        if state is None:
            start_position = 0
        else:
            self._validate_state(state)
            start_position = state.position
        if start_position + prompt_length > self.allocated_context:
            raise ValueError(
                f"serial prefill exceeds allocated cache capacity {self.allocated_context}; "
                f"semantic model limit remains {MAX_CONTEXT}"
            )
        current = self.allocate_state() if state is None else state
        self._validate_state(current)

        final = None
        for offset in range(prompt_length):
            last = offset == prompt_length - 1
            output = self.forward_decode(
                input_ids[:, offset : offset + 1],
                current,
                return_logits=return_logits if last else False,
                resolve_greedy=resolve_greedy if last else False,
                retain_hidden=retain_hidden if last else False,
                retain_hyper_residual=retain_hyper_residual if last else False,
                layer_observer=layer_observer,
            )
            current = output.state
            if last:
                final = output
            else:
                try:
                    output.release_tensors()
                except BaseException as error:
                    actions = [("prefill output", output.release_tensors)] if output.active else []
                    cause = _cleanup_failure_cause("serial prefill output", actions, error)
                    self._mark_poisoned("prefill_serial", BACKBONE_LAYERS, cause)
        if final is None:
            raise AssertionError("nonempty serial prefill did not produce a final row")
        return final

    # --- position-generic body: one trace captured at position 0 serves every position ---

    def _validate_generic_state(self, state: Qwen38TTNNTextModelGenericState) -> None:
        self._require_healthy()
        if not isinstance(state, Qwen38TTNNTextModelGenericState) or state._owner is not self._state_owner:
            raise ValueError("generic text-model state was not allocated by this model owner")
        if not isinstance(state.position, Qwen38TTNNDevicePosition):
            raise TypeError("generic text-model state requires a Qwen38TTNNDevicePosition")
        if len(state.layers) != BACKBONE_LAYERS:
            raise ValueError(f"generic text-model state must contain {BACKBONE_LAYERS} target layers")
        for layer_index, (layer, layer_state) in enumerate(zip(self.layers, state.layers)):
            if layer_state.namespace is not Qwen38TTNNLayerNamespace.BACKBONE or layer_state.layer_index != layer_index:
                raise ValueError(f"generic target state {layer_index} identity does not match its layer")
            layer.validate_generic_state(layer_state)

    def allocate_generic_state(self) -> Qwen38TTNNTextModelGenericState:
        """Allocate the device position and every layer's fixed-address generic state.

        The first call also builds the model-lifetime generic constants: the
        resident RoPE tables and the QSA position constants.
        """

        self._require_healthy()
        if self.rope_table is None:
            self.rope_table = Qwen38TTNNRoPETable.build(
                self.mesh_device, self.mesh_contract, self.config, self.allocated_context
            )
        if self.qsa_position_constants is None:
            qsa = next(layer.attention for layer in self.layers if layer.layer_type is Qwen38TTNNLayerType.QSA)
            self.qsa_position_constants = qsa_module.Qwen38TTNNQSAPositionConstants.build(
                self.mesh_device, self.mesh_contract, qsa.allocated_compressed_blocks
            )
        position = Qwen38TTNNDevicePosition.allocate(self.mesh_device, self.mesh_contract, position=0)
        allocated: list[Qwen38TTNNDecoderLayerGenericState] = []
        try:
            for layer in self.layers:
                allocated.append(layer.allocate_generic_state())
            result = Qwen38TTNNTextModelGenericState(position, tuple(allocated), self._state_owner)
            self._validate_generic_state(result)
            return result
        except BaseException as error:
            actions = [
                (
                    f"target layer {index} generic state",
                    lambda layer=layer, layer_state=layer_state: layer.release_generic_state(layer_state),
                )
                for index, (layer, layer_state) in reversed(tuple(enumerate(zip(self.layers, allocated))))
            ]
            actions.append(("device position", position.deallocate))
            cause = _cleanup_failure_cause("generic text-model state allocation", actions, error)
            if cause is not error:
                self._mark_poisoned("allocate_generic_state cleanup", 0, cause)
            raise

    def reset_generic_state_inplace(self, state: Qwen38TTNNTextModelGenericState) -> None:
        """Position 0 and position-zero buffer contents at every captured address (the whole prologue)."""

        self._validate_generic_state(state)
        try:
            state.position.reset(0, 0)
            for layer, layer_state in zip(self.layers, state.layers):
                layer.reset_generic_state_inplace(layer_state)
        except BaseException as error:
            self._mark_poisoned("reset_generic_state_inplace", 0, error)

    def release_generic_state(self, state: Qwen38TTNNTextModelGenericState) -> None:
        self._validate_generic_state(state)
        actions = [
            (
                f"target layer {index} generic state",
                lambda layer=layer, layer_state=layer_state: layer.release_generic_state(layer_state),
            )
            for index, (layer, layer_state) in reversed(tuple(enumerate(zip(self.layers, state.layers))))
        ]
        actions.append(("device position", state.position.deallocate))
        try:
            _run_cleanup_actions("generic text-model state", actions)
        except BaseException as error:
            self._mark_poisoned("release_generic_state", 0, error)

    # -- the prompt-end snapshot: the recurrent buffers of the generic state, copied on device ---------------------

    def allocate_generic_snapshot(
        self,
        state: Qwen38TTNNTextModelGenericState,
        *,
        extra_layers: Sequence[tuple[Qwen38TTNNDecoderLayer, Qwen38TTNNDecoderLayerGenericState]] = (),
    ) -> Qwen38TTNNTextModelGenericSnapshot:
        """One ``empty_like`` buffer per recurrent tensor of ``state`` (and of ``extra_layers``); allocate it before
        any capture so the traces' tracker sees no later allocation."""

        self._validate_generic_state(state)
        pairs: list[tuple[str, Any, Any]] = []
        gdn_states: list[Any] = []
        ple_states: list[Any] = []
        for layer, layer_state in (*zip(self.layers, state.layers), *extra_layers):
            layer.validate_generic_state(layer_state)
            attention = layer_state.attention
            name = f"layer {layer.layer_index}"
            if layer.layer_type is Qwen38TTNNLayerType.GDN:
                gdn_states.append(attention)
                pairs.append((f"{name} GDN recurrent", attention.recurrent, None))
                pairs.extend((f"{name} GDN conv[{slot}]", tensor, None) for slot, tensor in enumerate(attention.conv))
            else:
                pairs.append((f"{name} QSA KV staging", attention.kv_staging, None))
                pairs.append((f"{name} QSA raw-key ring", attention.raw_key_ring, None))
            if layer_state.ple is not None:
                ple_states.append(layer_state.ple)
                pairs.extend(
                    (f"{name} PLE conv[{slot}]", tensor, None) for slot, tensor in enumerate(layer_state.ple.conv)
                )
        allocated: list[Any] = []
        try:
            for index, (label, source, _) in enumerate(pairs):
                target = ttnn.empty_like(source, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                allocated.append(target)
                pairs[index] = (label, source, target)
        except BaseException:
            _deallocate_unique(*allocated)
            raise
        return Qwen38TTNNTextModelGenericSnapshot(tuple(pairs), tuple(gdn_states), tuple(ple_states), self._state_owner)

    def _validate_generic_snapshot(self, snapshot: Qwen38TTNNTextModelGenericSnapshot) -> None:
        if not isinstance(snapshot, Qwen38TTNNTextModelGenericSnapshot) or snapshot._owner is not self._state_owner:
            raise ValueError("generic text-model snapshot was not allocated by this model owner")

    def capture_generic_snapshot(
        self, state: Qwen38TTNNTextModelGenericState, snapshot: Qwen38TTNNTextModelGenericSnapshot, *, position: int
    ) -> None:
        """Copy every recurrent buffer into the snapshot: device ops only, nothing allocated, no host sync (enqueued
        behind the step whose state they record).  ``position`` is the host's count of consumed positions."""

        self._validate_generic_state(state)
        self._validate_generic_snapshot(snapshot)
        if isinstance(position, bool) or type(position) is not int or not 0 <= position <= self.allocated_context:
            raise ValueError(f"snapshot position must be an int in [0,{self.allocated_context}], got {position!r}")
        try:
            for label, source, target in snapshot.pairs:
                _copy_inplace(source, target, label=f"{label} snapshot capture")
        except BaseException as error:
            self._mark_poisoned("capture_generic_snapshot", 0, error)
        snapshot.position = position
        snapshot.rope_shift = state.position.rope_shift
        snapshot.captured = True

    def restore_generic_snapshot(
        self, snapshot: Qwen38TTNNTextModelGenericSnapshot, state: Qwen38TTNNTextModelGenericState
    ) -> None:
        """The snapshot's buffers back into the state, the position to the snapshot's, the GDN conv phases to
        ``position mod 4`` and the PLE host context cleared: the state of ``position`` consumed positions."""

        self._validate_generic_state(state)
        self._validate_generic_snapshot(snapshot)
        if not snapshot.captured:
            raise RuntimeError("cannot restore a generic snapshot that was never captured")
        try:
            for label, source, target in snapshot.pairs:
                _copy_inplace(target, source, label=f"{label} snapshot restore")
            state.position.reset(snapshot.position, snapshot.rope_shift)
        except BaseException as error:
            self._mark_poisoned("restore_generic_snapshot", 0, error)
        for gdn_state in snapshot.gdn_states:
            gdn_state.conv_phase = snapshot.position % gdn_module.CONV_KERNEL_SIZE
        for ple_state in snapshot.ple_states:
            ple_state.token_context = None

    def release_generic_snapshot(self, snapshot: Qwen38TTNNTextModelGenericSnapshot) -> None:
        self._validate_generic_snapshot(snapshot)
        _deallocate_unique(*(target for _, _, target in snapshot.pairs))
        snapshot.captured = False

    def prepare_generic_decode_inputs(
        self,
        token_id: int | torch.Tensor,
        state: Qwen38TTNNTextModelGenericState,
        *,
        device_token,
    ) -> Qwen38TTNNPreparedDecodeInputs:
        """The one persistent host-prepared input of the generic body: the PLE row of the seed token.

        ``device_token`` is the caller-owned token row the body embeds.  No RoPE
        upload and no host position: both are read on device.  The PLE row is
        prepared without an n-gram context (a fresh or reset state); the caller
        rewrites ``prepared.ple.embedding_sharded`` in place for later tokens
        and tracks the n-gram context itself.
        """

        self._validate_generic_state(state)
        if device_token is None:
            raise ValueError("generic decode inputs require the caller-owned device token row")
        value, host_token = _normalize_token_id(token_id)
        self.model_io.embedding.validate_token_row(device_token, label="prepared device token row")
        ple_layer = self.layers[PLE_CHECKPOINT_LAYER]
        ple_state = state.layers[PLE_CHECKPOINT_LAYER].ple
        if ple_layer.ple is None or ple_state is None:
            raise RuntimeError("checkpoint layer 1 PLE owner/state is unavailable")
        if ple_state.token_context is not None:
            raise ValueError("generic PLE row must be prepared from a freshly allocated or reset state")
        prepared_ple = ple_layer.ple.prepare_decode_input(host_token, ple_state, resident_lookup=True)
        return Qwen38TTNNPreparedDecodeInputs(
            input_token_id=value,
            position=None,
            host_token=host_token,
            residual_sharded=None,
            rope=None,
            ple=prepared_ple,
            device_token=device_token,
        )

    def _require_generic_decode_inputs(
        self, prepared: Qwen38TTNNPreparedDecodeInputs, state: Qwen38TTNNTextModelGenericState
    ) -> None:
        if (
            not isinstance(prepared, Qwen38TTNNPreparedDecodeInputs)
            or not prepared.active
            or prepared.device_token is None
            or prepared.rope is not None
            or prepared.position is not None
        ):
            raise TypeError("generic decode requires live device-token prepared inputs without host RoPE/position")
        self._validate_generic_state(state)
        if self.rope_table is None or self.qsa_position_constants is None:
            raise RuntimeError("generic constants are missing; allocate the generic state through this owner")

    def _validate_generic_head(self, head: Qwen38TTNNGenericHead) -> None:
        """Every HEAD -> TAIL handoff tensor against GENERIC_HEAD_HANDOFF, actual vs expected in the error."""

        if not isinstance(head, Qwen38TTNNGenericHead) or head._owner is not self._state_owner:
            raise ValueError("generic head was not produced by this model owner")
        if not head.active:
            raise RuntimeError("generic head tensors were already released")
        for handoff in GENERIC_HEAD_HANDOFF:
            tensor = getattr(head, handoff.attribute)
            actual, expected = handoff.describe(tensor), handoff.describe()
            if actual != expected:
                raise RuntimeError(f"HEAD/TAIL handoff {handoff.name}: actual {actual} vs expected {expected}")
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def forward_decode_generic_head(
        self,
        prepared: Qwen38TTNNPreparedDecodeInputs,
        state: Qwen38TTNNTextModelGenericState,
    ) -> Qwen38TTNNGenericHead:
        """HEAD of one position-generic step: embed the device token row, run layers[:GENERIC_HEAD_LAYERS].

        Reads the token row and layer 0's in-place GDN state only: no PLE row,
        no position, no RoPE, no host-written input, so a captured HEAD can be
        enqueued before the host knows the token.  Returns the handoff TAIL
        consumes.  Any failure poisons this owner.
        """

        self._require_generic_decode_inputs(prepared, state)
        processed_layers = 0
        try:
            residual = self._embed_residual_from_device_token(prepared.device_token)
            for layer_index in range(GENERIC_HEAD_LAYERS):
                layer, layer_state = self.layers[layer_index], state.layers[layer_index]
                residual = layer.forward_decode_generic(
                    residual, layer_state, prepared_ple=None, rope=None, qsa_position=None
                )
                processed_layers += 1
                self._validate_residual(residual, label=f"generic target layer {layer_index} output")
            head = Qwen38TTNNGenericHead(residual, self._state_owner)
            self._validate_generic_head(head)
            return head
        except BaseException as error:
            self._mark_poisoned("forward_decode_generic_head", processed_layers, error)

    def forward_decode_generic_tail(
        self,
        head: Qwen38TTNNGenericHead,
        prepared: Qwen38TTNNPreparedDecodeInputs,
        state: Qwen38TTNNTextModelGenericState,
        *,
        return_logits: bool = True,
        release_head: bool = True,
        retain_mtp_inputs: bool = False,
    ) -> Qwen38TTNNGenericDecodeOutput:
        """TAIL of one position-generic step: position derivation, layers[GENERIC_HEAD_LAYERS:], mixer, LM head, advance.

        Derives every position-dependent input from the device position (RoPE
        rows, QSA position inputs), runs the remaining layers in place (the
        first one consumes the PLE row), applies the final mixer and LM head,
        then advances the device position as the last op.  ``release_head=False``
        leaves the head residual allocated: the HEAD/TAIL traces bake its fixed
        address in, so the capture retains it for the life of the traces.
        ``retain_mtp_inputs`` returns the layer-47 residual, the RoPE rows and
        the QSA position inputs in the output instead of releasing them (the
        MTP-drafting server's epilogue runs the MTP layer's row on them).  Any
        failure poisons this owner: the in-place state cannot be rolled back.
        """

        self._require_generic_decode_inputs(prepared, state)
        self._validate_generic_head(head)
        if not isinstance(retain_mtp_inputs, bool):
            raise TypeError(f"retain_mtp_inputs must be a bool, got {retain_mtp_inputs!r}")
        processed_layers = GENERIC_HEAD_LAYERS
        try:
            if self._position_derive is None:
                index_row = state.position.index_row()
                # The RoPE rows read the table at P - S and (P & ~3) - S (S: the sequence's rotary shift, 0 for text).
                rope_index_row = state.position.rope_index_row(index_row)
                rope_block_start_row = state.position.rope_block_start_index_row(index_row)
                rope = self.rope_table.rows(rope_index_row, rope_block_start_row)
                _deallocate_unique(index_row, rope_index_row, rope_block_start_row)
                qsa_position = qsa_module.derive_qsa_position_inputs(state.position.scalar, self.qsa_position_constants)
            else:
                rope, qsa_position = self._position_derive(self, state)
            residual = head.residual
            for layer_index in range(GENERIC_HEAD_LAYERS, BACKBONE_LAYERS):
                layer, layer_state = self.layers[layer_index], state.layers[layer_index]
                first_tail_layer = layer_index == GENERIC_HEAD_LAYERS
                residual = layer.forward_decode_generic(
                    residual,
                    layer_state,
                    prepared_ple=prepared.ple if layer_index == PLE_CHECKPOINT_LAYER else None,
                    rope=rope,
                    qsa_position=qsa_position,
                    release_input=release_head or not first_tail_layer,
                )
                if first_tail_layer and release_head:
                    head.active = False
                processed_layers += 1
                self._validate_residual(residual, label=f"generic target layer {layer_index} output")
            if not retain_mtp_inputs:
                qsa_position.deallocate()
                rope.deallocate()
            hidden = self.final_mixer(residual)
            if not retain_mtp_inputs:
                _deallocate_unique(residual)
            self._validate_hidden(hidden, label="terminal hyper-connection output")
            logits = self.model_io.lm_head(hidden) if return_logits else None
            _deallocate_unique(hidden)
            retained = (residual, rope, qsa_position) if retain_mtp_inputs else (None, None, None)
            state.position.advance()
            return Qwen38TTNNGenericDecodeOutput(
                logits, residual=retained[0], rope=retained[1], qsa_position=retained[2]
            )
        except BaseException as error:
            self._mark_poisoned("forward_decode_generic_tail", processed_layers, error)

    def forward_decode_generic(
        self,
        prepared: Qwen38TTNNPreparedDecodeInputs,
        state: Qwen38TTNNTextModelGenericState,
        *,
        return_logits: bool = True,
    ) -> Qwen38TTNNGenericDecodeOutput:
        """One position-generic step: fixed op sequence, fixed shapes, no host ints, no host I/O.

        HEAD then TAIL in one body (:meth:`forward_decode_generic_head`,
        :meth:`forward_decode_generic_tail`); the first TAIL layer consumes the
        head residual as before, so the eager step and the single-body capture
        are the split's op sequence with nothing retained.
        """

        head = self.forward_decode_generic_head(prepared, state)
        return self.forward_decode_generic_tail(head, prepared, state, return_logits=return_logits)

    def _abort_trace_capture(self, trace_id: int, *, cq_id: int, part: str, error: BaseException) -> None:
        """A failure inside a trace capture: end the capture and release its trace so the failure path can
        synchronize and close the mesh (the op that failed is not in the trace; the capture's contents are dropped).
        Each step is best effort: the failure that matters is ``error``, re-raised by the caller."""

        for label, step in (
            ("end", lambda: ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=cq_id)),
            ("release", lambda: ttnn.release_trace(self.mesh_device, trace_id)),
        ):
            try:
                step()
            except BaseException as cleanup_error:  # noqa: BLE001
                warnings.warn(
                    f"generic {part} capture {label} after a failure inside the capture: "
                    f"{type(cleanup_error).__name__}: {cleanup_error} (the failure: {type(error).__name__})",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def capture_decode_generic(
        self,
        prepared: Qwen38TTNNPreparedDecodeInputs,
        state: Qwen38TTNNTextModelGenericState,
        *,
        residue: int,
        split: bool,
        guard: Callable[[str], AbstractContextManager[Any]],
        epilogue: Callable[[Qwen38TTNNGenericDecodeOutput], Any],
        phase_observer: Callable[[str], None] | None = None,
        regime: int = 0,
        cq_id: int = 0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        retain_mtp_inputs: bool = False,
    ) -> Qwen38TTNNGenericDecodeCapture:
        """Capture one residue class of the generic body: one single-body trace, or a HEAD and a TAIL trace.

        Each part is recorded inside ``corruptible_allocation_scope`` and the
        caller's ``guard(label)`` (a no-host-I/O, no-sync context; whatever it
        yields is collected into ``guard_attempts``).  ``epilogue(output)`` runs
        inside the last part's capture after the model (the caller's greedy
        candidates, device resolve and token-row copy) and its result is
        returned.  ``retain_mtp_inputs`` (the split form only) hands the epilogue
        the TAIL's retained residual, RoPE rows and QSA position inputs.  Capture
        records commands without executing them, so the device state is unchanged
        while the host ring-phase bookkeeping advances as if the body ran;
        ``phase_observer`` sees ``before-<part>-capture`` and
        ``after-<part>-capture`` for the caller's phase checks.  A body failure
        poisons this owner; the open capture is ended and its trace released
        before the error propagates (a synchronize inside an open capture is a
        TT_FATAL and the mesh close then hangs), then the process must exit.
        """

        if phase_observer is not None and not callable(phase_observer):
            raise TypeError("capture phase observer must be callable")
        if not isinstance(residue, int) or not isinstance(regime, int) or residue < 0 or regime < 0:
            raise ValueError(f"trace residue/regime must be non-negative ints, got {residue!r}/{regime!r}")
        if retain_mtp_inputs and not split:
            raise ValueError("retain_mtp_inputs needs the split (HEAD / TAIL) capture")
        parts = GENERIC_TRACE_PARTS_SPLIT if split else GENERIC_TRACE_PARTS_SINGLE
        head: Qwen38TTNNGenericHead | None = None
        output: Qwen38TTNNGenericDecodeOutput | None = None
        epilogue_result: Any = None
        trace_ids: dict[Qwen38TTNNGenericTraceKey, int] = {}
        capture_ns: dict[str, int] = {}
        guard_attempts: list[str] = []
        for part in parts:
            if phase_observer is not None:
                phase_observer(f"before-{part}-capture")
            with corruptible_allocation_scope(self.mesh_device):
                trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=cq_id)
                started_ns = clock_ns()
                try:
                    with guard(f"generic {part} capture residue {residue} regime {regime}") as blocked:
                        if part == "head":
                            head = self.forward_decode_generic_head(prepared, state)
                        else:
                            output = (
                                self.forward_decode_generic_tail(
                                    head, prepared, state, release_head=False, retain_mtp_inputs=retain_mtp_inputs
                                )
                                if part == "tail"
                                else self.forward_decode_generic(prepared, state)
                            )
                            epilogue_result = epilogue(output)
                except BaseException as error:
                    self._abort_trace_capture(trace_id, cq_id=cq_id, part=part, error=error)
                    raise
                ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=cq_id)
                capture_ns[part] = clock_ns() - started_ns
            guard_attempts.extend(blocked or ())
            trace_ids[Qwen38TTNNGenericTraceKey(part, residue, regime)] = trace_id
            if phase_observer is not None:
                phase_observer(f"after-{part}-capture")
        if output is None or (split and (head is None or not head.active)):
            raise RuntimeError("generic body capture completed without its output or its retained head")
        return Qwen38TTNNGenericDecodeCapture(
            residue=residue,
            regime=regime,
            parts=parts,
            trace_ids=trace_ids,
            head=head,
            output=output,
            epilogue=epilogue_result,
            capture_ns=capture_ns,
            guard_attempts=tuple(guard_attempts),
        )

    # ------------------------------------------------------------------ prefill chunk (32 or 128 rows)
    # forward_prefill_chunk_generic is the position-generic body over the positions P .. P + rows - 1 of one
    # chunk (P % 32 == 0), traced once per row count and replayed per chunk: no final mixer, no LM head, no
    # host ints; the chunk state carries the GDN/PLE histories and the hand-off sources between chunks, the
    # generic state is the decode's and is updated in place, and P += rows is the last op.  The 32-row form
    # serves full chunks and the padded tail through its accept scalar; the 128-row form (allocated beside the
    # 32-row state, sharing its histories) serves full chunks only.

    def _validate_chunk_state(self, chunk_state: Qwen38TTNNTextModelChunkState) -> None:
        self._require_healthy()
        if not isinstance(chunk_state, Qwen38TTNNTextModelChunkState) or chunk_state._owner is not self._state_owner:
            raise ValueError("text-model chunk state was not allocated by this model owner")
        rows = chunk_state.rows
        if rows not in CHUNK_ROW_COUNTS and not is_slab_rows(rows):
            raise ValueError(
                f"text-model chunk state rows must be one of {CHUNK_ROW_COUNTS} or a slab row count, got {rows!r}"
            )
        if (
            len(chunk_state.layers) != BACKBONE_LAYERS
            or chunk_state.rows_constants.rows != rows
            or chunk_state.qsa_chunk_constants.rows != rows
            or any(layer_state.rows != rows for layer_state in chunk_state.layers)
        ):
            raise ValueError(f"text-model chunk state must hold 48 layer states over the {rows}-row constants")
        self.model_io.embedding.validate_token_rows(chunk_state.token_row, rows=rows, label="chunk token rows")
        if (chunk_state.accepted is None) != (rows != CHUNK_ROWS):
            raise RuntimeError(
                f"the {rows}-row chunk state {'carries' if rows != CHUNK_ROWS else 'lacks'} an accept scalar"
            )
        if chunk_state.accepted is not None and (
            _shape(chunk_state.accepted) != (1, 1, 1, 1) or chunk_state.accepted.dtype != ttnn.float32
        ):
            raise RuntimeError(
                f"chunk accept scalar must be FP32 [1,1,1,1], got {tensor_metadata(chunk_state.accepted)}"
            )
        if (chunk_state.local_combine_output is None) != (rows == CHUNK_ROWS):
            raise RuntimeError(
                f"the {rows}-row chunk state's shared MoE combine buffer is {chunk_state.local_combine_output}"
            )
        if not chunk_state.ple_rows.active:
            raise RuntimeError("chunk PLE rows were released")
        rope = chunk_state.rope_rows
        block_rows = chunk_state.qsa_chunk_constants.block_tiles * ttnn.TILE_SIZE
        for name, tensor, count in (
            ("cos", rope.cos, rows),
            ("sin", rope.sin, rows),
            ("block-start cos", rope.block_start_cos, block_rows),
            ("block-start sin", rope.block_start_sin, block_rows),
        ):
            if (
                tensor is None
                or _shape(tensor) != (1, 1, count, QSA_ROPE_DIM)
                or tensor.dtype != ttnn.bfloat16
                or tensor.layout != ttnn.TILE_LAYOUT
            ):
                raise RuntimeError(
                    f"chunk RoPE {name} rows must be BF16 TILE [1,1,{count},{QSA_ROPE_DIM}], got "
                    f"{None if tensor is None else tensor_metadata(tensor)}"
                )
        features = chunk_state.feature_rows
        if (
            features.rows != rows
            or _shape(features.tensor) != (1, 1, rows, LOCAL_HIDDEN_SIZE)
            or features.tensor.dtype != ttnn.bfloat16
            or features.tensor.layout != ttnn.TILE_LAYOUT
        ):
            raise RuntimeError(
                f"chunk feature rows must be BF16 TILE [1,1,{rows},{LOCAL_HIDDEN_SIZE}] per device, got "
                f"{tensor_metadata(features.tensor)} for {features.rows} rows"
            )

    def _allocate_chunk_rope_rows(self, rows: int, block_tiles: int) -> Qwen38TTNNRoPEInputs:
        """The chunk's persistent RoPE rows (host-written per chunk): cos / sin BF16 TILE ``[1,1,rows,64]`` and the
        block-start rows ``[1,1,block_tiles*32,64]``, replicated; zero until the first chunk's write."""

        owned: list[Any] = []
        try:
            for count in (rows, rows, block_tiles * ttnn.TILE_SIZE, block_tiles * ttnn.TILE_SIZE):
                tensor = ttnn.from_torch(
                    torch.zeros((1, 1, count, QSA_ROPE_DIM), dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
                )
                owned.append(tensor)
                self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
        except BaseException:
            _deallocate_unique(*owned)
            raise
        return Qwen38TTNNRoPEInputs(None, *owned)

    def host_chunk_feature_rows(self, feature_image):
        """The host tensor of a chunk's feature rows (``vision_splice.feature_rows_image``: ``[1,1,rows,2560]`` BF16),
        hidden-sharded for the copy into a chunk's persistent feature rows (the backbone's or the MTP extension's)."""

        return ttnn.from_torch(
            feature_image,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
        )

    def _clean_feature_rows_host(self, rows: int):
        """The host tensor of a chunk's clean feature rows (every element -0.0), hidden-sharded like the upload."""

        cache = self._clean_feature_rows_hosts
        if rows not in cache:
            cache[rows] = ttnn.from_torch(
                vision_splice.clean_feature_rows(rows, hidden=HIDDEN_SIZE),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
            )
        return cache[rows]

    def _allocate_chunk_feature_rows(self, rows: int) -> Qwen38TTNNChunkFeatureRows:
        """The chunk's persistent feature rows (the vision splice): hidden-sharded BF16 TILE ``[1,1,rows,640]`` per
        device, clean (-0.0) until an image chunk's write."""

        tensor = ttnn.from_torch(
            vision_splice.clean_feature_rows(rows, hidden=HIDDEN_SIZE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
        )
        try:
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        except BaseException:
            _deallocate_unique(tensor)
            raise
        return Qwen38TTNNChunkFeatureRows(tensor, rows)

    def allocate_chunk_state(
        self,
        state: Qwen38TTNNTextModelGenericState,
        *,
        rows: int = CHUNK_ROWS,
        base: Qwen38TTNNTextModelChunkState | None = None,
    ) -> Qwen38TTNNTextModelChunkState:
        """Allocate the chunk constants and every layer's chunk buffers (before the decode captures).

        ``rows`` picks the form; the 128-row form and a slab need ``base``, the 32-row chunk state of the same
        generic state, whose per-layer GDN and PLE histories they share (one shared MoE combine buffer for the 48
        layers).  A slab's 35 GDN layers share one set of pass buffers (the first GDN layer's rows state owns them).
        """

        self._validate_generic_state(state)
        if self.rope_table is None or self.qsa_position_constants is None:
            raise RuntimeError("generic constants are missing; allocate the generic state through this owner")
        chunk_row_tiles(rows)
        if (base is None) != (rows == CHUNK_ROWS):
            raise ValueError(
                f"the {rows}-row chunk state {'needs' if rows != CHUNK_ROWS else 'takes no'} base 32-row chunk state"
            )
        if base is not None:
            self._validate_chunk_state(base)
            if base.rows != CHUNK_ROWS:
                raise ValueError(f"the base chunk state must be the {CHUNK_ROWS}-row one, got {base.rows} rows")
        gdn = next(layer.attention for layer in self.layers if layer.layer_type is Qwen38TTNNLayerType.GDN)
        qsa = next(layer.attention for layer in self.layers if layer.layer_type is Qwen38TTNNLayerType.QSA)
        rows_constants = gdn.allocate_rows_constants(rows)
        qsa_chunk_constants = None
        local_combine_output = None
        token_row = None
        accepted = None
        ple_rows = None
        rope_rows = None
        feature_rows = None
        allocated: list[Qwen38TTNNDecoderLayerChunkState] = []
        try:
            qsa_chunk_constants = qsa_module.Qwen38TTNNQSAChunkConstants.build(
                self.mesh_device, self.mesh_contract, qsa.allocated_compressed_blocks, rows=rows
            )
            if base is not None:
                # The rows-128 instances' shared combine buffer holds one moe_compute call's rows (128: the whole chunk).
                local_combine_output = moe_module.allocate_local_combine_output(
                    self.mesh_device, self.mesh_contract, moe_module.routed_tokens_per_call_for(rows)
                )
            gdn_body = None
            for index, layer in enumerate(self.layers):
                allocated.append(
                    layer.allocate_chunk_state(
                        rows_constants,
                        base=None if base is None else base.layers[index],
                        local_combine_output=local_combine_output,
                        gdn_body=gdn_body if layer.layer_type is Qwen38TTNNLayerType.GDN else None,
                    )
                )
                if is_slab_rows(rows) and gdn_body is None and layer.layer_type is Qwen38TTNNLayerType.GDN:
                    gdn_body = allocated[-1].attention
            token_row = self.model_io.embedding.upload_token_rows(rows)
            if rows == CHUNK_ROWS:
                accepted = ttnn.from_torch(
                    torch.full((1, 1, 1, 1), float(CHUNK_ROWS - 1)),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
                )
            ple_layer, ple_rows_state = self.layers[PLE_CHECKPOINT_LAYER], allocated[PLE_CHECKPOINT_LAYER].ple
            if ple_layer.ple is None or ple_rows_state is None:
                raise RuntimeError("checkpoint layer 1 PLE owner/rows state is unavailable")
            ple_rows = ple_layer.ple.prepare_rows_input([0] * rows, ple_rows_state, rows=rows)
            rope_rows = self._allocate_chunk_rope_rows(rows, qsa_chunk_constants.block_tiles)
            feature_rows = self._allocate_chunk_feature_rows(rows)
            result = Qwen38TTNNTextModelChunkState(
                rows_constants=rows_constants,
                qsa_chunk_constants=qsa_chunk_constants,
                layers=tuple(allocated),
                token_row=token_row,
                ple_rows=ple_rows,
                accepted=accepted,
                rope_rows=rope_rows,
                feature_rows=feature_rows,
                _owner=self._state_owner,
                rows=rows,
                local_combine_output=local_combine_output,
                position=state.position,
            )
            self._validate_chunk_state(result)
            return result
        except BaseException as error:
            actions = [
                (
                    f"target layer {index} chunk state",
                    lambda layer=layer, layer_state=layer_state: layer.release_chunk_state(layer_state),
                )
                for index, (layer, layer_state) in reversed(tuple(enumerate(zip(self.layers, allocated))))
            ]
            if feature_rows is not None:
                actions.append(("chunk feature rows", lambda: ttnn.deallocate(feature_rows.tensor)))
            if rope_rows is not None:
                actions.append(("chunk RoPE rows", rope_rows.deallocate))
            if ple_rows is not None:
                actions.append(("chunk PLE rows", ple_rows.release))
            for label, tensor in (
                ("chunk accept scalar", accepted),
                ("chunk token row", token_row),
                ("chunk MoE combine buffer", local_combine_output),
            ):
                if tensor is not None:
                    actions.append((label, lambda tensor=tensor: ttnn.deallocate(tensor)))
            if qsa_chunk_constants is not None:
                actions.append(("QSA chunk constants", qsa_chunk_constants.deallocate))
            actions.append(("GDN rows constants", rows_constants.deallocate))
            cause = _cleanup_failure_cause("chunk text-model state allocation", actions, error)
            if cause is not error:
                self._mark_poisoned("allocate_chunk_state cleanup", 0, cause)
            raise

    def release_chunk_state(self, chunk_state: Qwen38TTNNTextModelChunkState) -> None:
        """Release a chunk state; a 128-row state goes before the 32-row state whose histories it shares."""

        self._validate_chunk_state(chunk_state)
        actions = [
            (
                f"target layer {index} chunk state",
                lambda layer=layer, layer_state=layer_state: layer.release_chunk_state(layer_state),
            )
            for index, (layer, layer_state) in reversed(tuple(enumerate(zip(self.layers, chunk_state.layers))))
        ]
        actions.append(("chunk feature rows", lambda: ttnn.deallocate(chunk_state.feature_rows.tensor)))
        actions.append(("chunk RoPE rows", chunk_state.rope_rows.deallocate))
        actions.append(("chunk PLE rows", chunk_state.ple_rows.release))
        if chunk_state.accepted is not None:
            actions.append(("chunk accept scalar", lambda: ttnn.deallocate(chunk_state.accepted)))
        actions.append(("chunk token row", lambda: ttnn.deallocate(chunk_state.token_row)))
        if chunk_state.local_combine_output is not None:
            actions.append(("chunk MoE combine buffer", lambda: ttnn.deallocate(chunk_state.local_combine_output)))
        actions.append(("QSA chunk constants", chunk_state.qsa_chunk_constants.deallocate))
        actions.append(("GDN rows constants", chunk_state.rows_constants.deallocate))
        try:
            _run_cleanup_actions("chunk text-model state", actions)
        except BaseException as error:
            self._mark_poisoned("release_chunk_state", 0, error)

    def reset_chunk_state_inplace(
        self, state: Qwen38TTNNTextModelGenericState, chunk_state: Qwen38TTNNTextModelChunkState
    ) -> None:
        """Seed the chunk carry from the generic state at fixed addresses: after ``reset_generic_state_inplace``
        (position 0: zero histories) or after decode steps that ended at ``P % 32 == 0`` (the histories from
        the GDN ring and the PLE slots).  The accept scalar is set to a full chunk; the position is untouched.
        A 128-row state shares its histories with the 32-row state (reset that one too); only its QSA kept
        buffers are zeroed here."""

        self._validate_generic_state(state)
        self._validate_chunk_state(chunk_state)
        try:
            for layer, layer_state, layer_chunk in zip(self.layers, state.layers, chunk_state.layers):
                layer.reset_chunk_state_inplace(layer_chunk, layer_state)
            if chunk_state.accepted is not None:
                self.write_chunk_accepted(chunk_state, CHUNK_ROWS - 1)
        except BaseException as error:
            self._mark_poisoned("reset_chunk_state_inplace", 0, error)

    def write_chunk_accepted(self, chunk_state: Qwen38TTNNTextModelChunkState, accepted: int) -> None:
        """Host write of the accept scalar (outside any trace): 31 for a full chunk, r - 1 for r real rows."""

        if isinstance(accepted, bool) or type(accepted) is not int or not 0 <= accepted < CHUNK_ROWS:
            raise ValueError(f"chunk accept count must be an int in [0,{CHUNK_ROWS}), got {accepted!r}")
        if chunk_state.accepted is None:
            raise ValueError(f"the {chunk_state.rows}-row chunk state has no accept scalar")
        host = ttnn.from_torch(
            torch.full((1, 1, 1, 1), float(accepted)),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, chunk_state.accepted)

    def chunk_rope_host_rows(
        self, chunk_state: Qwen38TTNNTextModelChunkState, positions: torch.Tensor
    ) -> tuple[Any, Any, Any, Any]:
        """The host tensors of a chunk's RoPE rows from the rows' (t, h, w) positions ``positions`` (int64
        ``[3, rows]``): cos / sin ``[1,1,rows,64]`` and the block-start rows ``[1,1,block_tiles*32,64]`` (lane i =
        the row of the chunk's token 4 i, the first token of compressed block i; the lanes past rows / 4 repeat
        the last block's row), BF16 TILE replicated."""

        rows = chunk_state.rows
        if positions.dtype != torch.int64 or tuple(positions.shape) != (3, rows):
            raise ValueError(
                f"chunk positions must be int64 [3, {rows}], got {positions.dtype} {tuple(positions.shape)}"
            )
        if self.rope_table is None:
            raise RuntimeError("generic constants are missing; allocate the generic state through this owner")
        cos, sin = self.rope_table.host_rows_for(positions)
        block_cos, block_sin = self.rope_table.host_rows_for(positions[:, ::QSA_COMPRESS_RATIO])
        block_rows = chunk_state.qsa_chunk_constants.block_tiles * ttnn.TILE_SIZE
        if block_rows > block_cos.shape[0]:
            pad = block_rows - block_cos.shape[0]
            block_cos = torch.cat([block_cos, block_cos[-1:].expand(pad, -1)], dim=0)
            block_sin = torch.cat([block_sin, block_sin[-1:].expand(pad, -1)], dim=0)
        hosts = []
        for host, count in ((cos, rows), (sin, rows), (block_cos, block_rows), (block_sin, block_rows)):
            hosts.append(
                ttnn.from_torch(
                    host.reshape(1, 1, count, QSA_ROPE_DIM).contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
                )
            )
        return tuple(hosts)

    def prepare_chunk_inputs(
        self,
        chunk_state: Qwen38TTNNTextModelChunkState,
        token_ids: Sequence[int],
        *,
        ple_context,
        positions: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        start_position: int | None = None,
    ) -> Qwen38TTNNChunkHostInputs:
        """The host half of one chunk's input write (no device call): the ``rows`` token ids as the token-rows
        image (the zero sentinel at the image pads' lanes) and their n-gram rows, looked up from ``ple_context`` in
        one pass (the raw ids, pads included, as the reference hashes them), the RoPE rows of the rows' rotary
        positions ``positions`` (int64 ``[3, rows]``, ``mrope``; None: the plain positions of a text chunk from
        ``start_position``, or, when that is None too, from a readback of the device position ``P`` -- the eager
        tools' path; the prefill driver passes the positions) and the feature rows of the chunk's image pads
        (``features`` ``[n, 2560]`` BF16 in pad order; None when it has none), as host tensors shaped for the chunk
        state's persistent inputs."""

        self._validate_chunk_state(chunk_state)
        ple = self.layers[PLE_CHECKPOINT_LAYER].ple
        if ple is None:
            raise RuntimeError("checkpoint layer 1 PLE owner is unavailable")
        token_ids = [int(token) for token in token_ids]
        if len(token_ids) != chunk_state.rows:
            raise ValueError(
                f"the {chunk_state.rows}-row chunk takes {chunk_state.rows} token ids, got {len(token_ids)}"
            )
        lanes = vision_splice.image_lanes(token_ids)
        feature_image = vision_splice.feature_rows_image(token_ids, features, hidden=HIDDEN_SIZE)
        token_rows = ttnn.from_torch(
            vision_splice.sentinel_token_rows(self.model_io.embedding.host_token_rows(token_ids), lanes),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
        )
        rows, contexts = ple.host_rows(token_ids, ple_context)
        embedding_rows = ttnn.from_torch(
            rows.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
        )
        if positions is None:
            if start_position is None:
                if chunk_state.position is None:
                    raise ValueError("a chunk without rotary positions needs its start position (no device position)")
                start_position = chunk_state.position.read()  # the device's P: the chunk's first row
            if isinstance(start_position, bool) or type(start_position) is not int or start_position < 0:
                raise ValueError(f"start position must be a non-negative int, got {start_position!r}")
            plain = torch.arange(start_position, start_position + chunk_state.rows, dtype=torch.int64)
            positions = plain.reshape(1, -1).expand(3, -1)
        rope_rows = self.chunk_rope_host_rows(chunk_state, positions)
        feature_rows = None if feature_image is None else self.host_chunk_feature_rows(feature_image)
        return Qwen38TTNNChunkHostInputs(token_rows, embedding_rows, contexts, rope_rows, feature_rows, len(lanes))

    def upload_chunk_inputs(
        self, chunk_state: Qwen38TTNNTextModelChunkState, prepared: Qwen38TTNNChunkHostInputs
    ) -> None:
        """The device half: the copies into the chunk state's persistent inputs (the token rows, the PLE rows, the
        four RoPE rows, the feature rows when the chunk has image pads or the buffer still holds an earlier chunk's),
        queued on the command queue in order (behind a replay that reads the same buffers: the device finishes that
        replay before they land)."""

        self._validate_chunk_state(chunk_state)
        if len(prepared.contexts) != chunk_state.rows + 1:
            raise ValueError(
                f"prepared inputs of {len(prepared.contexts) - 1} rows vs the {chunk_state.rows}-row chunk state"
            )
        if len(prepared.rope_rows) != 4:
            raise ValueError("prepared chunk inputs carry no RoPE rows (prepare_chunk_inputs makes them)")
        ttnn.copy_host_to_device_tensor(prepared.token_rows, chunk_state.token_row)
        ttnn.copy_host_to_device_tensor(prepared.embedding_rows, chunk_state.ple_rows.embedding_rows)
        rope = chunk_state.rope_rows
        for host, target in zip(prepared.rope_rows, (rope.cos, rope.sin, rope.block_start_cos, rope.block_start_sin)):
            ttnn.copy_host_to_device_tensor(host, target)
        self.write_chunk_feature_rows(chunk_state.feature_rows, prepared.feature_rows)

    def write_chunk_feature_rows(self, features: Qwen38TTNNChunkFeatureRows, host_rows) -> None:
        """The copy into a chunk's persistent feature rows: ``host_rows`` (an image chunk's, ``host_chunk_feature_rows``)
        when given, else the clean image when the buffer still holds an earlier chunk's rows (a text chunk after a
        text chunk copies nothing)."""

        if host_rows is not None:
            ttnn.copy_host_to_device_tensor(host_rows, features.tensor)
            features.clean = False
        elif not features.clean:
            ttnn.copy_host_to_device_tensor(self._clean_feature_rows_host(features.rows), features.tensor)
            features.clean = True

    def write_chunk_inputs(
        self,
        chunk_state: Qwen38TTNNTextModelChunkState,
        token_ids: Sequence[int],
        *,
        ple_context,
        positions: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        start_position: int | None = None,
    ) -> tuple[tuple[int, int] | None, ...]:
        """Host writes of one chunk's inputs (outside any trace): :meth:`prepare_chunk_inputs` then
        :meth:`upload_chunk_inputs`.  Returns the rows + 1 contexts (``contexts[r]`` after committing r rows)."""

        prepared = self.prepare_chunk_inputs(
            chunk_state,
            token_ids,
            ple_context=ple_context,
            positions=positions,
            features=features,
            start_position=start_position,
        )
        self.upload_chunk_inputs(chunk_state, prepared)
        return prepared.contexts

    def forward_prefill_chunk_generic(
        self,
        chunk_state: Qwen38TTNNTextModelChunkState,
        state: Qwen38TTNNTextModelGenericState,
        *,
        gdn_step_anchor: bool = False,
        mtp=None,
    ) -> None:
        """One chunk: selectors (the 32-row form), RoPE rows and QSA chunk inputs from the device position, the
        rows-lane embedding, 48 layers in place, ``P += rows``.  No final mixer or LM head: the first decode
        replay after the prefill consumes the last prompt token.  ``gdn_step_anchor`` is every GDN layer's state
        re-anchor (the layer commits through ``commit_rows(step_committed_rows=True)``; a 32-row option).  ``mtp``
        (the MTP-drafting server's chunk extension of this chunk state's form, 32 or 128 rows; at slab rows the
        128-row twin allocated with the slab form, which runs the slab as 128-row slices) runs the MTP layer's rows
        on the layer-47 residual rows before they are released.  Any failure poisons this owner."""

        slab_mtp = mtp is not None and is_slab_rows(chunk_state.rows)
        if slab_mtp and not (mtp.rows == LONG_CHUNK_ROWS and mtp.slab_rows == chunk_state.rows):
            raise ValueError(
                f"a {chunk_state.rows}-row slab with MTP drafting runs the {LONG_CHUNK_ROWS}-row twin's slab form "
                f"(allocated with slab_rows={chunk_state.rows}), got a {mtp.rows}-row extension with slab_rows="
                f"{mtp.slab_rows}"
            )
        if mtp is not None and not slab_mtp and mtp.rows != chunk_state.rows:
            raise ValueError(
                f"the MTP chunk extension holds {mtp.rows} rows, the chunk state {chunk_state.rows}: the extension "
                "of the chunk's form runs in its body"
            )

        self._validate_generic_state(state)
        self._validate_chunk_state(chunk_state)
        rows = chunk_state.rows
        if gdn_step_anchor and rows != CHUNK_ROWS:
            raise ValueError("the GDN step anchor is a 32-row chunk option")
        processed_layers = 0
        try:
            selectors = (
                gdn_module.build_rows_selectors(chunk_state.accepted, chunk_state.rows_constants)
                if rows == CHUNK_ROWS
                else None
            )
            # The RoPE rows are the chunk state's host-written rows (prepare_chunk_inputs: the rows' rotary
            # positions, three-axis for image tokens, the table's rows for text); the cache, block and mask inputs
            # stay derived from the device position.
            rope = chunk_state.rope_rows
            qsa_chunk = qsa_module.derive_qsa_chunk_inputs(
                state.position.scalar, self.qsa_position_constants, chunk_state.qsa_chunk_constants
            )
            residual = self._embed_residual_rows_from_device_token(
                chunk_state.token_row, feature_rows=chunk_state.feature_rows.tensor
            )
            for layer_index in range(BACKBONE_LAYERS):
                layer = self.layers[layer_index]
                residual = layer.forward_chunk_generic(
                    residual,
                    state.layers[layer_index],
                    chunk_state.layers[layer_index],
                    prepared_ple_rows=chunk_state.ple_rows if layer_index == PLE_CHECKPOINT_LAYER else None,
                    rope_rows=rope,
                    qsa_chunk=qsa_chunk,
                    qsa_chunk_constants=chunk_state.qsa_chunk_constants,
                    selectors=selectors,
                    gdn_step_anchor=gdn_step_anchor,
                )
                processed_layers += 1
            if slab_mtp:
                # The 128-row twin over the slab's layer-47 residual, slice by slice (mtp_v2: the slab form), its
                # RoPE rows cut from the slab's host-written rows.
                mtp.forward_slab_rows(self, residual, rope_rows=rope, position_scalar=state.position.scalar)
            elif mtp is not None:
                mtp.forward_chunk_rows(
                    self,
                    residual,
                    rope_rows=rope,
                    qsa_chunk=qsa_chunk,
                    qsa_chunk_constants=chunk_state.qsa_chunk_constants,
                    selectors=selectors,
                )
            _deallocate_unique(residual)
            qsa_chunk.deallocate()  # the RoPE rows are the chunk state's persistent buffers: not released here
            if selectors is not None:
                selectors.deallocate()
            state.position.advance_by(rows)
        except BaseException as error:
            self._mark_poisoned("forward_prefill_chunk_generic", processed_layers, error)

    def capture_prefill_chunk(
        self,
        chunk_state: Qwen38TTNNTextModelChunkState,
        state: Qwen38TTNNTextModelGenericState,
        *,
        guard: Callable[[str], AbstractContextManager[Any]],
        cq_id: int = 0,
        gdn_step_anchor: bool = False,
        mtp=None,
    ) -> int:
        """Capture the chunk body once (after the decode captures, on the same generic state); returns the trace id.

        Same discipline as :meth:`capture_decode_generic`: ``corruptible_allocation_scope`` and the
        caller's no-host-I/O ``guard``.  Capture records without executing, so the device state is unchanged.
        ``gdn_step_anchor`` is baked into the trace: every replay commits the GDN state through the re-anchor.
        """

        with corruptible_allocation_scope(self.mesh_device):
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=cq_id)
            with guard("prefill chunk capture"):
                self.forward_prefill_chunk_generic(chunk_state, state, gdn_step_anchor=gdn_step_anchor, mtp=mtp)
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=cq_id)
        return trace_id

    def finish_prefill(
        self,
        state: Qwen38TTNNTextModelGenericState,
        chunk_state: Qwen38TTNNTextModelChunkState,
        prefilled: int,
        *,
        rope_shift: int = 0,
    ) -> None:
        """Eager hand-off from the chunk body to the 1-row generic body after a prefill of ``prefilled`` positions
        (0 .. prefilled - 1; the last chunk padded past prefilled % 32 with the accept scalar prefilled % 32 - 1).

        ``P`` becomes ``prefilled`` (the next replay's trace key: residue prefilled % 4, regime 0) and the rotary
        shift ``S`` becomes ``rope_shift`` (the prefilled sequence's ``mrope`` shift: 0 for text); the GDN ring
        slots and phase come from the committed history, the PLE slots from the rows history, the QSA staging
        tile and raw-key ring from the kept slab and raw keys; the committed GDN state and the QSA caches are
        already the decode's.  The accept scalar returns to a full chunk.  The caller keeps the n-gram context
        (the PLE row of position ``prefilled`` is its ordinary pre-replay write).  Any failure poisons this owner.
        """

        self._validate_generic_state(state)
        self._validate_chunk_state(chunk_state)
        if chunk_state.rows != CHUNK_ROWS:
            raise ValueError(f"the hand-off reads the {CHUNK_ROWS}-row chunk state, got {chunk_state.rows} rows")
        if isinstance(prefilled, bool) or type(prefilled) is not int or not 0 <= prefilled <= self.allocated_context:
            raise ValueError(
                f"prefilled position count must be an int in [0,{self.allocated_context}], got {prefilled!r}"
            )
        require_rope_shift(rope_shift, prefilled)
        try:
            ring_select = ttnn.from_torch(
                qsa_module.chunk_handoff_ring_select_rows(prefilled),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
            )
            try:
                state.position.reset(prefilled, rope_shift)
                for layer, layer_state, layer_chunk in zip(self.layers, state.layers, chunk_state.layers):
                    layer.finish_chunk_state_inplace(
                        layer_chunk, layer_state, prefilled=prefilled, qsa_ring_select=ring_select
                    )
            finally:
                _deallocate_unique(ring_select)
            self.write_chunk_accepted(chunk_state, CHUNK_ROWS - 1)
        except BaseException as error:
            self._mark_poisoned("finish_prefill", 0, error)

    # ------------------------------------------------------------------ lanes (batched decode, B = 1..32)
    # forward_decode_lanes is forward_decode_generic over B lanes: the per-lane RoPE rows and QSA lane inputs from
    # the position row, the token row's B lanes embedded, 48 layers on [1,4,B,640] rows in place, the final mixer
    # and LM head at B rows, the position row advanced as the last op.  One trace per GDN ring residue serves the
    # step; every resident lane's position is congruent to the ring phase (the residue rule), so a lane is admitted
    # at a step of its position's residue class.  The 1-row bodies above are untouched.

    def _validate_lane_state(self, state: Qwen38TTNNTextModelLaneState) -> int:
        self._require_healthy()
        if not isinstance(state, Qwen38TTNNTextModelLaneState) or state._owner is not self._state_owner:
            raise ValueError("text-model lane state was not allocated by this model owner")
        lanes = require_lane_count(state.lanes, label="text-model lanes")
        if not isinstance(state.position, Qwen38TTNNDevicePositionRow) or state.position.lanes != lanes:
            raise TypeError(f"text-model lane state requires a {lanes}-lane Qwen38TTNNDevicePositionRow")
        if len(state.layers) != BACKBONE_LAYERS:
            raise ValueError(f"text-model lane state must contain {BACKBONE_LAYERS} target layers")
        for layer_index, (layer, layer_state) in enumerate(zip(self.layers, state.layers)):
            if layer_state.namespace is not Qwen38TTNNLayerNamespace.BACKBONE or layer_state.layer_index != layer_index:
                raise ValueError(f"lane target state {layer_index} identity does not match its layer")
            if layer_state.lanes != lanes:
                raise ValueError(f"lane target state {layer_index} holds {layer_state.lanes} lanes, expected {lanes}")
        if (state.embedding_rows is None) != (lanes == CHUNK_ROWS):
            raise ValueError("the lane state carries an embedding rows buffer exactly below 32 lanes")
        if (
            state.qsa_lane_constants.lanes != lanes
            or state.qsa_lane_constants.allocated_context != self.allocated_context
        ):
            raise ValueError(
                f"QSA lane constants were built for {state.qsa_lane_constants.lanes} lanes at "
                f"{state.qsa_lane_constants.allocated_context}, the state has {lanes} at {self.allocated_context}"
            )
        if not state.ple_rows.active or tuple(state.ple_rows.source_contexts) != (None,) * lanes:
            raise ValueError("the lane PLE rows must be live and prepared without host n-gram contexts")
        return lanes

    def allocate_lane_state(self, lanes: int, *, indexer_form: str = "wide") -> Qwen38TTNNTextModelLaneState:
        """Allocate the position row, every layer's lane state, the lane constants and the host-written inputs of
        B lanes (before any capture).  The first call also builds the model-lifetime generic constants."""

        self._require_healthy()
        lanes = require_lane_count(lanes, label="text-model lanes")
        if self.rope_table is None:
            self.rope_table = Qwen38TTNNRoPETable.build(
                self.mesh_device, self.mesh_contract, self.config, self.allocated_context
            )
        qsa = next(layer.attention for layer in self.layers if layer.layer_type is Qwen38TTNNLayerType.QSA)
        if self.qsa_position_constants is None:
            self.qsa_position_constants = qsa_module.Qwen38TTNNQSAPositionConstants.build(
                self.mesh_device, self.mesh_contract, qsa.allocated_compressed_blocks
            )
        position = Qwen38TTNNDevicePositionRow.allocate(self.mesh_device, self.mesh_contract, [0] * lanes, lanes=lanes)
        allocated: list[Qwen38TTNNDecoderLayerLaneState] = []
        chunk_constants = None
        lane_constants = None
        token_row = None
        embedding_rows = None
        ple_rows = None
        try:
            chunk_constants = qsa_module.Qwen38TTNNQSAChunkConstants.build(
                self.mesh_device, self.mesh_contract, qsa.allocated_compressed_blocks
            )
            lane_constants = qsa_module.Qwen38TTNNQSALaneConstants.build(
                self.mesh_device,
                self.mesh_contract,
                lanes=lanes,
                allocated_context=self.allocated_context,
                indexer_form=indexer_form,
            )
            for layer in self.layers:
                allocated.append(layer.allocate_lane_state(lanes))
            token_row = self.model_io.embedding.upload_token_row(0)
            if lanes < CHUNK_ROWS:
                embedding_rows = ttnn.from_torch(
                    torch.zeros((1, 1, lanes, HIDDEN_SIZE), dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
                )
            ple_layer, ple_lanes_state = self.layers[PLE_CHECKPOINT_LAYER], allocated[PLE_CHECKPOINT_LAYER].ple
            if ple_layer.ple is None or ple_lanes_state is None:
                raise RuntimeError("checkpoint layer 1 PLE owner/lanes state is unavailable")
            ple_rows = ple_layer.ple.prepare_lanes_input([0] * lanes, ple_lanes_state)
            ple_lanes_state.token_contexts = (None,) * lanes
            result = Qwen38TTNNTextModelLaneState(
                lanes=lanes,
                position=position,
                layers=tuple(allocated),
                token_row=token_row,
                ple_rows=ple_rows,
                embedding_rows=embedding_rows,
                qsa_chunk_constants=chunk_constants,
                qsa_lane_constants=lane_constants,
                _owner=self._state_owner,
            )
            self._validate_lane_state(result)
            return result
        except BaseException as error:
            actions = [
                (
                    f"target layer {index} lane state",
                    lambda layer=layer, layer_state=layer_state: layer.release_lane_state(layer_state),
                )
                for index, (layer, layer_state) in reversed(tuple(enumerate(zip(self.layers, allocated))))
            ]
            if ple_rows is not None:
                actions.append(("lane PLE rows", ple_rows.release))
            for label, tensor in (("lane embedding rows", embedding_rows), ("lane token row", token_row)):
                if tensor is not None:
                    actions.append((label, lambda tensor=tensor: ttnn.deallocate(tensor)))
            for label, constants in (("QSA lane constants", lane_constants), ("QSA chunk constants", chunk_constants)):
                if constants is not None:
                    actions.append((label, constants.deallocate))
            actions.append(("device position row", position.deallocate))
            cause = _cleanup_failure_cause("lane text-model state allocation", actions, error)
            if cause is not error:
                self._mark_poisoned("allocate_lane_state cleanup", 0, cause)
            raise

    def release_lane_state(self, state: Qwen38TTNNTextModelLaneState) -> None:
        self._validate_lane_state(state)
        actions = [
            (
                f"target layer {index} lane state",
                lambda layer=layer, layer_state=layer_state: layer.release_lane_state(layer_state),
            )
            for index, (layer, layer_state) in reversed(tuple(enumerate(zip(self.layers, state.layers))))
        ]
        actions.append(("lane PLE rows", state.ple_rows.release))
        if state.embedding_rows is not None:
            actions.append(("lane embedding rows", lambda: ttnn.deallocate(state.embedding_rows)))
        actions.append(("lane token row", lambda: ttnn.deallocate(state.token_row)))
        actions.append(("QSA lane constants", state.qsa_lane_constants.deallocate))
        actions.append(("QSA chunk constants", state.qsa_chunk_constants.deallocate))
        actions.append(("device position row", state.position.deallocate))
        try:
            _run_cleanup_actions("lane text-model state", actions)
        except BaseException as error:
            self._mark_poisoned("release_lane_state", 0, error)

    def reset_lane_state_inplace(self, state: Qwen38TTNNTextModelLaneState, positions, shifts=None) -> None:
        """A new batch: every lane's buffers back to their position-zero contents at every captured address, the
        position row rewritten to ``positions`` (one residue class) and the rotary shift row to ``shifts`` (None: all
        0), the token row zeroed."""

        self._validate_lane_state(state)
        try:
            state.position.reset(positions, shifts)
            for layer, layer_state in zip(self.layers, state.layers):
                layer.reset_lane_state_inplace(layer_state)
            self.write_lane_tokens(state, [0] * state.lanes)
        except BaseException as error:
            self._mark_poisoned("reset_lane_state_inplace", 0, error)

    def admit_lane(self, state: Qwen38TTNNTextModelLaneState, lane: int, position: int, rope_shift: int = 0) -> None:
        """Admit one lane at ``position`` with the rotary shift ``rope_shift`` (0 for text): refused (nothing written)
        unless ``position mod 4`` is the resident residue at this step (``admission_wait_steps`` names the wait);
        then the lane's buffers in every layer are zeroed with keep-mask writes (the other lanes hold their state)
        and the position and shift rows take the lane."""

        self._validate_lane_state(state)
        state.position.admit(lane, position, rope_shift)  # the residue rule: a ValueError before any device write
        try:
            for layer, layer_state in zip(self.layers, state.layers):
                layer.reset_lane_inplace(layer_state, lane)
        except BaseException as error:
            self._mark_poisoned("admit_lane", 0, error)

    def lane_layout(self, state: Qwen38TTNNTextModelLaneState) -> Qwen38LaneLayout:
        """The lane state's per-lane device tensors by family for the lane pager (``lanes.Qwen38TTNNLanePager``:
        eviction to the host pool and re-admission): the 12 QSA caches, staging tiles and rings, the 36 GDN
        recurrent states and 144 ring slots, the 9 PLE slots.  The position row and the host-written inputs stay
        with the model (re-admission goes through ``state.position.admit`` first)."""

        lanes = self._validate_lane_state(state)
        gdn_states = [s.attention for s in state.layers if isinstance(s.attention, gdn_module.Qwen38TTNNGDNState)]
        qsa_states = [s.attention for s in state.layers if isinstance(s.attention, qsa_module.Qwen38TTNNQSALaneState)]
        ple_state = state.layers[PLE_CHECKPOINT_LAYER].ple
        if ple_state is None:
            raise RuntimeError("checkpoint layer 1 PLE lanes state is unavailable")
        layout = Qwen38LaneLayout(
            lanes=lanes,
            allocated_context=self.allocated_context,
            kv=tuple(s.packed_kv_cache for s in qsa_states),
            compressed=tuple(s.compressed_index_cache for s in qsa_states),
            staging=tuple(s.kv_staging for s in qsa_states),
            ring=tuple(s.raw_key_ring for s in qsa_states),
            recurrent=tuple(s.recurrent for s in gdn_states),
            conv=tuple(slot for s in gdn_states for slot in s.conv),
            ple=tuple(ple_state.conv),
        )
        layout.validate(self.mesh_contract)
        return layout

    def generic_lane_layout(self, state: Qwen38TTNNTextModelGenericState) -> Qwen38LaneLayout:
        """The generic (single-lane) state's tensors as a 1-lane layout for the lane pager: the state the chunk
        trace prefills is packed into a host slot and re-admitted into a lane (the lane server's prefill import).
        The device position scalar and the GDN ring phases stay with the model (the importer resets them)."""

        self._validate_generic_state(state)
        gdn_states = [s.attention for s in state.layers if isinstance(s.attention, gdn_module.Qwen38TTNNGDNState)]
        qsa_states = [
            s.attention for s in state.layers if isinstance(s.attention, qsa_module.Qwen38TTNNQSAGenericState)
        ]
        ple_state = state.layers[PLE_CHECKPOINT_LAYER].ple
        if ple_state is None:
            raise RuntimeError("checkpoint layer 1 PLE state is unavailable")
        layout = Qwen38LaneLayout(
            lanes=1,
            allocated_context=self.allocated_context,
            kv=tuple(s.packed_kv_cache for s in qsa_states),
            compressed=tuple(s.compressed_index_cache for s in qsa_states),
            staging=tuple(s.kv_staging for s in qsa_states),
            ring=tuple(s.raw_key_ring for s in qsa_states),
            recurrent=tuple(s.recurrent for s in gdn_states),
            conv=tuple(slot for s in gdn_states for slot in s.conv),
            ple=tuple(ple_state.conv),
        )
        layout.validate(self.mesh_contract)
        return layout

    def write_lane_tokens(self, state: Qwen38TTNNTextModelLaneState, tokens: Sequence[int]) -> None:
        """Host write of the token row (outside any trace): lane u = ``tokens[u]``, the idle lanes 0."""

        lanes = self._validate_lane_state(state)
        tokens = [int(token) for token in tokens]
        if len(tokens) != lanes:
            raise ValueError(f"lane tokens need one token per lane: got {len(tokens)}, expected {lanes}")
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                self.model_io.embedding.host_token_rows(tokens + [0] * (CHUNK_ROWS - lanes)),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
            ),
            state.token_row,
        )

    def write_lane_ple_rows(
        self, state: Qwen38TTNNTextModelLaneState, tokens: Sequence[int], contexts
    ) -> tuple[tuple[int, int], ...]:
        """Host write of the PLE rows (outside any trace): lane u's ``tokens[u]`` looked up under ``contexts[u]``
        in one batched read into the persistent rows.  Returns the next context per lane; the caller keeps them
        (the body's PLE state is pinned context-free)."""

        lanes = self._validate_lane_state(state)
        ple = self.layers[PLE_CHECKPOINT_LAYER].ple
        if ple is None:
            raise RuntimeError("checkpoint layer 1 PLE owner is unavailable")
        tokens = [int(token) for token in tokens]
        if len(tokens) != lanes:
            raise ValueError(f"lane PLE rows need one token per lane: got {len(tokens)}, expected {lanes}")
        rows, next_contexts = ple.host_lanes(tokens, contexts)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                rows.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
            ),
            state.ple_rows.embedding_rows,
        )
        return next_contexts

    def _embed_residual_lanes_from_device_token(self, state: Qwen38TTNNTextModelLaneState):
        """Trace-capturable branch-major ``[1,4,B,640]`` residual rows of the token row's B lanes: the 32-lane
        embedding, its lane rows sliced into the persistent embedding rows below 32 lanes."""

        owned: list[Any | None] = [self.model_io.embedding.embed_device_token_rows(state.token_row), None]
        try:
            hidden = owned[0]
            if state.embedding_rows is not None:
                landed = ttnn.slice(
                    hidden, (0, 0, 0, 0), (1, 1, state.lanes, LOCAL_HIDDEN_SIZE), output_tensor=state.embedding_rows
                )
                if landed is not None and _tensor_key(landed) != _tensor_key(state.embedding_rows):
                    raise RuntimeError("lane embedding slice did not land in the persistent embedding rows")
                hidden = state.embedding_rows
            expected = (1, 1, state.lanes, LOCAL_HIDDEN_SIZE)
            if _shape(hidden) != expected or hidden.dtype != ttnn.bfloat16 or hidden.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError(
                    f"lane embedding rows must be BF16 TILE {list(expected)}, got {tensor_metadata(hidden)}"
                )
            self.mesh_contract.validate_tensor(hidden, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
            owned[1] = ttnn.repeat_interleave(
                hidden, repeats=RESIDUAL_BRANCHES, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            residual = owned[1]
            expected = (1, RESIDUAL_BRANCHES, state.lanes, LOCAL_HIDDEN_SIZE)
            if _shape(residual) != expected or residual.dtype != ttnn.bfloat16 or residual.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError(
                    f"lane residual rows must be BF16 TILE {list(expected)}, got {tensor_metadata(residual)}"
                )
            self.mesh_contract.validate_tensor(residual, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        except BaseException as error:
            self._poisoned_device_owners.extend(tensor for tensor in owned if tensor is not None)
            self._mark_poisoned("lane residual rows", 0, error)
        _release_tensor_slot_indices(owned, (0,), label="lane embedding rows transient")
        return owned[1]

    def forward_decode_lanes(
        self, state: Qwen38TTNNTextModelLaneState, *, return_logits: bool = True
    ) -> Qwen38TTNNGenericDecodeOutput:
        """One batched step: fixed op sequence for every position row, no host ints, no host I/O.

        Per-lane RoPE rows and QSA lane inputs from the position row, the token row's lanes embedded, the 48 layers
        on ``[1,4,B,640]`` rows in place (the PLE layer consumes the persistent PLE rows), the final mixer and LM
        head at B rows (row u = lane u), then the position row advanced as the last op.  Any failure poisons this
        owner: the in-place state cannot be rolled back.
        """

        lanes = self._validate_lane_state(state)
        processed_layers = 0
        try:
            if self._position_derive_lanes is not None:
                rope, qsa_lanes = self._position_derive_lanes(self, state)
            else:
                index_row = state.position.index_row()
                rope_index_row = state.position.rope_index_row(index_row)
                rope_block_start_row = state.position.rope_block_start_index_row(index_row)
                rope = self.rope_table.rows_chunk(rope_index_row, rope_block_start_row)
                _deallocate_unique(index_row, rope_index_row, rope_block_start_row)
                qsa_lanes = qsa_module.derive_qsa_lane_inputs(
                    state.position.row, self.qsa_position_constants, state.qsa_chunk_constants, state.qsa_lane_constants
                )
            residual = self._embed_residual_lanes_from_device_token(state)
            for layer_index in range(BACKBONE_LAYERS):
                residual = self.layers[layer_index].forward_decode_lanes(
                    residual,
                    state.layers[layer_index],
                    prepared_ple=state.ple_rows if layer_index == PLE_CHECKPOINT_LAYER else None,
                    rope=rope,
                    qsa_lanes=qsa_lanes,
                    qsa_constants=state.qsa_chunk_constants,
                    qsa_lane_constants=state.qsa_lane_constants,
                )
                processed_layers += 1
            qsa_lanes.deallocate()
            rope.deallocate()
            hidden = self.final_mixer.rows(residual, flat_views=True)
            _deallocate_unique(residual)
            expected = (1, 1, lanes, LOCAL_HIDDEN_SIZE)
            if _shape(hidden) != expected or hidden.dtype != ttnn.bfloat16:
                raise RuntimeError(
                    f"lane terminal mixer output must be BF16 {list(expected)}, got {tensor_metadata(hidden)}"
                )
            logits = self.model_io.lm_head(hidden) if return_logits else None
            _deallocate_unique(hidden)
            state.position.advance()
            return Qwen38TTNNGenericDecodeOutput(logits)
        except BaseException as error:
            self._mark_poisoned("forward_decode_lanes", processed_layers, error)

    def resolve_lane_tokens(self, output: Qwen38TTNNGenericDecodeOutput, state: Qwen38TTNNTextModelLaneState):
        """The lane epilogue (trace-capturable): the per-row greedy candidates, the device resolve to the
        ``[1,1,1,32]`` token row (lane u = row u's id) and its copy into the state's token row for the next step.
        Returns ``(candidates, token_row)``; the caller owns them (a capture retains them for its traces' life)."""

        self._validate_lane_state(state)
        if output.logits is None:
            raise ValueError("the lane epilogue needs the step's logits")
        lm_head = self.model_io.lm_head
        candidates = lm_head.greedy_candidates_lanes(output.logits)
        token_row = lm_head.resolve_greedy_lanes_on_device(candidates)
        key = _tensor_key(state.token_row)
        copied = ttnn.copy(token_row, state.token_row)
        if _tensor_key(state.token_row) != key or (copied is not None and _tensor_key(copied) != key):
            raise RuntimeError("lane token row copy did not write the resident token row in place")
        return candidates, token_row

    def capture_decode_lanes(
        self,
        state: Qwen38TTNNTextModelLaneState,
        *,
        residue: int,
        guard: Callable[[str], AbstractContextManager[Any]],
        epilogue: Callable[[Qwen38TTNNGenericDecodeOutput], Any],
        phase_observer: Callable[[str], None] | None = None,
        cq_id: int = 0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> Qwen38TTNNGenericDecodeCapture:
        """Capture one residue class of the lane body as one single-body trace (:meth:`capture_decode_generic`'s
        discipline: ``corruptible_allocation_scope``, the caller's ``guard`` and ``epilogue``).  The position
        row's residue and every GDN lane state's ring phase must be ``residue`` (the residue rule made concrete);
        capture records without executing, so the host bookkeeping advances as if the body ran while the device
        state is unchanged.  A body failure poisons this owner and leaves the capture open; the process must exit."""

        if phase_observer is not None and not callable(phase_observer):
            raise TypeError("capture phase observer must be callable")
        self._validate_lane_state(state)
        if not isinstance(residue, int) or not 0 <= residue < gdn_module.CONV_KERNEL_SIZE:
            raise ValueError(f"lane trace residue must be an int in [0,{gdn_module.CONV_KERNEL_SIZE}), got {residue!r}")
        phases = {
            index: layer_state.attention.conv_phase
            for index, layer_state in enumerate(state.layers)
            if isinstance(layer_state.attention, gdn_module.Qwen38TTNNGDNState)
        }
        if state.position.residue != residue or set(phases.values()) != {residue}:
            raise RuntimeError(
                f"lane capture residue {residue}: the position row's residue is {state.position.residue} and the GDN "
                f"ring phases are {sorted(set(phases.values()))}; all must equal the residue"
            )
        if phase_observer is not None:
            phase_observer("before-body-capture")
        with corruptible_allocation_scope(self.mesh_device):
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=cq_id)
            started_ns = clock_ns()
            with guard(f"lanes body capture residue {residue}") as blocked:
                output = self.forward_decode_lanes(state)
                epilogue_result = epilogue(output)
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=cq_id)
            capture_ns = clock_ns() - started_ns
        if phase_observer is not None:
            phase_observer("after-body-capture")
        return Qwen38TTNNGenericDecodeCapture(
            residue=residue,
            regime=0,
            parts=GENERIC_TRACE_PARTS_SINGLE,
            trace_ids={Qwen38TTNNGenericTraceKey("body", residue, 0): trace_id},
            head=None,
            output=output,
            epilogue=epilogue_result,
            capture_ns={"body": capture_ns},
            guard_attempts=tuple(blocked or ()),
        )


def validate_model_static_contract() -> None:
    """No-device guard for model geometry, order, and RoPE block routing."""

    if (
        TP_SIZE,
        BACKBONE_LAYERS,
        HIDDEN_SIZE,
        LOCAL_HIDDEN_SIZE,
        RESIDUAL_BRANCHES,
        VOCAB_SIZE,
        LOCAL_VOCAB_SIZE,
        QSA_ROPE_DIM,
        QSA_COMPRESS_RATIO,
        ROPE_THETA,
        MAX_CONTEXT,
    ) != (4, 48, 2560, 640, 4, 248320, 62080, 64, 4, 10_000_000, 262144):
        raise RuntimeError("Qwen3.8 TTNN text-model geometry drifted")
    if RESIDUAL_LOCAL_SHAPE != (1, 4, 1, 640) or BLOCK_LOCAL_SHAPE != (1, 1, 1, 640):
        raise RuntimeError("Qwen3.8 TTNN text-model local shapes drifted")
    if EXPECTED_LAYER_PATTERN != tuple(LAYER_PATTERN) or len(EXPECTED_LAYER_PATTERN) != BACKBONE_LAYERS:
        raise RuntimeError("Qwen3.8 TTNN text-model layer order drifted")
    if tuple(index for index, kind in enumerate(EXPECTED_LAYER_PATTERN) if kind == "full_attention") != tuple(
        range(3, BACKBONE_LAYERS, QSA_COMPRESS_RATIO)
    ):
        raise RuntimeError("Qwen3.8 TTNN QSA layer locations drifted")
    if EXPECTED_VOCAB_RANGES != ((0, 62080), (62080, 124160), (124160, 186240), (186240, 248320)):
        raise RuntimeError("Qwen3.8 TTNN vocabulary placement drifted")
    # The split: HEAD is GDN-only (reads no position), the first TAIL layer is the PLE consumer.
    if GENERIC_HEAD_LAYERS != 1 or GENERIC_HEAD_LAYERS != PLE_CHECKPOINT_LAYER:
        raise RuntimeError("Qwen3.8 TTNN HEAD/TAIL split point drifted from the PLE checkpoint layer")
    if set(EXPECTED_LAYER_PATTERN[:GENERIC_HEAD_LAYERS]) != {"linear_attention"}:
        raise RuntimeError("Qwen3.8 TTNN HEAD layers must all be GDN layers")
    if [(handoff.name, handoff.shape, handoff.padded_shape) for handoff in GENERIC_HEAD_HANDOFF] != [
        ("layer_0_output_residual", (1, 4, 1, 640), (1, 4, 32, 640))
    ]:
        raise RuntimeError("Qwen3.8 TTNN HEAD/TAIL handoff drifted")

    inverse_frequency = 1.0 / (
        float(ROPE_THETA) ** (torch.arange(0, QSA_ROPE_DIM, 2, dtype=torch.float32) / QSA_ROPE_DIM)
    )
    cos, sin = _host_rope(0, inverse_frequency)
    if tuple(cos.shape) != (1, 1, 1, 64) or cos.dtype != torch.bfloat16 or not bool(torch.all(cos == 1)):
        raise RuntimeError("Qwen3.8 TTNN position-zero RoPE cosine drifted")
    if tuple(sin.shape) != (1, 1, 1, 64) or sin.dtype != torch.bfloat16 or not bool(torch.all(sin == 0)):
        raise RuntimeError("Qwen3.8 TTNN position-zero RoPE sine drifted")
    closing_positions = tuple(position for position in range(16) if position % QSA_COMPRESS_RATIO == 3)
    if closing_positions != (3, 7, 11, 15):
        raise RuntimeError("Qwen3.8 TTNN QSA block-start routing drifted")


__all__ = [
    "GENERIC_HEAD_HANDOFF",
    "GENERIC_HEAD_LAYERS",
    "GENERIC_TRACE_PARTS_SINGLE",
    "GENERIC_TRACE_PARTS_SPLIT",
    "Qwen38TTNNCleanupError",
    "Qwen38TTNNGenericDecodeCapture",
    "Qwen38TTNNGenericDecodeOutput",
    "Qwen38TTNNGenericHandoff",
    "Qwen38TTNNGenericHead",
    "Qwen38TTNNGenericTraceKey",
    "Qwen38TTNNGreedyStep",
    "Qwen38TTNNModelPoisonedError",
    "Qwen38TTNNPreparedDecodeInputs",
    "Qwen38TTNNRoPE",
    "Qwen38TTNNRoPEInputs",
    "Qwen38TTNNRoPETable",
    "Qwen38TTNNTextModel",
    "Qwen38TTNNTextModelGenericSnapshot",
    "Qwen38TTNNTextModelGenericState",
    "Qwen38TTNNTextModelLaneState",
    "Qwen38TTNNTextModelOutput",
    "Qwen38TTNNTextModelSnapshot",
    "Qwen38TTNNTextModelState",
    "validate_model_static_contract",
]
