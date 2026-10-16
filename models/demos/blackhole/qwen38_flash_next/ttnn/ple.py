# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact host-lookup/device-compute PLE for Qwen3.8-Flash-Next.

The 51.2B-parameter n-gram table remains on the host.  Only the sixteen
selected 160-wide rows are read for one token, concatenated to one BF16 vector,
and uploaded as four disjoint 640-wide ROW_MAJOR shards (1,280 B per device);
the captured device body tilizes the row as its first PLE op.  All projection,
gating, dilated-convolution, and residual arithmetic then runs on the exact
1x4 mesh.

This is the synchronous correctness path.  It intentionally establishes value
and ordering semantics before asynchronous host prefetch is introduced.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    INDEX_SHA256,
    Qwen38Checkpoint,
    Qwen38CheckpointRowReader,
)
from models.demos.blackhole.qwen38_flash_next.reference import ngram_token_ids
from models.demos.blackhole.qwen38_flash_next.tt.ple import Qwen38HostPLEEmbedding, Qwen38PLEWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    LONG_CHUNK_ROWS,
    MESH_SHAPE,
    Qwen38MeshContract,
    TensorPlacement,
    is_slab_rows,
    replicate_tensor_2d_mesh_mapper,
    require_lane_count,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import Qwen38TTNNRowsSelectors, Qwen38TTNNRowsSelectorsLanes

TP_AXIS = 1
TP_SIZE = 4
HIDDEN_SIZE = 2560
LOCAL_HIDDEN_SIZE = HIDDEN_SIZE // TP_SIZE
RESIDUAL_BRANCHES = 4
RESIDUAL_WIDTH = RESIDUAL_BRANCHES * HIDDEN_SIZE
EMBEDDING_WIDTH = 2560
NGRAM_SIZE = 3
CONV_KERNEL_SIZE = 4
CONV_DILATION = NGRAM_SIZE
CONV_STATE_LENGTH = (CONV_KERNEL_SIZE - 1) * CONV_DILATION
RMS_NORM_EPS = 1.0e-6
RESIDUAL_LOCAL_SHAPE = (1, 1, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE)


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _tensor_key(tensor) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


def _deallocate(*tensors) -> None:
    seen: set[tuple[str, int]] = set()
    for tensor in tensors:
        if tensor is None:
            continue
        key = _tensor_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        ttnn.deallocate(tensor)


def _copy_inplace(source, target, *, label: str) -> None:
    target_key = _tensor_key(target)
    copied = ttnn.copy(source, target)
    if _tensor_key(target) != target_key:
        raise RuntimeError(f"{label} changed the persistent target address")
    if copied is not None and _tensor_key(copied) != target_key:
        raise RuntimeError(f"{label} returned a different tensor")


def _require_hex_revision(value: str, *, label: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase 40-hex revision, got {value!r}")


def _cache_directory(
    root: str | Path,
    checkpoint: Qwen38Checkpoint,
    mesh_contract: Qwen38MeshContract,
    tt_metal_sha: str,
) -> Path:
    _require_hex_revision(tt_metal_sha, label="tt_metal_sha")
    physical = "-".join(str(value) for value in mesh_contract.physical_ids)
    path = (
        Path(root).resolve()
        / "ple"
        / f"index-{INDEX_SHA256}"
        / f"config-{checkpoint.config.config_sha256}"
        / f"tt-metal-{tt_metal_sha}"
        / f"mesh-1x4-physical-{physical}"
        / "layer-01"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_source(weights: Qwen38PLEWeights, checkpoint: Qwen38Checkpoint) -> None:
    config = checkpoint.config
    exact = {
        "layer_idx": (weights.layer_idx, 1),
        "hidden_size": (weights.hidden_size, HIDDEN_SIZE),
        "residual_branches": (weights.residual_branches, RESIDUAL_BRANCHES),
        "embedding_width": (weights.embedding_width, EMBEDDING_WIDTH),
        "conv_kernel": (weights.conv_kernel, CONV_KERNEL_SIZE),
        "conv_dilation": (weights.conv_dilation, CONV_DILATION),
        "rms_norm_eps": (weights.rms_norm_eps, RMS_NORM_EPS),
        "config_ple_layer": (config.ple_checkpoint_layer, 1),
        "config_ngram_size": (config.ngram_size, NGRAM_SIZE),
        "config_heads_per_ngram": (config.heads_per_ngram, 8),
        "config_ngram_shards": (config.ngram_shards, 128),
    }
    for name, (actual, expected) in exact.items():
        if actual != expected:
            raise ValueError(f"pinned PLE {name} must be {expected!r}, got {actual!r}")


def _prepare_key_weight(source: torch.Tensor) -> torch.Tensor:
    """Reorder N so each mesh column receives all four local branch slices."""

    branches = source.reshape(RESIDUAL_BRANCHES, HIDDEN_SIZE, EMBEDDING_WIDTH)
    local_weights = []
    for device_index in range(TP_SIZE):
        start = device_index * LOCAL_HIDDEN_SIZE
        end = start + LOCAL_HIDDEN_SIZE
        local = branches[:, start:end, :].reshape(RESIDUAL_BRANCHES * LOCAL_HIDDEN_SIZE, EMBEDDING_WIDTH)
        local_weights.append(local.transpose(0, 1).contiguous())
    result = torch.cat(local_weights, dim=1).reshape(1, 1, EMBEDDING_WIDTH, RESIDUAL_WIDTH)
    if tuple(result.shape) != (1, 1, EMBEDDING_WIDTH, RESIDUAL_WIDTH):
        raise AssertionError("PLE key packing changed shape")
    return result


@dataclass(frozen=True)
class Qwen38TTNNPLEWeights:
    """Resident BF16 weights for the one exact PLE layer."""

    key: Any
    value: Any
    norm_key: Any
    norm_query: Any
    norm_conv: Any
    conv_taps: tuple[Any, Any, Any, Any]
    replicated_anchor: Any
    layer_index: int

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        cache_root: str | Path,
        *,
        tt_metal_sha: str,
    ) -> "Qwen38TTNNPLEWeights":
        mesh_contract.validate_mesh(mesh_device)
        source = Qwen38PLEWeights.from_checkpoint(checkpoint)
        _validate_source(source, checkpoint)
        cache = _cache_directory(cache_root, checkpoint, mesh_contract, tt_metal_sha)
        output_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3))

        def upload(value: torch.Tensor, name: str, mapper=output_mapper, *, dtype=ttnn.bfloat16):
            host_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
            return ttnn.as_tensor(
                value.to(host_dtype).contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
                cache_file_name=cache / name,
            )

        key = upload(_prepare_key_weight(source.key), "key.bf16")
        value = upload(
            source.value.transpose(0, 1).reshape(1, 1, EMBEDDING_WIDTH, HIDDEN_SIZE),
            "value.bf16",
        )
        # Qwen4ExpTextRMSNorm is zero-centred: the checkpoint stores the
        # additive offset while TTNN rms_norm consumes the effective scale.
        norm_key = upload(
            (1.0 + source.norm_key.float()).reshape(1, 1, RESIDUAL_BRANCHES, HIDDEN_SIZE),
            "norm-key.fp32",
            dtype=ttnn.float32,
        )
        norm_query = upload(
            (1.0 + source.norm_query.float()).reshape(1, 1, RESIDUAL_BRANCHES, HIDDEN_SIZE),
            "norm-query.fp32",
            dtype=ttnn.float32,
        )
        norm_conv = upload(
            (1.0 + source.norm_conv.float()).reshape(1, 1, RESIDUAL_BRANCHES, HIDDEN_SIZE),
            "norm-conv.fp32",
            dtype=ttnn.float32,
        )
        conv = source.conv.reshape(RESIDUAL_BRANCHES, HIDDEN_SIZE, CONV_KERNEL_SIZE)
        conv_taps = tuple(
            upload(conv[:, :, tap].reshape(1, 1, RESIDUAL_BRANCHES, HIDDEN_SIZE), f"conv-tap-{tap}.bf16")
            for tap in range(CONV_KERNEL_SIZE)
        )
        replicated_anchor = ttnn.from_torch(
            torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
        )
        result = cls(key, value, norm_key, norm_query, norm_conv, conv_taps, replicated_anchor, source.layer_idx)
        result.validate(mesh_contract)
        return result

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        expected = {
            "key": ((1, 1, EMBEDDING_WIDTH, RESIDUAL_WIDTH // TP_SIZE), ttnn.bfloat16),
            "value": ((1, 1, EMBEDDING_WIDTH, LOCAL_HIDDEN_SIZE), ttnn.bfloat16),
            "norm_key": (RESIDUAL_LOCAL_SHAPE, ttnn.float32),
            "norm_query": (RESIDUAL_LOCAL_SHAPE, ttnn.float32),
            "norm_conv": (RESIDUAL_LOCAL_SHAPE, ttnn.float32),
        }
        for name, (shape, dtype) in expected.items():
            tensor = getattr(self, name)
            if _shape(tensor) != shape or tensor.dtype != dtype or tensor.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError(
                    f"PLE {name} must be TILE {dtype} {shape}, got {tensor.layout} {tensor.dtype} {_shape(tensor)}"
                )
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        for index, tensor in enumerate(self.conv_taps):
            if _shape(tensor) != RESIDUAL_LOCAL_SHAPE or tensor.dtype != ttnn.bfloat16:
                raise RuntimeError(f"PLE conv tap {index} has invalid shape/dtype")
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        mesh_contract.validate_tensor(self.replicated_anchor, placement=TensorPlacement.REPLICATED)
        if _shape(self.replicated_anchor) != (1, 1, 1, 1):
            raise RuntimeError("PLE replicated anchor must be [1,1,1,1]")
        if self.layer_index != 1:
            raise RuntimeError(f"PLE checkpoint layer must be zero-based 1, got {self.layer_index}")

    def deallocate(self) -> None:
        _deallocate(
            self.key,
            self.value,
            self.norm_key,
            self.norm_query,
            self.norm_conv,
            *self.conv_taps,
            self.replicated_anchor,
        )


@dataclass
class Qwen38TTNNPLESnapshot:
    conv: tuple[Any, ...]
    token_context: torch.Tensor | None = None
    captured: bool = False

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        if len(self.conv) != CONV_STATE_LENGTH:
            raise RuntimeError(f"PLE snapshot must contain {CONV_STATE_LENGTH} rows")
        for index, tensor in enumerate(self.conv):
            if _shape(tensor) != RESIDUAL_LOCAL_SHAPE or tensor.dtype != ttnn.bfloat16:
                raise RuntimeError(f"PLE snapshot row {index} has invalid shape/dtype")
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if self.token_context is not None and (
            self.token_context.device.type != "cpu"
            or self.token_context.dtype != torch.long
            or tuple(self.token_context.shape) != (1, NGRAM_SIZE - 1)
        ):
            raise RuntimeError("PLE snapshot token context must be host int64 [1,2]")

    def deallocate(self) -> None:
        _deallocate(*self.conv)


@dataclass
class Qwen38TTNNPLEState:
    """Host token history plus nine fixed-address hidden-sharded conv rows."""

    conv: tuple[Any, ...]
    zero_conv: Any
    mesh_contract: Qwen38MeshContract
    token_context: torch.Tensor | None = None

    @classmethod
    def allocate(cls, mesh_device, mesh_contract: Qwen38MeshContract) -> "Qwen38TTNNPLEState":
        mesh_contract.validate_mesh(mesh_device)
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3))
        host = torch.zeros((1, 1, RESIDUAL_BRANCHES, HIDDEN_SIZE), dtype=torch.bfloat16)

        def upload_zero():
            return ttnn.from_torch(
                host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        state = cls(tuple(upload_zero() for _ in range(CONV_STATE_LENGTH)), upload_zero(), mesh_contract)
        state.validate()
        return state

    def validate(self) -> None:
        if len(self.conv) != CONV_STATE_LENGTH:
            raise RuntimeError(f"PLE conv state must contain {CONV_STATE_LENGTH} rows")
        for index, tensor in enumerate((*self.conv, self.zero_conv)):
            if _shape(tensor) != RESIDUAL_LOCAL_SHAPE or tensor.dtype != ttnn.bfloat16:
                raise RuntimeError(f"PLE state row {index} has invalid shape/dtype")
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if self.token_context is not None:
            if self.token_context.device.type != "cpu" or self.token_context.dtype != torch.long:
                raise RuntimeError("PLE token context must be a host int64 tensor")
            if tuple(self.token_context.shape) != (1, NGRAM_SIZE - 1):
                raise RuntimeError(f"PLE token context must be [1,{NGRAM_SIZE - 1}]")

    def reset_inplace(self) -> None:
        self.validate()
        for index, tensor in enumerate(self.conv):
            _copy_inplace(self.zero_conv, tensor, label=f"PLE conv[{index}] reset")
        self.token_context = None

    def allocate_snapshot(self) -> Qwen38TTNNPLESnapshot:
        conv = tuple(
            ttnn.empty_like(
                tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for tensor in self.conv
        )
        snapshot = Qwen38TTNNPLESnapshot(conv)
        snapshot.validate(self.mesh_contract)
        return snapshot

    def capture_into(self, snapshot: Qwen38TTNNPLESnapshot) -> None:
        snapshot.validate(self.mesh_contract)
        for index, (source, target) in enumerate(zip(self.conv, snapshot.conv)):
            _copy_inplace(source, target, label=f"PLE conv[{index}] snapshot capture")
        snapshot.token_context = None if self.token_context is None else self.token_context.clone()
        snapshot.captured = True

    def restore_from(self, snapshot: Qwen38TTNNPLESnapshot) -> None:
        if not snapshot.captured:
            raise RuntimeError("cannot restore an uncaptured PLE snapshot")
        snapshot.validate(self.mesh_contract)
        for index, (source, target) in enumerate(zip(snapshot.conv, self.conv)):
            _copy_inplace(source, target, label=f"PLE conv[{index}] snapshot restore")
        self.token_context = None if snapshot.token_context is None else snapshot.token_context.clone()

    def deallocate(self) -> None:
        _deallocate(*self.conv, self.zero_conv)


@dataclass(frozen=True)
class Qwen38TTNNPLEResult:
    residual_delta: Any
    state: Qwen38TTNNPLEState


@dataclass
class Qwen38TTNNPLEPreparedInput:
    """Host-resolved PLE row uploaded before a captured device decode body.

    ``embedding_sharded`` is the persistent ROW_MAJOR BF16 ``[1,1,1,640]`` input
    per device; a runner rewrites it in place with ``copy_host_to_device_tensor``
    between replays and the body tilizes it.
    """

    embedding_sharded: Any
    source_token_context: torch.Tensor | None
    next_token_context: torch.Tensor
    active: bool = True

    def release(self) -> None:
        if not self.active:
            raise RuntimeError("prepared PLE input was already released")
        _deallocate(self.embedding_sharded)
        self.active = False


def _rows_residual_shape(rows: int) -> tuple[int, int, int, int]:
    return (1, rows, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE)


@dataclass
class Qwen38TTNNPLERowsState:
    """Persistent state of the ``rows``-row PLE path (MTP v2 verify): one history buffer, not nine slots.

    ``history`` holds the CONV_STATE_LENGTH normalized rows before the first new row (token-major
    ``[1,9,4,640]``); ``normalized`` keeps this pass's conv input rows so the commit can select the next
    history without recomputing them.  ``token_context`` is the host n-gram context ``(c0, c1)`` of the
    committed stream (``None`` = fresh EOS history).  Both buffers have fixed addresses.
    """

    rows: int
    history: Any
    normalized: Any
    mesh_contract: Qwen38MeshContract
    token_context: tuple[int, int] | None = None
    owns_history: bool = (
        True  # False: ``history`` is another rows state's buffer (the long chunk shares the 32-row one)
    )

    @classmethod
    def allocate(
        cls, mesh_device, mesh_contract: Qwen38MeshContract, *, rows: int, history=None
    ) -> "Qwen38TTNNPLERowsState":
        """``history`` hands over another rows state's history buffer: the two row forms then carry one history
        (and one host context, kept by the caller) and need no sync between them."""

        mesh_contract.validate_mesh(mesh_device)
        if not 1 <= rows <= 32 and rows != LONG_CHUNK_ROWS and not is_slab_rows(rows):
            raise ValueError(f"PLE rows path admits 1..32 rows, {LONG_CHUNK_ROWS} or a slab row count, got {rows}")
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3))

        def upload_zero(token_rows: int):
            return ttnn.from_torch(
                torch.zeros((1, token_rows, RESIDUAL_BRANCHES, HIDDEN_SIZE), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        owns_history = history is None
        if owns_history:
            history = upload_zero(CONV_STATE_LENGTH)
        try:
            state = cls(rows, history, upload_zero(rows), mesh_contract, owns_history=owns_history)
            state.validate()
            return state
        except BaseException:
            if owns_history:
                _deallocate(history)
            raise

    def validate(self) -> None:
        for name, tensor, token_rows in (
            ("history", self.history, CONV_STATE_LENGTH),
            ("normalized", self.normalized, self.rows),
        ):
            expected = _rows_residual_shape(token_rows)
            if _shape(tensor) != expected or tensor.dtype != ttnn.bfloat16:
                raise RuntimeError(f"PLE rows {name} must be BF16 {expected}, got {tensor.dtype} {_shape(tensor)}")
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _tensor_key(self.history) == _tensor_key(self.normalized):
            raise RuntimeError("PLE rows history and normalized rows must be distinct buffers")
        if self.token_context is not None and (
            len(self.token_context) != NGRAM_SIZE - 1
            or any(isinstance(value, bool) or type(value) is not int for value in self.token_context)
        ):
            raise RuntimeError(
                f"PLE rows token context must be {NGRAM_SIZE - 1} ints or None, got {self.token_context!r}"
            )

    def load_from_state(self, state: Qwen38TTNNPLEState) -> None:
        """Eager mode switch (1-row -> rows): the nine slots become the history buffer; host context copied."""

        self.validate()
        state.validate()
        combined = ttnn.concat(list(state.conv), dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _copy_inplace(combined, self.history, label="PLE rows history load")
        _deallocate(combined)
        self.token_context = None if state.token_context is None else tuple(int(v) for v in state.token_context[0])

    def store_to_state(self, state: Qwen38TTNNPLEState) -> None:
        """Eager mode switch (rows -> 1-row): the history buffer rows become the nine slots; host context copied."""

        self.validate()
        state.validate()
        for index, slot in enumerate(state.conv):
            row = ttnn.slice(
                self.history,
                (0, index, 0, 0),
                (1, index + 1, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _copy_inplace(row, slot, label=f"PLE rows history store {index}")
            _deallocate(row)
        state.token_context = (
            None if self.token_context is None else torch.tensor([list(self.token_context)], dtype=torch.long)
        )
        state.validate()

    def deallocate(self) -> None:
        _deallocate(*((self.history,) if self.owns_history else ()), self.normalized)


@dataclass
class Qwen38TTNNPLERowsPreparedInput:
    """Host-resolved PLE rows for one verify pass: the ``rows`` tokens ``[t_p, d_1 .. d_k]`` looked up with
    their own contexts, uploaded as one persistent ROW_MAJOR BF16 ``[1,1,rows,640]`` per device.

    ``contexts[c]`` is the host n-gram context after committing ``c`` of these rows (``c = 0 .. rows``); the
    commit picks ``contexts[accepted + 1]``.  A runner rewrites ``embedding_rows`` in place between replays
    (``copy_host_to_device_tensor`` of :meth:`Qwen38TTNNPLE.host_rows`); the body tilizes it.
    """

    embedding_rows: Any
    tokens: tuple[int, ...]
    contexts: tuple[tuple[int, int] | None, ...]
    active: bool = True

    def release(self) -> None:
        if not self.active:
            raise RuntimeError("prepared PLE rows input was already released")
        _deallocate(self.embedding_rows)
        self.active = False


@dataclass(frozen=True)
class Qwen38TTNNPLERowsResult:
    residual_delta: Any
    rows_state: Qwen38TTNNPLERowsState


@dataclass
class Qwen38TTNNPLELaneRowsState:
    """The rows state (:class:`Qwen38TTNNPLERowsState`) of B lanes, the MTP lanes verify: ``history`` ``[B,9,4,640]``
    holds lane u's CONV_STATE_LENGTH normalized rows before its first new row, ``normalized`` ``[B,R,4,640]`` lane u's
    conv input rows of this pass (kept for the commit's history select), one host n-gram context per lane.  Fixed
    addresses; every op of the lane rows body is per row or per element, so lane u is the rows path on its stream."""

    lanes: int
    rows: int
    history: Any
    normalized: Any
    mesh_contract: Qwen38MeshContract
    token_contexts: tuple[tuple[int, int] | None, ...]

    @classmethod
    def allocate(
        cls, mesh_device, mesh_contract: Qwen38MeshContract, *, lanes: int, rows: int
    ) -> "Qwen38TTNNPLELaneRowsState":
        mesh_contract.validate_mesh(mesh_device)
        lanes = require_lane_count(lanes, label="PLE lane rows lanes")
        if not 1 <= rows <= ttnn.TILE_SIZE or lanes * rows > ttnn.TILE_SIZE:
            raise ValueError(f"PLE lane rows admit B x R <= {ttnn.TILE_SIZE} rows in one tile, got {lanes} x {rows}")
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3))

        def upload_zero(token_rows: int):
            return ttnn.from_torch(
                torch.zeros((lanes, token_rows, RESIDUAL_BRANCHES, HIDDEN_SIZE), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        history = upload_zero(CONV_STATE_LENGTH)
        try:
            state = cls(lanes, rows, history, upload_zero(rows), mesh_contract, (None,) * lanes)
            state.validate()
            return state
        except BaseException:
            _deallocate(history)
            raise

    def validate(self) -> None:
        lanes = require_lane_count(self.lanes, label="PLE lane rows lanes", error_type=RuntimeError)
        for name, tensor, token_rows in (
            ("history", self.history, CONV_STATE_LENGTH),
            ("normalized", self.normalized, self.rows),
        ):
            expected = (lanes, token_rows, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE)
            if _shape(tensor) != expected or tensor.dtype != ttnn.bfloat16:
                raise RuntimeError(f"PLE lane rows {name} must be BF16 {expected}, got {tensor.dtype} {_shape(tensor)}")
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _tensor_key(self.history) == _tensor_key(self.normalized):
            raise RuntimeError("PLE lane rows history and normalized rows must be distinct buffers")
        self.token_contexts = _validate_lane_contexts(self.token_contexts, lanes, label="PLE lane rows token contexts")

    def deallocate(self) -> None:
        _deallocate(self.history, self.normalized)


def _lanes_residual_shape(lanes: int) -> tuple[int, int, int, int]:
    return (1, lanes, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE)


def _validate_lane_contexts(contexts, lanes: int, *, label: str) -> tuple[tuple[int, int] | None, ...]:
    values = tuple(contexts)
    if len(values) != lanes:
        raise ValueError(f"{label} needs one n-gram context per lane: got {len(values)}, expected {lanes}")
    for lane, context in enumerate(values):
        if context is not None and (
            len(context) != NGRAM_SIZE - 1
            or any(isinstance(value, bool) or type(value) is not int for value in context)
        ):
            raise ValueError(f"{label} lane {lane} must be {NGRAM_SIZE - 1} ints or None, got {context!r}")
    return values


@dataclass
class Qwen38TTNNPLELanesState:
    """The 1-row PLE state over ``lanes`` users: nine fixed-address conv slots of token-major ``[1,lanes,4,640]``
    (lane u = dim-1 index u, each lane its own nine-row dilated history, shifted per step exactly as the 1-row
    ring) and one host n-gram context per lane.  Lanes are independent: every op of the lanes body is per row or
    per element, so lane u is the 1-row path on user u's stream."""

    lanes: int
    conv: tuple[Any, ...]
    zero_conv: Any
    mesh_contract: Qwen38MeshContract
    token_contexts: tuple[tuple[int, int] | None, ...]

    @classmethod
    def allocate(cls, mesh_device, mesh_contract: Qwen38MeshContract, *, lanes: int) -> "Qwen38TTNNPLELanesState":
        mesh_contract.validate_mesh(mesh_device)
        lanes = require_lane_count(lanes, label="PLE lanes")
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3))
        host = torch.zeros((1, lanes, RESIDUAL_BRANCHES, HIDDEN_SIZE), dtype=torch.bfloat16)

        def upload_zero():
            return ttnn.from_torch(
                host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        state = cls(
            lanes, tuple(upload_zero() for _ in range(CONV_STATE_LENGTH)), upload_zero(), mesh_contract, (None,) * lanes
        )
        state.validate()
        return state

    def validate(self) -> None:
        lanes = require_lane_count(self.lanes, label="PLE lanes", error_type=RuntimeError)
        if len(self.conv) != CONV_STATE_LENGTH:
            raise RuntimeError(f"PLE lanes conv state holds {len(self.conv)} slots, expected {CONV_STATE_LENGTH}")
        expected = _lanes_residual_shape(lanes)
        for index, tensor in enumerate((*self.conv, self.zero_conv)):
            if _shape(tensor) != expected or tensor.dtype != ttnn.bfloat16:
                raise RuntimeError(
                    f"PLE lanes slot {index} must be BF16 {expected}, got {tensor.dtype} {_shape(tensor)}"
                )
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if len({_tensor_key(tensor) for tensor in (*self.conv, self.zero_conv)}) != CONV_STATE_LENGTH + 1:
            raise RuntimeError("PLE lanes slots must be distinct buffers")
        self.token_contexts = _validate_lane_contexts(self.token_contexts, lanes, label="PLE lanes token contexts")

    def reset_inplace(self) -> None:
        self.validate()
        for index, tensor in enumerate(self.conv):
            _copy_inplace(self.zero_conv, tensor, label=f"PLE lanes conv[{index}] reset")
        self.token_contexts = (None,) * self.lanes

    def reset_lane_inplace(self, lane: int) -> None:
        """Zero lane ``lane`` of every slot (a keep-mask multiply, the other lanes x 1.0) and clear its context, no
        address change.  Eager host upload: a lifecycle op between steps, never traced."""

        self.validate()
        if isinstance(lane, bool) or type(lane) is not int or not 0 <= lane < self.lanes:
            raise ValueError(f"PLE lane must be an int in [0,{self.lanes}), got {lane!r}")
        keep = torch.ones(1, self.lanes, 1, 1, dtype=torch.bfloat16)
        keep[0, lane] = 0.0
        device = self.zero_conv.device()
        keep_lanes = ttnn.from_torch(
            keep,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(device),
        )
        try:
            for index, slot in enumerate(self.conv):
                kept = ttnn.multiply(slot, keep_lanes, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                _copy_inplace(kept, slot, label=f"PLE lane {lane} conv[{index}] reset")
                _deallocate(kept)
        finally:
            _deallocate(keep_lanes)
        contexts = list(self.token_contexts)
        contexts[lane] = None
        self.token_contexts = tuple(contexts)

    def deallocate(self) -> None:
        _deallocate(*self.conv, self.zero_conv)


@dataclass
class Qwen38TTNNPLELanesPreparedInput:
    """Host-resolved PLE rows of one batched step: lane u's token looked up with lane u's own context, uploaded as
    one persistent ROW_MAJOR BF16 ``[1,1,lanes,640]`` per device (rewritten in place between replays from
    :meth:`Qwen38TTNNPLE.host_lanes`; the body tilizes it).  ``next_contexts[u]`` is lane u's context after
    this step."""

    embedding_rows: Any
    tokens: tuple[int, ...]
    source_contexts: tuple[tuple[int, int] | None, ...]
    next_contexts: tuple[tuple[int, int], ...]
    active: bool = True

    def release(self) -> None:
        if not self.active:
            raise RuntimeError("prepared PLE lanes input was already released")
        _deallocate(self.embedding_rows)
        self.active = False


def ngram_token_ids_decode_step(
    token_id: int,
    context: tuple[int, int] | None,
    *,
    eos_token_id: int,
    multipliers: Sequence[int],
    head_vocab_sizes: Sequence[int],
    head_offsets: Sequence[int],
) -> tuple[list[int], tuple[int, int]]:
    """``reference.ngram_token_ids`` for one decode step, in plain integers.

    ``context`` is the raw two-token history ``(c0, c1)``; ``None`` is a fresh
    history of EOS.  At the last history column the reference's EOS-aware shifts
    reduce to ``s0 = x``, ``s1 = c1`` and ``s2 = eos if c1 == eos else c0``.
    Every ``token * multiplier`` is below 2**63 by construction of the
    multipliers (``build_ngram_hash_spec``), so Python ints reproduce the int64
    mix, XOR and remainder bit for bit.  Returns the head ids (bigram heads
    first, then trigram heads) and the next context ``(c1, x)``.
    """

    c0, c1 = (eos_token_id, eos_token_id) if context is None else context
    bigram = (token_id * multipliers[0]) ^ (c1 * multipliers[1])
    trigram = bigram ^ ((eos_token_id if c1 == eos_token_id else c0) * multipliers[2])
    heads_per_ngram = len(head_vocab_sizes) // 2
    ids = [bigram % head_vocab_sizes[head] + head_offsets[head] for head in range(heads_per_ngram)]
    ids += [
        trigram % head_vocab_sizes[head] + head_offsets[head] for head in range(heads_per_ngram, 2 * heads_per_ngram)
    ]
    return ids, (c1, token_id)


class Qwen38ResidentPLELookup:
    """``Qwen38HostPLEEmbedding.lookup`` over one persistent descriptor per table part.

    Same hashed ids, same bytes, same row order as the host oracle; the 128
    parts are opened once through the checkpoint file guard and every token's
    sixteen rows are advised together before they are read.  ``lookup_token``
    is the decode-step form: the hash in plain integers, the rows as one
    payload, no torch.  A batch (:meth:`_read_rows`) re-proves the file guard's
    identity once per touched shard file before and after its reads (the
    identity is the file's, not the part's) and issues one WILLNEED and one
    ``pread`` per distinct row: an MTP pass's 80 rows cost about 130 syscalls
    where the per-part checked calls cost 344.
    """

    def __init__(self, host_embedding: Qwen38HostPLEEmbedding) -> None:
        self.host_embedding = host_embedding
        self.readers = tuple(
            Qwen38CheckpointRowReader(host_embedding.checkpoint, name) for name in host_embedding.table_names
        )
        for reader in self.readers:
            if reader.rows != host_embedding.rows_per_shard or reader.row_shape != (host_embedding.embedding_head_dim,):
                raise RuntimeError(f"PLE table part {reader.name} disagrees with the host table geometry")
        spec = host_embedding.spec
        if spec.ngram_size != NGRAM_SIZE or spec.layer_multipliers.numel() != NGRAM_SIZE:
            raise RuntimeError(f"PLE decode-step hash requires n-gram size {NGRAM_SIZE}, got {spec.ngram_size}")
        self.eos_token_id = int(host_embedding.config.eos_token_id)
        self.vocab_size = int(host_embedding.config.vocab_size)
        self.multipliers = tuple(int(value) for value in spec.layer_multipliers.tolist())
        self.head_vocab_sizes = tuple(int(value) for value in spec.head_vocab_sizes.tolist())
        self.head_offsets = tuple(int(value) for value in spec.head_offsets.tolist())
        if len(self.head_vocab_sizes) != (NGRAM_SIZE - 1) * spec.heads_per_ngram or len(self.head_offsets) != len(
            self.head_vocab_sizes
        ):
            raise RuntimeError("PLE hash spec head tables disagree with heads_per_ngram")
        if (self.vocab_size - 1) * max(self.multipliers) >= 1 << 63:
            raise RuntimeError("PLE hash multipliers would overflow int64 for this vocabulary")

    def _read_rows(self, hashed: Sequence[int]) -> bytearray:
        """The rows ``hashed`` in request order as one contiguous BF16 payload.

        One identity proof per touched shard file before and after the batch (the file guard's contract at the
        file, whose identity it is, instead of three ``fstat`` per touched part), every distinct row advised
        (WILLNEED: cold rows cost the batch one disk latency, not one per row) before any is read, one ``pread``
        per distinct row, the duplicates joined from the first read.  Same bytes in the same order as the per-part
        ``advise`` / ``read_rows`` calls."""

        rows_per_shard = self.host_embedding.rows_per_shard
        readers = self.readers
        rows_by_part: dict[int, list[int]] = {}
        for index in dict.fromkeys(hashed):
            rows_by_part.setdefault(index // rows_per_shard, []).append(index % rows_per_shard)
        proving = {readers[part].shard: readers[part] for part in rows_by_part}
        for reader in proving.values():
            reader.prove_identity("before")
        for part, rows in rows_by_part.items():
            readers[part].advise_proven(rows)
        payload_by_index: dict[int, bytes] = {}
        for part, rows in rows_by_part.items():
            base = part * rows_per_shard
            for row, payload in zip(rows, readers[part].read_rows_proven(rows)):
                payload_by_index[base + row] = payload
        for reader in proving.values():
            reader.prove_identity("after")
        return bytearray().join(payload_by_index[index] for index in hashed)

    def lookup(
        self, input_ids: torch.Tensor, previous_context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.device.type != "cpu":
            raise ValueError("PLE hash and table lookup are host-resident")
        host = self.host_embedding
        ids, next_context = ngram_token_ids(input_ids, previous_context, host.config.eos_token_id, host.spec)
        payload = self._read_rows(ids.reshape(-1).tolist())
        embeddings = torch.frombuffer(payload, dtype=torch.bfloat16).reshape(*ids.shape, host.embedding_head_dim)
        return embeddings.flatten(-2), next_context

    def lookup_token(self, token_id: int, context: tuple[int, int] | None) -> tuple[bytearray, tuple[int, int]]:
        """One decode step: the sixteen hashed rows as one 5,120 B payload and the next ``(c1, x)`` context."""

        for value in (token_id, *(() if context is None else context)):
            if isinstance(value, bool) or type(value) is not int or not 0 <= value < self.vocab_size:
                raise ValueError(
                    f"PLE decode-step tokens must be exact integers in [0,{self.vocab_size}), got {value!r}"
                )
        ids, next_context = ngram_token_ids_decode_step(
            token_id,
            context,
            eos_token_id=self.eos_token_id,
            multipliers=self.multipliers,
            head_vocab_sizes=self.head_vocab_sizes,
            head_offsets=self.head_offsets,
        )
        return self._read_rows(ids), next_context

    def lookup_tokens(
        self, tokens: Sequence[int], context: tuple[int, int] | None
    ) -> tuple[bytearray, tuple[tuple[int, int] | None, ...]]:
        """:meth:`lookup_token` over a token stream in one read: the decode-step hash per token, the contexts chained
        as the steps chain them (``contexts[c]`` after c tokens), then every row in one ``_read_rows`` batch (one
        WILLNEED pass per part; the same bytes in the same order as the per-token calls, about three times faster)."""

        for value in (*tokens, *(() if context is None else context)):
            if isinstance(value, bool) or type(value) is not int or not 0 <= value < self.vocab_size:
                raise ValueError(f"PLE tokens must be exact integers in [0,{self.vocab_size}), got {value!r}")
        hashed: list[int] = []
        contexts: list[tuple[int, int] | None] = [context]
        for token in tokens:
            ids, context = ngram_token_ids_decode_step(
                token,
                context,
                eos_token_id=self.eos_token_id,
                multipliers=self.multipliers,
                head_vocab_sizes=self.head_vocab_sizes,
                head_offsets=self.head_offsets,
            )
            hashed += ids
            contexts.append(context)
        return self._read_rows(hashed), tuple(contexts)

    def lookup_tokens_lanes(
        self, tokens_by_lane: Sequence[Sequence[int]], contexts: Sequence[tuple[int, int] | None]
    ) -> tuple[bytearray, tuple[tuple[tuple[int, int] | None, ...], ...]]:
        """:meth:`lookup_tokens` over B lanes in ONE read: lane u's tokens chained from ``contexts[u]``, every lane's
        rows in one ``_read_rows`` batch (lane-major payload: lane u's rows follow lane u - 1's) and per lane its
        R + 1 contexts (the MTP lanes' per-pass PLE input)."""

        if len(tokens_by_lane) != len(contexts) or not tokens_by_lane:
            raise ValueError(
                f"PLE lane rows need one context per lane: got {len(tokens_by_lane)} lanes, {len(contexts)} contexts"
            )
        hashed: list[int] = []
        lane_contexts = []
        for lane, (tokens, context) in enumerate(zip(tokens_by_lane, contexts)):
            for value in (*tokens, *(() if context is None else context)):
                if isinstance(value, bool) or type(value) is not int or not 0 <= value < self.vocab_size:
                    raise ValueError(
                        f"PLE lane {lane} tokens must be exact integers in [0,{self.vocab_size}), got {value!r}"
                    )
            chain: list[tuple[int, int] | None] = [context]
            for token in tokens:
                ids, context = ngram_token_ids_decode_step(
                    token,
                    context,
                    eos_token_id=self.eos_token_id,
                    multipliers=self.multipliers,
                    head_vocab_sizes=self.head_vocab_sizes,
                    head_offsets=self.head_offsets,
                )
                hashed += ids
                chain.append(context)
            lane_contexts.append(tuple(chain))
        return self._read_rows(hashed), tuple(lane_contexts)

    def lookup_lanes(
        self, tokens: Sequence[int], contexts: Sequence[tuple[int, int] | None]
    ) -> tuple[bytearray, tuple[tuple[int, int], ...]]:
        """One batched step: lane u's sixteen rows hashed from ``(tokens[u], contexts[u])``, every lane's rows
        advised and read in one pass (one payload of ``lanes x 5,120`` B, lane-major), and the next contexts."""

        if len(tokens) != len(contexts) or not tokens:
            raise ValueError(
                f"PLE lane lookup needs one context per token: got {len(tokens)} tokens, {len(contexts)} contexts"
            )
        ids: list[int] = []
        next_contexts = []
        for lane, (token_id, context) in enumerate(zip(tokens, contexts)):
            for value in (token_id, *(() if context is None else context)):
                if isinstance(value, bool) or type(value) is not int or not 0 <= value < self.vocab_size:
                    raise ValueError(
                        f"PLE lane {lane} tokens must be exact integers in [0,{self.vocab_size}), got {value!r}"
                    )
            lane_ids, next_context = ngram_token_ids_decode_step(
                token_id,
                context,
                eos_token_id=self.eos_token_id,
                multipliers=self.multipliers,
                head_vocab_sizes=self.head_vocab_sizes,
                head_offsets=self.head_offsets,
            )
            ids.extend(lane_ids)
            next_contexts.append(next_context)
        return self._read_rows(ids), tuple(next_contexts)

    def close(self) -> None:
        for reader in self.readers:
            reader.close()


class Qwen38TTNNPLE:
    """Synchronous global-B1 PLE decode for zero-based language layer one."""

    _fused_forward_prepared = None  # QWEN38_FUSED=ple binds ttnn/fused/ple per instance; the body is the chain
    _fused_forward_prepared_lanes = None  # its lane form (forward_prepared_lanes on the B row-blocks)

    def __init__(
        self,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        host_embedding: Qwen38HostPLEEmbedding,
        weights: Qwen38TTNNPLEWeights,
        *,
        collective_topology=None,
    ) -> None:
        mesh_contract.validate_mesh(mesh_device)
        weights.validate(mesh_contract)
        if host_embedding.config.ple_checkpoint_layer != weights.layer_index:
            raise ValueError("PLE host table and device weights refer to different layers")
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.host_embedding = host_embedding
        self.weights = weights
        self.collective_topology = collective_topology or ttnn.Topology.Linear
        self._poisoned_prepared_inputs: list[Qwen38TTNNPLEPreparedInput] = []
        self._resident_lookup: Qwen38ResidentPLELookup | None = None
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # QWEN38_FUSED=ple: the fused device body (ttnn/fused/ple) binds at construction; forward_prepared's chain stays.
        from models.demos.blackhole.qwen38_flash_next.ttnn import fused as fused_kernels

        if fused_kernels.enabled("ple"):
            self._fused_forward_prepared = functools.partial(fused_kernels.kernel("ple").fused, self)
            self._fused_forward_prepared_lanes = functools.partial(fused_kernels.ple.ple_lanes_fused, self)

    def allocate_state(self) -> Qwen38TTNNPLEState:
        return Qwen38TTNNPLEState.allocate(self.mesh_device, self.mesh_contract)

    @property
    def resident_lookup(self) -> Qwen38ResidentPLELookup:
        """Persistent-descriptor host lookup, opened on first use and kept for the model lifetime."""

        if self._resident_lookup is None:
            self._resident_lookup = Qwen38ResidentPLELookup(self.host_embedding)
        return self._resident_lookup

    def _validate_residual(self, tensor, *, label: str) -> None:
        if _shape(tensor) != RESIDUAL_LOCAL_SHAPE or tensor.dtype != ttnn.bfloat16:
            raise ValueError(f"{label} must be TILE BF16 {RESIDUAL_LOCAL_SHAPE}, got {_shape(tensor)} {tensor.dtype}")
        self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _distributed_group_norm(self, tensor, weight):
        self._validate_residual(tensor, label="PLE group-norm input")
        self.mesh_contract.validate_tensor(weight, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        stats = ttnn.rms_norm_pre_all_gather(
            tensor,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        stats = ttnn.reshape(stats, (1, 1, RESIDUAL_BRANCHES, 32))
        self.mesh_contract.validate_tensor(stats, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        gathered = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(stats)
        self.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
        normalized = ttnn.rms_norm_post_all_gather(
            tensor,
            gathered,
            epsilon=RMS_NORM_EPS,
            weight=weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
        )
        _deallocate(gathered)
        self._validate_residual(normalized, label="PLE group-norm output")
        return normalized

    def _validate_prepared_row(self, tensor, *, label: str) -> None:
        self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if (
            _shape(tensor) != (1, 1, 1, LOCAL_HIDDEN_SIZE)
            or tensor.dtype != ttnn.bfloat16
            or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
        ):
            raise RuntimeError(
                f"{label} must be ROW_MAJOR BF16 [1,1,1,{LOCAL_HIDDEN_SIZE}], "
                f"got {tensor.layout} {tensor.dtype} {_shape(tensor)}"
            )

    def _upload_embedding(self, embedding: torch.Tensor):
        """The host row as the persistent ROW_MAJOR input: 1,280 B per device, tilized inside the body."""

        if tuple(embedding.shape) != (1, 1, EMBEDDING_WIDTH) or embedding.dtype != torch.bfloat16:
            raise RuntimeError(f"host PLE lookup must return BF16 [1,1,{EMBEDDING_WIDTH}]")
        tensor = ttnn.from_torch(
            embedding.reshape(1, 1, 1, EMBEDDING_WIDTH).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
        )
        self._validate_prepared_row(tensor, label="PLE upload")
        return tensor

    def _project(self, embedding_tile):
        full_embedding = ttnn.all_gather(
            embedding_tile,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(embedding_tile)
        self.mesh_contract.validate_tensor(full_embedding, placement=TensorPlacement.REPLICATED)
        if _shape(full_embedding) != (1, 1, 1, EMBEDDING_WIDTH):
            raise RuntimeError("PLE embedding all-gather did not reconstruct width 2560")
        key_flat = ttnn.linear(
            full_embedding,
            self.weights.key,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        value = ttnn.linear(
            full_embedding,
            self.weights.value,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(full_embedding)
        self.mesh_contract.validate_tensor(key_flat, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        self.mesh_contract.validate_tensor(value, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(key_flat) != (1, 1, 1, RESIDUAL_WIDTH // TP_SIZE):
            raise RuntimeError(f"PLE key projection has local shape {_shape(key_flat)}")
        if _shape(value) != (1, 1, 1, LOCAL_HIDDEN_SIZE):
            raise RuntimeError(f"PLE value projection has local shape {_shape(value)}")
        key = ttnn.reshape(key_flat, RESIDUAL_LOCAL_SHAPE)
        _deallocate(key_flat)
        self._validate_residual(key, label="PLE key projection")
        return key, value

    @staticmethod
    def _same_host_context(left: torch.Tensor | None, right: torch.Tensor | None) -> bool:
        if left is None or right is None:
            return left is right
        return (
            left.device.type == "cpu"
            and right.device.type == "cpu"
            and left.dtype == torch.long
            and right.dtype == torch.long
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )

    def prepare_decode_input(
        self,
        token_id: torch.Tensor,
        state: Qwen38TTNNPLEState,
        *,
        resident_lookup: bool = False,
    ) -> Qwen38TTNNPLEPreparedInput:
        """Resolve the host n-gram row and upload it before trace capture/replay."""

        state.validate()
        if token_id.device.type != "cpu" or token_id.dtype != torch.long or tuple(token_id.shape) != (1, 1):
            raise ValueError("PLE token_id must be host int64 [1,1]")
        source_context = None if state.token_context is None else state.token_context.clone()
        lookup = self.resident_lookup.lookup if resident_lookup else self.host_embedding.lookup
        embedding, next_context = lookup(token_id, state.token_context)
        embedding_sharded = self._upload_embedding(embedding)
        return Qwen38TTNNPLEPreparedInput(
            embedding_sharded=embedding_sharded,
            source_token_context=source_context,
            next_token_context=next_context.clone(),
        )

    def forward_prepared(
        self,
        residual,
        prepared: Qwen38TTNNPLEPreparedInput,
        state: Qwen38TTNNPLEState,
    ) -> Qwen38TTNNPLEResult:
        """Run only device PLE work while retaining the persistent prepared row."""
        if self._fused_forward_prepared is not None:
            return self._fused_forward_prepared(residual, prepared, state)
        self._validate_residual(residual, label="PLE residual input")
        state.validate()
        if not isinstance(prepared, Qwen38TTNNPLEPreparedInput) or not prepared.active:
            raise TypeError("PLE prepared input must be a live Qwen38TTNNPLEPreparedInput")
        if not self._same_host_context(state.token_context, prepared.source_token_context):
            raise RuntimeError("prepared PLE input does not match the state's host token context")
        self._validate_prepared_row(prepared.embedding_sharded, label="prepared PLE row")
        # The body's one layout op: the 1,280 B row-major shard becomes the
        # 32x640 tile the projections consume (ttnn::tilize_with_val_padding,
        # zero pad rows; a data movement, no arithmetic).  The persistent row
        # itself is untouched so the host can rewrite it for the next token.
        embedding_tile = ttnn.to_layout(
            prepared.embedding_sharded, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # Metadata only, as after every local layout op in this model.
        embedding_tile.update_tensor_topology(prepared.embedding_sharded.tensor_topology())
        self.mesh_contract.validate_tensor(embedding_tile, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(embedding_tile) != (1, 1, 1, LOCAL_HIDDEN_SIZE) or embedding_tile.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"PLE row tilize produced {embedding_tile.layout} {_shape(embedding_tile)}")
        key, value = self._project(embedding_tile)
        gated = self._gate(key, residual, value)
        _deallocate(key)
        normalized = self._distributed_group_norm(gated, self.weights.norm_conv)
        convolution = self._convolve(normalized, state)
        _deallocate(normalized)
        output = ttnn.add(gated, convolution, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(gated, convolution)
        self._validate_residual(output, label="PLE residual delta")
        # The host context was cloned while preparing the persistent input.
        # Reuse that immutable owner here so the captured device body performs
        # no host tensor allocation.
        state.token_context = prepared.next_token_context
        state.validate()
        return Qwen38TTNNPLEResult(output, state)

    def _gate(self, key, query, value):
        key_norm = self._distributed_group_norm(key, self.weights.norm_key)
        query_norm = self._distributed_group_norm(query, self.weights.norm_query)
        key_global = ttnn.all_gather(
            key_norm,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query_global = ttnn.all_gather(
            query_norm,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(key_norm, query_norm)
        global_shape = (1, 1, RESIDUAL_BRANCHES, HIDDEN_SIZE)
        for label, tensor in (("key", key_global), ("query", query_global)):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
            if _shape(tensor) != global_shape:
                raise RuntimeError(f"PLE global {label} norm has shape {_shape(tensor)}, expected {global_shape}")
        key_fp32 = ttnn.typecast(key_global, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        query_fp32 = ttnn.typecast(query_global, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(key_global, query_global)
        products = ttnn.multiply(key_fp32, query_fp32, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(key_fp32, query_fp32)
        unscaled_gate = ttnn.sum(
            products,
            dim=3,
            keepdim=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(products)
        gate = ttnn.multiply(unscaled_gate, HIDDEN_SIZE**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(unscaled_gate)
        self.mesh_contract.validate_tensor(gate, placement=TensorPlacement.REPLICATED)

        magnitude = ttnn.abs(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        bounded = ttnn.clamp(magnitude, min=1.0e-6, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        root = ttnn.sqrt(bounded, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        direction = ttnn.sign(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(magnitude, bounded, gate)
        transformed = ttnn.multiply(root, direction, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(root, direction)
        coefficient = ttnn.sigmoid(transformed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(transformed)
        value_branches = ttnn.repeat(value, (1, 1, RESIDUAL_BRANCHES, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(value)
        gated = ttnn.multiply(value_branches, coefficient, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(value_branches, coefficient)
        self._validate_residual(gated, label="PLE gated value")
        return gated

    def _convolve(self, normalized, state: Qwen38TTNNPLEState):
        # PyTorch Conv1d cross-correlation for dilation=3 uses state positions
        # t-9, t-6, t-3 and the current row for taps 0,1,2,3 respectively.
        rows = (state.conv[0], state.conv[3], state.conv[6], normalized)
        convolution = ttnn.multiply(rows[0], self.weights.conv_taps[0], memory_config=ttnn.L1_MEMORY_CONFIG)
        for row, tap in zip(rows[1:], self.weights.conv_taps[1:]):
            previous = convolution
            convolution = ttnn.mac(row, tap, previous)
            if _tensor_key(convolution) != _tensor_key(previous):
                _deallocate(previous)
        convolution = ttnn.silu(convolution, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._validate_residual(convolution, label="PLE convolution")

        for index in range(CONV_STATE_LENGTH - 1):
            _copy_inplace(state.conv[index + 1], state.conv[index], label=f"PLE conv shift {index}")
        _copy_inplace(normalized, state.conv[-1], label="PLE conv newest row")
        return convolution

    def forward_decode(
        self,
        residual,
        token_id: torch.Tensor,
        state: Qwen38TTNNPLEState,
    ) -> Qwen38TTNNPLEResult:
        """Return the PLE residual delta for exactly one host token."""

        self._validate_residual(residual, label="PLE residual input")
        state.validate()
        if token_id.device.type != "cpu" or token_id.dtype != torch.long or tuple(token_id.shape) != (1, 1):
            raise ValueError("PLE token_id must be host int64 [1,1]")
        prepared = self.prepare_decode_input(token_id, state)
        try:
            result = self.forward_prepared(residual, prepared, state)
        except BaseException:
            # A device enqueue failure makes prepared-input lifetime uncertain.
            # Keep its owner until mesh teardown instead of scheduling a
            # potentially unsafe deallocation onto the failed queue.
            self._poisoned_prepared_inputs.append(prepared)
            raise
        prepared.release()
        return result

    # ------------------------------------------------------------------ rows path (MTP v2 verify)
    #
    # ``forward_prepared_rows`` runs the 1-row device PLE on ``[1,rows,4,640]`` (token-major: dim 1 is the
    # token, each token keeps its [4,640] branch block, so every op sees the 1-row tile structure per
    # token).  The dilated conv is a fixed window ``[9 history rows | rows new rows]`` read with four
    # slices (taps at window offsets 0, 3, 6, 9); ``commit_rows`` selects the next history
    # ``window[c : c + 9]`` with ``c = accepted + 1`` by an exact one-hot multiply/add.  The host
    # supplies the ``rows`` tokens looked up with their own sequential contexts (:meth:`host_rows`).
    # Assumption to confirm on hardware: rms_norm_post_all_gather applies the [1,1,4,640] per-branch
    # weight per tile row, so it is reused for every token (row 0 must match the 1-row path).

    def allocate_rows_state(self, rows: int, *, history=None) -> Qwen38TTNNPLERowsState:
        return Qwen38TTNNPLERowsState.allocate(self.mesh_device, self.mesh_contract, rows=rows, history=history)

    def _validate_rows_residual(self, tensor, rows: int, *, label: str) -> None:
        expected = _rows_residual_shape(rows)
        if _shape(tensor) != expected or tensor.dtype != ttnn.bfloat16:
            raise ValueError(f"{label} must be TILE BF16 {expected}, got {_shape(tensor)} {tensor.dtype}")
        self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def host_rows(
        self, tokens: Sequence[int], context: tuple[int, int] | None
    ) -> tuple[torch.Tensor, tuple[tuple[int, int] | None, ...]]:
        """The n-gram rows of ``tokens`` looked up sequentially from ``context``, as BF16 ``[1,1,rows,2560]``.

        Row i uses the context after rows 0 .. i-1; ``contexts`` has rows + 1 entries, ``contexts[c]`` being
        the context after committing c rows.  Same hash, bytes and row order as the 1-row decode step, read in one
        batch (:meth:`Qwen38ResidentPLELookup.lookup_tokens`).
        """

        if len(tokens) == 0:
            raise ValueError("PLE rows need at least one token")
        payload, contexts = self.resident_lookup.lookup_tokens([int(token) for token in tokens], context)
        host = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(1, 1, len(tokens), EMBEDDING_WIDTH)
        return host, contexts

    def host_rows_lanes(
        self, tokens_by_lane: Sequence[Sequence[int]], contexts: Sequence[tuple[int, int] | None]
    ) -> tuple[torch.Tensor, tuple[tuple[tuple[int, int] | None, ...], ...]]:
        """:meth:`host_rows` for B lanes in one table read: the lane-major ``[1,1,B*R,2560]`` rows (lane u's R rows at
        ``u*R``, looked up from lane u's own context) and per lane its R + 1 contexts."""

        if not tokens_by_lane or any(len(tokens) == 0 for tokens in tokens_by_lane):
            raise ValueError("PLE lane rows need at least one token per lane")
        payload, lane_contexts = self.resident_lookup.lookup_tokens_lanes(
            [[int(token) for token in tokens] for tokens in tokens_by_lane], contexts
        )
        rows = sum(len(tokens) for tokens in tokens_by_lane)
        host = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(1, 1, rows, EMBEDDING_WIDTH)
        return host, lane_contexts

    def _validate_prepared_rows(self, tensor, rows: int, *, label: str) -> None:
        self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        expected = (1, 1, rows, LOCAL_HIDDEN_SIZE)
        if _shape(tensor) != expected or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(
                f"{label} must be ROW_MAJOR BF16 {list(expected)}, got {tensor.layout} {tensor.dtype} {_shape(tensor)}"
            )

    def prepare_rows_input(
        self, tokens: Sequence[int], rows_state: Qwen38TTNNPLERowsState, *, rows: int | None = None
    ) -> Qwen38TTNNPLERowsPreparedInput:
        """Look up and upload the ``rows`` tokens of one pass (host work, before capture or between replays).

        ``rows`` (a prefill slab) sizes the upload past the rows state: the slab's PLE runs the rows state's 128-row
        pass per block of the uploaded rows."""

        rows_state.validate()
        rows = rows_state.rows if rows is None else rows
        if rows != rows_state.rows and (not is_slab_rows(rows) or rows_state.rows != LONG_CHUNK_ROWS):
            raise ValueError(f"PLE rows input of {rows} rows needs a slab row count over a 128-row rows state")
        if len(tokens) != rows:
            raise ValueError(f"PLE rows input needs {rows} tokens, got {len(tokens)}")
        host, contexts = self.host_rows(tokens, rows_state.token_context)
        tensor = ttnn.from_torch(
            host.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
        )
        self._validate_prepared_rows(tensor, rows, label="PLE rows upload")
        return Qwen38TTNNPLERowsPreparedInput(tensor, tuple(int(token) for token in tokens), contexts)

    def _distributed_group_norm_rows(self, tensor, weight, rows: int):
        self._validate_rows_residual(tensor, rows, label="PLE rows group-norm input")
        self.mesh_contract.validate_tensor(weight, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        stats = ttnn.rms_norm_pre_all_gather(
            tensor,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        stats = ttnn.reshape(stats, (1, rows, RESIDUAL_BRANCHES, 32))
        self.mesh_contract.validate_tensor(stats, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        gathered = ttnn.all_gather(stats, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(stats)
        self.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
        normalized = ttnn.rms_norm_post_all_gather(
            tensor,
            gathered,
            epsilon=RMS_NORM_EPS,
            weight=weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
        )
        _deallocate(gathered)
        self._validate_rows_residual(normalized, rows, label="PLE rows group-norm output")
        return normalized

    def _project_rows(self, embedding_tile, rows: int):
        full_embedding = ttnn.all_gather(
            embedding_tile, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        _deallocate(embedding_tile)
        self.mesh_contract.validate_tensor(full_embedding, placement=TensorPlacement.REPLICATED)
        if _shape(full_embedding) != (1, 1, rows, EMBEDDING_WIDTH):
            raise RuntimeError(
                f"PLE rows embedding all-gather gave {_shape(full_embedding)}, expected {(1, 1, rows, EMBEDDING_WIDTH)}"
            )
        key_flat = ttnn.linear(
            full_embedding,
            self.weights.key,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        value_flat = ttnn.linear(
            full_embedding,
            self.weights.value,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(full_embedding)
        self.mesh_contract.validate_tensor(key_flat, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        self.mesh_contract.validate_tensor(value_flat, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(key_flat) != (1, 1, rows, RESIDUAL_WIDTH // TP_SIZE):
            raise RuntimeError(f"PLE rows key projection has local shape {_shape(key_flat)}")
        if _shape(value_flat) != (1, 1, rows, LOCAL_HIDDEN_SIZE):
            raise RuntimeError(f"PLE rows value projection has local shape {_shape(value_flat)}")
        # Token t's 2560 key columns become its [4,640] branch block; the value row becomes [1,640] per token.  At one
        # row the value keeps its shape: a same-shape reshape is a view of its input on this runtime (releasing the
        # input freed the view: PLE lanes at B = 1, 2026-09-04), so the row is used as it is.
        key = ttnn.reshape(key_flat, _rows_residual_shape(rows))
        value_shape = (1, rows, 1, LOCAL_HIDDEN_SIZE)
        value = value_flat if _shape(value_flat) == value_shape else ttnn.reshape(value_flat, value_shape)
        _deallocate(key_flat, None if value is value_flat else value_flat)
        self._validate_rows_residual(key, rows, label="PLE rows key projection")
        if _shape(value) != (1, rows, 1, LOCAL_HIDDEN_SIZE):
            raise RuntimeError(
                f"PLE rows value has local shape {_shape(value)}, expected {(1, rows, 1, LOCAL_HIDDEN_SIZE)}"
            )
        return key, value

    def _gate_rows(self, key, query, value, rows: int):
        dram = ttnn.DRAM_MEMORY_CONFIG
        key_norm = self._distributed_group_norm_rows(key, self.weights.norm_key, rows)
        query_norm = self._distributed_group_norm_rows(query, self.weights.norm_query, rows)
        key_global = ttnn.all_gather(key_norm, dim=3, cluster_axis=TP_AXIS, memory_config=dram)
        query_global = ttnn.all_gather(query_norm, dim=3, cluster_axis=TP_AXIS, memory_config=dram)
        _deallocate(key_norm, query_norm)
        global_shape = (1, rows, RESIDUAL_BRANCHES, HIDDEN_SIZE)
        for label, tensor in (("key", key_global), ("query", query_global)):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
            if _shape(tensor) != global_shape:
                raise RuntimeError(f"PLE rows global {label} norm has shape {_shape(tensor)}, expected {global_shape}")
        key_fp32 = ttnn.typecast(key_global, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        query_fp32 = ttnn.typecast(query_global, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(key_global, query_global)
        products = ttnn.multiply(key_fp32, query_fp32, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(key_fp32, query_fp32)
        unscaled_gate = ttnn.sum(
            products, dim=3, keepdim=True, memory_config=dram, compute_kernel_config=self.compute_config
        )
        _deallocate(products)
        gate = ttnn.multiply(unscaled_gate, HIDDEN_SIZE**-0.5, memory_config=dram)
        _deallocate(unscaled_gate)
        self.mesh_contract.validate_tensor(gate, placement=TensorPlacement.REPLICATED)
        if _shape(gate) != (1, rows, RESIDUAL_BRANCHES, 1):
            raise RuntimeError(f"PLE rows gate has shape {_shape(gate)}, expected {(1, rows, RESIDUAL_BRANCHES, 1)}")

        magnitude = ttnn.abs(gate, memory_config=dram)
        bounded = ttnn.clamp(magnitude, min=1.0e-6, memory_config=dram)
        root = ttnn.sqrt(bounded, memory_config=dram)
        direction = ttnn.sign(gate, memory_config=dram)
        _deallocate(magnitude, bounded, gate)
        transformed = ttnn.multiply(root, direction, memory_config=dram)
        _deallocate(root, direction)
        coefficient = ttnn.sigmoid(transformed, memory_config=dram)
        _deallocate(transformed)
        value_branches = ttnn.repeat(value, (1, 1, RESIDUAL_BRANCHES, 1), memory_config=dram)
        _deallocate(value)
        gated = ttnn.multiply(value_branches, coefficient, memory_config=dram)
        _deallocate(value_branches, coefficient)
        self._validate_rows_residual(gated, rows, label="PLE rows gated value")
        return gated

    def _conv_window_rows(self, rows_state: Qwen38TTNNPLERowsState):
        """``[history | normalized]`` on the token dim: output row j reads window rows j, j+3, j+6, j+9."""

        window = ttnn.concat([rows_state.history, rows_state.normalized], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._validate_rows_residual(window, CONV_STATE_LENGTH + rows_state.rows, label="PLE rows conv window")
        return window

    def _convolve_rows(self, rows_state: Qwen38TTNNPLERowsState):
        rows = rows_state.rows
        window = self._conv_window_rows(rows_state)
        pieces = [
            ttnn.slice(
                window,
                (0, tap * CONV_DILATION, 0, 0),
                (1, tap * CONV_DILATION + rows, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for tap in range(CONV_KERNEL_SIZE)
        ]
        _deallocate(window)
        convolution = ttnn.multiply(pieces[0], self.weights.conv_taps[0], memory_config=ttnn.L1_MEMORY_CONFIG)
        for piece, tap in zip(pieces[1:], self.weights.conv_taps[1:]):
            previous = convolution
            convolution = ttnn.mac(piece, tap, previous)
            if _tensor_key(convolution) != _tensor_key(previous):
                _deallocate(previous)
        _deallocate(*pieces)
        convolution = ttnn.silu(convolution, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._validate_rows_residual(convolution, rows, label="PLE rows convolution")
        return convolution

    def forward_prepared_rows(
        self, residual_rows, prepared: Qwen38TTNNPLERowsPreparedInput, rows_state: Qwen38TTNNPLERowsState
    ) -> Qwen38TTNNPLERowsResult:
        """The residual delta for ``rows`` tokens ``[1,rows,4,640]``; the history is not advanced (see ``commit_rows``).

        The host context is not checked here (the verify body has no host token): the caller prepared the
        rows from ``rows_state.token_context`` and commits ``prepared.contexts[accepted + 1]`` afterwards.
        """

        rows_state.validate()
        rows = rows_state.rows
        self._validate_rows_residual(residual_rows, rows, label="PLE rows residual input")
        if not isinstance(prepared, Qwen38TTNNPLERowsPreparedInput) or not prepared.active:
            raise TypeError("PLE rows input must be a live Qwen38TTNNPLERowsPreparedInput")
        self._validate_prepared_rows(prepared.embedding_rows, rows, label="prepared PLE rows")
        embedding_tile = ttnn.to_layout(
            prepared.embedding_rows, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        embedding_tile.update_tensor_topology(prepared.embedding_rows.tensor_topology())
        self.mesh_contract.validate_tensor(embedding_tile, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(embedding_tile) != (1, 1, rows, LOCAL_HIDDEN_SIZE) or embedding_tile.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"PLE rows tilize produced {embedding_tile.layout} {_shape(embedding_tile)}")
        key, value = self._project_rows(embedding_tile, rows)
        gated = self._gate_rows(key, residual_rows, value, rows)
        _deallocate(key)
        normalized = self._distributed_group_norm_rows(gated, self.weights.norm_conv, rows)
        # The conv input rows are kept for the commit's history select.
        _copy_inplace(normalized, rows_state.normalized, label="PLE rows normalized")
        _deallocate(normalized)
        convolution = self._convolve_rows(rows_state)
        output = ttnn.add(gated, convolution, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(gated, convolution)
        self._validate_rows_residual(output, rows, label="PLE rows residual delta")
        return Qwen38TTNNPLERowsResult(output, rows_state)

    def inject_rows(self, residual_rows, prepared: Qwen38TTNNPLERowsPreparedInput, rows_state: Qwen38TTNNPLERowsState):
        """PLE injection for ``rows`` branch-major residual rows ``[1,4,rows,640]``; the input is consumed.

        The same permute pair as the layer's 1-row injection (``_apply_ple``), with the tokens on the
        branch-row layout's dim 1, around :meth:`forward_prepared_rows`; returns ``residual + delta``.
        """

        rows = rows_state.rows
        expected = (1, RESIDUAL_BRANCHES, rows, LOCAL_HIDDEN_SIZE)
        if _shape(residual_rows) != expected or residual_rows.dtype != ttnn.bfloat16:
            raise ValueError(
                f"rows residual must be BF16 TILE {expected}, got {_shape(residual_rows)} {residual_rows.dtype}"
            )
        self.mesh_contract.validate_tensor(residual_rows, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        token_rows = ttnn.permute(residual_rows, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        result = self.forward_prepared_rows(token_rows, prepared, rows_state)
        delta = ttnn.permute(result.residual_delta, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(token_rows, result.residual_delta)
        if _shape(delta) != expected:
            raise RuntimeError(f"PLE rows delta has shape {_shape(delta)}, expected {expected}")
        injected = ttnn.add(residual_rows, delta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(residual_rows, delta)
        if _shape(injected) != expected:
            raise RuntimeError(f"PLE-injected rows have shape {_shape(injected)}, expected {expected}")
        self.mesh_contract.validate_tensor(injected, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        return injected

    def commit_rows(self, rows_state: Qwen38TTNNPLERowsState, selectors: Qwen38TTNNRowsSelectors) -> None:
        """``history <- window[c : c + 9]`` with ``c = accepted + 1`` (exact one-hot multiply/add select)."""

        rows_state.validate()
        rows = rows_state.rows
        selectors.validate(rows)
        window = self._conv_window_rows(rows_state)
        terms = []
        for committed in range(1, rows + 1):
            candidate = ttnn.slice(
                window,
                (0, committed, 0, 0),
                (1, committed + CONV_STATE_LENGTH, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if rows == 1:
                landed = ttnn.multiply(candidate, selectors.onehot_bf16[0], output_tensor=rows_state.history)
                if landed is not None and _tensor_key(landed) != _tensor_key(rows_state.history):
                    raise RuntimeError("PLE rows history select did not land in its persistent buffer")
            else:
                terms.append(
                    ttnn.multiply(
                        candidate, selectors.onehot_bf16[committed - 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                )
            _deallocate(candidate)
        _deallocate(window)
        if rows == 1:
            return
        acc = terms[0]
        for term in terms[1:-1]:
            previous = acc
            acc = ttnn.add(previous, term, memory_config=ttnn.DRAM_MEMORY_CONFIG, fast_and_approximate_mode=False)
            _deallocate(previous, term)
        landed = ttnn.add(acc, terms[-1], output_tensor=rows_state.history, fast_and_approximate_mode=False)
        if landed is not None and _tensor_key(landed) != _tensor_key(rows_state.history):
            raise RuntimeError("PLE rows history select did not land in its persistent buffer")
        _deallocate(acc, terms[-1])

    # ------------------------------------------------------------------ lane rows path (MTP lanes verify)
    # B lanes of R rows in one 32-row tile, lane-major (row u*R + j = lane u's row j): the projection, gate and
    # group norm are the rows forms over the B*R token rows (row-independent); the dilated FIR runs on every lane's
    # own window ``[history_u | normalized_u]`` through the batch axis; the commit selects lane u's next history with
    # the pass's per-lane one-hots (the KEEP candidate for a lane that commits nothing).

    def allocate_lane_rows_state(self, lanes: int, rows: int) -> Qwen38TTNNPLELaneRowsState:
        return Qwen38TTNNPLELaneRowsState.allocate(self.mesh_device, self.mesh_contract, lanes=lanes, rows=rows)

    def _lane_rows_window(self, lanes_state: Qwen38TTNNPLELaneRowsState):
        """``[history | normalized]`` per lane on the token dim: ``[B, 9 + R, 4, 640]``."""

        window = ttnn.concat(
            [lanes_state.history, lanes_state.normalized], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        expected = (lanes_state.lanes, CONV_STATE_LENGTH + lanes_state.rows, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE)
        if _shape(window) != expected:
            raise RuntimeError(f"PLE lane rows conv window has shape {_shape(window)}, expected {expected}")
        return window

    def _convolve_rows_lanes(self, lanes_state: Qwen38TTNNPLELaneRowsState):
        """The rows path's dilated FIR on every lane's window: tap t reads window rows ``3t .. 3t + R - 1`` of every
        lane (one slice over the batch), the same tap arithmetic as ``_convolve_rows``; ``[B, R, 4, 640]``."""

        lanes, rows = lanes_state.lanes, lanes_state.rows
        window = self._lane_rows_window(lanes_state)
        pieces = [
            ttnn.slice(
                window,
                (0, tap * CONV_DILATION, 0, 0),
                (lanes, tap * CONV_DILATION + rows, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for tap in range(CONV_KERNEL_SIZE)
        ]
        _deallocate(window)
        convolution = ttnn.multiply(pieces[0], self.weights.conv_taps[0], memory_config=ttnn.L1_MEMORY_CONFIG)
        for piece, tap in zip(pieces[1:], self.weights.conv_taps[1:]):
            previous = convolution
            convolution = ttnn.mac(piece, tap, previous)
            if _tensor_key(convolution) != _tensor_key(previous):
                _deallocate(previous)
        _deallocate(*pieces)
        convolution = ttnn.silu(convolution, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        expected = (lanes, rows, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE)
        if _shape(convolution) != expected:
            raise RuntimeError(f"PLE lane rows convolution has shape {_shape(convolution)}, expected {expected}")
        return convolution

    def forward_prepared_rows_lanes(
        self, residual_rows, prepared: Qwen38TTNNPLERowsPreparedInput, lanes_state: Qwen38TTNNPLELaneRowsState
    ):
        """The residual delta for the B*R lane-major token rows ``[1,B*R,4,640]``; the histories are not advanced
        (see :meth:`commit_rows_lanes`).  ``prepared`` holds the B*R rows looked up per lane from the lanes' own
        contexts (the caller's host work); the conv input rows land in ``normalized`` per lane through a view."""

        lanes_state.validate()
        lanes, rows = lanes_state.lanes, lanes_state.rows
        total = lanes * rows
        self._validate_rows_residual(residual_rows, total, label="PLE lane rows residual input")
        if not isinstance(prepared, Qwen38TTNNPLERowsPreparedInput) or not prepared.active:
            raise TypeError("PLE lane rows input must be a live Qwen38TTNNPLERowsPreparedInput")
        self._validate_prepared_rows(prepared.embedding_rows, total, label="prepared PLE lane rows")
        embedding_tile = ttnn.to_layout(
            prepared.embedding_rows, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        embedding_tile.update_tensor_topology(prepared.embedding_rows.tensor_topology())
        self.mesh_contract.validate_tensor(embedding_tile, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(embedding_tile) != (1, 1, total, LOCAL_HIDDEN_SIZE) or embedding_tile.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"PLE lane rows tilize produced {embedding_tile.layout} {_shape(embedding_tile)}")
        key, value = self._project_rows(embedding_tile, total)
        gated = self._gate_rows(key, residual_rows, value, total)
        _deallocate(key)
        normalized = self._distributed_group_norm_rows(gated, self.weights.norm_conv, total)
        # [1, B*R, 4, 640] -> [B, R, 4, 640]: whole token tiles in the same order, the same pages (a view).
        per_lane = ttnn.experimental.view(normalized, (lanes, rows, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE))
        per_lane.update_tensor_topology(normalized.tensor_topology())
        _copy_inplace(per_lane, lanes_state.normalized, label="PLE lane rows normalized")
        _deallocate(normalized)
        convolution = self._convolve_rows_lanes(lanes_state)
        rows_conv = ttnn.experimental.view(convolution, (1, total, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE))
        rows_conv.update_tensor_topology(convolution.tensor_topology())
        output = ttnn.add(gated, rows_conv, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(gated, convolution)
        self._validate_rows_residual(output, total, label="PLE lane rows residual delta")
        return output

    def inject_rows_lanes(
        self, residual_rows, prepared: Qwen38TTNNPLERowsPreparedInput, lanes_state: Qwen38TTNNPLELaneRowsState
    ):
        """PLE injection for the B*R branch-major lane-major residual rows ``[1,4,B*R,640]``; the input is consumed."""

        total = lanes_state.lanes * lanes_state.rows
        expected = (1, RESIDUAL_BRANCHES, total, LOCAL_HIDDEN_SIZE)
        if _shape(residual_rows) != expected or residual_rows.dtype != ttnn.bfloat16:
            raise ValueError(
                f"lane rows residual must be BF16 TILE {expected}, got {_shape(residual_rows)} {residual_rows.dtype}"
            )
        self.mesh_contract.validate_tensor(residual_rows, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        token_rows = ttnn.permute(residual_rows, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        delta_rows = self.forward_prepared_rows_lanes(token_rows, prepared, lanes_state)
        delta = ttnn.permute(delta_rows, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(token_rows, delta_rows)
        if _shape(delta) != expected:
            raise RuntimeError(f"PLE lane rows delta has shape {_shape(delta)}, expected {expected}")
        injected = ttnn.add(residual_rows, delta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(residual_rows, delta)
        if _shape(injected) != expected:
            raise RuntimeError(f"PLE-injected lane rows have shape {_shape(injected)}, expected {expected}")
        self.mesh_contract.validate_tensor(injected, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        return injected

    def commit_rows_lanes(
        self, lanes_state: Qwen38TTNNPLELaneRowsState, selectors: Qwen38TTNNRowsSelectorsLanes
    ) -> None:
        """``history_u <- window_u[c_u : c_u + 9]`` for every lane (exact one-hot multiply/add select over the R + 1
        candidates; ``c_u = 0`` keeps the history)."""

        lanes_state.validate()
        lanes, rows = lanes_state.lanes, lanes_state.rows
        selectors.validate(lanes, rows)
        window = self._lane_rows_window(lanes_state)
        terms = []
        for committed in range(rows + 1):
            candidate = ttnn.slice(
                window,
                (0, committed, 0, 0),
                (lanes, committed + CONV_STATE_LENGTH, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            terms.append(
                ttnn.multiply(candidate, selectors.onehot_bf16[committed], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            )
            _deallocate(candidate)
        _deallocate(window)
        acc = terms[0]
        for term in terms[1:-1]:
            previous = acc
            acc = ttnn.add(previous, term, memory_config=ttnn.DRAM_MEMORY_CONFIG, fast_and_approximate_mode=False)
            _deallocate(previous, term)
        landed = ttnn.add(acc, terms[-1], output_tensor=lanes_state.history, fast_and_approximate_mode=False)
        if landed is not None and _tensor_key(landed) != _tensor_key(lanes_state.history):
            raise RuntimeError("PLE lane rows history select did not land in its persistent buffer")
        _deallocate(acc, terms[-1])

    def commit_rows_full(self, rows_state: Qwen38TTNNPLERowsState) -> None:
        """``history <- normalized[rows - 9 : rows]``: every row committed (the long chunk), one slice and one copy."""

        rows_state.validate()
        rows = rows_state.rows
        if rows < CONV_STATE_LENGTH:
            raise ValueError(f"a full commit needs at least {CONV_STATE_LENGTH} rows, the state has {rows}")
        history = ttnn.slice(
            rows_state.normalized,
            (0, rows - CONV_STATE_LENGTH, 0, 0),
            (1, rows, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._validate_rows_residual(history, CONV_STATE_LENGTH, label="PLE rows full-commit history")
        _copy_inplace(history, rows_state.history, label="PLE rows history")
        _deallocate(history)

    @staticmethod
    def commit_rows_host(
        rows_state: Qwen38TTNNPLERowsState, prepared: Qwen38TTNNPLERowsPreparedInput, accepted: int
    ) -> None:
        """Host bookkeeping after the readback: the committed stream's n-gram context is ``contexts[accepted + 1]``."""

        if isinstance(accepted, bool) or type(accepted) is not int or not 0 <= accepted < rows_state.rows:
            raise ValueError(f"accepted must be an int in [0,{rows_state.rows}), got {accepted!r}")
        if len(prepared.contexts) != rows_state.rows + 1 or prepared.contexts[0] != rows_state.token_context:
            raise RuntimeError("prepared PLE rows were not looked up from this state's host context")
        rows_state.token_context = prepared.contexts[accepted + 1]
        rows_state.validate()

    # ------------------------------------------------------------------ lanes path (batched decode)
    #
    # ``forward_prepared_lanes`` is the 1-row body on ``[1,lanes,4,640]`` (token-major: dim 1 is the lane, each
    # lane keeps its [4,640] branch block): the rows path's projection, gate and norm over ``lanes`` rows, then
    # the 1-row dilated conv on the nine per-lane slots (taps at slots 0, 3, 6 and the new row) with the same
    # shift; lane u is bitwise the 1-row path on user u's stream.  The host looks every lane's token up with
    # that lane's own context in one batched read (:meth:`host_lanes`).

    def allocate_lanes_state(self, lanes: int) -> Qwen38TTNNPLELanesState:
        return Qwen38TTNNPLELanesState.allocate(self.mesh_device, self.mesh_contract, lanes=lanes)

    def host_lanes(
        self, tokens: Sequence[int], contexts: Sequence[tuple[int, int] | None]
    ) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
        """The n-gram rows of lane u's ``tokens[u]`` under ``contexts[u]`` as BF16 ``[1,1,lanes,2560]`` and the next
        context per lane.  Same hash, bytes and row order as the 1-row decode step, lane by lane."""

        lanes = require_lane_count(len(tokens), label="PLE lanes token count")
        payload, next_contexts = self.resident_lookup.lookup_lanes(
            [int(token) for token in tokens], _validate_lane_contexts(contexts, lanes, label="PLE lanes contexts")
        )
        host = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(1, 1, lanes, EMBEDDING_WIDTH)
        return host, next_contexts

    def prepare_lanes_input(
        self, tokens: Sequence[int], lanes_state: Qwen38TTNNPLELanesState
    ) -> Qwen38TTNNPLELanesPreparedInput:
        """Look up and upload every lane's token (host work, before capture or between replays)."""

        lanes_state.validate()
        if len(tokens) != lanes_state.lanes:
            raise ValueError(f"PLE lanes input needs {lanes_state.lanes} tokens (one per lane), got {len(tokens)}")
        host, next_contexts = self.host_lanes(tokens, lanes_state.token_contexts)
        tensor = ttnn.from_torch(
            host.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
        )
        self._validate_prepared_rows(tensor, lanes_state.lanes, label="PLE lanes upload")
        return Qwen38TTNNPLELanesPreparedInput(
            tensor, tuple(int(token) for token in tokens), tuple(lanes_state.token_contexts), next_contexts
        )

    def _convolve_lanes(self, normalized, lanes_state: Qwen38TTNNPLELanesState):
        """The 1-row ``_convolve`` on the per-lane slots: taps at slots 0, 3, 6 and the new rows, then the shift."""

        rows = (lanes_state.conv[0], lanes_state.conv[3], lanes_state.conv[6], normalized)
        convolution = ttnn.multiply(rows[0], self.weights.conv_taps[0], memory_config=ttnn.L1_MEMORY_CONFIG)
        for row, tap in zip(rows[1:], self.weights.conv_taps[1:]):
            previous = convolution
            convolution = ttnn.mac(row, tap, previous)
            if _tensor_key(convolution) != _tensor_key(previous):
                _deallocate(previous)
        convolution = ttnn.silu(convolution, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._validate_rows_residual(convolution, lanes_state.lanes, label="PLE lanes convolution")
        for index in range(CONV_STATE_LENGTH - 1):
            _copy_inplace(lanes_state.conv[index + 1], lanes_state.conv[index], label=f"PLE lanes conv shift {index}")
        _copy_inplace(normalized, lanes_state.conv[-1], label="PLE lanes conv newest rows")
        return convolution

    def forward_prepared_lanes(
        self, residual_lanes, prepared: Qwen38TTNNPLELanesPreparedInput, lanes_state: Qwen38TTNNPLELanesState
    ) -> Any:
        """The residual delta ``[1,lanes,4,640]`` for one token per lane; the slots shift and the contexts advance."""

        if self._fused_forward_prepared_lanes is not None:
            return self._fused_forward_prepared_lanes(residual_lanes, prepared, lanes_state)
        lanes_state.validate()
        lanes = lanes_state.lanes
        self._validate_rows_residual(residual_lanes, lanes, label="PLE lanes residual input")
        if not isinstance(prepared, Qwen38TTNNPLELanesPreparedInput) or not prepared.active:
            raise TypeError("PLE lanes input must be a live Qwen38TTNNPLELanesPreparedInput")
        if prepared.source_contexts != lanes_state.token_contexts:
            raise RuntimeError(
                f"prepared PLE lanes were looked up from contexts {prepared.source_contexts}, the state holds "
                f"{lanes_state.token_contexts}"
            )
        self._validate_prepared_rows(prepared.embedding_rows, lanes, label="prepared PLE lanes")
        embedding_tile = ttnn.to_layout(
            prepared.embedding_rows, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        embedding_tile.update_tensor_topology(prepared.embedding_rows.tensor_topology())
        self.mesh_contract.validate_tensor(embedding_tile, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(embedding_tile) != (1, 1, lanes, LOCAL_HIDDEN_SIZE) or embedding_tile.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(
                f"PLE lanes tilize produced {embedding_tile.layout} {_shape(embedding_tile)}, expected "
                f"{ttnn.TILE_LAYOUT} {(1, 1, lanes, LOCAL_HIDDEN_SIZE)}"
            )
        key, value = self._project_rows(embedding_tile, lanes)
        gated = self._gate_rows(key, residual_lanes, value, lanes)
        _deallocate(key)
        normalized = self._distributed_group_norm_rows(gated, self.weights.norm_conv, lanes)
        convolution = self._convolve_lanes(normalized, lanes_state)
        _deallocate(normalized)
        output = ttnn.add(gated, convolution, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(gated, convolution)
        self._validate_rows_residual(output, lanes, label="PLE lanes residual delta")
        lanes_state.token_contexts = prepared.next_contexts
        lanes_state.validate()
        return output

    def inject_lanes(
        self, residual_lanes, prepared: Qwen38TTNNPLELanesPreparedInput, lanes_state: Qwen38TTNNPLELanesState
    ):
        """PLE injection for ``lanes`` branch-major residual rows ``[1,4,lanes,640]``; the input is consumed."""

        lanes = lanes_state.lanes
        expected = (1, RESIDUAL_BRANCHES, lanes, LOCAL_HIDDEN_SIZE)
        if _shape(residual_lanes) != expected or residual_lanes.dtype != ttnn.bfloat16:
            raise ValueError(
                f"lanes residual must be BF16 TILE {expected}, got {_shape(residual_lanes)} {residual_lanes.dtype}"
            )
        self.mesh_contract.validate_tensor(residual_lanes, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        lane_rows = ttnn.permute(residual_lanes, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        delta_rows = self.forward_prepared_lanes(lane_rows, prepared, lanes_state)
        delta = ttnn.permute(delta_rows, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(lane_rows, delta_rows)
        if _shape(delta) != expected:
            raise RuntimeError(f"PLE lanes delta has shape {_shape(delta)}, expected {expected}")
        injected = ttnn.add(residual_lanes, delta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(residual_lanes, delta)
        if _shape(injected) != expected:
            raise RuntimeError(f"PLE-injected lanes have shape {_shape(injected)}, expected {expected}")
        self.mesh_contract.validate_tensor(injected, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        return injected


def validate_ple_static_contract() -> None:
    actual = {
        "tp": TP_SIZE,
        "hidden": HIDDEN_SIZE,
        "local_hidden": LOCAL_HIDDEN_SIZE,
        "branches": RESIDUAL_BRANCHES,
        "residual": RESIDUAL_WIDTH,
        "embedding": EMBEDDING_WIDTH,
        "ngram": NGRAM_SIZE,
        "kernel": CONV_KERNEL_SIZE,
        "dilation": CONV_DILATION,
        "state": CONV_STATE_LENGTH,
    }
    expected = {
        "tp": 4,
        "hidden": 2560,
        "local_hidden": 640,
        "branches": 4,
        "residual": 10240,
        "embedding": 2560,
        "ngram": 3,
        "kernel": 4,
        "dilation": 3,
        "state": 9,
    }
    if actual != expected:
        raise RuntimeError(f"Qwen3.8 PLE static contract drifted: {actual} != {expected}")


__all__ = [
    "ngram_token_ids_decode_step",
    "Qwen38ResidentPLELookup",
    "Qwen38TTNNPLE",
    "Qwen38TTNNPLELanesPreparedInput",
    "Qwen38TTNNPLELanesState",
    "Qwen38TTNNPLEPreparedInput",
    "Qwen38TTNNPLEResult",
    "Qwen38TTNNPLERowsPreparedInput",
    "Qwen38TTNNPLERowsResult",
    "Qwen38TTNNPLERowsState",
    "Qwen38TTNNPLEWeights",
    "Qwen38TTNNPLEState",
    "Qwen38TTNNPLESnapshot",
    "validate_ple_static_contract",
]
