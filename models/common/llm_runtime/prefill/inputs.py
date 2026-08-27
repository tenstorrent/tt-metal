# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill host/device input staging and replay refresh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

import ttnn
from models.common.llm_runtime.prefill.plan import PrefillChunk, PrefillRequest
from models.common.llm_runtime.prefill.sampling_helpers import _TILE_SIZE
from models.common.llm_runtime.tensor_resources import (
    attach_cleanup_failures,
    best_effort_deallocate_owned_tensors,
    raise_cleanup_failures,
)


@dataclass(frozen=True)
class PrefillHostInputs:
    tokens: Any
    position_indices: Any
    page_table: Any
    chunk_page_table: Any | None
    chunk_start_idx: Any | None

    def values(self) -> tuple[Any, ...]:
        return (
            self.tokens,
            self.position_indices,
            self.page_table,
            self.chunk_page_table,
            self.chunk_start_idx,
        )


@dataclass(frozen=True)
class PrefillDeviceInputs:
    tokens: Any
    rotary_cos: Any
    rotary_sin: Any
    page_table: Any
    chunk_page_table: Any | None
    position_indices: Any
    chunk_start_idx: Any | None

    def model_values(self) -> tuple[Any, ...]:
        return (
            self.tokens,
            self.rotary_cos,
            self.rotary_sin,
            self.page_table,
            self.chunk_page_table,
            self.position_indices,
            self.chunk_start_idx,
        )

    def owned_tensor_values(self) -> tuple[Any, ...]:
        return self.model_values()


@dataclass(frozen=True)
class PrefillPositionInputs:
    slice_start: Any
    slice_end: Any
    row_index: Any

    def values(self) -> tuple[Any, ...]:
        return self.slice_start, self.slice_end, self.row_index

    def owned_tensor_values(self) -> tuple[Any, ...]:
        return self.values()


@dataclass(frozen=True)
class PrefillTraceInputs:
    """Named geometry for allocating one persistent trace input set."""

    chunk: PrefillChunk
    relative_last: int
    tokens: torch.Tensor
    start_pos: int
    chunk_page_table: torch.Tensor | None
    chunk_start_idx: int | None
    sequence_length: int


class PrefillInputStager:
    """Own TT prefill tensor geometry, allocation, and in-place refresh."""

    def __init__(
        self,
        *,
        model: Any,
        mesh_device: Any,
        release_transient: Callable[[Any], list[BaseException]],
    ) -> None:
        self.model = model
        self.mesh_device = mesh_device
        self._release_transient = release_transient

    def trace_inputs(self, request: PrefillRequest) -> PrefillTraceInputs:
        chunk = request.chunks[0]
        if request.uses_chunked_prefill:
            final_chunk = request.chunks[-1]
            relative_last = (request.last_token_indices[0] - final_chunk.chunk_start_idx) % final_chunk.chunk_size
            tokens = request.tokens[:, chunk.token_slice]
            start_pos = chunk.chunk_start_idx
            chunk_page_table = chunk.chunk_page_table
            chunk_start_idx = chunk.chunk_start_idx
            sequence_length = chunk.chunk_size
        else:
            relative_last = max(
                last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens)
            )
            tokens = request.tokens
            start_pos = 0
            chunk_page_table = None
            chunk_start_idx = None
            sequence_length = request.padded_sequence_length
        return PrefillTraceInputs(
            chunk=chunk,
            relative_last=relative_last,
            tokens=tokens,
            start_pos=start_pos,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            sequence_length=sequence_length,
        )

    def stage_step(
        self,
        request: PrefillRequest,
        chunk: PrefillChunk,
        final_relative_last: int,
    ) -> tuple[PrefillDeviceInputs, PrefillPositionInputs]:
        chunked = request.uses_chunked_prefill
        host_inputs = self.prepare_host_inputs(
            request.tokens[:, chunk.token_slice],
            request.page_table,
            start_pos=chunk.chunk_start_idx if chunked else 0,
            chunk_page_table=chunk.chunk_page_table if chunked else None,
            chunk_start_idx=chunk.chunk_start_idx if chunked else None,
            last_token_idx=max(request.last_token_indices),
        )
        device_inputs = None
        position_inputs = None
        try:
            device_inputs = self.stage_device_inputs(host_inputs)
            position_values = allocate_device_tensors(
                self.prepare_position_inputs_host(final_relative_last, chunk.chunk_size).values(),
                mesh_device=self.mesh_device,
            )
            position_inputs = PrefillPositionInputs(*position_values)
        except BaseException as primary:
            failures = self._release_transient((device_inputs, position_inputs))
            attach_cleanup_failures(primary, failures)
            raise
        return device_inputs, position_inputs

    def prepare_host_inputs(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        start_pos: int = 0,
        chunk_page_table: torch.Tensor | None = None,
        chunk_start_idx: int | None = None,
        last_token_idx: int | None = None,
    ) -> PrefillHostInputs:
        if tokens.ndim != 2:
            raise ValueError("prefill tokens must be rank 2")
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        tokens_tt = ttnn.from_torch(
            tokens.reshape(1, 1, 1, -1),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        rope = self.model.rope_setup
        rope.load_device_weights()
        matrix_length = int(rope.cos_matrix.shape[2])
        if matrix_length <= 0:
            raise ValueError("rotary position table must not be empty")
        start_pos = int(start_pos)
        sequence_length = int(tokens.shape[-1])
        if start_pos < 0:
            raise ValueError("prefill start position must be nonnegative")
        if last_token_idx is not None and int(last_token_idx) + 1 > matrix_length:
            raise ValueError(f"Sequence length {int(last_token_idx) + 1} exceeds rotary capacity {matrix_length}")
        position_indices = torch.arange(start_pos, start_pos + sequence_length, dtype=torch.long).clamp(
            max=matrix_length - 1
        )
        position_indices_tt = ttnn.from_torch(
            position_indices.reshape(1, -1),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        chunk_tt = (
            ttnn.from_torch(
                chunk_page_table,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            if chunk_page_table is not None
            else None
        )
        chunk_start_tt = (
            ttnn.from_torch(
                torch.tensor([int(chunk_start_idx)], dtype=torch.int32),
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            if chunk_start_idx is not None
            else None
        )
        return PrefillHostInputs(tokens_tt, position_indices_tt, page_table_tt, chunk_tt, chunk_start_tt)

    def prepare_position_inputs_host(self, relative_last: int, sequence_length: int) -> PrefillPositionInputs:
        relative_last = int(relative_last)
        sequence_length = int(sequence_length)
        if relative_last < 0 or relative_last >= sequence_length:
            raise ValueError("prefill last-token position must fall within the padded sequence")
        block_start = (relative_last // _TILE_SIZE) * _TILE_SIZE
        hidden_width = int(self.model.config.dim)
        bounds = ((0, 0, block_start, 0), (1, 1, block_start + _TILE_SIZE, hidden_width))
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        slice_bounds = tuple(
            ttnn.from_torch(
                torch.tensor(bound, dtype=torch.int32),
                device=None,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            for bound in bounds
        )
        row_index = ttnn.from_torch(
            torch.tensor([[relative_last % _TILE_SIZE]], dtype=torch.int32),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        return PrefillPositionInputs(slice_bounds[0], slice_bounds[1], row_index)

    def stage_device_inputs(self, host_inputs: PrefillHostInputs) -> PrefillDeviceInputs:
        raw_inputs = None
        rot_mats = None
        try:
            raw_inputs = allocate_device_tensors(host_inputs.values(), mesh_device=self.mesh_device)
            prepare_rot_mats = getattr(self.model, "prepare_prefill_rot_mats", None)
            if not callable(prepare_rot_mats):
                raise TypeError("model must provide prepare_prefill_rot_mats()")
            rot_mats = tuple(prepare_rot_mats(raw_inputs[1]))
            if len(rot_mats) != 2:
                raise ValueError("prepare_prefill_rot_mats() must return cosine and sine tensors")
        except BaseException as primary:
            failures = self._release_transient((rot_mats, raw_inputs))
            attach_cleanup_failures(primary, failures)
            raise
        return PrefillDeviceInputs(
            tokens=raw_inputs[0],
            rotary_cos=rot_mats[0],
            rotary_sin=rot_mats[1],
            page_table=raw_inputs[2],
            chunk_page_table=raw_inputs[3],
            position_indices=raw_inputs[1],
            chunk_start_idx=raw_inputs[4],
        )

    def refresh_regular_device_inputs(self, request: PrefillRequest, device_inputs: PrefillDeviceInputs) -> None:
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        host_tokens = ttnn.from_torch(
            request.tokens.reshape(1, 1, 1, -1),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        host_page_table = ttnn.from_torch(
            request.page_table,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        copy_into_device_tensors((host_tokens, host_page_table), (device_inputs.tokens, device_inputs.page_table))

    def refresh_chunk_device_inputs(
        self,
        request: PrefillRequest,
        chunk: PrefillChunk,
        device_inputs: PrefillDeviceInputs,
    ) -> None:
        host_inputs = self.prepare_host_inputs(
            request.tokens[:, chunk.token_slice],
            request.page_table,
            start_pos=chunk.chunk_start_idx,
            chunk_page_table=chunk.chunk_page_table,
            chunk_start_idx=chunk.chunk_start_idx,
            last_token_idx=max(request.last_token_indices),
        )
        copy_into_device_tensors(
            host_inputs.values(),
            (
                device_inputs.tokens,
                device_inputs.position_indices,
                device_inputs.page_table,
                device_inputs.chunk_page_table,
                device_inputs.chunk_start_idx,
            ),
        )
        self.copy_rotary_inputs(device_inputs)

    def copy_rotary_inputs(self, device_inputs: PrefillDeviceInputs) -> None:
        rot_mats = None
        try:
            rot_mats = tuple(self.model.prepare_prefill_rot_mats(device_inputs.position_indices))
            if len(rot_mats) != 2:
                raise ValueError("prepare_prefill_rot_mats() must return cosine and sine tensors")
            ttnn.copy(input_a=rot_mats[0], input_b=device_inputs.rotary_cos)
            ttnn.copy(input_a=rot_mats[1], input_b=device_inputs.rotary_sin)
        except BaseException as primary:
            failures = self._release_transient(rot_mats)
            attach_cleanup_failures(primary, failures)
            raise
        failures = self._release_transient(rot_mats)
        if failures:
            raise_cleanup_failures(failures)


def allocate_device_tensors(host_tensors: Sequence[Any], *, mesh_device: Any) -> list[Any]:
    """Allocate device tensors corresponding to one host tensor structure."""

    allocated = []
    try:
        for host_tensor in host_tensors:
            allocated.append(ttnn.to_device(host_tensor, device=mesh_device) if host_tensor is not None else None)
    except BaseException as primary:
        failures = best_effort_deallocate_owned_tensors(allocated)
        attach_cleanup_failures(primary, failures)
        raise
    return allocated


def copy_into_device_tensors(host_tensors: Sequence[Any], device_tensors: Sequence[Any]) -> Sequence[Any]:
    """Refresh an existing device tensor structure without allocating it."""

    if len(host_tensors) != len(device_tensors):
        raise ValueError("host/device tensor structures must have equal length")
    if any(
        (host_tensor is None) != (device_tensor is None)
        for host_tensor, device_tensor in zip(host_tensors, device_tensors)
    ):
        raise ValueError("host/device optional tensor structure changed")
    for host_tensor, device_tensor in zip(host_tensors, device_tensors):
        if host_tensor is None:
            continue
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
    return device_tensors
