# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Python-owned KDA graph orchestration.

The KDA layer keeps collective and state-flow decisions here; device leaves only
perform the bespoke kernels exposed by ``ttnn.experimental.kda``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.config import KDARecurrenceProgramConfig


def _output_memory_config(memory_config: ttnn.MemoryConfig | None) -> ttnn.MemoryConfig:
    return ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config


def _group_summary_memory_config(device: ttnn.Device, group_heads: int, key_dim: int) -> ttnn.MemoryConfig:
    worker_cores = ttnn.num_cores_to_corerangeset(
        group_heads,
        device.compute_with_storage_grid_size(),
        row_wise=True,
    )
    return ttnn.create_sharded_memory_config(
        (group_heads, key_dim, key_dim),
        core_grid=worker_cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def convolution_halo(
    projected_qkv: ttnn.Tensor,
    initial_carry: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    memory_config: ttnn.MemoryConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Exchange causal-convolution carries along the configured SP axis."""
    qkv_shape = tuple(projected_qkv.shape)
    carry_shape = tuple(initial_carry.shape)
    if sequence_parallel_axis not in (0, 1):
        raise ValueError(f"sequence_parallel_axis must be 0 or 1, got {sequence_parallel_axis}")
    if len(qkv_shape) != 3 or len(carry_shape) != 3:
        raise ValueError("KDA convolution halo expects rank-3 tensors")
    if qkv_shape[0] != carry_shape[0] or qkv_shape[2] != carry_shape[2]:
        raise ValueError("KDA convolution halo requires matching batch and channel dimensions")
    history = carry_shape[1]
    if history <= 0 or qkv_shape[1] < history:
        raise ValueError("KDA convolution halo requires 0 < history <= local T")
    if projected_qkv.dtype != initial_carry.dtype or projected_qkv.layout != initial_carry.layout:
        raise ValueError("KDA convolution halo requires matching dtypes and layouts")
    if history > ttnn.TILE_SIZE:
        raise ValueError("KDA convolution history must fit in one tile")

    mesh_device = projected_qkv.device()
    mesh_shape = tuple(mesh_device.shape)
    if len(mesh_shape) != 2 or mesh_shape[sequence_parallel_axis] <= 1:
        raise ValueError("KDA convolution halo requires a 2D mesh with SP > 1")
    sp_size = mesh_shape[sequence_parallel_axis]
    batch, local_sequence, channels = qkv_shape
    out_mem = _output_memory_config(memory_config)

    local_tail = ttnn.slice(
        projected_qkv,
        (0, local_sequence - history, 0),
        (batch, local_sequence, channels),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    padded_tail = ttnn.pad(
        local_tail,
        ((0, 0), (0, ttnn.TILE_SIZE - history), (0, 0)),
        value=0.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tiled_tail = ttnn.to_layout(padded_tail, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gathered_tails = ttnn.all_gather(
        tiled_tail,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    entry_carries = [initial_carry]
    for rank in range(sp_size - 1):
        tiled_rank_tail = ttnn.slice(
            gathered_tails,
            (0, rank * ttnn.TILE_SIZE, 0),
            (batch, (rank + 1) * ttnn.TILE_SIZE, channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rank_tail = ttnn.to_layout(tiled_rank_tail, ttnn.ROW_MAJOR_LAYOUT)
        entry_carries.append(ttnn.slice(rank_tail, (0, 0, 0), (batch, history, channels), memory_config=out_mem))
    replicated_entries = ttnn.concat(entry_carries, dim=1, memory_config=out_mem)
    partition_carry = ttnn.mesh_partition(
        replicated_entries,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=out_mem,
    )

    tiled_final_carry = ttnn.slice(
        gathered_tails,
        (0, (sp_size - 1) * ttnn.TILE_SIZE, 0),
        (batch, sp_size * ttnn.TILE_SIZE, channels),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    final_row_major = ttnn.to_layout(tiled_final_carry, ttnn.ROW_MAJOR_LAYOUT)
    final_carry = ttnn.slice(final_row_major, (0, 0, 0), (batch, history, channels), memory_config=out_mem)
    return partition_carry, final_carry


@dataclass(frozen=True)
class _RecurrenceGeometry:
    batch: int
    sequence: int
    heads: int
    key_dim: int
    value_dim: int
    chunk_size: int
    num_chunks: int

    @property
    def batch_heads(self) -> int:
        return self.batch * self.heads


@dataclass(frozen=True)
class _FlatRecurrenceInputs:
    q: ttnn.Tensor
    k: ttnn.Tensor
    v: ttnn.Tensor
    gate: ttnn.Tensor
    beta: ttnn.Tensor


@dataclass(frozen=True)
class _ChunkConstants:
    eye: ttnn.Tensor
    tril: ttnn.Tensor
    ones: ttnn.Tensor

    def as_kernel_args(self) -> tuple[ttnn.Tensor, ...]:
        return self.eye, self.tril, self.ones


@dataclass(frozen=True)
class _PreparedChunks:
    v_beta: ttnn.Tensor
    kd: ttnn.Tensor
    q_decay: ttnn.Tensor
    intra: ttnn.Tensor
    k_dec_t: ttnn.Tensor
    final_decay: ttnn.Tensor
    t_inv: ttnn.Tensor

    @classmethod
    def from_kernel_outputs(cls, outputs: list[ttnn.Tensor]) -> _PreparedChunks:
        if len(outputs) != 7:
            raise RuntimeError(f"KDA chunk preparation returned {len(outputs)} tensors, expected 7")
        return cls(*outputs)

    def as_kernel_args(self) -> tuple[ttnn.Tensor, ...]:
        return (
            self.v_beta,
            self.kd,
            self.q_decay,
            self.intra,
            self.k_dec_t,
            self.final_decay,
            self.t_inv,
        )


@dataclass(frozen=True)
class _ScanResult:
    output: ttnn.Tensor
    final_state: ttnn.Tensor


@dataclass(frozen=True)
class _RecurrenceComputeConfig:
    preparation: ttnn.DeviceComputeKernelConfig | None
    affine_prefix: ttnn.DeviceComputeKernelConfig | None
    grouped_scan: ttnn.DeviceComputeKernelConfig | None


def _validate_recurrence_geometry(
    inputs: _FlatRecurrenceInputs,
    *,
    program_config: KDARecurrenceProgramConfig,
    sequence_parallel_axis: int | None,
) -> _RecurrenceGeometry:
    """Validate the flat production contract and derive host-only execution metadata."""
    q_shape = tuple(inputs.q.shape)
    k_shape = tuple(inputs.k.shape)
    v_shape = tuple(inputs.v.shape)
    gate_shape = tuple(inputs.gate.shape)
    beta_shape = tuple(inputs.beta.shape)
    if len(beta_shape) != 3:
        raise ValueError("KDA recurrence beta must be [B,T,H]")
    batch, sequence, heads = beta_shape
    if any(len(shape) != 3 for shape in (q_shape, k_shape, v_shape, gate_shape)):
        raise ValueError("KDA recurrence q/k/v/g must be flat rank-3 tensors")
    if k_shape != q_shape or q_shape[:2] != (batch, sequence) or v_shape[:2] != (batch, sequence):
        raise ValueError("KDA recurrence q/k/v shapes are inconsistent")
    if gate_shape[:2] != (batch, sequence) or q_shape[2] != gate_shape[2]:
        raise ValueError("flat q/k/gate shapes are inconsistent")
    if q_shape[2] % heads or v_shape[2] % heads:
        raise ValueError("flat q/k/v widths must be divisible by the number of heads")
    key_dim = q_shape[2] // heads
    value_dim = v_shape[2] // heads
    if sequence_parallel_axis not in (None, 0, 1):
        raise ValueError("sequence_parallel_axis must be 0 or 1")
    if sequence <= 0 or sequence % program_config.chunk_size:
        raise ValueError(f"flat KDA recurrence requires T divisible by {program_config.chunk_size}")

    return _RecurrenceGeometry(
        batch=batch,
        sequence=sequence,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=program_config.chunk_size,
        num_chunks=sequence // program_config.chunk_size,
    )


def _prepare_chunk_inputs(
    inputs: _FlatRecurrenceInputs,
    geometry: _RecurrenceGeometry,
) -> _FlatRecurrenceInputs:
    beta_by_head = ttnn.permute(inputs.beta, (0, 2, 1))
    beta = ttnn.reshape(beta_by_head, (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, 1))
    return _FlatRecurrenceInputs(q=inputs.q, k=inputs.k, v=inputs.v, gate=inputs.gate, beta=beta)


def _prepare_chunk_terms(
    inputs: _FlatRecurrenceInputs,
    constants: _ChunkConstants,
    geometry: _RecurrenceGeometry,
    *,
    program_config: KDARecurrenceProgramConfig,
    compute_config: _RecurrenceComputeConfig,
) -> _PreparedChunks:
    eye_tile, tril_tile, ones_tile = constants.as_kernel_args()
    outputs = ttnn.experimental.kda.prepare_chunk_recurrence(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.gate,
        inputs.beta,
        eye_tile,
        tril_tile,
        ones_tile,
        geometry.heads,
        memory_config=program_config.preparation_memory_config,
        compute_kernel_config=compute_config.preparation,
        output_bf16_mask=program_config.prep_output_bf16_mask,
    )
    prepared = _PreparedChunks.from_kernel_outputs(outputs)
    for index, tensor in enumerate(outputs):
        expected_dtype = (
            program_config.qkv_dtype
            if program_config.prep_output_bf16_mask & (1 << index)
            else program_config.recurrent_state_dtype
        )
        assert tensor.dtype == expected_dtype
        assert tensor.memory_config() == program_config.preparation_memory_config
    return prepared


def _reshape_chunks_for_groups(
    prepared: _PreparedChunks,
    geometry: _RecurrenceGeometry,
    *,
    group_heads: int,
    summary_group_chunks: int,
) -> _PreparedChunks:
    return _PreparedChunks(
        v_beta=ttnn.reshape(
            prepared.v_beta, (group_heads, summary_group_chunks, geometry.chunk_size, geometry.value_dim)
        ),
        kd=ttnn.reshape(prepared.kd, (group_heads, summary_group_chunks, geometry.chunk_size, geometry.key_dim)),
        q_decay=ttnn.reshape(
            prepared.q_decay, (group_heads, summary_group_chunks, geometry.chunk_size, geometry.key_dim)
        ),
        intra=ttnn.reshape(
            prepared.intra, (group_heads, summary_group_chunks, geometry.chunk_size, geometry.chunk_size)
        ),
        k_dec_t=ttnn.reshape(
            prepared.k_dec_t, (group_heads, summary_group_chunks, geometry.key_dim, geometry.chunk_size)
        ),
        final_decay=ttnn.reshape(prepared.final_decay, (group_heads, summary_group_chunks, geometry.key_dim, 1)),
        t_inv=ttnn.reshape(
            prepared.t_inv, (group_heads, summary_group_chunks, geometry.chunk_size, geometry.chunk_size)
        ),
    )


def _summarize_chunk_groups(
    grouped: _PreparedChunks,
    geometry: _RecurrenceGeometry,
    *,
    program_config: KDARecurrenceProgramConfig,
    compute_config: _RecurrenceComputeConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    summary_memory_config = _group_summary_memory_config(
        grouped.v_beta.device(), grouped.v_beta.shape[0], geometry.key_dim
    )
    affine_a, affine_b = ttnn.experimental.kda.summarize_chunk_recurrence(
        *grouped.as_kernel_args(),
        memory_config=summary_memory_config,
        compute_kernel_config=compute_config.preparation,
    )
    assert affine_a.dtype == ttnn.float32
    assert affine_b.dtype == ttnn.float32
    # Precision boundary: summary-pair math is FP32; storage and transport are measured BF16.
    summary_a = ttnn.typecast(affine_a, program_config.affine_summary_dtype, memory_config=summary_memory_config)
    summary_b = ttnn.typecast(affine_b, program_config.affine_summary_dtype, memory_config=summary_memory_config)
    return summary_a, summary_b


class _RecurrenceScan(ABC):
    def __init__(
        self,
        program_config: KDARecurrenceProgramConfig,
        compute_config: _RecurrenceComputeConfig,
    ) -> None:
        self._program_config = program_config
        self._compute_config = compute_config

    @abstractmethod
    def run(
        self,
        prepared: _PreparedChunks,
        initial_state: ttnn.Tensor,
        identity_tile: ttnn.Tensor,
        geometry: _RecurrenceGeometry,
    ) -> _ScanResult:
        """Run the selected recurrence scan without changing the device operation order."""


class _DirectScan(_RecurrenceScan):
    def run(
        self,
        prepared: _PreparedChunks,
        initial_state: ttnn.Tensor,
        identity_tile: ttnn.Tensor,
        geometry: _RecurrenceGeometry,
    ) -> _ScanResult:
        del identity_tile
        output, final_state = ttnn.experimental.kda.recurrent_chunk_scan(
            *prepared.as_kernel_args(),
            initial_state,
            memory_config=self._program_config.output_memory_config,
            compute_kernel_config=self._compute_config.preparation,
        )
        assert output.dtype == self._program_config.scan_output_dtype
        assert final_state.dtype == self._program_config.recurrent_state_dtype
        return _ScanResult(output=output, final_state=final_state)


class _GroupedScan(_RecurrenceScan):
    def run(
        self,
        prepared: _PreparedChunks,
        initial_state: ttnn.Tensor,
        identity_tile: ttnn.Tensor,
        geometry: _RecurrenceGeometry,
    ) -> _ScanResult:
        group_chunks = self._program_config.summary_group_chunks
        if geometry.num_chunks % group_chunks:
            raise ValueError(
                f"local chunk count {geometry.num_chunks} must be divisible by " f"summary_group_chunks {group_chunks}"
            )
        if geometry.key_dim != geometry.value_dim:
            raise ValueError("grouped KDA affine prefix currently requires K == V")
        groups_per_head = geometry.num_chunks // group_chunks
        group_heads = geometry.batch_heads * groups_per_head
        grid = prepared.v_beta.device().compute_with_storage_grid_size()
        if group_heads > min(grid.x * grid.y, 128):
            raise ValueError(f"grouped KDA needs {group_heads} summary owners, but only 128 are supported")

        grouped = _reshape_chunks_for_groups(
            prepared,
            geometry,
            group_heads=group_heads,
            summary_group_chunks=group_chunks,
        )
        summary_a, summary_b = _summarize_chunk_groups(
            grouped,
            geometry,
            program_config=self._program_config,
            compute_config=self._compute_config,
        )
        group_initial_states, strategy_final_state = self._compute_group_entry_states(
            summary_a,
            summary_b,
            initial_state,
            groups_per_head,
        )
        grouped_output, grouped_final_states = ttnn.experimental.kda.recurrent_chunk_scan(
            *grouped.as_kernel_args(),
            group_initial_states,
            memory_config=self._program_config.output_memory_config,
            compute_kernel_config=self._compute_config.grouped_scan,
        )
        assert grouped_output.dtype == self._program_config.scan_output_dtype
        output = ttnn.reshape(
            grouped_output,
            (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, geometry.value_dim),
        )
        final_state = self._resolve_final_state(grouped_final_states, strategy_final_state, geometry, groups_per_head)
        return _ScanResult(output=output, final_state=final_state)

    @abstractmethod
    def _compute_group_entry_states(
        self,
        summary_a: ttnn.Tensor,
        summary_b: ttnn.Tensor,
        initial_state: ttnn.Tensor,
        groups_per_head: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        raise NotImplementedError

    @abstractmethod
    def _resolve_final_state(
        self,
        grouped_final_states: ttnn.Tensor,
        strategy_final_state: ttnn.Tensor | None,
        geometry: _RecurrenceGeometry,
        groups_per_head: int,
    ) -> ttnn.Tensor:
        raise NotImplementedError


class _LocalGroupedScan(_GroupedScan):
    def _compute_group_entry_states(
        self,
        summary_a: ttnn.Tensor,
        summary_b: ttnn.Tensor,
        initial_state: ttnn.Tensor,
        groups_per_head: int,
    ) -> tuple[ttnn.Tensor, None]:
        group_initial_states = ttnn.experimental.kda.affine_exclusive_scan(
            summary_a,
            summary_b,
            initial_state,
            groups_per_head,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._compute_config.affine_prefix,
        )
        return group_initial_states, None

    def _resolve_final_state(
        self,
        grouped_final_states: ttnn.Tensor,
        strategy_final_state: ttnn.Tensor | None,
        geometry: _RecurrenceGeometry,
        groups_per_head: int,
    ) -> ttnn.Tensor:
        if strategy_final_state is not None:
            raise RuntimeError("local grouped KDA scan unexpectedly produced a distributed final state")
        all_final_states = ttnn.reshape(
            grouped_final_states,
            (geometry.batch_heads, groups_per_head, geometry.key_dim, geometry.value_dim),
        )
        last_final_state = ttnn.slice(
            all_final_states,
            (0, groups_per_head - 1, 0, 0),
            (geometry.batch_heads, groups_per_head, geometry.key_dim, geometry.value_dim),
            memory_config=self._program_config.output_memory_config,
        )
        return ttnn.reshape(last_final_state, (geometry.batch_heads, geometry.key_dim, geometry.value_dim))


class _DistributedGroupedScan(_GroupedScan):
    def __init__(
        self,
        sequence_parallel_axis: int,
        program_config: KDARecurrenceProgramConfig,
        compute_config: _RecurrenceComputeConfig,
    ) -> None:
        super().__init__(program_config, compute_config)
        self._sequence_parallel_axis = sequence_parallel_axis

    def _compute_group_entry_states(
        self,
        summary_a: ttnn.Tensor,
        summary_b: ttnn.Tensor,
        initial_state: ttnn.Tensor,
        groups_per_head: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        partition_a, partition_b = ttnn.experimental.kda.reduce_affine_transforms(
            summary_a,
            summary_b,
            groups_per_head,
            memory_config=self._program_config.output_memory_config,
            compute_kernel_config=self._compute_config.affine_prefix,
        )
        assert partition_a.dtype == ttnn.float32
        assert partition_b.dtype == ttnn.float32
        partition_entry_state, distributed_final_state = _distributed_affine_prefix(
            partition_a,
            partition_b,
            initial_state,
            sequence_parallel_axis=self._sequence_parallel_axis,
            program_config=self._program_config,
            compute_config=self._compute_config.affine_prefix,
        )
        assert partition_entry_state.dtype == self._program_config.recurrent_state_dtype
        group_initial_states = ttnn.experimental.kda.affine_exclusive_scan(
            summary_a,
            summary_b,
            partition_entry_state,
            groups_per_head,
            memory_config=self._program_config.prefix_memory_config,
            compute_kernel_config=self._compute_config.affine_prefix,
        )
        return group_initial_states, distributed_final_state

    def _resolve_final_state(
        self,
        grouped_final_states: ttnn.Tensor,
        strategy_final_state: ttnn.Tensor | None,
        geometry: _RecurrenceGeometry,
        groups_per_head: int,
    ) -> ttnn.Tensor:
        del grouped_final_states, geometry, groups_per_head
        if strategy_final_state is None:
            raise RuntimeError("distributed grouped KDA scan did not produce a final state")
        return strategy_final_state


def _select_scan(
    *,
    num_chunks: int,
    program_config: KDARecurrenceProgramConfig,
    compute_config: _RecurrenceComputeConfig,
    sequence_parallel_axis: int | None,
) -> _RecurrenceScan:
    if sequence_parallel_axis is not None:
        return _DistributedGroupedScan(sequence_parallel_axis, program_config, compute_config)
    if num_chunks >= program_config.grouped_scan_min_chunks and num_chunks % program_config.summary_group_chunks == 0:
        return _LocalGroupedScan(program_config, compute_config)
    return _DirectScan(program_config, compute_config)


def _restore_recurrence_output(scan: _ScanResult, geometry: _RecurrenceGeometry) -> _ScanResult:
    output = ttnn.reshape(
        scan.output,
        (geometry.batch_heads, geometry.sequence, geometry.value_dim),
    )
    final_state = ttnn.reshape(
        scan.final_state,
        (geometry.batch, geometry.heads, geometry.key_dim, geometry.value_dim),
    )
    return _ScanResult(output=output, final_state=final_state)


def _assert_chunk_constants(constants: _ChunkConstants, chunk_size: int) -> None:
    expected_shapes = (
        (constants.eye, (1, 1, chunk_size, chunk_size)),
        (constants.tril, (1, 1, chunk_size, chunk_size)),
        (constants.ones, (1, 1, chunk_size, chunk_size)),
    )
    for tensor, expected_shape in expected_shapes:
        assert tuple(tensor.shape) == expected_shape
        assert tensor.dtype == ttnn.float32
        assert tensor.layout == ttnn.TILE_LAYOUT


def _chunk_recurrence(
    inputs: _FlatRecurrenceInputs,
    initial_state: ttnn.Tensor,
    constants: _ChunkConstants,
    *,
    program_config: KDARecurrenceProgramConfig,
    compute_config: _RecurrenceComputeConfig,
    sequence_parallel_axis: int | None,
) -> _ScanResult:
    """Run the fixed Kimi-K3 recurrence contract through the selected scan strategy."""
    geometry = _validate_recurrence_geometry(
        inputs,
        program_config=program_config,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    for tensor in (inputs.q, inputs.k, inputs.v, inputs.gate, inputs.beta, initial_state):
        assert tensor.layout == ttnn.TILE_LAYOUT
    assert inputs.q.dtype == program_config.qkv_dtype
    assert inputs.k.dtype == program_config.qkv_dtype
    assert inputs.v.dtype == program_config.qkv_dtype
    assert inputs.gate.dtype == program_config.gate_dtype
    assert inputs.beta.dtype == program_config.beta_dtype
    assert initial_state.dtype == program_config.recurrent_state_dtype
    assert tuple(initial_state.shape) == (
        geometry.batch,
        geometry.heads,
        geometry.key_dim,
        geometry.value_dim,
    )
    assert initial_state.memory_config() == program_config.output_memory_config
    _assert_chunk_constants(constants, program_config.chunk_size)

    state = ttnn.reshape(
        initial_state,
        (geometry.batch_heads, geometry.key_dim, geometry.value_dim),
    )
    scan_strategy = _select_scan(
        num_chunks=geometry.num_chunks,
        program_config=program_config,
        compute_config=compute_config,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    chunk_inputs = _prepare_chunk_inputs(inputs, geometry)
    prepared = _prepare_chunk_terms(
        chunk_inputs,
        constants,
        geometry,
        program_config=program_config,
        compute_config=compute_config,
    )
    scan = scan_strategy.run(prepared, state, constants.eye, geometry)
    result = _restore_recurrence_output(scan, geometry)
    assert result.output.dtype == program_config.scan_output_dtype
    assert result.final_state.dtype == program_config.recurrent_state_dtype
    return result


def _distributed_affine_prefix(
    transform_a: ttnn.Tensor,
    transform_b: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    program_config: KDARecurrenceProgramConfig,
    compute_config: ttnn.DeviceComputeKernelConfig | None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Compose SP partition affine summaries and return entry/final carries."""
    if sequence_parallel_axis not in (0, 1):
        raise ValueError(f"sequence_parallel_axis must be 0 or 1, got {sequence_parallel_axis}")
    shape = tuple(transform_a.shape)
    if tuple(transform_b.shape) != shape or tuple(initial_state.shape) != shape:
        raise ValueError("distributed KDA affine prefix requires equal batched [K,K] tensor shapes")
    if len(shape) != 3:
        raise ValueError("distributed KDA affine prefix requires rank-3 production transforms")
    if shape[-2] != shape[-1]:
        raise ValueError("distributed KDA affine prefix currently requires K == V")
    assert transform_a.dtype == ttnn.float32
    assert transform_b.dtype == ttnn.float32
    assert initial_state.dtype == program_config.recurrent_state_dtype

    mesh_device = transform_a.device()
    mesh_shape = tuple(mesh_device.shape)
    if len(mesh_shape) != 2 or mesh_shape[sequence_parallel_axis] <= 1:
        raise ValueError("distributed KDA affine prefix requires a 2D mesh with SP > 1")
    sp_size = mesh_shape[sequence_parallel_axis]
    batch_heads, key_dim = shape[0], shape[1]
    value_dim = transform_b.shape[-1]
    output_memory = program_config.output_memory_config
    working_memory = program_config.distributed_working_memory_config

    # Precision boundary: FP32 composition is transported as measured BF16.
    transport_a = ttnn.typecast(transform_a, program_config.affine_summary_dtype, memory_config=output_memory)
    transport_b = ttnn.typecast(transform_b, program_config.affine_summary_dtype, memory_config=output_memory)
    transport_a = ttnn.reshape(transport_a, (1, batch_heads, key_dim, key_dim))
    transport_b = ttnn.reshape(transport_b, (1, batch_heads, key_dim, value_dim))
    packed = ttnn.concat([transport_a, transport_b], dim=3, memory_config=output_memory)
    gathered = ttnn.all_gather(
        packed,
        dim=0,
        cluster_axis=sequence_parallel_axis,
        memory_config=output_memory,
    )

    carry = ttnn.to_memory_config(initial_state, working_memory)
    assert carry.dtype == program_config.recurrent_state_dtype
    carry = ttnn.reshape(carry, (1, batch_heads, key_dim, value_dim))
    entry_states = []
    for rank in range(sp_size):
        entry_states.append(carry)
        transported_rank_a = ttnn.slice(
            gathered,
            (rank, 0, 0, 0),
            (rank + 1, batch_heads, key_dim, key_dim),
            memory_config=working_memory,
        )
        transported_rank_b = ttnn.slice(
            gathered,
            (rank, 0, 0, key_dim),
            (rank + 1, batch_heads, key_dim, key_dim + value_dim),
            memory_config=working_memory,
        )
        # Precision boundary: BF16 collective payload is restored for FP32 carry math.
        rank_a_for_carry = ttnn.typecast(
            transported_rank_a,
            program_config.recurrent_state_dtype,
            memory_config=working_memory,
        )
        rank_b_for_carry = ttnn.typecast(
            transported_rank_b,
            program_config.recurrent_state_dtype,
            memory_config=working_memory,
        )
        carry = ttnn.matmul(
            rank_a_for_carry,
            carry,
            memory_config=working_memory,
            dtype=program_config.recurrent_state_dtype,
            compute_kernel_config=compute_config,
        )
        carry = ttnn.add(carry, rank_b_for_carry, memory_config=working_memory)

    replicated_entries = ttnn.concat(entry_states, dim=0, memory_config=output_memory)
    entry_state = ttnn.mesh_partition(
        replicated_entries,
        dim=0,
        cluster_axis=sequence_parallel_axis,
        memory_config=output_memory,
    )
    final_state = ttnn.to_memory_config(carry, output_memory)
    assert final_state.dtype == program_config.recurrent_state_dtype
    entry_state = ttnn.reshape(entry_state, (batch_heads, key_dim, value_dim))
    final_state = ttnn.reshape(final_state, (batch_heads, key_dim, value_dim))
    return entry_state, final_state
