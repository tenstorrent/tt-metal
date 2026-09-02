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
from models.demos.deepseek_v3_d_p.tt.kda.config import (
    KDA_AFFINE_SUMMARY_DTYPE,
    KDA_BETA_DTYPE,
    KDA_CHUNK_SIZE,
    KDA_DISTRIBUTED_PREFIX_MEMORY_CONFIG,
    KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG,
    KDA_GATE_DTYPE,
    KDA_LOCAL_PREFIX_MEMORY_CONFIG,
    KDA_OUTPUT_MEMORY_CONFIG,
    KDA_PREP_OUTPUT_BF16_MASK,
    KDA_PREPARATION_MEMORY_CONFIG,
    KDA_QKV_DTYPE,
    KDA_RECURRENT_STATE_DTYPE,
    KDA_SCAN_OUTPUT_DTYPE,
    KDARecurrenceProgramConfig,
)


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
    scan: ttnn.DeviceComputeKernelConfig | None


def _validate_recurrence_geometry(
    inputs: _FlatRecurrenceInputs,
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
    if sequence <= 0 or sequence % KDA_CHUNK_SIZE:
        raise ValueError(f"flat KDA recurrence requires T divisible by {KDA_CHUNK_SIZE}")

    return _RecurrenceGeometry(
        batch=batch,
        sequence=sequence,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=KDA_CHUNK_SIZE,
        num_chunks=sequence // KDA_CHUNK_SIZE,
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
    geometry: _RecurrenceGeometry,
    *,
    compute_config: _RecurrenceComputeConfig,
) -> _PreparedChunks:
    outputs = ttnn.experimental.kda.prepare_chunk_recurrence(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.gate,
        inputs.beta,
        geometry.heads,
        memory_config=KDA_PREPARATION_MEMORY_CONFIG,
        compute_kernel_config=compute_config.preparation,
        output_bf16_mask=KDA_PREP_OUTPUT_BF16_MASK,
    )
    prepared = _PreparedChunks.from_kernel_outputs(outputs)
    for index, tensor in enumerate(outputs):
        expected_dtype = KDA_QKV_DTYPE if KDA_PREP_OUTPUT_BF16_MASK & (1 << index) else KDA_RECURRENT_STATE_DTYPE
        assert tensor.dtype == expected_dtype
        assert tensor.memory_config() == KDA_PREPARATION_MEMORY_CONFIG
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
    compute_config: _RecurrenceComputeConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    summary_memory_config = _group_summary_memory_config(
        grouped.v_beta.device(), grouped.v_beta.shape[0], geometry.key_dim
    )
    affine_a, affine_b = ttnn.experimental.kda.summarize_chunk_recurrence(
        *grouped.as_kernel_args(),
        memory_config=summary_memory_config,
        # Summary generation is part of chunk preparation; the affine-prefix
        # fidelity knob applies only to composition of the emitted summaries.
        compute_kernel_config=compute_config.preparation,
    )
    assert affine_a.dtype == ttnn.float32
    assert affine_b.dtype == ttnn.float32
    # Precision boundary: summary-pair math is FP32; summaries are stored and transported as BF16.
    summary_a = ttnn.typecast(affine_a, KDA_AFFINE_SUMMARY_DTYPE, memory_config=summary_memory_config)
    summary_b = ttnn.typecast(affine_b, KDA_AFFINE_SUMMARY_DTYPE, memory_config=summary_memory_config)
    return summary_a, summary_b


def _validate_grouped_scan_capacity(
    *,
    batch_heads: int,
    num_chunks: int,
    summary_group_chunks: int,
    device: ttnn.Device | ttnn.MeshDevice,
) -> None:
    """Reject grouped scans that cannot assign one worker to each summary owner."""
    if num_chunks % summary_group_chunks:
        raise ValueError(
            f"local chunk count {num_chunks} must be divisible by summary_group_chunks {summary_group_chunks}"
        )
    group_heads = batch_heads * (num_chunks // summary_group_chunks)
    grid = device.compute_with_storage_grid_size()
    capacity = grid.x * grid.y
    if group_heads > capacity:
        raise ValueError(f"grouped KDA needs {group_heads} summary owners, but only {capacity} are supported")


def _effective_summary_group_chunks(num_chunks: int, configured_group_chunks: int) -> int:
    """Return the largest configured-or-smaller group size that divides the local chunk count."""
    for group_chunks in range(min(num_chunks, configured_group_chunks), 0, -1):
        if num_chunks % group_chunks == 0:
            return group_chunks
    raise AssertionError("positive chunk counts always have divisor 1")


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
        geometry: _RecurrenceGeometry,
    ) -> _ScanResult:
        """Run the selected recurrence scan without changing the device operation order."""


class _DirectScan(_RecurrenceScan):
    def run(
        self,
        prepared: _PreparedChunks,
        initial_state: ttnn.Tensor,
        geometry: _RecurrenceGeometry,
    ) -> _ScanResult:
        output, final_state = ttnn.experimental.kda.recurrent_chunk_scan(
            *prepared.as_kernel_args(),
            initial_state,
            memory_config=KDA_OUTPUT_MEMORY_CONFIG,
            compute_kernel_config=self._compute_config.scan,
        )
        assert output.dtype == KDA_SCAN_OUTPUT_DTYPE
        assert final_state.dtype == KDA_RECURRENT_STATE_DTYPE
        return _ScanResult(output=output, final_state=final_state)


class _GroupedScan(_RecurrenceScan):
    def run(
        self,
        prepared: _PreparedChunks,
        initial_state: ttnn.Tensor,
        geometry: _RecurrenceGeometry,
    ) -> _ScanResult:
        group_chunks = _effective_summary_group_chunks(
            geometry.num_chunks,
            self._program_config.summary_group_chunks,
        )
        if geometry.key_dim != geometry.value_dim:
            raise ValueError("grouped KDA affine prefix currently requires K == V")
        _validate_grouped_scan_capacity(
            batch_heads=geometry.batch_heads,
            num_chunks=geometry.num_chunks,
            summary_group_chunks=group_chunks,
            device=prepared.v_beta.device(),
        )
        groups_per_head = geometry.num_chunks // group_chunks
        group_heads = geometry.batch_heads * groups_per_head

        grouped = _reshape_chunks_for_groups(
            prepared,
            geometry,
            group_heads=group_heads,
            summary_group_chunks=group_chunks,
        )
        summary_a, summary_b = _summarize_chunk_groups(
            grouped,
            geometry,
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
            memory_config=KDA_OUTPUT_MEMORY_CONFIG,
            compute_kernel_config=self._compute_config.scan,
        )
        assert grouped_output.dtype == KDA_SCAN_OUTPUT_DTYPE
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
            memory_config=KDA_LOCAL_PREFIX_MEMORY_CONFIG,
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
            memory_config=KDA_OUTPUT_MEMORY_CONFIG,
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
            memory_config=KDA_OUTPUT_MEMORY_CONFIG,
            compute_kernel_config=self._compute_config.affine_prefix,
        )
        assert partition_a.dtype == ttnn.float32
        assert partition_b.dtype == ttnn.float32
        partition_entry_state, distributed_final_state = _distributed_affine_prefix(
            partition_a,
            partition_b,
            initial_state,
            sequence_parallel_axis=self._sequence_parallel_axis,
            compute_config=self._compute_config.affine_prefix,
        )
        assert partition_entry_state.dtype == KDA_RECURRENT_STATE_DTYPE
        group_initial_states = ttnn.experimental.kda.affine_exclusive_scan(
            summary_a,
            summary_b,
            partition_entry_state,
            groups_per_head,
            memory_config=KDA_DISTRIBUTED_PREFIX_MEMORY_CONFIG,
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
        if strategy_final_state is None:
            raise RuntimeError("distributed grouped KDA scan did not produce a final state")
        return strategy_final_state


def _select_scan(
    *,
    program_config: KDARecurrenceProgramConfig,
    compute_config: _RecurrenceComputeConfig,
    sequence_parallel_axis: int | None,
) -> _RecurrenceScan:
    if sequence_parallel_axis is None and program_config.local_scan_strategy != "grouped":
        return _DirectScan(program_config, compute_config)
    if sequence_parallel_axis is None:
        return _LocalGroupedScan(program_config, compute_config)
    return _DistributedGroupedScan(sequence_parallel_axis, program_config, compute_config)


class KDARecurrence:
    """Constructor-fixed KDA recurrence executor."""

    def __init__(
        self,
        device: ttnn.Device | ttnn.MeshDevice,
        program_config: KDARecurrenceProgramConfig,
        *,
        sequence_parallel_axis: int | None,
    ) -> None:
        if sequence_parallel_axis not in (None, 0, 1):
            raise ValueError("sequence_parallel_axis must be 0 or 1")
        preparation = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        affine_prefix = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=program_config.affine_prefix_math_fidelity,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        scan = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=program_config.scan_math_fidelity,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self._compute_config = _RecurrenceComputeConfig(
            preparation=preparation,
            affine_prefix=affine_prefix,
            scan=scan,
        )
        self._scan_strategy = _select_scan(
            program_config=program_config,
            compute_config=self._compute_config,
            sequence_parallel_axis=sequence_parallel_axis,
        )

    def __call__(
        self,
        *,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        gate: ttnn.Tensor,
        beta: ttnn.Tensor,
        initial_state: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return ``(new_state, output)`` for directly named recurrence tensors."""
        inputs = _FlatRecurrenceInputs(q=q, k=k, v=v, gate=gate, beta=beta)
        geometry = _validate_recurrence_geometry(inputs)
        tensors = {
            "q": q,
            "k": k,
            "v": v,
            "gate": gate,
            "beta": beta,
            "initial_state": initial_state,
        }
        for name, tensor in tensors.items():
            if tensor.layout != ttnn.TILE_LAYOUT:
                raise ValueError(f"{name} layout must be TILE_LAYOUT, got {tensor.layout}")
        expected_dtypes = {
            "q": KDA_QKV_DTYPE,
            "k": KDA_QKV_DTYPE,
            "v": KDA_QKV_DTYPE,
            "gate": KDA_GATE_DTYPE,
            "beta": KDA_BETA_DTYPE,
            "initial_state": KDA_RECURRENT_STATE_DTYPE,
        }
        for name, expected_dtype in expected_dtypes.items():
            actual_dtype = tensors[name].dtype
            if actual_dtype != expected_dtype:
                raise ValueError(f"{name} dtype must be {expected_dtype}, got {actual_dtype}")
        expected_state_shape = (geometry.batch, geometry.heads, geometry.key_dim, geometry.value_dim)
        if tuple(initial_state.shape) != expected_state_shape:
            raise ValueError(f"initial_state shape {tuple(initial_state.shape)} != {expected_state_shape}")
        if initial_state.memory_config() != KDA_OUTPUT_MEMORY_CONFIG:
            raise ValueError("initial_state memory config must be DRAM interleaved")

        state = ttnn.reshape(
            initial_state,
            (geometry.batch_heads, geometry.key_dim, geometry.value_dim),
        )
        chunk_inputs = _prepare_chunk_inputs(inputs, geometry)
        prepared = _prepare_chunk_terms(
            chunk_inputs,
            geometry,
            compute_config=self._compute_config,
        )
        scan = self._scan_strategy.run(prepared, state, geometry)
        output = ttnn.reshape(
            scan.output,
            (geometry.batch_heads, geometry.sequence, geometry.value_dim),
        )
        final_state = ttnn.reshape(
            scan.final_state,
            (geometry.batch, geometry.heads, geometry.key_dim, geometry.value_dim),
        )
        assert output.dtype == KDA_SCAN_OUTPUT_DTYPE
        assert final_state.dtype == KDA_RECURRENT_STATE_DTYPE
        return final_state, output


def _distributed_affine_prefix(
    transform_a: ttnn.Tensor,
    transform_b: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
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
    assert initial_state.dtype == KDA_RECURRENT_STATE_DTYPE

    mesh_device = transform_a.device()
    mesh_shape = tuple(mesh_device.shape)
    if len(mesh_shape) != 2 or mesh_shape[sequence_parallel_axis] <= 1:
        raise ValueError("distributed KDA affine prefix requires a 2D mesh with SP > 1")
    sp_size = mesh_shape[sequence_parallel_axis]
    batch_heads, key_dim = shape[0], shape[1]
    value_dim = transform_b.shape[-1]
    output_memory = KDA_OUTPUT_MEMORY_CONFIG
    working_memory = KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG

    # Precision boundary: FP32 composition is transported as BF16.
    transport_a = ttnn.typecast(transform_a, KDA_AFFINE_SUMMARY_DTYPE, memory_config=output_memory)
    transport_b = ttnn.typecast(transform_b, KDA_AFFINE_SUMMARY_DTYPE, memory_config=output_memory)
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
    assert carry.dtype == KDA_RECURRENT_STATE_DTYPE
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
            KDA_RECURRENT_STATE_DTYPE,
            memory_config=working_memory,
        )
        rank_b_for_carry = ttnn.typecast(
            transported_rank_b,
            KDA_RECURRENT_STATE_DTYPE,
            memory_config=working_memory,
        )
        carry = ttnn.matmul(
            rank_a_for_carry,
            carry,
            memory_config=working_memory,
            dtype=KDA_RECURRENT_STATE_DTYPE,
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
    assert final_state.dtype == KDA_RECURRENT_STATE_DTYPE
    entry_state = ttnn.reshape(entry_state, (batch_heads, key_dim, value_dim))
    final_state = ttnn.reshape(final_state, (batch_heads, key_dim, value_dim))
    return entry_state, final_state
