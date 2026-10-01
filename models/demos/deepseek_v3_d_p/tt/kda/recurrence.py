# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Python-owned KDA graph orchestration.

The KDA layer keeps collective and state-flow decisions here; device leaves only
perform the bespoke kernels exposed by ``ttnn.experimental.kda``.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections
from models.demos.deepseek_v3_d_p.tt.kda.config import (
    KDA_AFFINE_SUMMARY_DTYPE,
    KDA_CHUNK_SIZE,
    KDA_DISTRIBUTED_PREFIX_MEMORY_CONFIG,
    KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG,
    KDA_LOCAL_PREFIX_MEMORY_CONFIG,
    KDA_OUTPUT_MEMORY_CONFIG,
    KDA_PREP_OUTPUT_BF16_MASK,
    KDA_PREPARATION_MEMORY_CONFIG,
    KDA_RECURRENT_STATE_DTYPE,
    KDARecurrenceProgramConfig,
)


def _group_summary_memory_config(device: ttnn.Device, group_heads: int, key_dim: int) -> ttnn.MemoryConfig:
    grid = device.compute_with_storage_grid_size()
    capacity = grid.x * grid.y
    if group_heads > capacity:
        raise ValueError(f"grouped KDA needs {group_heads} summary owners, but only {capacity} are supported")
    worker_cores = ttnn.num_cores_to_corerangeset(group_heads, grid, row_wise=True)
    return ttnn.create_sharded_memory_config(
        (group_heads, key_dim, key_dim),
        core_grid=worker_cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@dataclass(frozen=True)
class _AffineTransform:
    """State-space affine map ``state -> a @ state + b``."""

    a: ttnn.Tensor
    b: ttnn.Tensor


@dataclass(frozen=True)
class _RecurrenceGeometry:
    batch: int
    local_rows: int
    heads: int
    key_dim: int
    value_dim: int
    chunk_size: int
    num_chunks: int

    @property
    def batch_heads(self) -> int:
        return self.batch * self.heads


@dataclass(frozen=True)
class _PreparedChunks:
    v_beta: ttnn.Tensor
    kd: ttnn.Tensor
    q_decay: ttnn.Tensor
    intra: ttnn.Tensor
    k_dec_t: ttnn.Tensor
    final_decay: ttnn.Tensor
    t_inv: ttnn.Tensor

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
class RecurrenceResult:
    output: ttnn.Tensor
    final_state: ttnn.Tensor


@dataclass(frozen=True)
class _RecurrenceComputeConfig:
    preparation: ttnn.DeviceComputeKernelConfig
    affine_prefix: ttnn.DeviceComputeKernelConfig
    scan: ttnn.DeviceComputeKernelConfig


def _prepare_chunk_terms(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    gate: ttnn.Tensor,
    beta: ttnn.Tensor,
    geometry: _RecurrenceGeometry,
    *,
    compute_config: _RecurrenceComputeConfig,
    actual_start: ttnn.Tensor,
    actual_end: ttnn.Tensor | None,
    sequence_parallel_axis: int,
) -> _PreparedChunks:
    beta_by_head = ttnn.permute(beta, (0, 2, 1))
    beta_by_chunk = ttnn.reshape(
        beta_by_head,
        (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, 1),
    )
    outputs = ttnn.experimental.kda.prepare_chunk_recurrence(
        q,
        k,
        v,
        gate,
        beta_by_chunk,
        geometry.heads,
        memory_config=KDA_PREPARATION_MEMORY_CONFIG,
        compute_kernel_config=compute_config.preparation,
        output_bf16_mask=KDA_PREP_OUTPUT_BF16_MASK,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    return _PreparedChunks(*outputs)


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
    *,
    actual_start: ttnn.Tensor,
    actual_end: ttnn.Tensor | None,
    sequence_parallel_axis: int,
    groups_per_head: int,
    summary_memory_config: ttnn.MemoryConfig,
    compute_config: _RecurrenceComputeConfig,
) -> _AffineTransform:
    """Summarize whole groups for the single-rank path at the BF16 prefix boundary."""
    a, b, tail_a, tail_b = ttnn.experimental.kda.summarize_chunk_recurrence(
        *grouped.as_kernel_args(),
        groups_per_head=groups_per_head,
        memory_config=summary_memory_config,
        compute_kernel_config=compute_config.preparation,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    # A single sequence rank has no split tail, so only the head summaries are needed.
    ttnn.deallocate(tail_a)
    ttnn.deallocate(tail_b)
    return _AffineTransform(a, b)


def _scan_chunks(
    prepared: _PreparedChunks,
    group_entry_states: ttnn.Tensor,
    tail_entry_states: ttnn.Tensor,
    *,
    actual_start: ttnn.Tensor,
    actual_end: ttnn.Tensor | None,
    sequence_parallel_axis: int,
    compute_config: ttnn.DeviceComputeKernelConfig,
    groups_per_head: int = 1,
) -> RecurrenceResult:
    output, final_states = ttnn.experimental.kda.recurrent_chunk_scan(
        *prepared.as_kernel_args(),
        group_entry_states,
        groups_per_head=groups_per_head,
        memory_config=KDA_OUTPUT_MEMORY_CONFIG,
        compute_kernel_config=compute_config,
        actual_start=actual_start,
        actual_end=actual_end,
        tail_entry_states=tail_entry_states,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    return RecurrenceResult(output=output, final_state=final_states)


def _distributed_prefix(
    transform: _AffineTransform,
    initial_state: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    selections: ChronologicalSelections,
    compute_config: ttnn.DeviceComputeKernelConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Compose one affine transform per chip in chronological order.

    Entry states are stored in chronological order; the selector maps the
    local physical rank to its entry while the carry advances in that order.
    Return local entry and the replicated final carry on each independent TP line.
    """
    transform_a, transform_b = transform.a, transform.b
    batch_heads, key_dim = tuple(transform_a.shape)[0], tuple(transform_a.shape)[1]
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
    carry = ttnn.reshape(carry, (1, batch_heads, key_dim, value_dim))
    entry_states: list[ttnn.Tensor] = []
    # Apply chip transforms chronologically: O(sp_size) graph nodes, bounded by mesh size, not token count.
    for step in range(gathered.shape[0]):
        # Keep chronological slots until the final device-indexed entry selection.
        entry_states.append(carry)
        selected = selections.select_affine_transform(gathered, step, memory_config=working_memory)
        transported_a = ttnn.slice(
            selected,
            (0, 0, 0, 0),
            (1, batch_heads, key_dim, key_dim),
            memory_config=working_memory,
        )
        transported_b = ttnn.slice(
            selected,
            (0, 0, 0, key_dim),
            (1, batch_heads, key_dim, key_dim + value_dim),
            memory_config=working_memory,
        )
        # Precision boundary: BF16 collective payload is restored for FP32 carry math.
        a_for_carry = ttnn.typecast(transported_a, KDA_RECURRENT_STATE_DTYPE, memory_config=working_memory)
        b_for_carry = ttnn.typecast(transported_b, KDA_RECURRENT_STATE_DTYPE, memory_config=working_memory)
        carry = ttnn.matmul(
            a_for_carry,
            carry,
            memory_config=working_memory,
            dtype=KDA_RECURRENT_STATE_DTYPE,
            compute_kernel_config=compute_config,
        )
        carry = ttnn.add(carry, b_for_carry, memory_config=working_memory)

    chronological_entries = ttnn.concat(entry_states, dim=0, memory_config=working_memory)
    local_entries = selections.select_local_entry_state(chronological_entries, memory_config=working_memory)
    local_entry_state = ttnn.reshape(local_entries, (batch_heads, key_dim, value_dim))
    final_state = ttnn.reshape(ttnn.to_memory_config(carry, output_memory), (batch_heads, key_dim, value_dim))
    return local_entry_state, final_state


def _last_group_state(
    grouped_final_states: ttnn.Tensor,
    geometry: _RecurrenceGeometry,
    groups_per_head: int,
) -> ttnn.Tensor:
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


def _ordinary_group_scan(
    grouped: _PreparedChunks,
    summary: _AffineTransform,
    initial_state: ttnn.Tensor,
    *,
    actual_start: ttnn.Tensor,
    actual_end: ttnn.Tensor | None,
    sequence_parallel_axis: int,
    groups_per_head: int,
    prefix_memory_config: ttnn.MemoryConfig,
    compute_config: _RecurrenceComputeConfig,
) -> RecurrenceResult:
    """Scan groups on a single sequence rank, where no split-tail reset is needed."""
    # Tail inputs are unused on this path; reuse the head summaries and initial state.
    group_entry_states = ttnn.experimental.kda.affine_exclusive_scan(
        summary.a,
        summary.b,
        initial_state,
        groups_per_head,
        memory_config=prefix_memory_config,
        compute_kernel_config=compute_config.affine_prefix,
        actual_start=actual_start,
        actual_end=actual_end,
        tail_a=summary.a,
        tail_b=summary.b,
        tail_entry_states=initial_state,
        local_rows=grouped.v_beta.shape[1] * grouped.v_beta.shape[2] * groups_per_head,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    return _scan_chunks(
        grouped,
        group_entry_states,
        initial_state,
        groups_per_head=groups_per_head,
        compute_config=compute_config.scan,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
    )


def _scan_local_grouped_chunks(
    prepared: _PreparedChunks,
    initial_state: ttnn.Tensor,
    geometry: _RecurrenceGeometry,
    *,
    actual_start: ttnn.Tensor,
    actual_end: ttnn.Tensor | None,
    sequence_parallel_axis: int,
    summary_group_chunks: int,
    groups: int,
    memory: ttnn.MemoryConfig,
    compute_config: _RecurrenceComputeConfig,
) -> RecurrenceResult:
    grouped = _reshape_chunks_for_groups(
        prepared, geometry, group_heads=geometry.batch_heads * groups, summary_group_chunks=summary_group_chunks
    )
    summary = _summarize_chunk_groups(
        grouped,
        groups_per_head=groups,
        summary_memory_config=memory,
        compute_config=compute_config,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    scan = _ordinary_group_scan(
        grouped,
        summary,
        initial_state,
        groups_per_head=groups,
        prefix_memory_config=KDA_LOCAL_PREFIX_MEMORY_CONFIG,
        compute_config=compute_config,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    output = ttnn.reshape(
        scan.output, (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, geometry.value_dim)
    )
    return RecurrenceResult(output, _last_group_state(scan.final_state, geometry, groups))


def _partition_prefix(
    summary: _AffineTransform,
    initial_state: ttnn.Tensor,
    *,
    groups_per_head: int,
    local_rows: int,
    sequence_parallel_axis: int,
    selections: ChronologicalSelections,
    actual_start: ttnn.Tensor,
    actual_end: ttnn.Tensor | None,
    compute_config: _RecurrenceComputeConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    a, b = ttnn.experimental.kda.reduce_affine_transforms(
        summary.a,
        summary.b,
        groups_per_head,
        local_rows=local_rows,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
        memory_config=KDA_OUTPUT_MEMORY_CONFIG,
        compute_kernel_config=compute_config.affine_prefix,
    )
    return _distributed_prefix(
        _AffineTransform(a, b),
        initial_state,
        sequence_parallel_axis=sequence_parallel_axis,
        selections=selections,
        compute_config=compute_config.affine_prefix,
    )


def _scan_sp_grouped_chunks(
    prepared: _PreparedChunks,
    initial_state: ttnn.Tensor,
    geometry: _RecurrenceGeometry,
    *,
    summary_group_chunks: int,
    groups: int,
    memory: ttnn.MemoryConfig,
    sequence_parallel_axis: int,
    selections: ChronologicalSelections,
    actual_start: ttnn.Tensor,
    actual_end: ttnn.Tensor | None,
    compute_config: _RecurrenceComputeConfig,
) -> RecurrenceResult:
    grouped = _reshape_chunks_for_groups(
        prepared, geometry, group_heads=geometry.batch_heads * groups, summary_group_chunks=summary_group_chunks
    )
    parts = ttnn.experimental.kda.summarize_chunk_recurrence(
        *grouped.as_kernel_args(),
        groups_per_head=groups,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
        memory_config=memory,
        compute_kernel_config=compute_config.preparation,
    )
    head_a, head_b, tail_a, tail_b = parts
    head = _AffineTransform(head_a, head_b)
    local_entry_state, prefix_final_state = _partition_prefix(
        head,
        initial_state,
        groups_per_head=groups,
        local_rows=geometry.local_rows,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
        selections=selections,
        compute_config=compute_config,
    )
    # The chronological prefix ends where the split tail begins, supplying its seed.
    group_entry_states = ttnn.experimental.kda.affine_exclusive_scan(
        head_a,
        head_b,
        local_entry_state,
        groups,
        local_rows=geometry.local_rows,
        tail_a=tail_a,
        tail_b=tail_b,
        tail_entry_states=prefix_final_state,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
        memory_config=KDA_DISTRIBUTED_PREFIX_MEMORY_CONFIG,
        compute_kernel_config=compute_config.affine_prefix,
    )
    scan = _scan_chunks(
        grouped,
        group_entry_states,
        prefix_final_state,
        groups_per_head=groups,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
        compute_config=compute_config.scan,
    )
    output = ttnn.reshape(
        scan.output, (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, geometry.value_dim)
    )
    # Keep the gather in the fixed graph: the device selector chooses the completed
    # tail when split, or the prefix's final state when unsplit, at runtime.
    gathered = ttnn.all_gather(
        _last_group_state(scan.final_state, geometry, groups),
        dim=0,
        cluster_axis=sequence_parallel_axis,
        memory_config=KDA_OUTPUT_MEMORY_CONFIG,
    )
    final_state = selections.select_final_state(gathered, prefix_final_state)
    return RecurrenceResult(
        output, ttnn.reshape(final_state, (geometry.batch_heads, geometry.key_dim, geometry.value_dim))
    )


class KDARecurrence:
    """Constructor-fixed KDA recurrence executor."""

    def __init__(
        self,
        device: ttnn.Device | ttnn.MeshDevice,
        program_config: KDARecurrenceProgramConfig,
        *,
        sequence_parallel_axis: int,
        local_rows: int,
        heads: int,
        key_dim: int,
        value_dim: int,
        batch: int = 1,
    ) -> None:
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
        if local_rows <= 0 or local_rows % KDA_CHUNK_SIZE:
            raise ValueError("recurrence local_rows must be positive and divisible by the chunk size")
        if min(batch, heads, key_dim, value_dim) <= 0:
            raise ValueError("recurrence dimensions must be positive")
        self._geometry = _RecurrenceGeometry(
            batch, local_rows, heads, key_dim, value_dim, KDA_CHUNK_SIZE, local_rows // KDA_CHUNK_SIZE
        )
        self._sequence_parallel_axis = sequence_parallel_axis
        self._sequence_parallel = (
            isinstance(device, ttnn.MeshDevice) and tuple(device.shape)[sequence_parallel_axis] > 1
        )
        grouped = self._sequence_parallel or program_config.local_scan_strategy == "grouped"
        if grouped:
            if key_dim != value_dim:
                raise ValueError("grouped KDA affine prefix currently requires K == V")
            self._summary_group_chunks = program_config.summary_group_chunks
            if self._geometry.num_chunks % self._summary_group_chunks:
                raise ValueError("summary_group_chunks must divide the constructed local chunk count")
            self._groups = self._geometry.num_chunks // self._summary_group_chunks
            self._summary_memory = _group_summary_memory_config(device, batch * heads * self._groups, key_dim)
        if self._sequence_parallel:
            self._execute = self._run_sp
        elif grouped:
            self._execute = self._run_grouped
        else:
            self._execute = self._run_direct

    def _prepare(
        self,
        *,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        gate: ttnn.Tensor,
        beta: ttnn.Tensor,
        initial_state: ttnn.Tensor,
    ) -> tuple[_PreparedChunks, ttnn.Tensor, _RecurrenceGeometry]:
        geometry = self._geometry
        if tuple(beta.shape) != (geometry.batch, geometry.local_rows, geometry.heads):
            raise ValueError("recurrence beta shape does not match constructed geometry")
        for name, tensor, width in (
            ("q", q, geometry.heads * geometry.key_dim),
            ("k", k, geometry.heads * geometry.key_dim),
            ("v", v, geometry.heads * geometry.value_dim),
            ("gate", gate, geometry.heads * geometry.key_dim),
        ):
            if tuple(tensor.shape) != (geometry.batch, geometry.local_rows, width):
                raise ValueError(f"recurrence {name} shape does not match constructed geometry")

        state = ttnn.reshape(
            initial_state,
            (geometry.batch_heads, geometry.key_dim, geometry.value_dim),
        )
        prepared = _prepare_chunk_terms(
            q,
            k,
            v,
            gate,
            beta,
            geometry,
            compute_config=self._compute_config,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=self._sequence_parallel_axis,
        )
        return prepared, state, geometry

    @staticmethod
    def _finish(scan: RecurrenceResult, geometry: _RecurrenceGeometry) -> RecurrenceResult:
        output = ttnn.reshape(scan.output, (geometry.batch_heads, geometry.local_rows, geometry.value_dim))
        final_state = ttnn.reshape(
            scan.final_state, (geometry.batch, geometry.heads, geometry.key_dim, geometry.value_dim)
        )
        return RecurrenceResult(output, final_state)

    def __call__(
        self,
        *,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None = None,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        gate: ttnn.Tensor,
        beta: ttnn.Tensor,
        initial_state: ttnn.Tensor,
        selections: ChronologicalSelections | None = None,
    ) -> RecurrenceResult:
        """Execute the constructed graph using caller-owned state and chronology."""
        if self._sequence_parallel != (selections is not None):
            raise ValueError("chronological selections must be provided exactly for sequence-parallel recurrence")
        prepared, state, geometry = self._prepare(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=beta,
            initial_state=initial_state,
            actual_start=actual_start,
            actual_end=actual_end,
        )
        return self._finish(self._execute(prepared, state, actual_start, actual_end, selections), geometry)

    def _run_direct(
        self,
        prepared: _PreparedChunks,
        state: ttnn.Tensor,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None,
        selections: ChronologicalSelections | None,
    ) -> RecurrenceResult:
        return _scan_chunks(
            prepared,
            state,
            state,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=self._sequence_parallel_axis,
            compute_config=self._compute_config.scan,
        )

    def _run_grouped(
        self,
        prepared: _PreparedChunks,
        state: ttnn.Tensor,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None,
        selections: ChronologicalSelections | None,
    ) -> RecurrenceResult:
        return _scan_local_grouped_chunks(
            prepared,
            state,
            self._geometry,
            summary_group_chunks=self._summary_group_chunks,
            groups=self._groups,
            memory=self._summary_memory,
            compute_config=self._compute_config,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=self._sequence_parallel_axis,
        )

    def _run_sp(
        self,
        prepared: _PreparedChunks,
        state: ttnn.Tensor,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None,
        selections: ChronologicalSelections | None,
    ) -> RecurrenceResult:
        return _scan_sp_grouped_chunks(
            prepared,
            state,
            self._geometry,
            summary_group_chunks=self._summary_group_chunks,
            groups=self._groups,
            memory=self._summary_memory,
            compute_config=self._compute_config,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=self._sequence_parallel_axis,
            selections=selections,
        )
