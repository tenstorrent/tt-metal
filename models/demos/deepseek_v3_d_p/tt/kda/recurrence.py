# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Python-owned KDA graph orchestration.

The KDA layer keeps collective and state-flow decisions here; device leaves only
perform the bespoke kernels exposed by ``ttnn.experimental.kda``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
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
class _ScanResult:
    output: ttnn.Tensor
    final_state: ttnn.Tensor


@dataclass(frozen=True)
class _RecurrenceComputeConfig:
    preparation: ttnn.DeviceComputeKernelConfig
    affine_prefix: ttnn.DeviceComputeKernelConfig
    scan: ttnn.DeviceComputeKernelConfig


def _recurrence_geometry(
    q: ttnn.Tensor,
    v: ttnn.Tensor,
    beta: ttnn.Tensor,
) -> _RecurrenceGeometry:
    """Derive host-only execution metadata from layer-produced tensors."""
    q_shape = tuple(q.shape)
    v_shape = tuple(v.shape)
    batch, sequence, heads = tuple(beta.shape)
    key_dim = q_shape[2] // heads
    value_dim = v_shape[2] // heads
    return _RecurrenceGeometry(
        batch=batch,
        sequence=sequence,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=KDA_CHUNK_SIZE,
        num_chunks=sequence // KDA_CHUNK_SIZE,
    )


def _prepare_chunk_terms(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    gate: ttnn.Tensor,
    beta: ttnn.Tensor,
    geometry: _RecurrenceGeometry,
    *,
    compute_config: _RecurrenceComputeConfig,
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
    # Precision boundary: summary-pair math is FP32; summaries are stored and transported as BF16.
    summary_a = ttnn.typecast(affine_a, KDA_AFFINE_SUMMARY_DTYPE, memory_config=summary_memory_config)
    summary_b = ttnn.typecast(affine_b, KDA_AFFINE_SUMMARY_DTYPE, memory_config=summary_memory_config)
    return summary_a, summary_b


def _effective_summary_group_chunks(num_chunks: int, configured_group_chunks: int) -> int:
    """Return the largest configured-or-smaller group size that divides the local chunk count."""
    for group_chunks in range(min(num_chunks, configured_group_chunks), 0, -1):
        if num_chunks % group_chunks == 0:
            return group_chunks
    return 1


def _scan_chunks(
    prepared: _PreparedChunks,
    initial_states: ttnn.Tensor,
    *,
    compute_config: ttnn.DeviceComputeKernelConfig,
) -> _ScanResult:
    output, final_states = ttnn.experimental.kda.recurrent_chunk_scan(
        *prepared.as_kernel_args(),
        initial_states,
        memory_config=KDA_OUTPUT_MEMORY_CONFIG,
        compute_kernel_config=compute_config,
    )
    return _ScanResult(output=output, final_state=final_states)


def _distributed_affine_prefix(
    transform_a: ttnn.Tensor,
    transform_b: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    compute_config: ttnn.DeviceComputeKernelConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Compose SP partition affine summaries and return entry/final carries."""
    shape = tuple(transform_a.shape)

    mesh_device = transform_a.device()
    mesh_shape = tuple(mesh_device.shape)
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
    entry_state = ttnn.reshape(entry_state, (batch_heads, key_dim, value_dim))
    final_state = ttnn.reshape(final_state, (batch_heads, key_dim, value_dim))
    return entry_state, final_state


def _slice_prepared_chunks(prepared: _PreparedChunks, begin: int, end: int) -> _PreparedChunks:
    def slice_chunks(tensor: ttnn.Tensor) -> ttnn.Tensor:
        start = [0] * len(tensor.shape)
        stop = list(tensor.shape)
        start[1] = begin
        stop[1] = end
        return ttnn.slice(tensor, tuple(start), tuple(stop), memory_config=KDA_PREPARATION_MEMORY_CONFIG)

    return _PreparedChunks(*(slice_chunks(tensor) for tensor in prepared.as_kernel_args()))


def _summarize_one_segment(
    prepared: _PreparedChunks,
    geometry: _RecurrenceGeometry,
    *,
    compute_config: _RecurrenceComputeConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    grouped = _reshape_chunks_for_groups(
        prepared,
        geometry,
        group_heads=geometry.batch_heads,
        summary_group_chunks=geometry.num_chunks,
    )
    return _summarize_chunk_groups(grouped, geometry, compute_config=compute_config)


def _gather_affine_summary(
    transform_a: ttnn.Tensor,
    transform_b: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    batch_heads, key_dim, _ = transform_a.shape
    value_dim = transform_b.shape[-1]
    transport_a = ttnn.reshape(
        ttnn.to_memory_config(
            transform_a,
            KDA_OUTPUT_MEMORY_CONFIG,
        ),
        (1, batch_heads, key_dim, key_dim),
    )
    transport_b = ttnn.reshape(
        ttnn.to_memory_config(
            transform_b,
            KDA_OUTPUT_MEMORY_CONFIG,
        ),
        (1, batch_heads, key_dim, value_dim),
    )
    return (
        ttnn.all_gather(
            transport_a,
            dim=0,
            cluster_axis=sequence_parallel_axis,
            memory_config=KDA_OUTPUT_MEMORY_CONFIG,
        ),
        ttnn.all_gather(
            transport_b,
            dim=0,
            cluster_axis=sequence_parallel_axis,
            memory_config=KDA_OUTPUT_MEMORY_CONFIG,
        ),
    )


def _apply_rank_affine(
    carry: ttnn.Tensor,
    gathered_a: ttnn.Tensor,
    gathered_b: ttnn.Tensor,
    rank: int,
    geometry: _RecurrenceGeometry,
    *,
    compute_config: ttnn.DeviceComputeKernelConfig,
) -> ttnn.Tensor:
    working_memory = KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG
    rank_a = ttnn.slice(
        gathered_a,
        (rank, 0, 0, 0),
        (rank + 1, geometry.batch_heads, geometry.key_dim, geometry.key_dim),
        memory_config=working_memory,
    )
    rank_b = ttnn.slice(
        gathered_b,
        (rank, 0, 0, 0),
        (rank + 1, geometry.batch_heads, geometry.key_dim, geometry.value_dim),
        memory_config=working_memory,
    )
    updated = ttnn.matmul(
        ttnn.typecast(rank_a, KDA_RECURRENT_STATE_DTYPE, memory_config=working_memory),
        carry,
        memory_config=working_memory,
        dtype=KDA_RECURRENT_STATE_DTYPE,
        compute_kernel_config=compute_config,
    )
    return ttnn.add(
        updated,
        ttnn.typecast(rank_b, KDA_RECURRENT_STATE_DTYPE, memory_config=working_memory),
        memory_config=working_memory,
    )


def _partition_entry_states_prototype(
    entries: list[ttnn.Tensor],
    geometry: _RecurrenceGeometry,
    *,
    sequence_parallel_axis: int,
) -> ttnn.Tensor:
    replicated = ttnn.concat(entries, dim=0, memory_config=KDA_OUTPUT_MEMORY_CONFIG)
    partitioned = ttnn.mesh_partition(
        replicated,
        dim=0,
        cluster_axis=sequence_parallel_axis,
        memory_config=KDA_OUTPUT_MEMORY_CONFIG,
    )
    return ttnn.reshape(partitioned, (geometry.batch_heads, geometry.key_dim, geometry.value_dim))


def _scan_offset_segments_prototype(
    prepared: _PreparedChunks,
    initial_state: ttnn.Tensor,
    geometry: _RecurrenceGeometry,
    *,
    actual_start: int,
    sequence_parallel_axis: int,
    compute_config: _RecurrenceComputeConfig,
) -> _ScanResult:
    """Scan MLA-placed rows as head, intervening ranks, then boundary tail.

    This intentionally favors a direct proof of the proposed schedule over
    launch efficiency. Every rank is split at the boundary row so all mesh
    shards keep equal shapes; on non-boundary ranks the two scans are adjacent.
    Only affine summaries and entry states cross SP.
    """
    sp_size = tuple(prepared.v_beta.device().shape)[sequence_parallel_axis]
    local_sequence = geometry.sequence
    split = actual_start % local_sequence
    boundary_rank = (actual_start // local_sequence) % sp_size
    head_chunks = (local_sequence - split) // geometry.chunk_size

    if split == 0:
        summary_a, summary_b = _summarize_one_segment(prepared, geometry, compute_config=compute_config)
        gathered_a, gathered_b = _gather_affine_summary(
            summary_a, summary_b, sequence_parallel_axis=sequence_parallel_axis
        )
        carry = ttnn.reshape(
            ttnn.to_memory_config(initial_state, KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG),
            (1, geometry.batch_heads, geometry.key_dim, geometry.value_dim),
        )
        entries: list[ttnn.Tensor | None] = [None] * sp_size
        for step in range(sp_size):
            rank = (boundary_rank + step) % sp_size
            entries[rank] = carry
            carry = _apply_rank_affine(
                carry, gathered_a, gathered_b, rank, geometry, compute_config=compute_config.affine_prefix
            )
        partitioned = _partition_entry_states_prototype(
            [entry for entry in entries if entry is not None], geometry, sequence_parallel_axis=sequence_parallel_axis
        )
        scan = _scan_chunks(prepared, partitioned, compute_config=compute_config.scan)
        return _ScanResult(output=scan.output, final_state=ttnn.to_memory_config(carry, KDA_OUTPUT_MEMORY_CONFIG))

    tail_chunks = geometry.num_chunks - head_chunks
    head_geometry = replace(geometry, sequence=head_chunks * geometry.chunk_size, num_chunks=head_chunks)
    tail_geometry = replace(geometry, sequence=tail_chunks * geometry.chunk_size, num_chunks=tail_chunks)
    head_prepared = _slice_prepared_chunks(prepared, 0, head_chunks)
    tail_prepared = _slice_prepared_chunks(prepared, head_chunks, geometry.num_chunks)
    head_a, head_b = _summarize_one_segment(head_prepared, head_geometry, compute_config=compute_config)
    tail_a, tail_b = _summarize_one_segment(tail_prepared, tail_geometry, compute_config=compute_config)
    gathered_head_a, gathered_head_b = _gather_affine_summary(
        head_a, head_b, sequence_parallel_axis=sequence_parallel_axis
    )
    gathered_tail_a, gathered_tail_b = _gather_affine_summary(
        tail_a, tail_b, sequence_parallel_axis=sequence_parallel_axis
    )

    carry = ttnn.reshape(
        ttnn.to_memory_config(initial_state, KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG),
        (1, geometry.batch_heads, geometry.key_dim, geometry.value_dim),
    )
    head_entries: list[ttnn.Tensor | None] = [None] * sp_size
    tail_entries: list[ttnn.Tensor | None] = [None] * sp_size
    head_entries[boundary_rank] = carry
    carry = _apply_rank_affine(
        carry,
        gathered_head_a,
        gathered_head_b,
        boundary_rank,
        geometry,
        compute_config=compute_config.affine_prefix,
    )
    for step in range(1, sp_size):
        rank = (boundary_rank + step) % sp_size
        head_entries[rank] = carry
        carry = _apply_rank_affine(
            carry, gathered_head_a, gathered_head_b, rank, geometry, compute_config=compute_config.affine_prefix
        )
        tail_entries[rank] = carry
        carry = _apply_rank_affine(
            carry, gathered_tail_a, gathered_tail_b, rank, geometry, compute_config=compute_config.affine_prefix
        )
    tail_entries[boundary_rank] = carry
    carry = _apply_rank_affine(
        carry,
        gathered_tail_a,
        gathered_tail_b,
        boundary_rank,
        geometry,
        compute_config=compute_config.affine_prefix,
    )

    head_initial = _partition_entry_states_prototype(
        [entry for entry in head_entries if entry is not None],
        geometry,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    tail_initial = _partition_entry_states_prototype(
        [entry for entry in tail_entries if entry is not None],
        geometry,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    head_scan = _scan_chunks(head_prepared, head_initial, compute_config=compute_config.scan)
    tail_scan = _scan_chunks(tail_prepared, tail_initial, compute_config=compute_config.scan)
    return _ScanResult(
        output=ttnn.concat([head_scan.output, tail_scan.output], dim=1, memory_config=KDA_OUTPUT_MEMORY_CONFIG),
        final_state=ttnn.to_memory_config(carry, KDA_OUTPUT_MEMORY_CONFIG),
    )


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


def _scan_grouped_chunks(
    prepared: _PreparedChunks,
    initial_state: ttnn.Tensor,
    geometry: _RecurrenceGeometry,
    *,
    summary_group_chunks: int,
    sequence_parallel_axis: int | None,
    compute_config: _RecurrenceComputeConfig,
) -> _ScanResult:
    group_chunks = _effective_summary_group_chunks(geometry.num_chunks, summary_group_chunks)
    groups_per_head = geometry.num_chunks // group_chunks
    group_heads = geometry.batch_heads * groups_per_head

    grouped = _reshape_chunks_for_groups(
        prepared,
        geometry,
        group_heads=group_heads,
        summary_group_chunks=group_chunks,
    )
    summary_a, summary_b = _summarize_chunk_groups(grouped, geometry, compute_config=compute_config)

    prefix_initial_state = initial_state
    prefix_memory_config = KDA_LOCAL_PREFIX_MEMORY_CONFIG
    if sequence_parallel_axis is not None:
        partition_a, partition_b = ttnn.experimental.kda.reduce_affine_transforms(
            summary_a,
            summary_b,
            groups_per_head,
            memory_config=KDA_OUTPUT_MEMORY_CONFIG,
            compute_kernel_config=compute_config.affine_prefix,
        )
        prefix_initial_state, distributed_final_state = _distributed_affine_prefix(
            partition_a,
            partition_b,
            initial_state,
            sequence_parallel_axis=sequence_parallel_axis,
            compute_config=compute_config.affine_prefix,
        )
        prefix_memory_config = KDA_DISTRIBUTED_PREFIX_MEMORY_CONFIG

    group_initial_states = ttnn.experimental.kda.affine_exclusive_scan(
        summary_a,
        summary_b,
        prefix_initial_state,
        groups_per_head,
        memory_config=prefix_memory_config,
        compute_kernel_config=compute_config.affine_prefix,
    )
    grouped_scan = _scan_chunks(grouped, group_initial_states, compute_config=compute_config.scan)
    output = ttnn.reshape(
        grouped_scan.output,
        (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, geometry.value_dim),
    )

    if sequence_parallel_axis is not None:
        return _ScanResult(output=output, final_state=distributed_final_state)
    return _ScanResult(
        output=output,
        final_state=_last_group_state(grouped_scan.final_state, geometry, groups_per_head),
    )


class KDARecurrence:
    """Constructor-fixed KDA recurrence executor."""

    def __init__(
        self,
        device: ttnn.Device | ttnn.MeshDevice,
        program_config: KDARecurrenceProgramConfig,
        *,
        sequence_parallel_axis: int | None,
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
        self._summary_group_chunks = program_config.summary_group_chunks
        self._sequence_parallel_axis = sequence_parallel_axis
        self._use_grouped_scan = sequence_parallel_axis is not None or program_config.local_scan_strategy == "grouped"

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
        geometry = _recurrence_geometry(q, v, beta)

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
        )
        if self._use_grouped_scan:
            scan = _scan_grouped_chunks(
                prepared,
                state,
                geometry,
                summary_group_chunks=self._summary_group_chunks,
                sequence_parallel_axis=self._sequence_parallel_axis,
                compute_config=self._compute_config,
            )
        else:
            scan = _scan_chunks(prepared, state, compute_config=self._compute_config.scan)
        output = ttnn.reshape(
            scan.output,
            (geometry.batch_heads, geometry.sequence, geometry.value_dim),
        )
        final_state = ttnn.reshape(
            scan.final_state,
            (geometry.batch, geometry.heads, geometry.key_dim, geometry.value_dim),
        )
        return final_state, output

    def offset_sequential_tail_prototype(
        self,
        *,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        gate: ttnn.Tensor,
        beta: ttnn.Tensor,
        initial_state: ttnn.Tensor,
        actual_start: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Throwaway head/full-ranks/tail recurrence schedule."""
        if self._sequence_parallel_axis is None:
            raise ValueError("sequential-tail offset prototype requires sequence parallelism")
        geometry = _recurrence_geometry(q, v, beta)
        state = ttnn.reshape(initial_state, (geometry.batch_heads, geometry.key_dim, geometry.value_dim))
        prepared = _prepare_chunk_terms(
            q,
            k,
            v,
            gate,
            beta,
            geometry,
            compute_config=self._compute_config,
        )
        scan = _scan_offset_segments_prototype(
            prepared,
            state,
            geometry,
            actual_start=actual_start,
            sequence_parallel_axis=self._sequence_parallel_axis,
            compute_config=self._compute_config,
        )
        output = ttnn.reshape(scan.output, (geometry.batch_heads, geometry.sequence, geometry.value_dim))
        final_state = ttnn.reshape(
            scan.final_state,
            (geometry.batch, geometry.heads, geometry.key_dim, geometry.value_dim),
        )
        return final_state, output
