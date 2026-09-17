# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Python-owned KDA graph orchestration.

The KDA layer keeps collective and state-flow decisions here; device leaves only
perform the bespoke kernels exposed by ``ttnn.experimental.kda``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

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
    batch, local_rows, heads = tuple(beta.shape)
    key_dim = q_shape[2] // heads
    value_dim = v_shape[2] // heads
    return _RecurrenceGeometry(
        batch=batch,
        local_rows=local_rows,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=KDA_CHUNK_SIZE,
        num_chunks=local_rows // KDA_CHUNK_SIZE,
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
    *,
    groups_per_head: int,
    summary_memory_config: ttnn.MemoryConfig,
    compute_config: _RecurrenceComputeConfig,
) -> _AffineTransform:
    """Summarize whole groups, transported at the affine-prefix BF16 boundary."""
    a, b = ttnn.experimental.kda.summarize_chunk_recurrence(
        *grouped.as_kernel_args(),
        groups_per_head=groups_per_head,
        memory_config=summary_memory_config,
        compute_kernel_config=compute_config.preparation,
    )
    return _AffineTransform(
        *(ttnn.typecast(t, KDA_AFFINE_SUMMARY_DTYPE, memory_config=summary_memory_config) for t in (a, b))
    )


def _effective_summary_group_chunks(
    num_chunks: int,
    configured_group_chunks: int,
    max_groups: int,
) -> int:
    """Return a group size that divides the chunk count and fits the worker budget.

    ``configured_group_chunks`` is a performance ceiling; ``max_groups`` is a
    hardware limit, since every group needs its own summary owner core. A
    fragment whose chunk count is prime -- which an offset split can easily
    produce -- has no divisor at or below the ceiling other than one, and one
    chunk per group can demand far more owners than exist. When the ceiling and
    the budget conflict, the budget wins and the group grows past the ceiling.
    """
    preferred = 1
    for group_chunks in range(min(num_chunks, configured_group_chunks), 0, -1):
        if num_chunks % group_chunks == 0:
            preferred = group_chunks
            break
    if num_chunks // preferred <= max_groups:
        return preferred
    for group_chunks in range(preferred + 1, num_chunks + 1):
        if num_chunks % group_chunks == 0 and num_chunks // group_chunks <= max_groups:
            return group_chunks
    return num_chunks


def _scan_chunks(
    prepared: _PreparedChunks,
    initial_states: ttnn.Tensor,
    *,
    compute_config: ttnn.DeviceComputeKernelConfig,
    groups_per_head: int = 1,
) -> _ScanResult:
    output, final_states = ttnn.experimental.kda.recurrent_chunk_scan(
        *prepared.as_kernel_args(),
        initial_states,
        groups_per_head=groups_per_head,
        memory_config=KDA_OUTPUT_MEMORY_CONFIG,
        compute_kernel_config=compute_config,
    )
    return _ScanResult(output=output, final_state=final_states)


def _distributed_prefix(
    transform: _AffineTransform,
    initial_state: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    selections: ChronologicalSelections,
    compute_config: ttnn.DeviceComputeKernelConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Compose one affine transform per chip in chronological order.

    The entry tensor is stored by physical rank before mesh partitioning, while
    the carry advances in chronological rank order. Return local entry and the
    replicated final carry on each independent TP line.
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
    entry = ttnn.reshape(local_entries, (batch_heads, key_dim, value_dim))
    final_state = ttnn.reshape(ttnn.to_memory_config(carry, output_memory), (batch_heads, key_dim, value_dim))
    return entry, final_state


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


def _prepare_grouped_chunks(
    prepared: _PreparedChunks,
    geometry: _RecurrenceGeometry,
    *,
    summary_group_chunks: int,
) -> tuple[_PreparedChunks, int, ttnn.MemoryConfig]:
    grid = prepared.v_beta.device().compute_with_storage_grid_size()
    group_chunks = _effective_summary_group_chunks(
        geometry.num_chunks,
        summary_group_chunks,
        max_groups=max((grid.x * grid.y) // geometry.batch_heads, 1),
    )
    groups_per_head = geometry.num_chunks // group_chunks

    grouped = _reshape_chunks_for_groups(
        prepared,
        geometry,
        group_heads=geometry.batch_heads * groups_per_head,
        summary_group_chunks=group_chunks,
    )
    summary_memory_config = _group_summary_memory_config(
        prepared.v_beta.device(), geometry.batch_heads * groups_per_head, geometry.key_dim
    )

    return grouped, groups_per_head, summary_memory_config


def _ordinary_group_scan(
    grouped: _PreparedChunks,
    summary: _AffineTransform,
    initial_state: ttnn.Tensor,
    *,
    groups_per_head: int,
    prefix_memory_config: ttnn.MemoryConfig,
    compute_config: _RecurrenceComputeConfig,
) -> _ScanResult:
    entries = ttnn.experimental.kda.affine_exclusive_scan(
        summary.a,
        summary.b,
        initial_state,
        groups_per_head,
        memory_config=prefix_memory_config,
        compute_kernel_config=compute_config.affine_prefix,
    )
    return _scan_chunks(grouped, entries, groups_per_head=groups_per_head, compute_config=compute_config.scan)


def _scan_local_grouped_chunks(
    prepared: _PreparedChunks,
    initial_state: ttnn.Tensor,
    geometry: _RecurrenceGeometry,
    *,
    summary_group_chunks: int,
    compute_config: _RecurrenceComputeConfig,
) -> _ScanResult:
    grouped, groups, memory = _prepare_grouped_chunks(prepared, geometry, summary_group_chunks=summary_group_chunks)
    summary = _summarize_chunk_groups(
        grouped, groups_per_head=groups, summary_memory_config=memory, compute_config=compute_config
    )
    scan = _ordinary_group_scan(
        grouped,
        summary,
        initial_state,
        groups_per_head=groups,
        prefix_memory_config=KDA_LOCAL_PREFIX_MEMORY_CONFIG,
        compute_config=compute_config,
    )
    output = ttnn.reshape(
        scan.output, (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, geometry.value_dim)
    )
    return _ScanResult(output, _last_group_state(scan.final_state, geometry, groups))


def _partition_prefix(
    summary: _AffineTransform,
    initial_state: ttnn.Tensor,
    *,
    groups_per_head: int,
    local_rows: int,
    sequence_parallel_axis: int,
    selections: ChronologicalSelections,
    actual_start: ttnn.Tensor,
    compute_config: _RecurrenceComputeConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    a, b = ttnn.experimental.kda.reduce_affine_transforms(
        summary.a,
        summary.b,
        groups_per_head,
        local_rows=local_rows,
        actual_start=actual_start,
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
    sequence_parallel_axis: int,
    selections: ChronologicalSelections,
    actual_start: ttnn.Tensor,
    compute_config: _RecurrenceComputeConfig,
) -> _ScanResult:
    grouped, groups, memory = _prepare_grouped_chunks(prepared, geometry, summary_group_chunks=summary_group_chunks)
    parts = ttnn.experimental.kda.summarize_chunk_recurrence(
        *grouped.as_kernel_args(),
        groups_per_head=groups,
        actual_start=actual_start,
        sequence_parallel_axis=sequence_parallel_axis,
        memory_config=memory,
        compute_kernel_config=compute_config.preparation,
    )
    head_a, head_b, tail_a, tail_b = parts
    head = _AffineTransform(head_a, head_b)
    entry, tail_state = _partition_prefix(
        head,
        initial_state,
        groups_per_head=groups,
        local_rows=geometry.local_rows,
        actual_start=actual_start,
        sequence_parallel_axis=sequence_parallel_axis,
        selections=selections,
        compute_config=compute_config,
    )
    entries = ttnn.experimental.kda.affine_exclusive_scan(
        head_a,
        head_b,
        entry,
        groups,
        local_rows=geometry.local_rows,
        tail_a=tail_a,
        tail_b=tail_b,
        tail_state=tail_state,
        actual_start=actual_start,
        sequence_parallel_axis=sequence_parallel_axis,
        memory_config=KDA_DISTRIBUTED_PREFIX_MEMORY_CONFIG,
        compute_kernel_config=compute_config.affine_prefix,
    )
    output, final_states = ttnn.experimental.kda.recurrent_chunk_scan(
        *grouped.as_kernel_args(),
        entries,
        groups_per_head=groups,
        tail_state=tail_state,
        actual_start=actual_start,
        sequence_parallel_axis=sequence_parallel_axis,
        memory_config=KDA_OUTPUT_MEMORY_CONFIG,
        compute_kernel_config=compute_config.scan,
    )
    output = ttnn.reshape(output, (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, geometry.value_dim))
    gathered = ttnn.all_gather(
        _last_group_state(final_states, geometry, groups),
        dim=0,
        cluster_axis=sequence_parallel_axis,
        memory_config=KDA_OUTPUT_MEMORY_CONFIG,
    )
    final = selections.select_final_state(gathered, tail_state)
    return _ScanResult(output, ttnn.reshape(final, (geometry.batch_heads, geometry.key_dim, geometry.value_dim)))


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

    def _prepare(
        self,
        *,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        gate: ttnn.Tensor,
        beta: ttnn.Tensor,
        initial_state: ttnn.Tensor,
    ) -> tuple[_PreparedChunks, ttnn.Tensor, _RecurrenceGeometry]:
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
        return prepared, state, geometry

    @staticmethod
    def _finish(scan: _ScanResult, geometry: _RecurrenceGeometry) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        output = ttnn.reshape(scan.output, (geometry.batch_heads, geometry.local_rows, geometry.value_dim))
        final_state = ttnn.reshape(
            scan.final_state, (geometry.batch, geometry.heads, geometry.key_dim, geometry.value_dim)
        )
        return final_state, output

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
        """Run local direct or grouped recurrence without SP routing metadata."""
        prepared, state, geometry = self._prepare(q=q, k=k, v=v, gate=gate, beta=beta, initial_state=initial_state)
        scan = (
            _scan_local_grouped_chunks(
                prepared,
                state,
                geometry,
                summary_group_chunks=self._summary_group_chunks,
                compute_config=self._compute_config,
            )
            if self._use_grouped_scan
            else _scan_chunks(prepared, state, compute_config=self._compute_config.scan)
        )
        return self._finish(scan, geometry)

    def sequence_parallel(
        self,
        *,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        gate: ttnn.Tensor,
        beta: ttnn.Tensor,
        initial_state: ttnn.Tensor,
        selections: ChronologicalSelections,
        actual_start: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run SP recurrence from the layer's normalized chronological topology."""
        prepared, state, geometry = self._prepare(q=q, k=k, v=v, gate=gate, beta=beta, initial_state=initial_state)
        scan = _scan_sp_grouped_chunks(
            prepared,
            state,
            geometry,
            summary_group_chunks=self._summary_group_chunks,
            sequence_parallel_axis=cast(int, self._sequence_parallel_axis),
            selections=selections,
            actual_start=actual_start,
            compute_config=self._compute_config,
        )
        return self._finish(scan, geometry)
