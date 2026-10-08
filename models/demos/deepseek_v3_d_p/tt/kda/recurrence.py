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


def _pack_transform(transform: _AffineTransform) -> ttnn.Tensor:
    """Pack a FP32 transform into the BF16 [1, BH, K, K+V] rows of [A | B] that the prefix gathers."""
    transform_a, transform_b = transform.a, transform.b
    batch_heads, key_dim = tuple(transform_a.shape)[0], tuple(transform_a.shape)[1]
    value_dim = transform_b.shape[-1]
    # Precision boundary: FP32 composition is transported as BF16. Rounding is elementwise, so
    # packing in FP32 (in L1) and narrowing once matches narrowing each half before packing.
    packed = ttnn.concat(
        [
            ttnn.reshape(transform_a, (1, batch_heads, key_dim, key_dim)),
            ttnn.reshape(transform_b, (1, batch_heads, key_dim, value_dim)),
        ],
        dim=3,
        memory_config=KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG,
    )
    return ttnn.typecast(packed, KDA_AFFINE_SUMMARY_DTYPE, memory_config=KDA_OUTPUT_MEMORY_CONFIG)


def _distributed_prefix(
    packed: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    compute_config: ttnn.DeviceComputeKernelConfig,
    actual_start: ttnn.Tensor,
    local_rows: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Compose one affine transform per chip in chronological order.

    ``packed`` is this chip's BF16 [1, BH, K, K+V] transform of [A | B] rows. The transforms are
    gathered as BF16; ``chain_affine_transforms`` derives the order from ``actual_start`` and
    applies them with FP32 accumulation. Return the local entry state and the replicated final
    carry on each independent TP line.
    """
    output_memory = KDA_OUTPUT_MEMORY_CONFIG
    gathered = ttnn.all_gather(
        packed,
        dim=0,
        cluster_axis=sequence_parallel_axis,
        memory_config=output_memory,
    )
    return ttnn.experimental.kda.chain_affine_transforms(
        gathered,
        initial_state,
        actual_start=actual_start,
        local_rows=local_rows,
        memory_config=output_memory,
        compute_kernel_config=compute_config,
        sequence_parallel_axis=sequence_parallel_axis,
    )


def select_final_state(
    rank_final: ttnn.Tensor,
    prefix_final_state: ttnn.Tensor,
    *,
    selections: ChronologicalSelections,
    actual_start: ttnn.Tensor,
    actual_end: ttnn.Tensor | None,
    local_rows: int,
    sequence_parallel_axis: int,
    fused: bool = True,
) -> ttnn.Tensor:
    """State after the last valid token: the owning rank's final for a separated tail, else the prefix carry.

    ``rank_final`` may hold every group's state as ``[B*H, groups, K, V]``; the last group is the final one.
    ``fused`` moves only the owner's state over the fabric, and nothing when the interval is unsplit;
    gathering every rank's final and selecting on device is its bit-exact reference.
    """
    if not fused:
        if len(rank_final.shape) == 4:
            batch_heads, groups, key_dim, value_dim = rank_final.shape
            rank_final = ttnn.reshape(
                ttnn.slice(
                    rank_final,
                    (0, groups - 1, 0, 0),
                    (batch_heads, groups, key_dim, value_dim),
                    memory_config=KDA_OUTPUT_MEMORY_CONFIG,
                ),
                (batch_heads, key_dim, value_dim),
            )
        gathered = ttnn.all_gather(
            rank_final, dim=0, cluster_axis=sequence_parallel_axis, memory_config=KDA_OUTPUT_MEMORY_CONFIG
        )
        return selections.select_final_state(gathered, prefix_final_state)
    return ttnn.experimental.kda.select_final_carry(
        rank_final,
        prefix_final_state,
        actual_start=actual_start,
        actual_end=actual_end,
        local_rows=local_rows,
        memory_config=KDA_OUTPUT_MEMORY_CONFIG,
        sequence_parallel_axis=sequence_parallel_axis,
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
        _pack_transform(_AffineTransform(a, b)),
        initial_state,
        sequence_parallel_axis=sequence_parallel_axis,
        compute_config=compute_config.affine_prefix,
        actual_start=actual_start,
        local_rows=local_rows,
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
    if groups == 1:
        # One group per head: the summary writes the packed transform the prefix gathers (the identity for a rank
        # without a head), and the group enters at the rank's own entry state.
        (packed,) = ttnn.experimental.kda.summarize_chunk_recurrence(
            *grouped.as_kernel_args(),
            groups_per_head=groups,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=sequence_parallel_axis,
            memory_config=KDA_OUTPUT_MEMORY_CONFIG,
            compute_kernel_config=compute_config.preparation,
            packed_head=True,
        )
        group_entry_states, prefix_final_state = _distributed_prefix(
            packed,
            initial_state,
            sequence_parallel_axis=sequence_parallel_axis,
            compute_config=compute_config.affine_prefix,
            actual_start=actual_start,
            local_rows=geometry.local_rows,
        )
    else:
        head_a, head_b, tail_a, tail_b = ttnn.experimental.kda.summarize_chunk_recurrence(
            *grouped.as_kernel_args(),
            groups_per_head=groups,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=sequence_parallel_axis,
            memory_config=memory,
            compute_kernel_config=compute_config.preparation,
        )
        local_entry_state, prefix_final_state = _partition_prefix(
            _AffineTransform(head_a, head_b),
            initial_state,
            groups_per_head=groups,
            local_rows=geometry.local_rows,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=sequence_parallel_axis,
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
    # The device chooses the completed tail when split, or the prefix's final state when unsplit, at runtime.
    final_state = select_final_state(
        ttnn.reshape(scan.final_state, (geometry.batch_heads, groups, geometry.key_dim, geometry.value_dim)),
        prefix_final_state,
        selections=selections,
        actual_start=actual_start,
        actual_end=actual_end,
        local_rows=geometry.local_rows,
        sequence_parallel_axis=sequence_parallel_axis,
    )
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
        result = self._execute(prepared, state, actual_start, actual_end, selections)
        # Release the chunk terms as soon as the scan consumed them, so every call sees the same free memory.
        for tensor in prepared.as_kernel_args():
            ttnn.deallocate(tensor)
        return self._finish(result, geometry)

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
