# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Python-owned KDA graph orchestration.

The KDA layer keeps collective and state-flow decisions here; device leaves only
perform the bespoke kernels exposed by ``ttnn.experimental.kda``.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from models.demos.deepseek_v3_d_p.tt.kda.offset import OffsetTopology, fragment_order


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


def _distributed_fragment_prefix(
    fragment_transforms: list[tuple[ttnn.Tensor, ttnn.Tensor]],
    initial_state: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    order: tuple[tuple[int, int], ...],
    sp_size: int,
    compute_config: ttnn.DeviceComputeKernelConfig,
) -> tuple[list[ttnn.Tensor], ttnn.Tensor]:
    """Compose fragment affine summaries in chronological order.

    ``fragment_transforms`` holds this chip's summaries in local fragment order;
    ``order`` lists ``(chip, fragment)`` chronologically. One fragment per chip is
    the whole-partition case and reduces exactly to composing over SP rank order
    when the boundary chip is rank zero.

    Entry states are returned per local fragment, delivered by ``mesh_partition``
    so each device receives only its own.
    """
    per_chip = len(fragment_transforms)
    transform_a, transform_b = fragment_transforms[0]
    batch_heads, key_dim = tuple(transform_a.shape)[0], tuple(transform_a.shape)[1]
    value_dim = transform_b.shape[-1]
    output_memory = KDA_OUTPUT_MEMORY_CONFIG
    working_memory = KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG

    packed_fragments = []
    for fragment_a, fragment_b in fragment_transforms:
        # Precision boundary: FP32 composition is transported as BF16.
        transport_a = ttnn.typecast(fragment_a, KDA_AFFINE_SUMMARY_DTYPE, memory_config=output_memory)
        transport_b = ttnn.typecast(fragment_b, KDA_AFFINE_SUMMARY_DTYPE, memory_config=output_memory)
        transport_a = ttnn.reshape(transport_a, (1, batch_heads, key_dim, key_dim))
        transport_b = ttnn.reshape(transport_b, (1, batch_heads, key_dim, value_dim))
        packed_fragments.append(ttnn.concat([transport_a, transport_b], dim=3, memory_config=output_memory))
    packed = packed_fragments[0] if per_chip == 1 else ttnn.concat(packed_fragments, dim=0, memory_config=output_memory)
    gathered = ttnn.all_gather(
        packed,
        dim=0,
        cluster_axis=sequence_parallel_axis,
        memory_config=output_memory,
    )

    carry = ttnn.to_memory_config(initial_state, working_memory)
    carry = ttnn.reshape(carry, (1, batch_heads, key_dim, value_dim))
    entry_states: list[ttnn.Tensor | None] = [None] * (sp_size * per_chip)
    for chip, fragment in order:
        index = chip * per_chip + fragment
        # Stored by physical slot so mesh_partition still hands each device its
        # own entry states, while the carry advances chronologically.
        entry_states[index] = carry
        transported_a = ttnn.slice(
            gathered,
            (index, 0, 0, 0),
            (index + 1, batch_heads, key_dim, key_dim),
            memory_config=working_memory,
        )
        transported_b = ttnn.slice(
            gathered,
            (index, 0, 0, key_dim),
            (index + 1, batch_heads, key_dim, key_dim + value_dim),
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

    replicated_entries = ttnn.concat(entry_states, dim=0, memory_config=output_memory)
    local_entries = ttnn.mesh_partition(
        replicated_entries,
        dim=0,
        cluster_axis=sequence_parallel_axis,
        memory_config=output_memory,
    )
    entries = [
        ttnn.reshape(
            ttnn.slice(
                local_entries,
                (fragment, 0, 0, 0),
                (fragment + 1, batch_heads, key_dim, value_dim),
                memory_config=output_memory,
            ),
            (batch_heads, key_dim, value_dim),
        )
        for fragment in range(per_chip)
    ]
    final_state = ttnn.reshape(ttnn.to_memory_config(carry, output_memory), (batch_heads, key_dim, value_dim))
    return entries, final_state


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


def _slice_chunk_range(prepared: _PreparedChunks, start_chunk: int, end_chunk: int) -> _PreparedChunks:
    """Restrict every prepared term to a half-open chunk range.

    Chunk preparation is token-local, so a fragment is a plain slice of its
    outputs -- no term needs recomputing.
    """

    def cut(term: ttnn.Tensor) -> ttnn.Tensor:
        shape = tuple(term.shape)
        return ttnn.slice(
            term,
            (0, start_chunk, 0, 0),
            (shape[0], end_chunk, shape[2], shape[3]),
            memory_config=KDA_PREPARATION_MEMORY_CONFIG,
        )

    return _PreparedChunks(*(cut(term) for term in prepared.as_kernel_args()))


def _group_fragment(
    fragment: _PreparedChunks,
    geometry: _RecurrenceGeometry,
    *,
    num_chunks: int,
    summary_group_chunks: int,
    compute_config: _RecurrenceComputeConfig,
) -> tuple[_PreparedChunks, ttnn.Tensor, ttnn.Tensor, int]:
    """Group one fragment's chunks and summarize each group."""
    group_chunks = _effective_summary_group_chunks(num_chunks, summary_group_chunks)
    groups_per_head = num_chunks // group_chunks
    grouped = _reshape_chunks_for_groups(
        fragment,
        geometry,
        group_heads=geometry.batch_heads * groups_per_head,
        summary_group_chunks=group_chunks,
    )
    summary_a, summary_b = _summarize_chunk_groups(grouped, geometry, compute_config=compute_config)
    return grouped, summary_a, summary_b, groups_per_head


def _scan_grouped_chunks(
    prepared: _PreparedChunks,
    initial_state: ttnn.Tensor,
    geometry: _RecurrenceGeometry,
    *,
    summary_group_chunks: int,
    sequence_parallel_axis: int | None,
    topology: OffsetTopology | None,
    compute_config: _RecurrenceComputeConfig,
) -> _ScanResult:
    """Scan the local partition, splitting it into fragments when the offset splits it.

    When the boundary chip holds two causally non-adjacent pieces, every chip
    splits at the same row. Mesh operations take group counts as scalars shared
    by the whole mesh, so a ragged split -- two fragments on one chip and one
    elsewhere -- is not expressible; splitting uniformly keeps every shape equal
    while only the boundary chip's fragments are actually non-adjacent.
    """
    split_chunk = topology.head_rows // geometry.chunk_size if topology is not None and topology.is_split else None
    if split_chunk is None:
        fragment_chunks = [(0, geometry.num_chunks)]
    else:
        fragment_chunks = [(0, split_chunk), (split_chunk, geometry.num_chunks)]

    fragments = [
        _group_fragment(
            prepared if len(fragment_chunks) == 1 else _slice_chunk_range(prepared, start, end),
            geometry,
            num_chunks=end - start,
            summary_group_chunks=summary_group_chunks,
            compute_config=compute_config,
        )
        for start, end in fragment_chunks
    ]

    prefix_initial_states = [initial_state] * len(fragments)
    distributed_final_state = None
    prefix_memory_config = KDA_LOCAL_PREFIX_MEMORY_CONFIG
    if sequence_parallel_axis is not None:
        if topology is None:
            raise ValueError("sequence-parallel recurrence requires an offset topology")
        partition_transforms = [
            ttnn.experimental.kda.reduce_affine_transforms(
                summary_a,
                summary_b,
                groups_per_head,
                memory_config=KDA_OUTPUT_MEMORY_CONFIG,
                compute_kernel_config=compute_config.affine_prefix,
            )
            for _, summary_a, summary_b, groups_per_head in fragments
        ]
        order = (
            fragment_order(topology) if split_chunk is not None else tuple((chip, 0) for chip in topology.chip_order)
        )
        prefix_initial_states, distributed_final_state = _distributed_fragment_prefix(
            partition_transforms,
            initial_state,
            sequence_parallel_axis=sequence_parallel_axis,
            order=order,
            sp_size=topology.sp_size,
            compute_config=compute_config.affine_prefix,
        )
        prefix_memory_config = KDA_DISTRIBUTED_PREFIX_MEMORY_CONFIG

    outputs = []
    last_local_state = None
    for (grouped, summary_a, summary_b, groups_per_head), entry_state, (start, end) in zip(
        fragments, prefix_initial_states, fragment_chunks, strict=True
    ):
        group_initial_states = ttnn.experimental.kda.affine_exclusive_scan(
            summary_a,
            summary_b,
            entry_state,
            groups_per_head,
            memory_config=prefix_memory_config,
            compute_kernel_config=compute_config.affine_prefix,
        )
        fragment_scan = _scan_chunks(grouped, group_initial_states, compute_config=compute_config.scan)
        outputs.append(
            ttnn.reshape(
                fragment_scan.output,
                (geometry.batch_heads, end - start, geometry.chunk_size, geometry.value_dim),
            )
        )
        last_local_state = _last_group_state(fragment_scan.final_state, geometry, groups_per_head)

    output = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1, memory_config=KDA_OUTPUT_MEMORY_CONFIG)
    if distributed_final_state is not None:
        return _ScanResult(output=output, final_state=distributed_final_state)
    return _ScanResult(output=output, final_state=last_local_state)


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
        topology: OffsetTopology | None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return ``(new_state, output)`` for directly named recurrence tensors.

        ``topology`` carries the chronological SP segment order and is required
        whenever sequence parallelism is enabled.
        """
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
                topology=topology,
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
