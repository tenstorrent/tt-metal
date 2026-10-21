# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Python-owned KDA graph orchestration.

The KDA layer keeps collective and state-flow decisions here; device leaves only
perform the bespoke kernels exposed by ``ttnn.experimental.kda``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.config import (
    KDA_AFFINE_SUMMARY_DTYPE,
    KDA_CHUNK_SIZE,
    KDA_DISTRIBUTED_PREFIX_L1_BYTES_PER_CORE,
    KDA_DISTRIBUTED_PREFIX_MEMORY_CONFIG,
    KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG,
    KDA_HIGH_BW_PREFIX_GATHER_LINKS,
    KDA_LOCAL_PREFIX_MEMORY_CONFIG,
    KDA_OUTPUT_MEMORY_CONFIG,
    KDA_PREP_OUTPUT_BF16_MASK,
    KDA_PREPARATION_L1_BYTES_PER_CORE,
    KDARecurrenceProgramConfig,
    l1_when_it_fits,
    preparation_bytes,
)
from models.demos.deepseek_v3_d_p.tt.kda.entry_correction import EntryCorrection
from models.demos.deepseek_v3_d_p.tt.kda.relay_affine_prefix import (
    RelayAffinePrefix,
    RelayOutputGate,
    fabric_wraps_axis,
    persistent_semaphores,
)


def _group_summary_memory_config(
    device: ttnn.Device, tail: bool, group_heads: int, key_dim: int, value_dim: int
) -> ttnn.MemoryConfig:
    """L1 shards of each group-head's [S, 1, K, K+V] summaries, on the core its prefix worker occupies; S = 2 with
    tail transforms."""
    slots = 2 if tail else 1
    grid = device.compute_with_storage_grid_size()
    capacity = grid.x * grid.y
    if group_heads > capacity:
        raise ValueError(f"grouped KDA needs {group_heads} summary owners, but only {capacity} are supported")
    worker_cores = ttnn.num_cores_to_corerangeset(group_heads, grid, row_wise=True)
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[slots, 1, key_dim, key_dim + value_dim],
            grid=worker_cores,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


@dataclass(frozen=True)
class AffineTransform:
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
    final_state: ttnn.Tensor | PendingFinalCarry
    # The output of the relay's RelayOutputGate, when one ran.
    output_gate: ttnn.Tensor | None = None


@dataclass(frozen=True)
class PendingFinalCarry:
    """The relay's final carry, left for the caller to run later: run(projection) dispatches it, with the
    CarryProjection in its dispatch when given, and returns the final state and the projection's output."""

    run: object


@dataclass(frozen=True)
class CarryProjection:
    """input @ weight, run in the final carry's dispatch on the worker cores the carry leaves
    (ttnn.experimental.kda.select_final_carry_with_projection)."""

    input: ttnn.Tensor
    weight: ttnn.Tensor
    program_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig  # ttnn.linear's, on the cores it names
    compute_config: ttnn.DeviceComputeKernelConfig
    memory_config: ttnn.MemoryConfig


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
    gate_scale: float,
    beta_logits_column_offset: int,
    memory_config: ttnn.MemoryConfig,
) -> _PreparedChunks:
    # Chunk preparation reads each head's logit column in place from the token-major projection and activates it.
    outputs = ttnn.experimental.kda.prepare_chunk_recurrence(
        q,
        k,
        v,
        gate,
        beta,
        geometry.heads,
        memory_config=memory_config,
        compute_kernel_config=compute_config.preparation,
        output_bf16_mask=KDA_PREP_OUTPUT_BF16_MASK,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
        gate_scale=gate_scale,
        beta_logits_column_offset=beta_logits_column_offset,
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
    summary_tail: bool,
    summary_memory_config: ttnn.MemoryConfig,
    compute_config: _RecurrenceComputeConfig,
) -> ttnn.Tensor:
    """Summarize whole groups for the single-rank path at the BF16 prefix boundary."""
    # The scan never reads a group without a valid chunk, so those slots need no identity.
    return ttnn.experimental.kda.summarize_chunk_recurrence(
        *grouped.as_kernel_args(),
        groups_per_head=groups_per_head,
        memory_config=summary_memory_config,
        compute_kernel_config=compute_config.preparation,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
        tail=summary_tail,
        init=False,
    )


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


def pack_transform(transform: AffineTransform) -> ttnn.Tensor:
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


def _sp_links(device: ttnn.MeshDevice, sequence_parallel_axis: int) -> int:
    """Ethernet links between neighbours on the SP axis; 0 for a system the link table does not know."""
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_num_links

    try:
        return get_num_links(device, sequence_parallel_axis)
    except (KeyError, ValueError):
        return 0


def distributed_prefix(
    packed: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    compute_config: ttnn.DeviceComputeKernelConfig,
    actual_start: ttnn.Tensor,
    local_rows: int,
    gather_buffer: ttnn.Tensor | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Compose one affine transform per chip in chronological order.

    ``packed`` is this chip's BF16 [1, BH, K, K+V] transform of [A | B] rows. The transforms are
    gathered as BF16; ``chain_affine_transforms`` derives the order from ``actual_start`` and
    applies them with FP32 accumulation. Return the local entry state and the replicated final
    carry on each independent TP line.
    """
    device = packed.device()
    _, batch_heads, key_dim, row_width = tuple(packed.shape)
    # The chain's entry state and final carry share initial_state's FP32 [B*H, K, V] shape.
    chain_output_bytes = 2 * batch_heads * key_dim * (row_width - key_dim) * initial_state.element_size()
    if gather_buffer is not None:
        # The gather lands in the persistent DRAM buffer, so only the chain's outputs count against L1.
        output_memory = l1_when_it_fits(device, chain_output_bytes, KDA_DISTRIBUTED_PREFIX_L1_BYTES_PER_CORE)
        gathered = ttnn.experimental.high_bw_all_gather(
            packed,
            dim=0,
            output_tensor=gather_buffer,
            cluster_axis=sequence_parallel_axis,
            num_links=KDA_HIGH_BW_PREFIX_GATHER_LINKS,
        )
    else:
        gathered_bytes = tuple(device.shape)[sequence_parallel_axis] * packed.logical_volume() * packed.element_size()
        output_memory = l1_when_it_fits(
            device, gathered_bytes + chain_output_bytes, KDA_DISTRIBUTED_PREFIX_L1_BYTES_PER_CORE
        )
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
    summaries: ttnn.Tensor,
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
    group_entry_states = ttnn.experimental.kda.affine_exclusive_scan(
        summaries,
        initial_state,
        groups_per_head,
        memory_config=prefix_memory_config,
        compute_kernel_config=compute_config.affine_prefix,
        actual_start=actual_start,
        actual_end=actual_end,
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
    summary_tail: bool,
    memory: ttnn.MemoryConfig,
    compute_config: _RecurrenceComputeConfig,
) -> RecurrenceResult:
    grouped = _reshape_chunks_for_groups(
        prepared, geometry, group_heads=geometry.batch_heads * groups, summary_group_chunks=summary_group_chunks
    )
    summaries = _summarize_chunk_groups(
        grouped,
        groups_per_head=groups,
        summary_tail=summary_tail,
        summary_memory_config=memory,
        compute_config=compute_config,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    scan = _ordinary_group_scan(
        grouped,
        summaries,
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
    summaries: ttnn.Tensor,
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
        summaries,
        groups_per_head,
        local_rows=local_rows,
        actual_start=actual_start,
        actual_end=actual_end,
        sequence_parallel_axis=sequence_parallel_axis,
        memory_config=KDA_OUTPUT_MEMORY_CONFIG,
        compute_kernel_config=compute_config.affine_prefix,
    )
    return distributed_prefix(
        pack_transform(AffineTransform(a, b)),
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
    summary_tail: bool,
    memory: ttnn.MemoryConfig,
    sequence_parallel_axis: int,
    actual_start: ttnn.Tensor,
    actual_end: ttnn.Tensor | None,
    compute_config: _RecurrenceComputeConfig,
    gather_buffer: ttnn.Tensor | None = None,
    affine_relay: RelayAffinePrefix | None = None,
    final_carry: _RelayFinalCarry | None = None,
    output_gate: RelayOutputGate | None = None,
    defer_final_carry: bool = False,
    correction: EntryCorrection | None = None,
) -> RecurrenceResult:
    grouped = _reshape_chunks_for_groups(
        prepared, geometry, group_heads=geometry.batch_heads * groups, summary_group_chunks=summary_group_chunks
    )
    correct = correction is not None and groups == 1 and affine_relay is not None
    if groups == 1:
        # One group per head: the summary writes the packed transform the prefix gathers (the identity for a rank
        # without a head), and the group enters at the rank's own entry state. With the entry correction it also
        # leaves each chunk's output terms, from which the correction forms the outputs.
        summary = ttnn.experimental.kda.summarize_chunk_recurrence(
            *grouped.as_kernel_args(),
            groups_per_head=groups,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=sequence_parallel_axis,
            memory_config=correction.memory_config if correct else KDA_OUTPUT_MEMORY_CONFIG,
            compute_kernel_config=compute_config.preparation,
            emit_outputs=correct,
            emit_memory_config=correction.terms_memory_config if correct else None,
        )
        packed, *emitted = summary if correct else [summary]
        if correct:
            # The summary is the chunk terms' last reader: free their L1 before the relay and its companion.
            for tensor in grouped.as_kernel_args():
                ttnn.deallocate(tensor)
        if affine_relay is not None:
            group_entry_states, prefix_final_state, output_gate = affine_relay(
                packed, initial_state, actual_start, output_gate=output_gate
            )
        else:
            group_entry_states, prefix_final_state = distributed_prefix(
                packed,
                initial_state,
                sequence_parallel_axis=sequence_parallel_axis,
                compute_config=compute_config.affine_prefix,
                actual_start=actual_start,
                local_rows=geometry.local_rows,
                gather_buffer=gather_buffer,
            )
    else:
        # The reduction and the scan read only slots with a valid segment (the reduction writes its own identity for
        # a rank without a head), so the other slots need no identity.
        summaries = ttnn.experimental.kda.summarize_chunk_recurrence(
            *grouped.as_kernel_args(),
            groups_per_head=groups,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=sequence_parallel_axis,
            memory_config=memory,
            compute_kernel_config=compute_config.preparation,
            tail=summary_tail,
            init=False,
        )
        local_entry_state, prefix_final_state = _partition_prefix(
            summaries,
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
            summaries,
            local_entry_state,
            groups,
            local_rows=geometry.local_rows,
            tail_entry_states=prefix_final_state,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=sequence_parallel_axis,
            memory_config=KDA_DISTRIBUTED_PREFIX_MEMORY_CONFIG,
            compute_kernel_config=compute_config.affine_prefix,
        )
    if correct:
        output, rank_final = correction(*emitted, group_entry_states, prefix_final_state, actual_start, actual_end)
        for tensor in (*emitted, group_entry_states):
            ttnn.deallocate(tensor)
        scan = RecurrenceResult(output=output, final_state=rank_final)
    else:
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
    rank_final = ttnn.reshape(scan.final_state, (geometry.batch_heads, groups, geometry.key_dim, geometry.value_dim))
    if final_carry is not None and defer_final_carry:
        # Nothing in the layer reads the final state: the caller runs the carry beside a later program.
        return RecurrenceResult(
            output,
            PendingFinalCarry(
                lambda projection=None: final_carry(
                    rank_final, prefix_final_state, actual_start, actual_end, geometry.local_rows, projection
                )
            ),
            output_gate,
        )
    if final_carry is not None:
        # The relay leaves the prefix's final state on the first and last chronological ranks only: without a split
        # tail the last rank multicasts it, with one the tail's owner multicasts its own.
        final_state, _ = final_carry(rank_final, prefix_final_state, actual_start, actual_end, geometry.local_rows)
        return RecurrenceResult(output, final_state, output_gate)
    # The device chooses the completed tail when split, or the prefix's final state when unsplit, at runtime: only
    # the owner of a separated tail sends its last group's state over the fabric.
    final_state = ttnn.experimental.kda.select_final_carry(
        rank_final,
        prefix_final_state,
        actual_start=actual_start,
        actual_end=actual_end,
        local_rows=geometry.local_rows,
        memory_config=KDA_OUTPUT_MEMORY_CONFIG,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    return RecurrenceResult(output, final_state)


class _RelayFinalCarry:
    """select_final_carry with the prefix's final state on the last chronological rank, where the relay leaves it."""

    _shared: dict = {}

    @classmethod
    def shared(cls, device: ttnn.MeshDevice, sequence_parallel_axis: int, num_links: int) -> _RelayFinalCarry:
        """One carry per mesh and line (mesh ids are never reused): layers run one after another, so they share its
        semaphores and program."""
        key = (device.id(), sequence_parallel_axis, num_links)
        if key not in cls._shared:
            cls._shared[key] = cls(device, sequence_parallel_axis, num_links)
        return cls._shared[key]

    def __init__(self, device: ttnn.MeshDevice, sequence_parallel_axis: int, num_links: int):
        grid = device.compute_with_storage_grid_size()
        # The bottom of the last worker column: the fabric senders are the first num_links of these, the rest copy.
        self._cores = [ttnn.CoreCoord(grid.x - 1, y) for y in range(grid.y - num_links - 1, grid.y)]
        self._axis = sequence_parallel_axis
        self._num_links = num_links
        # The line barrier and arrival counters persist across calls.
        self._semaphores = persistent_semaphores(device, 2)

    def __call__(self, rank_final, prefix_final, actual_start, actual_end, local_rows, projection=None):
        """Return the final carry and, with ``projection`` (on the worker cores above the carry's, run in its
        dispatch), the projection's output."""
        kwargs = dict(
            actual_start=actual_start,
            actual_end=actual_end,
            local_rows=local_rows,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            sequence_parallel_axis=self._axis,
            num_links=self._num_links,
            cores=self._cores,
            prefix_on_last_rank=True,
            semaphores=self._semaphores,
        )
        if projection is None:
            return ttnn.experimental.kda.select_final_carry(rank_final, prefix_final, **kwargs), None
        return ttnn.experimental.kda.select_final_carry_with_projection(
            rank_final,
            prefix_final,
            projection.input,
            projection.weight,
            projection_program_config=projection.program_config,
            projection_memory_config=projection.memory_config,
            projection_compute_kernel_config=projection.compute_config,
            **kwargs,
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
        gate_scale: float = 1.0,
        affine_relay_cores: list[ttnn.CoreCoord] | None = None,
    ) -> None:
        """``affine_relay_cores``: the worker cores the SP relay takes its head and sender cores from, in order
        (default: the grid in row-major order)."""
        # Preparation multiplies the gate by gate_scale before its cumulative sum.
        self._gate_scale = gate_scale
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
        # Both scans read the chunk terms chunk by chunk; keeping them in L1 when they fit spares those DRAM reads
        # and chunk preparation's DRAM writes.
        preparation_size = preparation_bytes(batch * heads, local_rows // KDA_CHUNK_SIZE, key_dim, value_dim)
        self._preparation_memory = l1_when_it_fits(device, preparation_size, KDA_PREPARATION_L1_BYTES_PER_CORE)
        if self._preparation_memory.buffer_type != ttnn.BufferType.L1:
            grid = device.compute_with_storage_grid_size()
            logger.warning(
                f"KDA chunk terms need {preparation_size // (grid.x * grid.y) // 1024} KiB per core, over the "
                f"{KDA_PREPARATION_L1_BYTES_PER_CORE // 1024} KiB L1 budget; keeping them in DRAM, so chunk "
                f"preparation and the scan run slower ({heads} heads, {local_rows} rows per device)."
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
            # Only a sequence-parallel run can split chronologically, so only its summaries carry tail transforms.
            self._summary_tail = self._sequence_parallel
            self._summary_memory = _group_summary_memory_config(
                device, self._summary_tail, batch * heads * self._groups, key_dim, value_dim
            )
        # A persistent, replicated output for high_bw_all_gather of the per-rank packed [A | B] summaries; only the
        # one-group-per-head summary gathers them directly.
        self._gather_buffer = None
        if (
            self._sequence_parallel
            and program_config.high_bw_prefix_gather
            and grouped
            and self._groups == 1
            and _sp_links(device, sequence_parallel_axis) >= KDA_HIGH_BW_PREFIX_GATHER_LINKS
        ):
            self._gather_buffer = ttnn.from_torch(
                torch.zeros(
                    (tuple(device.shape)[sequence_parallel_axis], batch * heads, key_dim, key_dim + value_dim),
                    dtype=torch.bfloat16,
                ),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        # The SP relay carries the prefix rank to rank when the fabric wraps the SP axis (the last rank
        # sends the final state on to the first); the final carry then runs as its own program.
        self._affine_relay = None
        self._relay_final_carry = None
        self._entry_correction = None
        if (
            self._sequence_parallel
            and program_config.relay_affine_prefix
            and grouped
            and self._groups == 1
            and fabric_wraps_axis(sequence_parallel_axis)
            and tuple(device.shape)[sequence_parallel_axis] > 2
            and _sp_links(device, sequence_parallel_axis) > 0
        ):
            links = _sp_links(device, sequence_parallel_axis)
            self._affine_relay = RelayAffinePrefix.shared(
                device,
                sequence_parallel_axis=sequence_parallel_axis,
                batch_heads=batch * heads,
                key_dim=key_dim,
                value_dim=value_dim,
                num_links=links,
                local_rows=local_rows,
                math_fidelity=program_config.affine_prefix_math_fidelity,
                cores=affine_relay_cores,
            )
            self._relay_final_carry = _RelayFinalCarry.shared(device, sequence_parallel_axis, links)
            # The relay needs no summary gather buffer.
            if self._gather_buffer is not None:
                ttnn.deallocate(self._gather_buffer)
                self._gather_buffer = None
            if program_config.entry_correction:
                grid = device.compute_with_storage_grid_size()
                # Its products count toward the Blackhole di/dt bound on cores in matmuls at once.
                max_cores = program_config.entry_correction_max_cores
                self._entry_correction = EntryCorrection(
                    device,
                    sequence_parallel_axis=sequence_parallel_axis,
                    batch_heads=batch * heads,
                    num_chunks=local_rows // KDA_CHUNK_SIZE,
                    key_dim=key_dim,
                    local_rows=local_rows,
                    cores=[ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][:max_cores],
                    math_fidelity=program_config.scan_math_fidelity,
                )
                # Every correction core reads its head's entry state: L1 spares those DRAM reads.
                self._affine_relay.entry_memory_config = ttnn.L1_MEMORY_CONFIG
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
        beta_logits_column_offset: int,
        release_gate: bool,
    ) -> tuple[_PreparedChunks, ttnn.Tensor, _RecurrenceGeometry]:
        geometry = self._geometry
        # Beta's logits are a column window of a wider projection.
        if (
            tuple(beta.shape)[:2] != (geometry.batch, geometry.local_rows)
            or beta.shape[-1] < beta_logits_column_offset + geometry.heads
        ):
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
            gate_scale=self._gate_scale,
            beta_logits_column_offset=beta_logits_column_offset,
            memory_config=self._preparation_memory,
        )
        if release_gate:
            # Only chunk preparation reads the gate; freeing it here lowers L1 use during the scans.
            ttnn.deallocate(gate)
        return prepared, state, geometry

    @staticmethod
    def _finish(scan: RecurrenceResult, geometry: _RecurrenceGeometry) -> RecurrenceResult:
        output = ttnn.reshape(scan.output, (geometry.batch_heads, geometry.local_rows, geometry.value_dim))
        shape = (geometry.batch, geometry.heads, geometry.key_dim, geometry.value_dim)
        if isinstance(scan.final_state, PendingFinalCarry):
            pending = scan.final_state

            def run(projection=None):
                final_state, projected = pending.run(projection)
                return ttnn.reshape(final_state, shape), projected

            return RecurrenceResult(output, PendingFinalCarry(run), scan.output_gate)
        return RecurrenceResult(output, ttnn.reshape(scan.final_state, shape), scan.output_gate)

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
        beta_logits_column_offset: int,
        release_gate: bool = False,
        output_gate: RelayOutputGate | None = None,
        defer_final_carry: bool = False,
    ) -> RecurrenceResult:
        """Execute the constructed graph using caller-owned state and chronology.

        ``beta`` is a token-major BF16 tensor whose ``heads`` columns from ``beta_logits_column_offset`` hold beta's
        pre-sigmoid logits, such as the fused input projection; chunk preparation applies the sigmoid.
        With ``release_gate`` the recurrence takes ownership of ``gate`` and frees it once chunk preparation has
        read it. ``output_gate`` is a projection run in the SP relay's dispatch on the worker cores the relay leaves
        idle (``affine_relay_cores``); its output is the result's output_gate. With ``defer_final_carry`` (the relay
        only) the result's final_state is a PendingFinalCarry the caller runs.
        """
        if (output_gate is not None or defer_final_carry) and not self.uses_affine_relay:
            raise ValueError("the output gate and a deferred final carry need the SP relay")
        prepared, state, geometry = self._prepare(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=beta,
            initial_state=initial_state,
            actual_start=actual_start,
            actual_end=actual_end,
            beta_logits_column_offset=beta_logits_column_offset,
            release_gate=release_gate,
        )
        if self.uses_affine_relay:
            result = self._run_sp(prepared, state, actual_start, actual_end, output_gate, defer_final_carry)
        else:
            result = self._execute(prepared, state, actual_start, actual_end)
        # Release the chunk terms as soon as the scan consumed them, so every call sees the same free memory (the
        # summary already released them when the entry correction runs).
        for tensor in prepared.as_kernel_args():
            if tensor.is_allocated():
                ttnn.deallocate(tensor)
        return self._finish(result, geometry)

    def _run_direct(
        self,
        prepared: _PreparedChunks,
        state: ttnn.Tensor,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None,
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
    ) -> RecurrenceResult:
        return _scan_local_grouped_chunks(
            prepared,
            state,
            self._geometry,
            summary_group_chunks=self._summary_group_chunks,
            groups=self._groups,
            summary_tail=self._summary_tail,
            memory=self._summary_memory,
            compute_config=self._compute_config,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=self._sequence_parallel_axis,
        )

    @property
    def uses_affine_relay(self) -> bool:
        return self._affine_relay is not None

    @property
    def final_carry_cores(self) -> list[ttnn.CoreCoord]:
        """Worker cores the relay's final carry runs on."""
        return list(self._relay_final_carry._cores)

    @property
    def affine_relay_head_cores(self) -> list[ttnn.CoreCoord]:
        """The relay's head cores: one head's state update (a small matmul) each."""
        return list(self._affine_relay.head_cores)

    @property
    def affine_relay_cores(self) -> list[ttnn.CoreCoord]:
        """Worker cores the SP relay runs on (one per head, then one per link)."""
        return list(self._affine_relay.cores)

    def _run_sp(
        self,
        prepared: _PreparedChunks,
        state: ttnn.Tensor,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None,
        output_gate: RelayOutputGate | None = None,
        defer_final_carry: bool = False,
    ) -> RecurrenceResult:
        return _scan_sp_grouped_chunks(
            prepared,
            state,
            self._geometry,
            summary_group_chunks=self._summary_group_chunks,
            groups=self._groups,
            summary_tail=self._summary_tail,
            memory=self._summary_memory,
            compute_config=self._compute_config,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=self._sequence_parallel_axis,
            gather_buffer=self._gather_buffer,
            affine_relay=self._affine_relay,
            final_carry=self._relay_final_carry,
            output_gate=output_gate,
            defer_final_carry=defer_final_carry,
            correction=self._entry_correction,
        )
