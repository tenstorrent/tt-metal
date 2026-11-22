# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The sequence-parallel prefix as a relay along the SP axis (ttnn.experimental.kda.relay_chunk_scan): each rank
applies its own summary, in complement form [E | B] (E = A - I), to the entry state S its predecessor sends as
S + (E S + B) and forwards the result, and the last rank hands the final state to the first. On a ring that hand-off
is one hop; on a line it crosses the SP axis (2D routing, or sp_size - 1 hops back on a 1D fabric). The first rank
then forms the chunk's final carry from it and multicasts it to every rank.

The same program forms the chunk outputs from the summary's emitted terms (the entry correction): each head's chunks
are spread over `cores_per_head` correction cores, and relay cores, each serving two heads of a link, run the hops and
copy each landed entry state (and, for a separated tail, the final state) to the heads' correction cores, which form
the gated norm's output gate x @ W for their chunk rows while the hops are on their way. The summary writes the terms
sharded so each core's chunk rows, F(0) then F(I), are in its own L1 (terms_memory_config); when a core's share is
over KDA_RELAY_TERMS_L1_BYTES_PER_CORE they stay in DRAM and each core streams its chunk rows.

The relay's landing buffer is an L1 tensor sharded over the relay cores, at the same address on every device. A
rank's predecessor writes into it before the rank's own call starts, so a layer reserves it before the SP barrier that
precedes the recurrence (reserve) and the call frees it once the relay has consumed it: between layers that L1 is free
for the other attention layers (ring_mla's circular buffers span nearly all of L1).

Load-bearing: the entry barrier of the layer's qkv convolution history exchange runs after the reservation and before
the relay, so no rank relays until every rank has reserved its landing buffer and finished the programs before it,
whose circular buffers may overlap that L1. A layer that drops that exchange needs another SP barrier there. Layers run
one after another, so one relay per mesh and geometry is shared; every rank waiting for the final carry, which the
first rank sends after the lap, separates consecutive calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class SlabEntry:
    """Entry ``index * scale + offset`` of a ``[entries, ...]`` state slab; ``index`` is a replicated UINT32 device
    scalar the KDA ops read on device, so a captured trace follows the index of each replay. The ops read the carry
    from the entry (zeros where ``actual_start`` reads 0) and write the replacement back in place."""

    slab: ttnn.Tensor
    index: ttnn.Tensor
    scale: int = 1
    offset: int = 0

    def op_kwargs(self) -> dict:
        return {"slab_index": self.index, "slab_index_scale": self.scale, "slab_index_offset": self.offset}


@dataclass(frozen=True)
class OutputNorm:
    """The gated RMSNorm the relay applies to each output row x: x / sqrt(mean(x^2) + norm_epsilon) * weight *
    sigmoid(gate), with the gate ``gate_input @ gate_weight`` ([B, T, X] @ [X, num_heads * V]) formed in the relay at
    ``gate_math_fidelity``. With ``gate_local``, this rank's TP slice of the input's columns is read from it ([B, T,
    X / TP]) instead. The output is token-major [B, T, num_heads * V] in ``output_dtype`` and ``memory_config``."""

    gate_input: ttnn.Tensor
    gate_weight: ttnn.Tensor
    weight: ttnn.Tensor
    num_heads: int
    norm_epsilon: float
    output_dtype: ttnn.DataType
    memory_config: ttnn.MemoryConfig
    gate_local: ttnn.Tensor | None = None
    gate_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2


# The relay's persistent counters (relay_semaphores in relay_chunk_scan_device_operation_types.hpp).
_RELAY_SEMAPHORES = 4

# A sender core's count semaphores, one per head it serves (relay_chunk_scan_max_heads_per_link).
_MAX_HEADS_PER_LINK = 14

# Heads per relay core (relay_chunk_scan_heads_per_relay_core).
_HEADS_PER_RELAY_CORE = 2

# The output gate's working L1 per correction core (its K blocks of x and W in flight). Allocated right before the
# relay, it takes L1 the layer has freed by then (its q, k, v and projection); with the terms in L1 and more than 5
# chunk rows a core, less is left (relay_chunk_scan_program_factory.cpp takes the K blocks it fits).
KDA_GATE_WORKSPACE_BYTES_PER_CORE = 94 * 4096
KDA_GATE_WORKSPACE_BYTES_PER_CORE_CROWDED = 64 * 4096

# A core's share of the emitted FP32 terms (F(0) and F(I) of its chunks) that stays in its L1; beyond it they stream
# from DRAM (Kimi-K3: 320 KiB per core at 1280 rows per chip).
KDA_RELAY_TERMS_L1_BYTES_PER_CORE = 320 * 1024


# Every persistent counter per mesh id, for reset_persistent_semaphores.
_persistent_semaphores: dict[int, list] = {}


def persistent_semaphores(mesh_device, count: int, cores=None) -> list:
    """Zeroed global semaphores that persist across calls, on `cores` (default: every worker core), created up front so
    they sit with the model's other long-lived L1 allocations. L1_SMALL, when the device has one, keeps a fabric mux
    from clobbering them (ttnn::ccl::prefer_l1_small_buffer_type). The kernels consume their counts by subtraction
    (persistent_counter.hpp), so they stay zero between calls."""
    if cores is None:
        grid = mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    l1_small = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL)
    buffer_type = ttnn.BufferType.L1_SMALL if l1_small.total_bytes_per_bank > 0 else ttnn.BufferType.L1
    semaphores = [ttnn.create_global_semaphore(mesh_device, cores, 0, buffer_type) for _ in range(count)]
    _persistent_semaphores.setdefault(mesh_device.id(), []).extend(semaphores)
    # The zeroed counters must land before any device's program can increment a peer's.
    ttnn.synchronize_device(mesh_device)
    return semaphores


def reset_persistent_semaphores(mesh_device) -> None:
    """Zero every KDA persistent counter on the mesh. A run that stopped mid-call leaves counts behind that the next call
    would take for its peers' arrivals; call this with the mesh idle before the next call."""
    ttnn.synchronize_device(mesh_device)
    for semaphore in _persistent_semaphores.get(mesh_device.id(), []):
        ttnn.reset_global_semaphore_value(semaphore, 0)
    ttnn.synchronize_device(mesh_device)


class RelayChunkScan:
    """The SP relay's landing buffer, semaphores and core placement for one mesh and geometry: a call forms the chunk
    outputs and their gated norm from the relayed entry state."""

    _shared: dict = {}

    @classmethod
    def shared(cls, mesh_device, **kwargs) -> RelayChunkScan:
        """One relay per mesh and geometry (mesh ids are never reused): layers run one after another, so they share
        its L1 landing buffer."""
        key = (mesh_device.id(), tuple(sorted(kwargs.items())))
        if key not in cls._shared:
            cls._shared[key] = cls(mesh_device, **kwargs)
        return cls._shared[key]

    def __init__(
        self,
        mesh_device,
        *,
        sequence_parallel_axis,
        batch_heads,
        key_dim,
        value_dim,
        num_links,
        local_rows,
        max_cores,
        math_fidelity=ttnn.MathFidelity.HiFi3,
    ):
        if key_dim != value_dim:
            raise ValueError("relay prefix requires K == V")
        # The gated norm's FPU products lose too much below HiFi3; the op also refuses it at its first call.
        if math_fidelity not in (ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4):
            raise ValueError(f"relay_chunk_scan needs math_fidelity HiFi3 or HiFi4, got {math_fidelity}")
        heads_per_link = -(-batch_heads // num_links)
        if heads_per_link > _MAX_HEADS_PER_LINK:
            raise ValueError(
                f"relay prefix: {batch_heads} heads over {num_links} links put {heads_per_link} on a sender, "
                f"which has semaphores for {_MAX_HEADS_PER_LINK}: use more links"
            )
        self.mesh_device = mesh_device
        self.axis = sequence_parallel_axis
        self.num_links = num_links
        self.local_rows = local_rows
        # Each head's correction cores, then two senders per link (the relay's and the final carry's second), then the
        # relay cores. The senders and relay cores take the grid's first cores, nearest the Ethernet cores, clear of the
        # correction cores' multicasts. Head h's correction cores are a column from the next row (correction_core.hpp),
        # as the summary places a head's value blocks. Up to max_cores correction cores: with the relay cores, theirs
        # are the matmuls that count toward the Blackhole di/dt bound.
        grid = mesh_device.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        senders = 2 * num_links
        relay_count = sum(
            -(-len(range(link, batch_heads, num_links)) // _HEADS_PER_RELAY_CORE) for link in range(num_links)
        )
        first_row = math.ceil((senders + relay_count) / grid.x)
        # Cores per head: the most that divide the head's chunks evenly (so each core's chunk rows are one shard), fit
        # the budget and the grid's columns, and leave the relay cores room.
        num_chunks = local_rows // ttnn.TILE_SIZE
        if num_chunks == 0:
            raise ValueError(f"relay prefix needs at least one chunk of {ttnn.TILE_SIZE} rows, has {local_rows} rows")
        budget = min(max_cores, len(cores) - senders - relay_count)

        def fits(d):
            per_column = (grid.y - first_row) // d
            return batch_heads * d <= budget and per_column > 0 and -(-batch_heads // per_column) <= grid.x

        self.cores_per_head = cores_per_head = max(
            (d for d in range(1, num_chunks + 1) if num_chunks % d == 0 and fits(d)), default=0
        )
        if cores_per_head == 0:
            raise ValueError(
                f"relay prefix needs {batch_heads} correction cores, {senders} senders and {relay_count} relay cores; "
                f"max_cores={max_cores}, grid is {grid.x}x{grid.y}"
            )
        per_column = (grid.y - first_row) // cores_per_head
        correction = [
            ttnn.CoreCoord(h // per_column, first_row + h % per_column * cores_per_head + q)
            for h in range(batch_heads)
            for q in range(cores_per_head)
        ]
        relay = cores[senders : senders + relay_count]
        self.cores = correction + cores[:senders] + relay

        def core_set(selected):
            # One range per row run, in the cores' order: a sharded tensor places shard i on the set's i-th core.
            runs = []
            for core in selected:
                if runs and runs[-1][1].y == core.y and runs[-1][1].x + 1 == core.x:
                    runs[-1][1] = core
                else:
                    runs.append([core, core])
            return ttnn.CoreRangeSet([ttnn.CoreRange(lo, hi) for lo, hi in runs])

        # Per relay core, a K x K entry state per head slot, then a final state per slot, while reserved.
        landing_rows = 2 * _HEADS_PER_RELAY_CORE * key_dim
        self._landing_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_set(relay), (landing_rows, key_dim), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self._landing_shape = ttnn.Shape([relay_count * landing_rows, key_dim])
        self._landing = None
        # A peer can bump the entry, final and carry arrival counters before this rank's call starts.
        self.semaphores = persistent_semaphores(mesh_device, _RELAY_SEMAPHORES)
        # The hop's, the correction's and the norm's products run at math_fidelity, accumulating in an FP32 DST.
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True
        )
        # The emitted terms land in the L1 of the core that consumes them: each core takes whole chunks, so a core's
        # shard is its chunk rows of F(0) and F(I), and nothing is read over the NoC for them.
        core_rows = num_chunks // cores_per_head * ttnn.TILE_SIZE
        streamed = 2 * core_rows * key_dim * 4 > KDA_RELAY_TERMS_L1_BYTES_PER_CORE
        self.terms_memory_config = (
            ttnn.DRAM_MEMORY_CONFIG
            if streamed
            else ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(core_set(correction), (2 * core_rows, key_dim), ttnn.ShardOrientation.ROW_MAJOR),
            )
        )
        crowded = not streamed and num_chunks // cores_per_head > 5
        workspace_bytes = KDA_GATE_WORKSPACE_BYTES_PER_CORE_CROWDED if crowded else KDA_GATE_WORKSPACE_BYTES_PER_CORE
        workspace_rows = workspace_bytes // (ttnn.TILE_SIZE * 4)
        self._gate_workspace_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_set(correction), (workspace_rows, ttnn.TILE_SIZE), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self._gate_workspace_shape = ttnn.Shape([len(correction) * workspace_rows, ttnn.TILE_SIZE])

    def reserve(self):
        """Allocate the landing buffer for the next call, before the SP barrier that precedes the recurrence (a no-op
        while reserved)."""
        if self._landing is None:
            self._landing = ttnn.allocate_tensor_on_device(
                self._landing_shape, ttnn.float32, ttnn.TILE_LAYOUT, self.mesh_device, self._landing_memory_config
            )

    def __call__(
        self,
        packed,
        terms,
        initial_state: ttnn.Tensor | SlabEntry,
        final_e,
        final_b,
        output_norm: OutputNorm,
        actual_start,
        actual_end=None,
    ):
        """Return the gated-norm outputs (see OutputNorm) and the chunk's final carry [BH, K, K] FP32 on every rank.
        Frees the landing buffer once the relay has run. A slab entry ``initial_state`` is read in place (zeros at a
        request's head) and the final carry is written back into it: the carry returned is then the slab."""
        # Reserving here suffices only when no program of this rank runs between its predecessor's relay and its own.
        self.reserve()
        slab_kwargs = {}
        if isinstance(initial_state, SlabEntry):
            slab_kwargs = initial_state.op_kwargs()
            initial_state = initial_state.slab
        gate_workspace = ttnn.allocate_tensor_on_device(
            self._gate_workspace_shape,
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            self._gate_workspace_memory_config,
        )
        result = ttnn.experimental.kda.relay_chunk_scan(
            packed,
            terms,
            initial_state,
            final_e,
            final_b,
            output_norm.gate_input,
            output_norm.gate_weight,
            output_norm.weight,
            actual_start,
            self._landing,
            gate_workspace,
            self.semaphores,
            local_rows=self.local_rows,
            num_links=self.num_links,
            cores=self.cores,
            actual_end=actual_end,
            sequence_parallel_axis=self.axis,
            compute_kernel_config=self.compute_config,
            num_heads=output_norm.num_heads,
            gate_local=output_norm.gate_local,
            gate_math_fidelity=output_norm.gate_math_fidelity,
            norm_epsilon=output_norm.norm_epsilon,
            output_dtype=output_norm.output_dtype,
            memory_config=output_norm.memory_config,
            **slab_kwargs,
        )
        # Every write into it is consumed within the call: the predecessor's entry columns by this rank's heads, and
        # the last rank's final state by the first rank.
        ttnn.deallocate(self._landing)
        self._landing = None
        ttnn.deallocate(gate_workspace)
        return result
