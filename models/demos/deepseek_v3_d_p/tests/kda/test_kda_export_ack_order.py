# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The per-layer migration ack orders after EVERY chip's export of the SP-sharded KDA state.

Once the state is sharded over SP, each chip stores different heads, so a migration reader that starts on a
layer's ack must find every chip's export landed, not only one replica's. This drives the production pieces
on Galaxy SP8xTP4: ``KdaStates.export_layer`` followed by ``send_layer_ack`` over a metadata-only
``D2HStreamService`` (one socket per chip, as the runner builds it), bridged by the real ``LayerAckService``
into a layer-completion ring. Reads go through a real ``KvChunkAddressTable`` (``read_device_chunk``: each
segment's DRAM bytes read straight off the chip its entry names, outside the command queue). (The migration
worker's own transport, a device-less UMD cluster, pins 1 GiB of host memory per chip, ~32 GiB on a Galaxy.)

Every (chunk, SP row) writes values that differ in every element, recurrent AND convolution, from the previous
write to the same slot (self-checked), so a read that runs ahead of a chip's export of either kind sees stale bytes.

One chip per chunk (rotating) is made slow: a spin kernel runs on that mesh coordinate only, ahead of its export,
so the other 31 chips export and send their ack records long before it. Then:

* control: once the other 31 records are in, the slow chip still holds the previous chunk (both kinds, both
  layers) and no ring message has arrived;
* on each ring message, a fast probe of all 32 chips (both kinds) runs first and must see this chunk on every
  chip, the slow one included; the full layer read follows.

An ack that fired on the first (or any one) chip's record would fail the control or the probe on the slow chip.
The per-socket arm reads chip k the moment chip k's own record arrives (without the service).
"""

from __future__ import annotations

import math
import os
import time

import pytest
import torch
from loguru import logger
from ttnn._experimental.layer_completion import LayerCompletionQueue

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.utils import kda_device_params, kda_mesh_param
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import (
    KdaContractGeometry,
    KdaStates,
    assemble_convolution,
    assemble_recurrent,
    convolution_segment_to_torch,
    device_scalar,
    recurrent_segment_to_torch,
)
from models.demos.deepseek_v3_d_p.tt.kda.state_contract import KDA_CONVOLUTION_PAGE_CHANNELS
from models.demos.deepseek_v3_d_p.tt.kv_ack import send_layer_ack
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    kda_chunk_n_tokens,
    kda_position,
    kda_segment_bytes,
    kda_window,
    populate_kv_chunk_address_table_kda,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical, encode_tail
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch

pytestmark = [
    run_for_blackhole(),
    skip_for_slow_dispatch(),
    pytest.mark.skipif(
        ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
        reason="D2HStreamService runs on Blackhole Galaxy clusters only (#47750)",
    ),
]

SP_AXIS, TP_AXIS = 0, 1
LAYER_IDS = (1, 2)  # one ack record per KDA layer per chunk, in this order
NUM_SLOTS = 2
NUM_CHUNKS = 8
RECORD_BYTES = 12  # {slot_id, pos_start, pos_end}, the runner's packed chunk metadata
CHUNK_POSITIONS = 128
# How long the one slow chip spins ahead of its export: long enough for the other 31 chips to export, send their
# records and be observed, and for the slow chip to be probed, all while it still spins.
SLOW_CHIP_MS = 200.0
RING_TIMEOUT_S = 60.0
CONV_EXPONENT_BIAS = 48  # convolution value exponent = tag - bias, inside BF16's normal range for every tag used


def _geometry(mesh_device) -> KdaContractGeometry:
    return KdaContractGeometry.from_kda_config(
        kimi_k3_kda_config(), mesh_shape=tuple(mesh_device.shape), sp_axis=SP_AXIS, tp_axis=TP_AXIS
    )


def _tag(chunk: int, sp_row: int) -> int:
    return chunk * 8 + sp_row + 1


def _row_patterns(geometry: KdaContractGeometry, chunk: int, sp_row: int):
    """Global recurrent ``[H, D, D]`` and golden-order convolution ``[K-1, 3*H*D]`` the chips of ``sp_row`` carry.

    Exact in the stored dtypes and tagged by ``_tag``: recurrent ``(i % 509) + 1024 * tag`` (integers below 2**24),
    convolution ``(1 + (i % 127) / 128) * 2**(tag - 48)`` (8 significant bits, so exact in BF16), whose binary
    exponent is the tag. Two different tags differ in every element of both kinds.
    """
    tag = _tag(chunk, sp_row)
    heads, dim = geometry.num_heads, geometry.head_dim
    recurrent = (torch.arange(heads * dim * dim) % 509).float().reshape(heads, dim, dim) + 1024.0 * tag
    width = 3 * heads * dim
    mantissa = 1.0 + (torch.arange(geometry.conv_history * width) % 127).float() / 128.0
    convolution = (mantissa * 2.0 ** (tag - CONV_EXPONENT_BIAS)).to(torch.bfloat16)
    return recurrent, convolution.reshape(geometry.conv_history, width)


def _recurrent_tag(value: float) -> int:
    return int(value) // 1024


def _convolution_tag(value: float) -> int:
    return math.frexp(abs(value))[1] - 1 + CONV_EXPONENT_BIAS


def _assert_writes_differ(geometry: KdaContractGeometry, writes_per_slot: dict[int, list[int]]) -> None:
    """Every write differs from the previous write to its slot in EVERY element of both kinds, on every SP row."""
    for slot, chunks in writes_per_slot.items():
        for before, after in zip(chunks, chunks[1:]):
            for row in range(geometry.sequence_parallel_size):
                old, new = _row_patterns(geometry, before, row), _row_patterns(geometry, after, row)
                for kind, a, b in (("recurrent", old[0], new[0]), ("convolution", old[1], new[1])):
                    assert (a != b).all(), f"slot {slot} row {row}: {kind} of chunk {after} repeats chunk {before}"


def _per_chip_convolution(geometry: KdaContractGeometry, convolution: torch.Tensor, tp_col: int) -> torch.Tensor:
    """The chip's ``[q_local | k_local | v_local]`` rows from the golden-order global tensor."""
    heads, dim, local = geometry.num_heads, geometry.head_dim, geometry.local_heads
    return torch.cat(
        [
            convolution[
                :, branch * heads * dim + tp_col * local * dim : branch * heads * dim + (tp_col + 1) * local * dim
            ]
            for branch in range(3)
        ],
        dim=-1,
    )


def _chunk_carry(mesh_device, geometry: KdaContractGeometry, chunk: int) -> KdaState:
    """A native carry whose SP rows hold different states: chip (r, c) gets row r's pattern, TP column c's heads."""
    sp, tp = geometry.sequence_parallel_size, geometry.tensor_parallel_size
    patterns = [_row_patterns(geometry, chunk, row) for row in range(sp)]
    recurrent = torch.stack([rec for rec, _ in patterns])  # [SP, H, D, D] -> chip (r, c): [1, H_local, D, D]
    convolution = torch.stack(
        [
            encode_tail(
                torch.cat([_per_chip_convolution(geometry, conv, col) for col in range(tp)], dim=-1),
                KDA_CONVOLUTION_PAGE_CHANNELS,
            )
            for _, conv in patterns
        ]
    )  # [SP, TP * SEG, 192] -> chip (r, c): [1, SEG, 192]
    mesh_shape = tuple(mesh_device.shape)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=mesh_shape)
    return KdaState(
        recurrent=ttnn.from_torch(
            recurrent,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        ),
        convolution=ttnn.from_torch(
            convolution,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        ),
    )


def _expected_global(geometry: KdaContractGeometry, chunk: int):
    """What the table must reassemble after ``chunk``: head g from the pattern of the SP row that stores it."""
    sp, local, stored, dim = (
        geometry.sequence_parallel_size,
        geometry.local_heads,
        geometry.stored_heads,
        geometry.head_dim,
    )
    patterns = [_row_patterns(geometry, chunk, row) for row in range(sp)]
    recurrent = torch.empty_like(patterns[0][0])
    convolution = torch.empty_like(patterns[0][1])
    heads = geometry.num_heads
    for g in range(heads):
        row = (g % local) // stored
        recurrent[g] = patterns[row][0][g]
        for branch in range(3):
            lo = branch * heads * dim + g * dim
            convolution[:, lo : lo + dim] = patterns[row][1][:, lo : lo + dim]
    return recurrent, convolution


class _TableReader:
    """Out-of-band reads of whole KDA layers through a real address table, as the migration worker reads them."""

    def __init__(self, mesh_device, geometry: KdaContractGeometry, slabs: KdaStates):
        disagg = ttnn.experimental.disaggregation
        rows, cols = tuple(mesh_device.shape)
        self.geometry = geometry
        configs = {}
        for name, kind in (("01", "kda_recurrent"), ("02", "kda_convolution")):
            cfg = disagg.KvChunkAddressTableConfig()
            cfg.num_layers = max(LAYER_IDS) + 1
            cfg.max_sequence_length = kda_window(geometry)
            cfg.num_slots = NUM_SLOTS
            cfg.chunk_n_tokens = kda_chunk_n_tokens(geometry, kind)
            cfg.chunk_size_bytes = kda_segment_bytes(geometry, kind)
            cfg.tensor_type = disagg.KvChunkTensorType.Snapshot
            configs[name] = cfg
        self.table = disagg.KvChunkAddressTable(configs)
        fnids = [[mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for c in range(cols)] for r in range(rows)]
        for name, kind, slab in (
            ("01", "kda_recurrent", slabs.recurrent),
            ("02", "kda_convolution", slabs.convolution),
        ):
            layout = [
                {
                    "rank": 0,
                    "first_layer": 0,
                    "count": len(LAYER_IDS),
                    "base_addr": int(slab.buffer_address()),
                    "num_banks": mesh_device.dram_grid_size().x,
                    "host_tag": 0,
                    "fnids": fnids,
                }
            ]
            populate_kv_chunk_address_table_kda(
                self.table,
                configs[name],
                (rows, cols),
                SP_AXIS,
                TP_AXIS,
                geometry,
                kind,
                num_users=NUM_SLOTS,
                config_id=self.table.config_id_of(name),
                stage_layout=layout,
                layer_rows=list(LAYER_IDS),
            )
        self.coord_of = {
            (int(fnids[r][c].mesh_id), int(fnids[r][c].chip_id)): (r, c) for r in range(rows) for c in range(cols)
        }
        self._plans = {}

    def _plan(self, layer: int, slot: int, kind: str):
        """(config, position, owning chip) of every segment of one layer and slot, in segment order."""
        key = (layer, slot, kind)
        if key not in self._plans:
            config = self.table.config_id_of("01" if kind == "kda_recurrent" else "02")
            count = (
                self.geometry.recurrent_segments_per_layer
                if kind == "kda_recurrent"
                else self.geometry.convolution_segments_per_layer
            )
            plan = []
            for segment in range(count):
                position = kda_position(self.geometry, kind, segment)
                loc = self.table.lookup(layer, position, slot, config)
                members = self.table.get_device_group(loc.device_group_index).fabric_node_ids
                assert len(members) == 1, f"{kind} L{layer} segment {segment}: group of {len(members)} chips"
                plan.append((config, position, self.coord_of[(int(members[0].mesh_id), int(members[0].chip_id))]))
            self._plans[key] = plan
        return self._plans[key]

    def _read(self, layer: int, slot: int, config: int, position: int) -> bytes:
        # The table's own read: the segment's DRAM bytes straight off its chip, outside the command queue.
        return bytes(self.table.read_device_chunk(layer, position, slot, config))

    def read_layer(self, layer: int, slot: int):
        """The layer's global recurrent ``[H, D, D]`` and golden-order convolution, read through the table."""
        out = []
        for kind, decode, assemble in (
            ("kda_recurrent", recurrent_segment_to_torch, assemble_recurrent),
            ("kda_convolution", convolution_segment_to_torch, assemble_convolution),
        ):
            segments = {
                segment: decode(self._read(layer, slot, config, position), self.geometry)
                for segment, (config, position, _) in enumerate(self._plan(layer, slot, kind))
            }
            out.append(assemble(segments, self.geometry))
        return tuple(out)

    def probe(self, layer: int, slot: int, chips=None) -> dict[tuple[int, int], tuple[int, int]]:
        """(recurrent tag, convolution tag) of each chip, from its first segment of each kind: two small reads."""
        tags = {}
        for kind, decode, tag_of, index in (
            ("kda_recurrent", recurrent_segment_to_torch, _recurrent_tag, 0),
            ("kda_convolution", convolution_segment_to_torch, _convolution_tag, 1),
        ):
            seen = set()
            for config, position, owner in self._plan(layer, slot, kind):
                if owner in seen or (chips is not None and owner not in chips):
                    continue
                seen.add(owner)
                value = float(decode(self._read(layer, slot, config, position), self.geometry).flatten()[0])
                tags.setdefault(owner, [None, None])[index] = tag_of(value)
        return {chip: tuple(pair) for chip, pair in tags.items()}


def _record(mesh_device, chunk: int) -> ttnn.Tensor:
    slot = chunk % NUM_SLOTS
    start = chunk * CHUNK_POSITIONS
    return ttnn.from_torch(
        torch.tensor([slot, start, start + CHUNK_POSITIONS], dtype=torch.int64).reshape(1, 1, 1, 3),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _d2h_service(mesh_device):
    # Metadata-only (global_spec=None): one socket per mesh coordinate, as the runner builds it.
    return ttnn.D2HStreamService(
        mesh_device,
        global_spec=None,
        fifo_size_bytes=4 * 64,
        worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
        metadata_size_bytes=RECORD_BYTES,
        metadata_ring=True,
    )


def _unlink(shm_name: str) -> None:
    try:
        os.remove(f"/dev/shm/{shm_name.lstrip('/')}")
    except FileNotFoundError:
        pass


_SPIN_KERNEL = """
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t loops = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < loops; i++) {
        riscv_wait(1000000);
    }
}
"""


class _SlowChip:
    """A spin kernel on ONE mesh coordinate: that chip's command queue stalls while the other chips run on."""

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        # generic_op wants an input and an output tensor; the spin kernel touches neither.
        self.dummies = [
            ttnn.from_torch(
                torch.zeros(1, 1, 32, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for _ in range(2)
        ]
        self.measure_ms((0, 0), loops=1)  # compile first: a cold JIT would inflate the calibration
        per_loop_ms = self.measure_ms((0, 0), loops=64) / 64
        self.loops = max(1, int(SLOW_CHIP_MS / per_loop_ms))
        logger.info(f"spin: {per_loop_ms:.4f} ms per loop -> {self.loops} loops for {SLOW_CHIP_MS} ms")

    def enqueue(self, chip: tuple[int, int], loops: int | None = None) -> None:
        core = ttnn.CoreCoord(0, 0)
        runtime_args = ttnn.RuntimeArgs()
        runtime_args[core.x][core.y] = [self.loops if loops is None else loops]
        kernel = ttnn.KernelDescriptor(
            kernel_source=_SPIN_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
            compile_time_args=[],
            runtime_args=runtime_args,
            config=ttnn.ReaderConfigDescriptor(),
        )
        program = ttnn.MeshProgramDescriptor()
        coord = ttnn.MeshCoordinate(*chip)
        program[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(
            kernels=[kernel], semaphores=[], cbs=[]
        )
        ttnn.generic_op(self.dummies, program)

    def measure_ms(self, chip: tuple[int, int], loops: int | None = None) -> float:
        ttnn.synchronize_device(self.mesh_device)
        start = time.perf_counter()
        self.enqueue(chip, loops)
        ttnn.synchronize_device(self.mesh_device)
        return (time.perf_counter() - start) * 1000

    def deallocate(self) -> None:
        for tensor in self.dummies:
            ttnn.deallocate(tensor)


def _slow_chip(chunk: int, rows: int, cols: int) -> tuple[int, int]:
    """Rotate the slow chip over rows and columns (13 is coprime with 32)."""
    index = (chunk * 13) % (rows * cols)
    return index // cols, index % cols


def _stale_chips(tags, chunk: int, chips) -> list:
    """Chips whose probe does not show this chunk in BOTH kinds."""
    return sorted(chip for chip in chips if tags[chip] != (_tag(chunk, chip[0]),) * 2)


def _pop(ring, deadline_s: float = RING_TIMEOUT_S):
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        message = ring.try_pop()
        if message is not None:
            return message
        time.sleep(0.0002)
    raise AssertionError(f"no layer-completion message within {deadline_s} s")


def _check_layer(reader: _TableReader, geometry, layer: int, chunk: int, name: str) -> None:
    want_rec, want_conv = _expected_global(geometry, chunk)
    got_rec, got_conv = reader.read_layer(layer, chunk % NUM_SLOTS)
    stale = {}
    rows = geometry.sequence_parallel_size
    for g in range(geometry.num_heads):
        if not torch.equal(want_rec[g], got_rec[g]):
            chip = ((g % geometry.local_heads) // geometry.stored_heads, g // geometry.local_heads)
            stale.setdefault(chip, int(got_rec[g][0, 0].item()) // 1024)
    assert not stale, (
        f"{name}: chunk {chunk} (tags {[_tag(chunk, r) for r in range(rows)]}) L{layer}: chips whose recurrent "
        f"segments did not hold this chunk when the ack arrived (chip -> tag read): {stale}"
    )
    assert_bit_identical(want_rec, got_rec, name=f"{name} recurrent chunk {chunk} L{layer}")
    assert_bit_identical(want_conv, got_conv, name=f"{name} convolution chunk {chunk} L{layer}")


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [kda_mesh_param((8, 4), "mesh-8x4", "8x4", kda_device_params((8, 4), trace_region_size=64 * 1024 * 1024))],
    indirect=True,
)
@pytest.mark.parametrize("traced", [False, True], ids=["eager", "traced"])
def test_layer_ack_orders_after_every_chip_export(mesh_device, device_params, traced):
    geometry = _geometry(mesh_device)
    rows, cols = tuple(mesh_device.shape)
    chips = [(r, c) for r in range(rows) for c in range(cols)]
    assert geometry.stored_heads * geometry.sequence_parallel_size == geometry.local_heads
    seeds = {slot: NUM_CHUNKS + slot for slot in range(NUM_SLOTS)}  # never acked; each slot's first predecessor
    warm_up = NUM_CHUNKS + NUM_SLOTS  # never checked; written to slot 0 after its seed
    writes = {slot: [seeds[slot]] for slot in range(NUM_SLOTS)}
    writes[0].append(warm_up)
    for chunk in range(NUM_CHUNKS):
        writes[chunk % NUM_SLOTS].append(chunk)
    _assert_writes_differ(geometry, writes)

    slabs = KdaStates.allocate(mesh_device, geometry, sp_axis=SP_AXIS, layer_ids=LAYER_IDS, num_slots=NUM_SLOTS)
    reader = _TableReader(mesh_device, geometry, slabs)
    slow = _SlowChip(mesh_device)
    for slot, chunk in seeds.items():
        carry = _chunk_carry(mesh_device, geometry, chunk)
        slot_tt = device_scalar(mesh_device, slot)
        for layer in LAYER_IDS:
            slabs.export_layer(carry, slot_tt, layer)
        for tensor in (slot_tt, carry.recurrent, carry.convolution):
            ttnn.deallocate(tensor)
    ttnn.synchronize_device(mesh_device)

    ring_name = f"/tt_lcq_kda_ack_{os.getpid()}_{int(traced)}"
    _unlink(ring_name)
    ring = LayerCompletionQueue.create(ring_name, len(LAYER_IDS))
    d2h = _d2h_service(mesh_device)
    sockets = d2h.get_sockets()
    assert len(sockets) == len(chips), f"{len(sockets)} sockets for {len(chips)} chips"  # row-major coordinates
    service = ttnn.LayerAckService(
        d2h, ring_name, first_record=0, record_count=len(LAYER_IDS), layer_ids=list(LAYER_IDS)
    )
    service.start()

    # Persistent homes for what a capture bakes in: the carry, the slot and the D2H record.
    carry = _chunk_carry(mesh_device, geometry, warm_up)
    slot_tt = device_scalar(mesh_device, 0)
    record = _record(mesh_device, NUM_CHUNKS)

    def export_and_ack():
        for layer in LAYER_IDS:
            slabs.export_layer(carry, slot_tt, layer)
            send_layer_ack(
                mesh_device=mesh_device,
                global_layer_idx=layer,
                d2h_service=d2h,
                metadata_msg=record,
                on_layer_complete=None,
                trace_controller=None,
            )

    name = "traced" if traced else "eager"
    trace_id = None
    try:
        # Warm-up: one eager pass compiles every program of the chain (a capture may not load new binaries) and
        # proves the ring is live; its messages are drained here.
        export_and_ack()
        for layer in LAYER_IDS:
            assert _pop(ring)[3] == layer
        if traced:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            export_and_ack()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
        for chunk in range(NUM_CHUNKS):
            slot = chunk % NUM_SLOTS
            slow_chip = _slow_chip(chunk, rows, cols)
            others = [chip for chip in chips if chip != slow_chip]
            staged = _chunk_carry(mesh_device, geometry, chunk)
            staged_slot = device_scalar(mesh_device, slot)
            staged_record = _record(mesh_device, chunk)
            ttnn.synchronize_device(mesh_device)  # the staged data is on device; only the chain below is in flight
            for layer in LAYER_IDS:  # every chip holds the slot's predecessor before the chain starts
                assert _stale_chips(reader.probe(layer, slot), chunk, chips) == chips, f"chunk {chunk} L{layer}"

            started = time.perf_counter()
            slow.enqueue(slow_chip)
            ttnn.copy(staged.recurrent, carry.recurrent)
            ttnn.copy(staged.convolution, carry.convolution)
            ttnn.copy(staged_slot, slot_tt)
            ttnn.copy(staged_record, record)
            if traced:
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            else:
                export_and_ack()
            enqueued = time.perf_counter()

            # Control: the other 31 chips have sent their records; the slow chip has not, still holds the previous
            # chunk in both kinds and both layers, and the service has emitted nothing.
            slow_socket = sockets[chips.index(slow_chip)]
            deadline = time.monotonic() + RING_TIMEOUT_S
            while not all(sockets[chips.index(chip)].has_data() for chip in others):
                assert time.monotonic() < deadline, f"chunk {chunk}: the 31 fast chips' records never arrived"
                time.sleep(0.0002)
            others_in = time.perf_counter()
            for layer in LAYER_IDS:
                tags = reader.probe(layer, slot, chips={slow_chip})
                assert _stale_chips(tags, chunk, [slow_chip]) == [slow_chip], (
                    f"chunk {chunk} L{layer}: slow chip {slow_chip} already holds this chunk ({tags}); the control "
                    "cannot show a missing ack"
                )
            assert not slow_socket.has_data(), f"chunk {chunk}: slow chip {slow_chip} sent its record too early"
            assert (
                ring.try_pop() is None
            ), f"chunk {chunk}: an ack arrived while slow chip {slow_chip} had neither exported nor sent its record"
            # A socket with data guarantees only that chip's FIRST record, i.e. layer 1's export.
            fast_tags = reader.probe(LAYER_IDS[0], slot, chips=set(others))
            assert not _stale_chips(fast_tags, chunk, others), f"chunk {chunk}: fast chips stale after their records"
            control_done = time.perf_counter()
            assert not slow_socket.has_data(), (
                f"chunk {chunk}: the slow chip finished during the control ({1000 * (control_done - started):.1f} ms); "
                f"raise SLOW_CHIP_MS"
            )

            for layer in LAYER_IDS:
                slot_id, pos_start, pos_end, layer_start, layer_end, _ = _pop(ring)
                acked = time.perf_counter()
                tags = reader.probe(layer, slot)  # the fast 32-chip probe first, right on the message
                assert (slot_id, pos_start, pos_end) == (slot, chunk * CHUNK_POSITIONS, (chunk + 1) * CHUNK_POSITIONS)
                assert (layer_start, layer_end) == (layer, layer + 1)
                stale = _stale_chips(tags, chunk, chips)
                assert not stale, f"{name} chunk {chunk} L{layer}: ack arrived while chips {stale} were stale: {tags}"
                logger.info(
                    f"chunk {chunk} slot {slot} L{layer} slow chip {slow_chip}, ms after the spin was queued: chain "
                    f"enqueued {1000 * (enqueued - started):.1f}, 31 records in {1000 * (others_in - started):.1f}, "
                    f"control done {1000 * (control_done - started):.1f}, ack {1000 * (acked - started):.1f}; "
                    "all 32 chips held this chunk (both kinds)"
                )
            for layer in LAYER_IDS:
                _check_layer(reader, geometry, layer, chunk, name)
            service.check()
            for tensor in (staged.recurrent, staged.convolution, staged_slot, staged_record):
                ttnn.deallocate(tensor)
            ttnn.synchronize_device(mesh_device)
    finally:
        service.stop()
        ring.shutdown()
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)
        for tensor in (carry.recurrent, carry.convolution, slot_tt, record):
            ttnn.deallocate(tensor)
        slow.deallocate()
        slabs.deallocate()


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [kda_mesh_param((8, 4), "mesh-8x4", "8x4", kda_device_params((8, 4)))],
    indirect=True,
)
def test_each_chip_record_follows_its_own_export(mesh_device, device_params):
    """Without the service: read chip k the moment chip k's own socket holds the layer's record."""
    geometry = _geometry(mesh_device)
    rows, cols = tuple(mesh_device.shape)
    chips = [(r, c) for r in range(rows) for c in range(cols)]  # socket order = row-major mesh coordinates
    seeds = {slot: NUM_CHUNKS + slot for slot in range(NUM_SLOTS)}
    writes = {slot: [seeds[slot]] for slot in range(NUM_SLOTS)}
    for chunk in range(NUM_CHUNKS):
        writes[chunk % NUM_SLOTS].append(chunk)
    _assert_writes_differ(geometry, writes)
    slabs = KdaStates.allocate(mesh_device, geometry, sp_axis=SP_AXIS, layer_ids=LAYER_IDS, num_slots=NUM_SLOTS)
    reader = _TableReader(mesh_device, geometry, slabs)
    slow = _SlowChip(mesh_device)
    for slot, chunk in seeds.items():
        carry = _chunk_carry(mesh_device, geometry, chunk)
        slot_tt = device_scalar(mesh_device, slot)
        for layer in LAYER_IDS:
            slabs.export_layer(carry, slot_tt, layer)
        for tensor in (slot_tt, carry.recurrent, carry.convolution):
            ttnn.deallocate(tensor)
    ttnn.synchronize_device(mesh_device)

    d2h = _d2h_service(mesh_device)
    sockets = d2h.get_sockets()
    assert len(sockets) == len(chips), f"{len(sockets)} sockets for {len(chips)} chips"
    layer = LAYER_IDS[-1]
    for chunk in range(NUM_CHUNKS):
        slot = chunk % NUM_SLOTS
        slow_chip = _slow_chip(chunk, rows, cols)
        carry = _chunk_carry(mesh_device, geometry, chunk)
        slot_tt = device_scalar(mesh_device, slot)
        record = _record(mesh_device, chunk)
        ttnn.synchronize_device(mesh_device)
        slow.enqueue(slow_chip)
        slabs.export_layer(carry, slot_tt, layer)
        send_layer_ack(
            mesh_device=mesh_device,
            global_layer_idx=layer,
            d2h_service=d2h,
            metadata_msg=record,
            on_layer_complete=None,
            trace_controller=None,
        )
        arrived = []
        deadline = time.monotonic() + RING_TIMEOUT_S
        while len(arrived) < len(sockets) and time.monotonic() < deadline:
            for index, sock in enumerate(sockets):
                if index not in arrived and sock.has_data():
                    arrived.append(index)
                    chip = chips[index]
                    tags = reader.probe(layer, slot, chips={chip})
                    assert not _stale_chips(tags, chunk, [chip]), (
                        f"chunk {chunk}: chip {chip}'s record arrived while its slab held {tags[chip]}, "
                        f"not this chunk's tag {_tag(chunk, chip[0])}"
                    )
        assert len(arrived) == len(sockets), f"chunk {chunk}: only {len(arrived)} of {len(sockets)} records arrived"
        assert chips[arrived[-1]] == slow_chip, f"chunk {chunk}: slow chip {slow_chip} was not the last record"
        d2h.read_metadata()  # drain one record from every socket
        _check_layer(reader, geometry, layer, chunk, "per-socket")
        logger.info(f"chunk {chunk}: 32 records, slow chip {slow_chip} last; every chip held its export on arrival")
        for tensor in (carry.recurrent, carry.convolution, slot_tt, record):
            ttnn.deallocate(tensor)
    slow.deallocate()
    slabs.deallocate()
