# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Deterministic debugging tests for the Python point_to_point CCL op.

DO NOT DELETE — this file documents the debugging process.

Two families:

A. ``test_fabric_mesh_open_close_only`` — the CONTROL. It opens the (1, 4)
   FABRIC_1D mesh through the same function-scoped ``mesh_device`` fixture the
   acceptance test uses, runs NO op at all, and does that three times. The
   acceptance suite showed exactly one test passing per pytest process, with every
   subsequent mesh open dying in
   ``RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset`` ->
   "Device 0: Timed out while waiting for active ethernet core 24-25 to become
   active again". If THIS control reproduces that with no op in the picture, the
   wedge belongs to repeated fabric mesh open/close on this board, not to the op.

   MEASURED (Blackhole 1x4 QuietBox, 2026-08-19) -- the full bisect:
     * 16x op-free fabric mesh open/close ................. PASS (10.2 s)
     * GlobalSemaphore created per cycle, no traffic, 4x ... PASS ( 4.2 s)
     * THIS op, 1 hop  (0,0)->(0,1), 4 cycles ............. PASS ( 4.5 s)
     * THIS op, 2 hops (0,0)->(0,2), 4 cycles ............. 1 PASS then wedge
     * BOUND C++ ttnn.point_to_point, 1 hop,  4 cycles .... PASS ( 7.0 s)
     * BOUND C++ ttnn.point_to_point, 2 hops, 4 cycles .... 1 PASS then wedge
   CONCLUSION: the wedge is a MULTI-HOP fabric-teardown limitation of this board,
   reproduced identically by the pre-existing C++ op (an independent implementation).
   It is NOT specific to this Python op. In both cases the FIRST multi-hop transfer
   delivers correct data and only the NEXT mesh open fails, so it is a teardown/
   ethernet-firmware issue, not a data-path issue. Single-hop traffic is unaffected
   and repeats indefinitely. Practical consequence: any pytest process that performs
   a multi-hop transfer must be the last one to use the board before a reset.

B. Deterministic-payload transfers whose every byte is hand-checkable:
     * all-ones      — every element of the receiver shard must be exactly 1.0
     * monotonic     — unique values, so ANY reordering/mis-addressing shows up
     * tile-position — t[r][c] = r*100 + c, so a wrong page stride is visible as
                       a whole row landing at the wrong offset
   The monotonic/tile-position patterns are the ones that catch the
   TensorAccessor page-stride bug (op_design.md Key Risk #1): a 96 B or 48 B
   row-major row on Blackhole (64 B DRAM alignment) mis-addresses every page past
   the first bank row, which random data hides behind a still-highish PCC but
   exact-value comparison does not.
"""

from math import prod

import pytest
import torch

import ttnn

from ttnn.operations.point_to_point import point_to_point

MESH_SHAPE = (1, 4)
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _require_mesh(mesh_device):
    if prod(tuple(mesh_device.shape)) < prod(MESH_SHAPE):
        pytest.skip(f"point_to_point debug needs a {MESH_SHAPE} mesh")


# --------------------------------------------------------------------------------------
# A. CONTROL — fabric mesh open/close only, no op
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("iteration", list(range(16)))
def test_fabric_mesh_open_close_only(mesh_device, iteration):
    """No op, no fabric traffic — just prove the mesh can be opened repeatedly.

    Isolates 'repeated fabric mesh open/close wedges the ethernet cores' from
    'point_to_point wedges the ethernet cores'.
    """
    _require_mesh(mesh_device)
    assert tuple(mesh_device.shape) == MESH_SHAPE


# --------------------------------------------------------------------------------------
# A2. DIAGNOSTIC — is id(mesh_device) reused across the fixture's open/close cycle?
# --------------------------------------------------------------------------------------
# The op caches its GlobalSemaphore per mesh device. Keying that cache on
# id(mesh_device) is only sound if CPython cannot hand the SAME id to a DIFFERENT
# MeshDevice object after the previous one is collected. If it can, a later call
# reuses a GlobalSemaphore that belongs to a CLOSED device -- a stale L1 address
# that the kernels then atomic-inc and zero, which can land on whatever the new
# device allocated there (e.g. an EDM flow-control semaphore) and wedge an ethernet
# core. This test records the ids so the reuse is observable, not theorised.
_SEEN_DEVICE_IDS = []


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("probe_iteration", [0, 1, 2, 3])
def test_mesh_device_id_reuse_probe(mesh_device, probe_iteration):
    """Record id(mesh_device) across successive fixture open/close cycles."""
    _require_mesh(mesh_device)
    import ttnn as _ttnn
    from ttnn.operations.point_to_point.point_to_point import _get_or_create_semaphore

    sem = _get_or_create_semaphore(mesh_device)
    addr = _ttnn.get_global_semaphore_address(sem)
    _SEEN_DEVICE_IDS.append((probe_iteration, id(mesh_device), addr))
    print(f"P2P_ID_PROBE iteration={probe_iteration} id={id(mesh_device)} sem_addr={addr}")
    if probe_iteration == 3:
        ids = [d for _, d, _ in _SEEN_DEVICE_IDS]
        print(f"P2P_ID_PROBE all_ids={ids} unique={len(set(ids))} of {len(ids)}")


# --------------------------------------------------------------------------------------
# A3. CONTROL — the BOUND C++ point_to_point under the same fixture cycle
# --------------------------------------------------------------------------------------
# The op-free control (A) survives 16 mesh open/close cycles, but a cycle that also
# runs REAL fabric traffic wedges "active ethernet core 24-25" after 1-4 iterations.
# That leaves two candidates: (a) this Python op leaves fabric state behind, or
# (b) any worker<->EDM connection followed by a fabric teardown wedges this board.
# ttnn.point_to_point (the pre-existing C++ op, which this Python op deliberately
# does NOT wrap or dispatch to) is an independent implementation of the same
# traffic pattern, so running IT under the identical fixture separates the two.
# Referenced here for DIAGNOSIS ONLY -- the op itself never touches it.
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
# ids without "=" so `pytest -k` can select them.
@pytest.mark.parametrize("cpp_hops", [1, 2, 3], ids=["ch1", "ch2", "ch3"])
@pytest.mark.parametrize("cpp_payload", [(1, 1, 32, 32), (1, 1, 64, 128)], ids=["cp1tile", "cp8tile"])
@pytest.mark.parametrize("cpp_iteration", list(range(20)), ids=[f"c{i}" for i in range(20)])
def test_cpp_reference_op_fixture_cycle(mesh_device, cpp_iteration, cpp_hops, cpp_payload):
    """Same fixture cycle + same fabric traffic + same hop count, C++ op."""
    _require_mesh(mesh_device)
    if not hasattr(ttnn, "point_to_point"):
        pytest.skip("bound C++ ttnn.point_to_point not available")

    num_devices = prod(tuple(mesh_device.shape))
    torch.manual_seed(7)
    full = torch.randn((num_devices * cpp_payload[0], *cpp_payload[1:]), dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    # MEASURED signature of the bound C++ op (op_design.md's "(receiver, sender)"
    # note is stale): (input, sender_coord, receiver_coord, *, output_tensor,
    # intermediate_tensor, topology) -- topology is keyword-only there.
    ttnn.point_to_point(inp, ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, cpp_hops), topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh_device)


# --------------------------------------------------------------------------------------
# A4. DIAGNOSTIC — does any acceptance case frame a packet LARGER than the EDM slot?
# --------------------------------------------------------------------------------------
# `ccl_packet_dims` caps a packet at get_tt_fabric_channel_buffer_size_bytes(), but
# that value is header + max_payload (fabric_context.cpp:157-159), while the worker
# writes the header at the channel-slot base and the payload at
# base + sizeof(PACKET_HEADER_TYPE) (edm_fabric_worker_adapters.hpp:705-708). The
# only guard is an ASSERT that omits the header and is compiled out in Release. So a
# packet_size in (max_payload, max_payload + header] overruns the ETH CORE's L1 —
# which corrupts the router so it never polls its run flag, and device close then
# fails with "Timed out while waiting for active ethernet core ... to become active
# again" (llrt.cpp:529-569). That is our exact symptom, so this test enumerates the
# whole acceptance matrix and reports any case that oversteps.
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_packet_geometry_within_edm_slot(mesh_device):
    """Enumerate every acceptance (dtype, layout, shape) and print the framing."""
    _require_mesh(mesh_device)
    max_payload = ttnn._ttnn.fabric.get_tt_fabric_max_payload_size_bytes()
    header = ttnn._ttnn.fabric.get_tt_fabric_packet_header_size_bytes()
    l1a = ttnn.get_l1_alignment()
    print(f"P2P_GEOM max_payload={max_payload} header={header} channel_buffer={header + max_payload}")

    dtype_layouts = [
        (ttnn.bfloat16, ttnn.TILE_LAYOUT),
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.float32, ttnn.TILE_LAYOUT),
        (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ]
    shapes = [
        (1, 1, 32, 32),
        (1, 1, 64, 128),
        (1, 1, 96, 64),
        (2, 1, 32, 64),
        (1, 1, 48, 64),
        (1, 1, 32, 48),
        (1, 1, 24, 24),
        # golden-suite shapes too
        (1, 1, 32, 64),
        (1, 1, 128, 128),
        (1, 1, 256, 64),
        (2, 4, 64, 64),
        (1, 8, 32, 32),
        (32, 64),
        (4, 32, 96),
        (1, 1, 512, 512),
        (1, 4, 256, 128),
        (1, 1, 56, 88),
        (2, 1, 48, 64),
    ]
    offenders = []
    for dtype, layout in dtype_layouts:
        for shape in shapes:
            t = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.float32),
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            page = t.buffer_page_size()
            npages = t.buffer_num_pages()
            d = ttnn._ttnn.fabric.ccl_packet_dims(dtype, page, npages, l1a)
            over = d.packet_size_bytes > max_payload
            print(
                f"P2P_GEOM {dtype} {layout} {shape} page={page} npages={npages} "
                f"pkt={d.packet_size_bytes} ppp={d.pages_per_packet} seg={d.page_segments} "
                f"total={d.total_packets} OVER={over}"
            )
            if over:
                offenders.append((dtype, layout, shape, page, npages, d.packet_size_bytes))
            ttnn.deallocate(t)
    print(f"P2P_GEOM offenders={offenders}")
    assert not offenders, f"packet_size exceeds EDM max_payload ({max_payload} B) for: {offenders}"


# --------------------------------------------------------------------------------------
# A5. BISECT — is the ethernet wedge triggered by HOP COUNT rather than call count?
# --------------------------------------------------------------------------------------
# Evidence from the acceptance runs: a chunk whose ONLY op call was a 2-hop transfer
# ((0,0)->(0,2)) wedged immediately, while a chunk of three consecutive 1-hop
# transfers survived and only the fourth call wedged. A multi-hop send routes the
# payload AND the trailing atomic-inc through the eth cores of the intervening chip,
# which has NO program of its own. This parametrizes hop count with a fixed payload so
# the two variables (hops vs call count) are separated.
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
# ids without "=" so `pytest -k` can select them (its expression parser rejects "=").
@pytest.mark.parametrize("hops", [1, 2, 3], ids=["h1", "h2", "h3"])
@pytest.mark.parametrize("payload", [(1, 1, 32, 32), (1, 1, 64, 128)], ids=["p1tile", "p8tile"])
@pytest.mark.parametrize("rep", list(range(20)), ids=[f"r{i}" for i in range(20)])
def test_hop_count_stress(mesh_device, hops, rep, payload):
    """Fixed payload, varying hop distance, several fixture cycles each."""
    _require_mesh(mesh_device)
    sender = ttnn.MeshCoordinate(0, 0)
    receiver = ttnn.MeshCoordinate(0, hops)

    num_devices = prod(tuple(mesh_device.shape))
    torch.manual_seed(3)
    full = torch.randn((num_devices * payload[0], *payload[1:]), dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    expected = ttnn.to_torch(ttnn.get_device_tensors(inp)[0]).float()

    out = point_to_point(inp, sender, receiver)
    ttnn.synchronize_device(mesh_device)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[hops]).float()
    diff = (got - expected).abs().max().item()
    assert diff == 0.0, f"hops={hops} rep={rep}: max diff {diff}"


# --------------------------------------------------------------------------------------
# B. Deterministic payloads — exact-value checks
# --------------------------------------------------------------------------------------
def _shard_deterministic(mesh_device, shard_shape, dtype, layout, make_shard):
    """Shard `make_shard(device_index)` across the mesh on dim 0.

    Returns (ttnn_tensor, [per-device torch shards as float32]).
    """
    num_devices = prod(tuple(mesh_device.shape))
    per_device = [make_shard(i) for i in range(num_devices)]
    full = torch.cat(per_device, dim=0)
    tensor = ttnn.from_torch(
        full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return tensor, [s.float() for s in per_device]


def _read_shards(tensor):
    return [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(tensor)]


def _run_exact(mesh_device, shard_shape, dtype, layout, make_shard, label):
    """Send shard 0 -> device 1 and require the receiver shard to match EXACTLY."""
    _require_mesh(mesh_device)
    sender, receiver = ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1)

    tensor, expected_shards = _shard_deterministic(mesh_device, shard_shape, dtype, layout, make_shard)
    out = point_to_point(tensor, sender, receiver)
    ttnn.synchronize_device(mesh_device)
    actual = _read_shards(out)

    # Receiver got the sender's shard, bit-for-bit (pure byte copy -> exact).
    diff = (actual[1] - expected_shards[0]).abs().max().item()
    assert diff == 0.0, f"{label}: receiver shard != sender shard, max diff {diff}\n{actual[1]}"

    # Every other device is byte-for-byte its own (seeded) input shard.
    for i, exp in enumerate(expected_shards):
        if i == 1:
            continue
        d = (actual[i] - exp).abs().max().item()
        assert d == 0.0, f"{label}: device {i} shard was disturbed, max diff {d}"


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_all_ones_single_tile(mesh_device):
    """All-ones payload: every received element must be exactly 1.0.

    Hand-calculated: the op does no arithmetic, so out == in == 1.0 everywhere.
    """
    shape = (1, 1, 32, 32)
    _run_exact(
        mesh_device,
        shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        # device i gets a constant (i+1) so a cross-device mix-up is obvious.
        lambda i: torch.full(shape, float(i + 1), dtype=torch.bfloat16),
        "all_ones_single_tile",
    )


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),  # single tile / single RM page group
        (1, 1, 64, 128),  # multi-tile, multi-packet
        (1, 1, 32, 48),  # 96 B bf16 row: NOT 64 B aligned (page-stride trap)
        (1, 1, 24, 24),  # 48 B bf16 row: NOT 64 B aligned (page-stride trap)
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_monotonic_exact(mesh_device, shape, layout):
    """Monotonic payload: unique values expose ANY reordering or page mis-addressing.

    Values are exactly representable in bfloat16 up to 256, so the comparison is
    exact rather than tolerance-based. Each device's block is offset by 1000*i so a
    shard landing from the wrong device is unmistakable.
    """
    n = int(torch.tensor(shape).prod().item())

    def make(i):
        # 0..n-1 modulo 256 keeps every value exactly representable in bf16.
        vals = (torch.arange(n) % 256).float() + 1000.0 * i
        return vals.reshape(shape).to(torch.bfloat16)

    _run_exact(mesh_device, shape, ttnn.bfloat16, layout, make, f"monotonic {shape} {layout}")


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_tile_position_encoding_row_major(mesh_device):
    """t[r][c] = r*100 + c in a 96 B row-major row (the page-stride trap).

    If the output TensorAccessor's per-bank stride were the raw 96 B logical page
    instead of the 128 B aligned page, whole rows land at the wrong offset and this
    encoding shows exactly which row went where.
    """
    H, W = 32, 48  # bfloat16 row = 96 B, not a multiple of the 64 B DRAM alignment
    shape = (1, 1, H, W)

    def make(i):
        t = torch.zeros(shape, dtype=torch.float32)
        for r in range(H):
            for c in range(W):
                t[0, 0, r, c] = r * 100 + c + 10000 * i
        return t.to(torch.bfloat16)

    # r*100+c maxes at 3147; bf16 has 8 mantissa bits, so pick values that round
    # trip: compare against the SAME bf16-rounded tensor (make() is the reference).
    _run_exact(mesh_device, shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, make, "tile_position_rm_96B")
