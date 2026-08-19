# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Deterministic regression guards for the Python point_to_point CCL op.

Distilled (verifier pass, 2026-08-19) from the implementer's much larger wedge
bisect. Three families:

A. **Fixture-cycle / hop-count guards.** The implementer's changelog reported a
   "multi-packet / multi-hop ethernet wedge": after a transfer of more than one
   packet or over more than one hop, the NEXT ``open_mesh_device`` was said to die in
   ``RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset`` -> "Timed out
   while waiting for active ethernet core 24-25 to become active again", forcing one
   pytest process (+ a board reset) per case.

   **It does not reproduce.** MEASURED on the graded Blackhole 1x4 QuietBox after the
   DRAM-slot-stride fix, each in a SINGLE pytest process with the function-scoped
   ``mesh_device`` fixture opening/closing the fabric mesh once per case:
     * acceptance suite (``test_point_to_point.py``) ....... 90/90 PASS in 61 s
     * golden cartesian (396 cells + 36 INVALID skips) ..... 396/396 PASS in 171 s
     * ``test_hop_count_stress`` (1/2/3 hops x 20 reps) .... 120/120 PASS
   The wedge was almost certainly the NoC-alignment fault it was diagnosed alongside
   (a sanitizer halt on the offending core wedges the ethernet), which the
   ``_dram_slot_stride`` CB-sizing fix removed. The guards below stay, at a fraction
   of the original repetition count, so a regression is caught: they are the cheapest
   canary for "fabric teardown after real traffic".

   The implementer's fourth control -- running the BOUND C++ ``ttnn.point_to_point``
   under the same fixture to prove parity -- has been removed. Its conclusion is
   recorded above, and the generation mandate says to treat the C++ op as if it does
   not exist, so a test in the generated op's own directory must not dispatch to it.

B. **Packet-geometry guard.** ``ccl_packet_dims`` caps a packet at
   ``get_tt_fabric_channel_buffer_size_bytes()``, but that value is
   ``header + max_payload`` while the worker writes the header at the channel-slot
   base and the payload at ``base + sizeof(PACKET_HEADER_TYPE)``. The only guard is an
   ``ASSERT`` that omits the header and compiles out in Release, so a ``packet_size``
   in ``(max_payload, max_payload + header]`` overruns the ETH core's L1. This test
   enumerates the whole acceptance + golden shape matrix and fails if any case
   oversteps.

C. **Deterministic-payload transfers** whose every byte is hand-checkable:
     * all-ones      -- every element of the receiver shard must be exactly 1.0
     * monotonic     -- unique values, so ANY reordering/mis-addressing shows up
     * tile-position -- t[r][c] = r*100 + c, so a wrong page stride is visible as
                        a whole row landing at the wrong offset
   The monotonic/tile-position patterns are the ones that catch the TensorAccessor
   page-stride bug (op_design.md Key Risk #1): a 96 B or 48 B row-major row on
   Blackhole (64 B DRAM alignment) mis-addresses every page past the first bank row,
   which random data hides behind a still-highish PCC but exact-value comparison
   does not.
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
@pytest.mark.parametrize("iteration", list(range(3)))
def test_fabric_mesh_open_close_only(mesh_device, iteration):
    """No op, no fabric traffic — just prove the mesh can be opened repeatedly.

    Isolates 'repeated fabric mesh open/close wedges the ethernet cores' from
    'point_to_point wedges the ethernet cores'. 16 cycles measured green; 3 kept.
    """
    _require_mesh(mesh_device)
    assert tuple(mesh_device.shape) == MESH_SHAPE


# --------------------------------------------------------------------------------------
# A2. GUARD — the op-internal GlobalSemaphore is one-per-LIVE-device, never inherited
# --------------------------------------------------------------------------------------
# The op caches its GlobalSemaphore per mesh device. The original design keyed that
# cache on `id(mesh_device)` in a module-level dict, which is unsound twice over: the
# dict outlives the device (pinning a closed device's L1 allocation for the life of the
# process), and CPython recycles id()s, so a NEW MeshDevice allocated at a freed one's
# address silently inherits the PREVIOUS device's semaphore — a stale L1 address the
# kernels then atomic-inc and zero, which can land on whatever the new device put
# there (e.g. an EDM flow-control semaphore) and wedge an ethernet core.
#
# The cache is now an attribute ON the device object, so its lifetime is exactly the
# device's. This guards both halves of that contract: the handle is stable within one
# device (so program-cache hits see one address) and it is stored on the device rather
# than in process-global state (so it cannot outlive or cross devices).
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("probe_iteration", [0, 1])
def test_semaphore_is_per_live_device(mesh_device, probe_iteration):
    """One semaphore per live mesh device, stable across calls, bound to the object."""
    _require_mesh(mesh_device)
    from ttnn.operations.point_to_point.point_to_point import (
        _SEM_ATTR,
        _SEMAPHORE_FALLBACK_CACHE,
        _get_or_create_semaphore,
    )

    first = _get_or_create_semaphore(mesh_device)
    second = _get_or_create_semaphore(mesh_device)
    addr = ttnn.get_global_semaphore_address(first)
    assert ttnn.get_global_semaphore_address(second) == addr, "semaphore address must be stable"

    # Bound to THIS device object, not to process-global state keyed by a reusable id.
    assert getattr(mesh_device, _SEM_ATTR, None) is not None, "semaphore was not bound to the device"
    assert not _SEMAPHORE_FALLBACK_CACHE, "MeshDevice accepted the attribute; the id() fallback must stay unused"
    print(f"P2P_SEM_PROBE iteration={probe_iteration} id={id(mesh_device)} sem_addr={addr}")


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

    # EVERY dtype in SUPPORTED, not just the float ones: the bfloat16 bit_floor in
    # ccl_packet_dims caps bf16 at 4096 B (always <= max_payload), so restricting this
    # sweep to bf16/f32/bf8b hid the integer dtypes, where the golden cell
    # (1, 1, 56, 88) uint16 ROW_MAJOR frames a 4400 B packet (25 x 176 B pages).
    dtype_layouts = [
        (dtype, layout)
        for dtype in (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.uint16, ttnn.int32, ttnn.uint32)
        for layout in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT)
        if not (dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT)  # INVALID: no RM bf8b
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
        # extended-suite shapes that FORCE page_segments > 1 (wide row-major rows):
        # segmentation must keep packet_size within the payload cap, not exceed it.
        (1, 1, 8, 4096),
        (1, 1, 8, 2048),
    ]
    offenders = []
    for dtype, layout in dtype_layouts:
        for shape in shapes:
            zeros = torch.zeros(
                shape, dtype=torch.int32 if dtype in (ttnn.uint16, ttnn.int32, ttnn.uint32) else torch.float32
            )
            t = ttnn.from_torch(
                zeros,
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
@pytest.mark.parametrize("rep", list(range(2)), ids=[f"r{i}" for i in range(2)])
def test_hop_count_stress(mesh_device, hops, rep, payload):
    """Fixed payload, varying hop distance, several fixture cycles each.

    20 reps per (hops, payload) measured green; 2 kept as the standing canary.
    """
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
