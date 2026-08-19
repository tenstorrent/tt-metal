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

   MEASURED (Blackhole 1x4 QuietBox, 2026-08-19): 16 op-free open/close cycles are
   green in ~4 s, so the fixture cycle alone is NOT the wedge. Mixed with real op
   traffic the wedge appears after ~6 cycles, i.e. it is fabric-teardown state, and
   it is why the 90-case acceptance file cannot complete in one pytest process on
   this board -- see changelog / the run notes.

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
@pytest.mark.parametrize("cpp_iteration", [0, 1, 2, 3, 4, 5])
def test_cpp_reference_op_fixture_cycle(mesh_device, cpp_iteration):
    """Same fixture cycle + same fabric traffic, but through the C++ op."""
    _require_mesh(mesh_device)
    if not hasattr(ttnn, "point_to_point"):
        pytest.skip("bound C++ ttnn.point_to_point not available")

    num_devices = prod(tuple(mesh_device.shape))
    torch.manual_seed(7)
    full = torch.randn((num_devices, 1, 32, 32), dtype=torch.bfloat16)
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
    ttnn.point_to_point(inp, ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1), topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh_device)


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
