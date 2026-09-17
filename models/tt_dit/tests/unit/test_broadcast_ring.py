# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness + timing for the broadcast_ring CCL op (experimental).

Checks the property ttnn.broadcast can't provide: broadcast along the RING axis while the ORTHOGONAL
(tp) axis stays SHARDED, so each tp row broadcasts its own data. Also compares the result to
all_gather+slice (same golden) and logs a rough speedup.

Requires the op to be built (ttnn.experimental.broadcast_ring). Correctness + host wall-clock:
  pytest models/tt_dit/tests/unit/test_broadcast_ring.py -k bh_4x8_ring -s

Host wall-clock is dispatch-bound at these payloads and does NOT resolve the data-movement difference.
For DEVICE kernel time, run under the tracy device-op profiler and read the per-op CSV:
  python -m tracy -r -p -o bcast_ring \\
    -m "pytest models/tt_dit/tests/unit/test_broadcast_ring.py -k '1024tiles and bh_4x8_ring' \\
        -s -p no:cacheprovider --timeout=0"
The signpost regions ("broadcast_ring" / "all_gather") delimit each op's warm loop; compare the
BroadcastRingDeviceOperation device time vs the AllGather op device time in the summary CSV.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.utils.test import ring_params

try:  # tracy signpost is only present under `python -m tracy`; no-op otherwise.
    from tracy import signpost
except ImportError:

    def signpost(_name):
        pass


T = ttnn.TILE_SIZE
OWNER = 5  # sender index along the ring (cluster) axis


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "device_params", "topology"),
    [pytest.param((4, 8), 1, 0, ring_params, ttnn.Topology.Ring, id="bh_4x8_ring")],
    indirect=["mesh_device", "device_params"],
)
# (tiles_per_shard, chunk_size_tiles): 1 tile = correctness sanity; 1024 tiles exposes the bandwidth
# comparison (overhead-bound at 1 tile). chunk_size 0 = auto (one fabric packet); the 1024-tile rows sweep
# the chunk size to find the pipeline-overlap vs per-chunk-overhead sweet spot.
@pytest.mark.parametrize(
    # (tiles_per_shard, chunk_size_tiles, bcast_offset_tiles, bcast_num_tiles, use_l1_relay): offset/num 0/0 =
    # whole shard, a nonzero pair broadcasts only that sub-range (pre-slice). use_l1_relay picks the L1 relay
    # (no per-hop DRAM read); the matched chunk128 rows give a DRAM-vs-L1 steady-state comparison in one CSV.
    ("tiles_per_shard", "chunk_size_tiles", "bcast_offset_tiles", "bcast_num_tiles", "use_l1_relay", "num_slots"),
    [
        pytest.param(1, 0, 0, 0, False, 0, id="1tile"),
        pytest.param(1024, 0, 0, 0, False, 0, id="1024tiles_chunkauto"),
        pytest.param(1024, 8, 0, 0, False, 0, id="1024tiles_chunk8"),
        pytest.param(1024, 32, 0, 0, False, 0, id="1024tiles_chunk32"),
        pytest.param(1024, 128, 0, 0, False, 0, id="1024tiles_chunk128"),
        pytest.param(1024, 0, 300, 400, False, 0, id="1024tiles_subrange"),  # pre-slice: broadcast tiles [300, 700)
        # L1-relay chunk sweep. sp=8 knee was chunk64 (L1 is overlap-limited, wants smaller chunks, not larger).
        pytest.param(1024, 32, 0, 0, True, 0, id="1024tiles_chunk32_l1"),
        pytest.param(1024, 64, 0, 0, True, 0, id="1024tiles_chunk64_l1"),
        pytest.param(1024, 128, 0, 0, True, 0, id="1024tiles_chunk128_l1"),
        # L1-relay credit-window (num_slots) sweep at the best chunk (64). Deeper window = more overlap; L1
        # cost is num_slots * chunk * page_size (slots=8 x chunk64 x 2KB ~ 1MB, still within a Blackhole core).
        pytest.param(1024, 64, 0, 0, True, 4, id="1024tiles_chunk64_slots4_l1"),
        pytest.param(1024, 64, 0, 0, True, 6, id="1024tiles_chunk64_slots6_l1"),
        pytest.param(1024, 64, 0, 0, True, 8, id="1024tiles_chunk64_slots8_l1"),
    ],
)
def test_broadcast_ring(
    mesh_device,
    sp_axis,
    tp_axis,
    device_params,
    topology,
    tiles_per_shard,
    chunk_size_tiles,
    bcast_offset_tiles,
    bcast_num_tiles,
    use_l1_relay,
    num_slots,
):
    rows, cols = tuple(mesh_device.shape)
    tp_factor, sp_factor = rows, cols
    N = tiles_per_shard
    logger.info(
        f"[bcast_ring] mesh={rows}x{cols} ring_axis={sp_axis}(={sp_factor}) tp_axis={tp_axis}(={tp_factor}) "
        f"owner={OWNER} tiles_per_shard={N}"
    )

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    wsd_id = ttnn.SubDeviceId(0)
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([wsd_id])

    # 2D-sharded input: device (r, c) holds an N-tile-wide block, constant = r*10 + c. tp SHARDED.
    host = torch.zeros(1, 1, tp_factor * T, sp_factor * N * T, dtype=torch.float32)
    for r in range(tp_factor):
        for c in range(sp_factor):
            host[0, 0, r * T : (r + 1) * T, c * N * T : (c + 1) * N * T] = r * 10 + c
    shard_dims = [None, None]
    shard_dims[tp_axis] = 2
    shard_dims[sp_axis] = 3
    rm = ttnn.from_torch(
        host.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    tt_in = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    # Sample a tile inside the broadcast range (offset 0 for a whole-shard broadcast).
    sample_tile = bcast_offset_tiles

    def readback(t):
        out = ttnn.to_torch(
            ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT),
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        )
        return [
            [round(out[0, 0, r * T, (c * N + sample_tile) * T].item()) for c in range(sp_factor)]
            for r in range(tp_factor)
        ]

    from models.tt_dit.parallel.manager import CCLManager  # local import to keep collection light

    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=topology)

    def run_bcast():
        return ttnn.experimental.broadcast_ring(
            tt_in,
            sender_ring_index=OWNER,
            cluster_axis=sp_axis,
            topology=topology,
            chunk_size_tiles=chunk_size_tiles,
            broadcast_offset_tiles=bcast_offset_tiles,
            broadcast_num_tiles=bcast_num_tiles,
            use_l1_relay=use_l1_relay,
            num_slots=num_slots,
        )

    def run_ag():
        return ccl.all_gather(tt_in, dim=3, mesh_axis=sp_axis, use_hyperparams=False)

    # --- correctness (also warms up / JIT-compiles both ops) ---
    tt_out = run_bcast()
    ttnn.synchronize_device(mesh_device, sub_device_ids=[wsd_id])
    grid_out = readback(tt_out)
    logger.info("[bcast_ring] OUTPUT grid (row=tp, col=ring):")
    for r in range(tp_factor):
        logger.info(f"[bcast_ring]   tp{r}: {grid_out[r]}  (per-line expects all {r * 10 + OWNER})")
    # Per-line correctness: device (r, c) holds the sender ring-shard's data for ITS tp row = r*10 + OWNER.
    per_line = all(grid_out[r][c] == r * 10 + OWNER for r in range(tp_factor) for c in range(sp_factor))

    gathered = run_ag()
    ttnn.synchronize_device(mesh_device, sub_device_ids=[wsd_id])
    g = ttnn.to_torch(
        ttnn.to_layout(gathered, ttnn.ROW_MAJOR_LAYOUT),
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    matches_ag = all(
        round(g[0, 0, r * T, (OWNER * N + sample_tile) * T].item()) == r * 10 + OWNER for r in range(tp_factor)
    )

    # --- steady-state timing (ops are warm; first call above already JIT-compiled + built the program) ---
    # Each op's warm loop is bracketed by a named signpost so, under `python -m tracy -r -p`, the device-op
    # profiler CSV can be filtered per op. Host wall-clock below is dispatch-bound and only a rough sanity.
    iters = 10

    def _timed(fn, name):
        fn()  # extra warm iter so the program-cache hit path is what we measure
        ttnn.synchronize_device(mesh_device, sub_device_ids=[wsd_id])
        signpost(name)
        t0 = time.time()
        for _ in range(iters):
            fn()
        ttnn.synchronize_device(mesh_device, sub_device_ids=[wsd_id])
        dt = (time.time() - t0) / iters
        signpost(f"{name}_stop")
        return dt

    dt_bcast = _timed(run_bcast, "broadcast_ring")
    dt_ag = _timed(run_ag, "all_gather")

    logger.info(f"[bcast_ring] per_line={per_line}  matches_all_gather_slice={matches_ag}")
    logger.info(
        f"[bcast_ring] tiles_per_shard={N} chunk_size_tiles={chunk_size_tiles} "
        f"warm host wall-clock ({iters} iters, DISPATCH-BOUND, not device time): "
        f"broadcast_ring={dt_bcast*1e3:.3f} ms  all_gather={dt_ag*1e3:.3f} ms"
    )
    logger.info(
        "[bcast_ring] for device kernel time, run under `python -m tracy -r -p -o bcast_ring` and read the "
        "per-op CSV (BroadcastRingDeviceOperation vs AllGather); see the module docstring."
    )
    assert per_line, "broadcast_ring did not deliver each tp row's own ring-shard to all ring devices"
    assert matches_ag, "broadcast_ring result disagrees with all_gather+slice"


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "device_params", "topology"),
    [pytest.param((4, 8), 1, 0, ring_params, ttnn.Topology.Ring, id="bh_4x8_ring")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("n_per_shard", "rng_lo", "rng_hi"),
    [
        pytest.param(4, 10, 14, id="straddle_2_3"),
        pytest.param(8, 14, 18, id="straddle_1_2_small"),
        pytest.param(6, 10, 16, id="straddle_1_2"),
    ],
)
def test_broadcast_ring_straddle(mesh_device, sp_axis, tp_axis, device_params, topology, n_per_shard, rng_lo, rng_hi):
    rows, cols = tuple(mesh_device.shape)
    tp_factor, sp_factor = rows, cols
    N = n_per_shard
    owner_lo, owner_hi = rng_lo // N, (rng_hi - 1) // N
    assert owner_lo != owner_hi and owner_hi < sp_factor

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    wsd_id = ttnn.SubDeviceId(0)
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([wsd_id])

    host = torch.zeros(1, 1, tp_factor * T, sp_factor * N * T, dtype=torch.float32)
    for r in range(tp_factor):
        for c in range(sp_factor):
            host[0, 0, r * T : (r + 1) * T, c * N * T : (c + 1) * N * T] = r * 10 + c
    shard_dims = [None, None]
    shard_dims[tp_axis] = 2
    shard_dims[sp_axis] = 3
    rm = ttnn.from_torch(
        host.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    tt_in = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    def sample(t, st):
        out = ttnn.to_torch(
            ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT),
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        )
        return [[round(out[0, 0, r * T, (c * N + st) * T].item()) for c in range(sp_factor)] for r in range(tp_factor)]

    ok = True
    for owner in range(owner_lo, owner_hi + 1):
        lo = max(rng_lo, owner * N) - owner * N
        hi = min(rng_hi, (owner + 1) * N) - owner * N
        full = ttnn.experimental.broadcast_ring(
            tt_in,
            sender_ring_index=owner,
            cluster_axis=sp_axis,
            broadcast_offset_tiles=lo,
            broadcast_num_tiles=hi - lo,
            topology=topology,
            subdevice_id=wsd_id,
            num_links=2,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[wsd_id])
        for st in range(lo, hi):
            grid_out = sample(full, st)
            good = all(grid_out[r][c] == r * 10 + owner for r in range(tp_factor) for c in range(sp_factor))
            if not good:
                ok = False
                logger.info(f"[straddle] owner={owner} tile={st}: {grid_out}")
    assert ok, "straddle sub-range broadcast delivered wrong data"


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "device_params", "topology"),
    [pytest.param((4, 8), 1, 0, ring_params, ttnn.Topology.Ring, id="bh_4x8_ring")],
    indirect=["mesh_device", "device_params"],
)
# L1-relay path (use_l1_relay=True): same per-line correctness, credit-bounded L1 recv buffer. A few
# tiles/chunk/subrange configs exercise <slots, >slots (credit wraps), and pre-slice.
@pytest.mark.parametrize(
    ("tiles_per_shard", "chunk_size_tiles", "bcast_offset_tiles", "bcast_num_tiles"),
    [
        pytest.param(1, 0, 0, 0, id="l1_1tile"),
        pytest.param(64, 8, 0, 0, id="l1_64tiles_chunk8"),
        pytest.param(1024, 128, 0, 0, id="l1_1024tiles_chunk128"),
        pytest.param(1024, 0, 300, 400, id="l1_1024tiles_subrange"),
    ],
)
def test_broadcast_ring_l1(
    mesh_device,
    sp_axis,
    tp_axis,
    device_params,
    topology,
    tiles_per_shard,
    chunk_size_tiles,
    bcast_offset_tiles,
    bcast_num_tiles,
):
    rows, cols = tuple(mesh_device.shape)
    tp_factor, sp_factor = rows, cols
    N = tiles_per_shard

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    wsd_id = ttnn.SubDeviceId(0)
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([wsd_id])

    host = torch.zeros(1, 1, tp_factor * T, sp_factor * N * T, dtype=torch.float32)
    for r in range(tp_factor):
        for c in range(sp_factor):
            host[0, 0, r * T : (r + 1) * T, c * N * T : (c + 1) * N * T] = r * 10 + c
    shard_dims = [None, None]
    shard_dims[tp_axis] = 2
    shard_dims[sp_axis] = 3
    rm = ttnn.from_torch(
        host.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    tt_in = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    sample_tile = bcast_offset_tiles
    tt_out = ttnn.experimental.broadcast_ring(
        tt_in,
        sender_ring_index=OWNER,
        cluster_axis=sp_axis,
        topology=topology,
        chunk_size_tiles=chunk_size_tiles,
        broadcast_offset_tiles=bcast_offset_tiles,
        broadcast_num_tiles=bcast_num_tiles,
        use_l1_relay=True,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=[wsd_id])
    out = ttnn.to_torch(
        ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT),
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    grid_out = [
        [round(out[0, 0, r * T, (c * N + sample_tile) * T].item()) for c in range(sp_factor)] for r in range(tp_factor)
    ]
    for r in range(tp_factor):
        logger.info(f"[bcast_ring_l1]   tp{r}: {grid_out[r]}  (per-line expects all {r * 10 + OWNER})")
    per_line = all(grid_out[r][c] == r * 10 + OWNER for r in range(tp_factor) for c in range(sp_factor))
    assert per_line, "L1-relay broadcast_ring did not deliver the sender shard to all ring devices"
