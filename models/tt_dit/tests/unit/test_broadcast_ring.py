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
from models.tt_dit.utils.test import ring_params, ring_params_8k

try:  # tracy signpost is only present under `python -m tracy`; no-op otherwise.
    from tracy import signpost
except ImportError:

    def signpost(_name):
        pass


T = ttnn.TILE_SIZE
OWNER = 5  # sender index along the ring (cluster) axis


# The tests above fill each shard with a single CONSTANT (r*10+c), so they cannot see an intra-shard
# tile shuffle or drop -- every tile reads the same value. This test fills each tile DISTINCTLY
# (per-element random) and compares the full broadcast output against the sender shard tile-for-tile,
# so a multi-tile / multi-chunk data-movement bug (which 1-tile misses) is caught. Parametrized over
# DRAM vs L1 relay and chunk sizes that force many chunks.
@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "device_params", "topology"),
    [
        pytest.param((4, 8), 1, 0, ring_params, ttnn.Topology.Ring, id="bh_4x8_ring"),
        pytest.param((4, 32), 1, 0, ring_params_8k, ttnn.Topology.Ring, id="bh_4x32_ring"),
    ],
    indirect=["mesh_device", "device_params"],
)
# sender=-1 means "last ring index" (sp_factor-1) -- the wrap edge, which the production reference-frame
# owner actually hits (the reference is the last frame). chunk64 matches the production chunk size.
@pytest.mark.parametrize("sender", [pytest.param(5, id="owner5"), pytest.param(-1, id="owner_last")])
@pytest.mark.parametrize(
    ("tiles_per_shard", "chunk_size_tiles", "use_l1_relay"),
    [
        pytest.param(128, 0, False, id="128tiles_chunkauto_dram"),
        pytest.param(128, 64, False, id="128tiles_chunk64_dram"),
        pytest.param(128, 64, True, id="128tiles_chunk64_l1"),
        pytest.param(128, 32, True, id="128tiles_chunk32_l1"),
    ],
)
def test_broadcast_ring_wholeshard(
    mesh_device, sp_axis, tp_axis, device_params, topology, tiles_per_shard, chunk_size_tiles, use_l1_relay, sender
):
    tp_factor, sp_factor = tuple(mesh_device.shape)
    N = tiles_per_shard
    owner = sender if sender >= 0 else sp_factor - 1

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    wsd_id = ttnn.SubDeviceId(0)
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([wsd_id])

    # Per-element-distinct data (bf16-roundtripped so the golden compare is bit-exact).
    torch.manual_seed(0)
    host = torch.randn(1, 1, tp_factor * T, sp_factor * N * T, dtype=torch.float32).to(torch.bfloat16).float()
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

    tt_out = ttnn.experimental.broadcast_ring(
        tt_in,
        sender_ring_index=owner,
        cluster_axis=sp_axis,
        topology=topology,
        subdevice_id=wsd_id,
        num_links=2,
        chunk_size_tiles=chunk_size_tiles,
        use_l1_relay=use_l1_relay,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=[wsd_id])
    out = ttnn.to_torch(
        ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT),
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    ).float()

    # Every ring device must hold the sender's shard, tile-for-tile, for its own tp row.
    bad = []
    for r in range(tp_factor):
        golden = host[0, 0, r * T : (r + 1) * T, owner * N * T : (owner + 1) * N * T]
        for c in range(sp_factor):
            block = out[0, 0, r * T : (r + 1) * T, c * N * T : (c + 1) * N * T]
            if not torch.equal(block, golden):
                first_bad = next(
                    (
                        st
                        for st in range(N)
                        if not torch.equal(block[:, st * T : (st + 1) * T], golden[:, st * T : (st + 1) * T])
                    ),
                    -1,
                )
                bad.append((r, c, first_bad))
    if bad:
        logger.info(f"[wholeshard] {len(bad)} device-shards wrong; first few (tp,ring,first_bad_tile): {bad[:8]}")
    assert not bad, f"broadcast_ring whole-shard mismatch on {len(bad)} device-shards, first: {bad[0]}"


# Exact production kv layout: [2b, nh_local, seq, head_dim] -- heads(dim1)->tp, seq(dim2)->sp, head_dim
# on dim3. SR transformer is dim=5120/num_heads=40 -> head_dim=128 (4 E-tiles), 10 heads/device at tp=4.
# This is the shape sparse_attention actually feeds; the tests above use different dims/layout.
@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "device_params", "topology"),
    [
        pytest.param((4, 8), 1, 0, ring_params, ttnn.Topology.Ring, id="bh_4x8_ring"),
        pytest.param((4, 32), 1, 0, ring_params_8k, ttnn.Topology.Ring, id="bh_4x32_ring"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("sender", [pytest.param(5, id="owner5"), pytest.param(-1, id="owner_last")])
@pytest.mark.parametrize("use_l1_relay", [pytest.param(False, id="dram"), pytest.param(True, id="l1")])
def test_broadcast_ring_production_layout(mesh_device, sp_axis, tp_axis, device_params, topology, sender, use_l1_relay):
    tp_factor, sp_factor = tuple(mesh_device.shape)
    n_batch = 2  # k|v stacked on the unsharded batch dim, as sparse_attention does
    nh_local = 10  # 40 heads / tp=4
    head_dim_tiles = 4  # head_dim 128 / 32
    per_dev = 8  # seq tiles per device -> 2*10*8*4 = 640 pages/device, chunk64 -> 10 chunks
    chunk_size_tiles = 64
    owner = sender if sender >= 0 else sp_factor - 1

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    wsd_id = ttnn.SubDeviceId(0)
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([wsd_id])

    n_heads = nh_local * tp_factor
    n_seq = per_dev * sp_factor
    torch.manual_seed(0)
    host = torch.randn(n_batch, n_heads, n_seq * T, head_dim_tiles * T, dtype=torch.float32).to(torch.bfloat16).float()
    shard_dims = [None, None]
    shard_dims[tp_axis] = 1  # heads
    shard_dims[sp_axis] = 2  # seq
    rm = ttnn.from_torch(
        host.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    tt_in = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    tt_out = ttnn.experimental.broadcast_ring(
        tt_in,
        sender_ring_index=owner,
        cluster_axis=sp_axis,
        topology=topology,
        subdevice_id=wsd_id,
        num_links=2,
        chunk_size_tiles=chunk_size_tiles,
        use_l1_relay=use_l1_relay,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=[wsd_id])
    out = ttnn.to_torch(
        ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT),
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    ).float()

    # Every sp device must hold the owner's seq shard, tile-for-tile, for its own heads and batch.
    bad = []
    for b in range(n_batch):
        for h in range(n_heads):
            golden = host[b, h, owner * per_dev * T : (owner + 1) * per_dev * T, :]
            for c in range(sp_factor):
                block = out[b, h, c * per_dev * T : (c + 1) * per_dev * T, :]
                if not torch.equal(block, golden):
                    first_bad = next(
                        (
                            st
                            for st in range(per_dev)
                            if not torch.equal(block[st * T : (st + 1) * T, :], golden[st * T : (st + 1) * T, :])
                        ),
                        -1,
                    )
                    bad.append((b, h, c, first_bad))
    if bad:
        logger.info(f"[prod_layout] {len(bad)} shards wrong; first few (batch,head,ring,first_bad_tile): {bad[:8]}")
    assert not bad, f"broadcast_ring production-layout mismatch on {len(bad)} shards, first: {bad[0]}"


# The production path passes caller-owned ping-pong global semaphores (get_broadcast_ping_pong_semaphore)
# instead of the op creating its own. This exercises that path over SEVERAL calls: the pool has 2 sets, so
# calls 0/2/4 reuse set 0 -- and the op must zero its recv/credit sems in-kernel each call, else set 0's
# accumulated count from call 0 satisfies call 2's wait_min early and it reads stale data. Verifying every
# call catches a missing reset (and the per-call override_runtime_arguments that rotates the semaphore).
@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "device_params", "topology"),
    [
        pytest.param((4, 8), 1, 0, ring_params, ttnn.Topology.Ring, id="bh_4x8_ring"),
        pytest.param((4, 32), 1, 0, ring_params_8k, ttnn.Topology.Ring, id="bh_4x32_ring"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("use_l1_relay", [pytest.param(False, id="dram"), pytest.param(True, id="l1")])
def test_broadcast_ring_ccl_semaphore(mesh_device, sp_axis, tp_axis, device_params, topology, use_l1_relay):
    from models.tt_dit.parallel.manager import CCLManager

    tp_factor, sp_factor = tuple(mesh_device.shape)
    N = 128
    chunk = 64
    owner = 5

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    wsd_id = ttnn.SubDeviceId(0)
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([wsd_id])

    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=topology)

    torch.manual_seed(0)
    host = torch.randn(1, 1, tp_factor * T, sp_factor * N * T, dtype=torch.float32).to(torch.bfloat16).float()
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

    # Several calls so the ping-pong reuses set 0 (calls 0/2/4) -- a missing in-kernel reset fails call 2.
    for it in range(5):
        tt_out = ttnn.experimental.broadcast_ring(
            tt_in,
            sender_ring_index=owner,
            cluster_axis=sp_axis,
            topology=topology,
            subdevice_id=wsd_id,
            num_links=2,
            chunk_size_tiles=chunk,
            use_l1_relay=use_l1_relay,
            multi_device_global_semaphore=ccl.get_broadcast_ping_pong_semaphore(sp_axis),
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[wsd_id])
        out = ttnn.to_torch(
            ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT),
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        ).float()
        bad = []
        for r in range(tp_factor):
            golden = host[0, 0, r * T : (r + 1) * T, owner * N * T : (owner + 1) * N * T]
            for c in range(sp_factor):
                block = out[0, 0, r * T : (r + 1) * T, c * N * T : (c + 1) * N * T]
                if not torch.equal(block, golden):
                    bad.append((it, r, c))
        assert (
            not bad
        ), f"iter {it}: broadcast_ring (ccl semaphore) mismatch on {len(bad)} device-shards, first: {bad[0]}"


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "device_params", "topology"),
    [
        pytest.param((4, 8), 1, 0, ring_params, ttnn.Topology.Ring, id="bh_4x8_ring"),
        pytest.param((4, 32), 1, 0, ring_params_8k, ttnn.Topology.Ring, id="bh_4x32_ring"),
    ],
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
    [
        pytest.param((4, 8), 1, 0, ring_params, ttnn.Topology.Ring, id="bh_4x8_ring"),
        pytest.param((4, 32), 1, 0, ring_params_8k, ttnn.Topology.Ring, id="bh_4x32_ring"),
    ],
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


# --- reference-frame layout: the sparse_attention usage the op is actually fed --------------------
# sparse_attention broadcasts kv=[2b, nh, N, E] sharded heads(dim1)->tp, seq(dim2)->sp, then slices a
# seq sub-range (the reference frame) out on dim 2. That differs from the tests above in three ways at
# once: broadcast dim is 2 (not the innermost), dim0 (2b) and dim3 (E) are > 1 tile, and sp can be 32.
#
# broadcast_offset_tiles/broadcast_num_tiles are a FLAT contiguous page range over the whole shard
# (row-major, width tiles innermost), so a CONTIGUOUS dim-2 sub-range can only be expressed when E/32==1
# and dim0*dim1==1. The "subrange" case below therefore fails (xfail: contiguous-range limitation). The
# "blocked" case uses broadcast_stride_pages/broadcast_num_blocks -- one block of frame pages per
# (dim0, head) with stride = shard rows -- which expresses the dim-2 sub-range exactly (L1 relay only).
# "outremap" adds broadcast_out_offset/stride_pages on top: the blocks are persisted compactly into a
# frame-shaped persistent buffer instead of at their input page ids -- this blocked+remapped form is what
# the production reference-frame assembly uses (no post-broadcast slice/concat). "wholeshard" broadcasts
# the whole owner shard and slices dim 2 on host -- correct, but moves the whole shard.
@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "device_params", "topology"),
    [
        pytest.param((4, 8), 1, 0, ring_params, ttnn.Topology.Ring, id="bh_4x8_ring"),
        pytest.param((4, 32), 1, 0, ring_params_8k, ttnn.Topology.Ring, id="bh_4x32_ring"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("wholeshard", id="wholeshard"),
        pytest.param("blocked", id="blocked"),
        pytest.param("outremap", id="outremap"),
        pytest.param(
            "subrange",
            id="subrange",
            marks=pytest.mark.xfail(
                reason="broadcast_offset/num_tiles is a flat page range; a dim-2 sub-range across "
                "dim0/E outer tiles is disjoint pages it cannot express (use the blocked range)",
                strict=False,
            ),
        ),
    ],
)
def test_broadcast_ring_reference_frame(mesh_device, sp_axis, tp_axis, device_params, topology, mode):
    tp_factor, sp_factor = tuple(mesh_device.shape)
    per_dev = 4  # seq tiles per device shard
    e_tiles = 2  # dim3 (E) = 2 tiles, so E/32 > 1 (exposes the flat-page stride)
    n_batch = 2  # dim0 (stacked k|v), so dim0 > 1
    owner = sp_factor // 2
    off, flen = 1, 2  # frame = owner seq tiles [off, off+flen), lives inside one shard

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    wsd_id = ttnn.SubDeviceId(0)
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([wsd_id])

    # value(b, head, global_seq_tile, e_tile) is unique so a mis-addressed page is detectable.
    def val(b, head, seqt, et):
        return b * 100000 + head * 1000 + seqt * 10 + et

    n_seq = sp_factor * per_dev
    host = torch.zeros(n_batch, tp_factor, n_seq * T, e_tiles * T, dtype=torch.float32)
    for b in range(n_batch):
        for head in range(tp_factor):
            for seqt in range(n_seq):
                for et in range(e_tiles):
                    host[b, head, seqt * T : (seqt + 1) * T, et * T : (et + 1) * T] = val(b, head, seqt, et)
    # val() exceeds bf16's exact-integer range (>256), so compare against the bf16-rounded golden, not
    # the raw ints -- otherwise pure bf16 quantization (e.g. 1170 -> 1168) reads as a false mismatch.
    host_bf16 = host.to(torch.bfloat16).float()

    shard_dims = [None, None]
    shard_dims[tp_axis] = 1  # heads
    shard_dims[sp_axis] = 2  # seq
    rm = ttnn.from_torch(
        host.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    tt_in = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    bcast_kwargs = dict(
        sender_ring_index=owner, cluster_axis=sp_axis, topology=topology, subdevice_id=wsd_id, num_links=2
    )
    if mode == "subrange":
        bcast_kwargs.update(broadcast_offset_tiles=off, broadcast_num_tiles=flen)
    elif mode == "blocked":
        # Per-device shard is [n_batch, 1 head, per_dev*T, e_tiles*T]: the frame's seq-tile range
        # [off, off+flen) is n_batch*1 blocks of flen*e_tiles pages, stride = per_dev*e_tiles pages.
        bcast_kwargs.update(
            broadcast_offset_tiles=off * e_tiles,
            broadcast_num_tiles=flen * e_tiles,
            broadcast_stride_pages=per_dev * e_tiles,
            broadcast_num_blocks=n_batch,
            use_l1_relay=True,
        )
    elif mode == "outremap":
        # Blocked input range as above PLUS output remap: block b is persisted at out page b*flen*e_tiles
        # of a COMPACT [n_batch, 1, flen*T, e_tiles*T] buffer (frame rows only, no untouched pages) --
        # the production reference-frame assembly writes reference_kv this way, with no post-op slicing.
        out_buf = ttnn.allocate_tensor_on_device(
            ttnn.Shape([n_batch, 1, flen * T, e_tiles * T]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        bcast_kwargs.update(
            broadcast_offset_tiles=off * e_tiles,
            broadcast_num_tiles=flen * e_tiles,
            broadcast_stride_pages=per_dev * e_tiles,
            broadcast_num_blocks=n_batch,
            broadcast_out_offset_pages=0,
            broadcast_out_stride_pages=flen * e_tiles,
            persistent_output_buffer=out_buf,
            use_l1_relay=True,
        )
    full = ttnn.experimental.broadcast_ring(tt_in, **bcast_kwargs)
    ttnn.synchronize_device(mesh_device, sub_device_ids=[wsd_id])

    out = ttnn.to_torch(
        ttnn.to_layout(full, ttnn.ROW_MAJOR_LAYOUT),
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    # Every sp device must hold the OWNER's frame tiles for its own tp head, across all batch/E tiles.
    # outremap persists compactly (frame rows only), so its per-device row j is just j; the other modes
    # persist at the input page ids, so the frame sits at rows [off, off+flen) of each shard.
    bad = []
    for b in range(n_batch):
        for head in range(tp_factor):
            for c in range(sp_factor):
                for j in range(flen):
                    for et in range(e_tiles):
                        out_row = (c * flen + j) if mode == "outremap" else (c * per_dev + off + j)
                        got = out[b, head, out_row * T, et * T].item()
                        want = host_bf16[b, head, (owner * per_dev + off + j) * T, et * T].item()
                        if got != want:
                            bad.append((b, head, c, off + j, et, got, want))
    assert not bad, f"broadcast_ring reference-frame mismatch ({len(bad)} tiles), first: {bad[0]}"
