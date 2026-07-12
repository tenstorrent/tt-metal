# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""qr-ring sparse-MLA — MEASURED transport perf diff (LoudBox, Blackhole).

The plan's central claim: sparse-MLA prefill today all-gathers the KV latent
cache across the SP ring EVERY layer — bytes grow O(T). qr-ring keeps KV
stationary and gathers the QUERIES (fixed size) + reduce-scatters the output —
bytes FLAT in T. This measures both transports on a real SP ring so the flat-vs-
O(T) asymmetry is a number, not a projection.

  KV-gather (today):  all_gather( KVPE [1,1,T,K_DIM] ) over SP   -> grows with T
  Q-gather (qr MVP):  all_gather( Q    [1,H,Sq,K_DIM] ) over SP   -> flat in T
                    + reduce component: O [1,H,Sq,V_DIM]          -> flat in T

Run:  scripts/run_safe_pytest.sh --run-all tests/nightly/blackhole/sdpa/test_qr_ring_transport_perf.py

This is a TRANSPORT microbenchmark (wall-clock of the collective on the SP ring),
not the fused op. It isolates the one thing the plan is really trading: bytes on
the ethernet ring per layer. Compute (sparse attention over the fixed top-k=2048)
is flat in T by construction and measured separately by the op tests.
"""
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import create_global_semaphores

K_DIM = 576  # latent(512) + rope(64) — the KVPE / absorbed-Q width
V_DIM = 512  # latent V width (the reduce-scatter output width)
H = 128  # DeepSeek head count (absorbed Q is per-head)
SQ = 512  # per-chunk query count (fixed — this is why Q-gather is flat in T)


def _timed_all_gather(mesh_device, num_devices, shape, dim, cluster_axis, dtype, num_iters=6, warmup=2):
    """Median wall-clock (ms) of all_gather_async of `shape` (sharded on `dim`) across `cluster_axis`.
    Native (2,4) mesh: shard the seq dim along the SP cluster axis, replicate along the other axis."""
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])
    sem = create_global_semaphores(mesh_device, num_devices, crs, 0)

    mesh_shape = tuple(mesh_device.shape)
    # 2D mapper: shard `dim` on the cluster (SP) axis, replicate (None) on the other axis.
    dims = [None, None]
    dims[cluster_axis] = dim
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=mesh_shape)

    torch.manual_seed(0)
    full = torch.rand(shape).bfloat16()
    inp = ttnn.from_torch(
        full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    def run():
        return ttnn.experimental.all_gather_async(
            inp,
            dim=dim,
            multi_device_global_semaphore=sem,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=cluster_axis,
            subdevice_id=sub_id,
        )

    times = []
    for i in range(warmup + num_iters):
        t0 = time.perf_counter()
        out = run()
        ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])
        dt = (time.perf_counter() - t0) * 1e3
        ttnn.deallocate(out)
        if i >= warmup:
            times.append(dt)
    ttnn.deallocate(inp)
    times.sort()
    return times[len(times) // 2]  # median ms


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True, ids=["fabric"]
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True, ids=["lb2x4"])
def test_qr_ring_transport_diff(mesh_device):
    """Measure KV-gather (O(T)) vs Q-gather (flat) transport across the SP=4 ring on the LoudBox (native 2x4)."""
    sp = tuple(mesh_device.shape)[1]  # cluster axis 1 (the size-4 row) = SP ring
    cluster_axis = 1
    logger.info(f"SP ring size = {sp}")

    # Context-length sweep. KVPE is sharded on seq (dim 2) so each chip holds T/sp; the gather rebuilds full T.
    # T must be tile-aligned (32) and divisible by sp.
    Ts = [4096, 16384, 65536, 131072]

    kv_rows, q_ms, o_ms = [], None, None

    # Q-gather transport is INDEPENDENT of T: measure once. Q [1,H,SQ,K_DIM] gathered over SP (dim 2 = Sq).
    q_ms = _timed_all_gather(mesh_device, sp, [1, H, SQ, K_DIM], dim=2, cluster_axis=cluster_axis, dtype=ttnn.bfloat16)
    o_ms = _timed_all_gather(mesh_device, sp, [1, H, SQ, V_DIM], dim=2, cluster_axis=cluster_axis, dtype=ttnn.bfloat16)

    for T in Ts:
        ms = _timed_all_gather(mesh_device, sp, [1, 1, T, K_DIM], dim=2, cluster_axis=cluster_axis, dtype=ttnn.bfloat16)
        kv_rows.append((T, ms))
        logger.info(f"KV-gather  T={T:>7}  {ms:8.3f} ms")

    logger.info("")
    logger.info("================ qr-ring transport perf diff (SP=%d, measured) ================" % sp)
    logger.info(f"Q-gather (all_gather Q  [1,{H},{SQ},{K_DIM}]) : {q_ms:8.3f} ms   [FLAT in T]")
    logger.info(f"Q-gather (reduce O      [1,{H},{SQ},{V_DIM}]) : {o_ms:8.3f} ms   [FLAT in T]")
    qr_total = q_ms + o_ms
    logger.info(f"Q-gather TOTAL transport                      : {qr_total:8.3f} ms   [FLAT in T]")
    logger.info("-" * 78)
    logger.info(f"{'T (ctx)':>10} {'KV-gather ms':>14} {'qr ms':>10} {'speedup':>10}")
    for T, ms in kv_rows:
        logger.info(f"{T:>10} {ms:>14.3f} {qr_total:>10.3f} {ms / qr_total:>9.2f}x")
    logger.info("=" * 78)

    # Sanity: KV-gather transport must GROW with T (O(T)); qr stays flat. Assert the trend so the test
    # fails loudly if the ring degenerates rather than silently reporting a flat KV curve.
    assert kv_rows[-1][1] > kv_rows[0][1], "KV-gather transport should grow with context (O(T))"
