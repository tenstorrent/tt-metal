# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""qr-ring transport for GLM-5.1 at TP=4, on one SP=8 ring (LoudBox = one Galaxy SP ring; TP=4 is 4 copies).

GLM-5.1: 64 attention heads -> TP=4 -> 16 heads/chip. Absorbed-Q width 576 (kv_lora 512 + rope 64), latent-V
width 512, q_lora_rank 2048. The KV latent cache is MQA-shared (one latent head), so KV-gather is
head-count-independent — identical for GLM or DeepSeek.

Measures, over the 8-chip ring (FABRIC_1D_RING):
  main   (KV-gather): all_gather(KVPE [1,1,T,576])                         — O(T)
  branch (Q-gather, classic per-head): all_gather(Q [1,16,S,576]) + O[1,16,S,512]  — flat
  branch (Q-gather, qr-latent):        all_gather(q_lora [1,1,S,2048]) + O[1,16,S,512]  — flat, smaller Q
"""
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import create_global_semaphores

# GLM-5.1 @ TP=4
H_TP4 = 64 // 4  # 16 heads/chip
K_DIM = 512 + 64  # absorbed Q width = kv_lora_rank + qk_rope_head_dim = 576
V_DIM = 512  # latent-V / O width (wkv_b2 -> v_head_dim=256 happens AFTER the op)
Q_LORA = 2048  # GLM q_lora_rank (the compact latent, no head dim)
SQ = 512  # per-chunk query count


def _timed(mesh_device, sp, ca, shape, dim, nl, iters=6, warmup=2):
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])
    sem = create_global_semaphores(mesh_device, sp, crs, 0)
    dims = [None, None]
    dims[ca] = dim
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape))
    torch.manual_seed(0)
    inp = ttnn.from_torch(
        torch.rand(shape).bfloat16(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    def run():
        return ttnn.experimental.all_gather_async(
            inp,
            dim=dim,
            multi_device_global_semaphore=sem,
            num_links=nl,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            cluster_axis=ca,
            subdevice_id=sub_id,
        )

    ts = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        o = run()
        ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])
        dt = (time.perf_counter() - t0) * 1e3
        ttnn.deallocate(o)
        if i >= warmup:
            ts.append(dt)
    ttnn.deallocate(inp)
    ts.sort()
    return ts[len(ts) // 2]


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["sp8"])
def test_qr_glm_tp4(mesh_device):
    sp, ca, nl = 8, 1, 2
    logger.info(f"GLM-5.1 @ TP=4 ({H_TP4} heads/chip) on SP={sp} ring, num_links={nl}")

    # branch: flat, T-independent
    q_head = _timed(mesh_device, sp, ca, [1, H_TP4, SQ, K_DIM], 2, nl)  # classic per-head Q
    q_lat = _timed(mesh_device, sp, ca, [1, 1, SQ, Q_LORA], 2, nl)  # qr-latent q_lora
    o_red = _timed(mesh_device, sp, ca, [1, H_TP4, SQ, V_DIM], 2, nl)  # output reduce
    branch_classic = q_head + o_red
    branch_qr = q_lat + o_red

    # main: KV-gather, O(T), head-independent
    rows = []
    for T in [4096, 16384, 65536, 131072, 262144, 524288]:
        kv = _timed(mesh_device, sp, ca, [1, 1, T, K_DIM], 2, nl)
        rows.append((T, kv))
        logger.info(f"main KV-gather T={T:>7} {kv:8.3f} ms")

    logger.info("=" * 68)
    logger.info(f"GLM-5.1 TP=4, SP=8 ring — branch (flat in T):")
    logger.info(
        f"  classic per-head Q [1,{H_TP4},{SQ},{K_DIM}]  {q_head:6.3f} ms  + O {o_red:6.3f} = {branch_classic:6.3f} ms"
    )
    logger.info(
        f"  qr-latent q_lora   [1,1,{SQ},{Q_LORA}]      {q_lat:6.3f} ms  + O {o_red:6.3f} = {branch_qr:6.3f} ms"
    )
    logger.info("-" * 68)
    logger.info(f"{'T (ctx)':>10} {'main KV':>9} {'branch cls':>11} {'branch qr':>10} {'cls x':>7} {'qr x':>7}")
    for T, kv in rows:
        logger.info(
            f"{T:>10} {kv:>9.3f} {branch_classic:>11.3f} {branch_qr:>10.3f} {kv/branch_classic:>6.2f}x {kv/branch_qr:>6.2f}x"
        )
    logger.info("=" * 68)
    assert rows[-1][1] > rows[0][1]
