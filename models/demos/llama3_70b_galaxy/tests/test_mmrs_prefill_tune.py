# Fused matmul + reduce-scatter (ttnn.experimental.matmul_reduce_scatter_prefill) vs the unfused matmul + reduce-scatter
# the TP32 BH prefill runs today, at the per-chip 4K prefill shapes of Llama-3.3-70B on the 8x4 galaxy:
#   ff13: x [M, 2048] @ w1 or w3 [2048, 3584] (bf4, LoFi), partials summed over cluster_axis 1 (4 chips), scatter N
#   wo:   attn [M, 1024] @ wo [1024, 2048] (bf8, HiFi2), partials summed over cluster_axis 0 (8 chips), scatter M (seq)
#   ff2:  h [M, 3584] @ w2 [3584, 2048] (bf8, HiFi2), partials summed over cluster_axis 0 (8 chips), scatter M (seq)
# A and W are replicated, so the group sum is G * (A @ W); device 0's output is checked against its chunk (PCC).
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_mmrs_prefill_tune.py -s
# env: MMRS_M (4096), MMRS_FABRIC (ring1d | torus2d), MMRS_PAYLOAD (default | big: 14 KiB + 64 router payload),
#      MMRS_LINKS (2)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

N_PER_TRACE, REPLAYS = 10, 5
M = int(os.environ.get("MMRS_M", "4096"))
LINKS = int(os.environ.get("MMRS_LINKS", "2"))


def _device_params():
    fab = {
        "ring1d": ttnn.FabricConfig.FABRIC_1D_RING,
        "torus2d": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    }[os.environ.get("MMRS_FABRIC", "ring1d")]
    p = {"fabric_config": fab, "trace_region_size": 200_000_000}
    if os.environ.get("MMRS_PAYLOAD", "default") == "big":
        rc = ttnn._ttnn.fabric.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = 14 * 1024 + 64
        p["fabric_router_config"] = rc
    return p


def _timed(mesh, fn):
    out = fn()
    ttnn.synchronize_device(mesh)
    first = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [fn() for _ in range(N_PER_TRACE)]
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t0) / (REPLAYS * N_PER_TRACE) * 1e6
    ttnn.release_trace(mesh, tid)
    for o in outs:
        ttnn.deallocate(o)
    return us, first


SHAPES = {
    # name: (K, N, cluster_axis, scatter_dim, weight dtype, fidelity, baseline minimal_matmul block config or None)
    "ff13": (2048, 3584, 1, -1, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, (11, 9, 8, 8, 8, 1, 8)),
    "wo": (1024, 2048, 0, -2, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, None),
    "ff2": (3584, 2048, 0, -2, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, (11, 10, 8, 8, 4, 2, 4)),
}


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("shape", ["ff13", "wo", "ff2"])
def test_mmrs_prefill_tune(mesh_device, device_params, shape):
    mesh = mesh_device
    K, N, axis, sdim, wdt, fid, mm = SHAPES[shape]
    G = tuple(mesh.shape)[axis]
    tag = f"MMRS {shape} M={M} fabric={os.environ.get('MMRS_FABRIC', 'ring1d')} payload={os.environ.get('MMRS_PAYLOAD', 'default')} links={LINKS}"
    # baseline as the model runs it (fp32 DEST off: the tuned minimal_matmul subblocks need the full DEST)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    # the fused op takes a ComputeConfigDescriptor (fp32 DEST and full sync always on)
    fused_ckc = ttnn.ComputeConfigDescriptor(math_fidelity=fid)
    torch.manual_seed(0)
    ta = torch.randn(1, 1, M, K)
    tw = torch.randn(K, N) / K**0.5
    rep = ttnn.ReplicateTensorToMesh(mesh)
    a = ttnn.from_torch(
        ta,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    w = ttnn.from_torch(
        tw.reshape(1, 1, K, N),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=wdt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    w2d = ttnn.reshape(w, (K, N))
    golden = G * (ta[0, 0] @ tw)
    # device 0 = mesh (0, 0): chunk 0 of the scattered dim
    ref = torch.chunk(golden, G, dim=sdim)[0]

    def check(name, got):
        got = got.reshape(ref.shape) if got.numel() == ref.numel() else got
        if tuple(got.shape[-2:]) != tuple(ref.shape):
            logger.warning(f"{tag} {name}: shape {tuple(got.shape)} vs ref {tuple(ref.shape)}")
            return
        _, pcc = comp_pcc(ref, got[..., : ref.shape[0], : ref.shape[1]].reshape(ref.shape), 0.99)
        logger.info(f"{tag} {name} pcc={pcc}")

    def baseline():
        if mm is None:
            o = ttnn.linear(
                a, w, compute_kernel_config=ckc, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        else:
            gx, gy, mb, kb, nb, sh, sw = mm
            o = ttnn.experimental.minimal_matmul(
                input_tensor=a,
                weight_tensor=w,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=mb,
                    K_block_size=kb,
                    N_block_size=nb,
                    subblock_h=sh,
                    subblock_w=sw,
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                ),
                compute_kernel_config=ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        r = ttnn.reduce_scatter(
            o,
            4 + sdim,
            cluster_axis=axis,
            num_links=LINKS,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(o)
        return r

    try:
        us, got = _timed(mesh, baseline)
        logger.info(f"{tag} baseline matmul+reduce_scatter: {us:.1f} us")
        check("baseline", got)
    except Exception as e:
        logger.warning(f"{tag} baseline ERROR {str(e)[:300]}")

    for wname, wt in (("w4d", w), ("w2d", w2d)):

        def fused():
            return ttnn.experimental.matmul_reduce_scatter_prefill(
                a,
                wt,
                cluster_axis=axis,
                scatter_dim=sdim,
                topology=ttnn.Topology.Ring,
                num_links=LINKS,
                compute_kernel_config=fused_ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        try:
            us, got = _timed(mesh, fused)
            logger.info(f"{tag} fused matmul_reduce_scatter_prefill ({wname}): {us:.1f} us")
            check(f"fused-{wname}", got)
            break
        except Exception as e:
            logger.warning(f"{tag} fused ({wname}) ERROR {str(e)[:400]}")
