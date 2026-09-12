# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH roofline microbenchmark for the fused all_gather_minimal_matmul_async (AGMM) in-projection at the
Qwen3.6-27B TP=4 prefill shape, production layout (QWEN36_AGMM_LAYOUT=nt11x8: 11x8 grid, untransposed, 2 links x 4 workers,
M_block 4, K_block 8, N_block 12, subblock 1x4, HiFi2, fp32 DEST, packer_l1_acc).

Question: what bounds the kernel at ~34 cycles per tile-matmul? Candidates: FPU passes (HiFi2 = 32), operand unpack bytes
(1 bf16 in0 + 4 bf8 in1 tiles per 4 MACs), fp32 DEST spill/reload traffic per K block (interm fp32 pack + packer_l1_acc), or
the DEST sync. Each variant changes exactly one of these. Wall-clock over iterations on the (1,4) mesh + PCC vs torch.

  MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_agmm_roofline_scratch.py -s
  AGMM_RL_SHAPE=gdn_in|attn_in|mlp_gateup (default gdn_in), AGMM_RL_ITERS (default 20)
"""
import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.tt_transformers.tt.ccl import TT_CCL

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2, "fabric_config": ttnn.FabricConfig.FABRIC_1D}]
S, K_TOTAL = 2048, 5120
N_LOCALS = {"gdn_in": 4120, "attn_in": 3584, "mlp_gateup": 8704}


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _ckc(fidelity, fp32_dest, full_sync=False):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_dest,
        packer_l1_acc=True,
        dst_full_sync_en=full_sync,
    )


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_agmm_roofline(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    tt_ccl = TT_CCL(mesh)
    topo = ttnn.Topology.Ring
    name = os.environ.get("AGMM_RL_SHAPE", "gdn_in")
    iters = int(os.environ.get("AGMM_RL_ITERS", "20"))
    n_local = N_LOCALS[name]
    torch.manual_seed(0)
    x = torch.randn(1, 1, S, K_TOTAL, dtype=torch.bfloat16)
    w = torch.randn(K_TOTAL, n_local * 4, dtype=torch.bfloat16) * 0.02

    x_sh = {}
    for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
        x_sh[dt] = ttnn.from_torch(
            x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3)
        )  # K-sharded [1,1,S,1280]
    w_sh = {}
    for dt in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        w_sh[dt] = ttnn.from_torch(
            w, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1)
        )  # [5120, n_local] per device
    # torch reference uses the same quantised weights as the bf8 device path (device 0 shard)
    refs = {}
    for dt in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        w_dev0 = ttnn.to_torch(ttnn.get_device_tensors(w_sh[dt])[0]).float()
        refs[dt] = x.float().reshape(S, K_TOTAL) @ w_dev0

    n_tiles = (n_local + 31) // 32
    grid = (11, 8)
    per_core = max(1, -(-n_tiles // grid[0]))
    n_block_default = ((per_core + 3) // 4) * 4  # 12 for gdn_in / attn_in

    def run(
        label,
        fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest=True,
        sub=(1, 4),
        in0=ttnn.bfloat16,
        w_dt=ttnn.bfloat8_b,
        m_block=4,
        n_block=None,
        k_block=None,
        full_sync=False,
        out_dtype=ttnn.bfloat16,
        links=2,
        nbuf=8,
        fuse_swiglu=False,
    ):
        n_block = n_block or n_block_default
        k_block = k_block or tpc.agmm_k_block_size(K_TOTAL // 4)
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=m_block,
            K_block_size=k_block,
            N_block_size=n_block,
            subblock_h=sub[0],
            subblock_w=sub[1],
            compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
        )
        ckc = _ckc(fidelity, fp32_dest, full_sync)

        def f():
            return ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x_sh[in0],
                weight_tensor=w_sh[w_dt],
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
                config=cfg,
                compute_kernel_config=ckc,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(1),
                num_links=links,
                topology=topo,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=out_dtype,
                force_transpose=False,
                num_workers_per_link=grid[1] // links,
                num_buffers_per_channel=nbuf,
                fuse_swiglu=fuse_swiglu,
            )[0]

        ref0 = refs[w_dt]
        signpost(f"AGMM_RL {label}")
        try:
            out = f()
            ttnn.synchronize_device(mesh)
            got0 = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(S, -1)[:, :n_local]
            pcc = _pcc(got0, ref0[:, : got0.shape[1]])
            if os.environ.get("AGMM_RL_ERRMAP", "1") == "1":
                nt = (got0.shape[1] // 32) * 32
                nt = min(nt, got0.shape[1])
                e = (got0[:, :nt] - ref0[:, :nt]).abs().reshape(S // 32, 32, nt // 32, 32).amax(dim=(1, 3))
                scale = ref0.abs().mean().item()
                bad = e > 0.05 * scale
                rows = bad.any(dim=1).nonzero().flatten().tolist()
                cols = bad.any(dim=0).nonzero().flatten().tolist()
                logger.info(
                    f"AGMM_RL_ERR {name:10s} {label:28s} bad_tiles={int(bad.sum())}/{bad.numel()} "
                    f"max_rel={e.max().item() / scale:.3f} rows={rows[:12]}{'...' if len(rows) > 12 else ''} "
                    f"cols={cols[:16]}{'...' if len(cols) > 16 else ''}"
                )
            ttnn.deallocate(out)
            for _ in range(2):
                ttnn.deallocate(f())
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            outs = [f() for _ in range(iters)]
            ttnn.synchronize_device(mesh)
            us = (time.time() - t0) / iters * 1e6
            for o in outs:
                ttnn.deallocate(o)
            macs = (S // 32) * (K_TOTAL // 32) * n_tiles
            cyc = us * 1e-6 * 1.35e9 * 88 / macs
            logger.info(f"AGMM_RL {name:10s} {label:28s} {us:8.1f} us  {cyc:5.1f} cyc/tile-mm/core  PCC={pcc:.6f}")
        except Exception as e:  # noqa: BLE001
            logger.info(f"AGMM_RL {name:10s} {label:28s} ERR {str(e).splitlines()[0][:140]}")

    x_full = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )

    def run_plain(label, fn):
        signpost(f"AGMM_RL {label}")
        try:
            ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            outs = [fn() for _ in range(iters)]
            ttnn.synchronize_device(mesh)
            us = (time.time() - t0) / iters * 1e6
            for o in outs:
                ttnn.deallocate(o)
            logger.info(f"AGMM_RL {name:10s} {label:28s} {us:8.1f} us (wall)")
        except Exception as e:  # noqa: BLE001
            logger.info(f"AGMM_RL {name:10s} {label:28s} ERR {str(e).splitlines()[0][:140]}")

    def plain_mm(ckc, cfg):
        return lambda: ttnn.experimental.minimal_matmul(
            x_full,
            w_sh[ttnn.bfloat8_b],
            config=cfg,
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

    def plain_ag(links):
        return lambda: ttnn.experimental.all_gather_async(
            x_sh[ttnn.bfloat16],
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(1),
            num_links=links,
            topology=topo,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    sel = os.environ.get("AGMM_RL_SET", "fid,dest,io,links,plain").split(",")
    if "fid" in sel:
        run("prod hifi2 fp32d 1x4")
        run("lofi fp32d 1x4", fidelity=ttnn.MathFidelity.LoFi)
        run("hifi4 fp32d 1x4", fidelity=ttnn.MathFidelity.HiFi4)
    if "dest" in sel:
        run("hifi2 bf16d 1x4", fp32_dest=False)
        run("hifi2 bf16d 2x4", fp32_dest=False, sub=(2, 4))
        run("hifi2 bf16d 4x2", fp32_dest=False, sub=(4, 2))
        run("lofi bf16d 2x4", fidelity=ttnn.MathFidelity.LoFi, fp32_dest=False, sub=(2, 4))
    if "io2" in sel:
        run("lofi bf16d 4x2", fidelity=ttnn.MathFidelity.LoFi, fp32_dest=False, sub=(4, 2))
        run("lofi bf16d 4x2 w=bf4", fidelity=ttnn.MathFidelity.LoFi, fp32_dest=False, sub=(4, 2), w_dt=ttnn.bfloat4_b)
        run("lofi bf16d 2x4 w=bf4", fidelity=ttnn.MathFidelity.LoFi, fp32_dest=False, sub=(2, 4), w_dt=ttnn.bfloat4_b)
        run("hifi2 bf16d 4x2 w=bf4", fp32_dest=False, sub=(4, 2), w_dt=ttnn.bfloat4_b)
        run("lofi fp32d 1x4 w=bf4", fidelity=ttnn.MathFidelity.LoFi, w_dt=ttnn.bfloat4_b)
        run("lofi bf16d 1x4", fidelity=ttnn.MathFidelity.LoFi, fp32_dest=False)
        run("lofi bf16d 1x8 Nb8", fidelity=ttnn.MathFidelity.LoFi, fp32_dest=False, sub=(1, 8), n_block=8)
        run("hifi2 fp32d 1x4 fullsync 2x4", sub=(2, 4), full_sync=True)
    if "io" in sel:
        run("prod out=bf8", out_dtype=ttnn.bfloat8_b)
        run("prod w=bf4", w_dt=ttnn.bfloat4_b)
        run("prod fuse_swiglu", fuse_swiglu=True)
        run(
            "lofi bf16d 4x2 out=bf8 w=bf4",
            fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest=False,
            sub=(4, 2),
            out_dtype=ttnn.bfloat8_b,
            w_dt=ttnn.bfloat4_b,
        )
    if "links" in sel:
        run("prod links=1 workers=8", links=1)
        run("prod nbuf=4", nbuf=4)
        run("prod nbuf=16", nbuf=16)
    if "plain" in sel:
        cfg0 = ttnn.MinimalMatmulConfig(
            M_block_size=4,
            K_block_size=tpc.agmm_k_block_size(K_TOTAL // 4),
            N_block_size=n_block_default,
            subblock_h=1,
            subblock_w=4,
            compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
        )
        run_plain("minimal_matmul pre-gathered", plain_mm(_ckc(ttnn.MathFidelity.HiFi2, True), cfg0))
        run_plain("minimal_matmul lofi bf16d", plain_mm(_ckc(ttnn.MathFidelity.LoFi, False), cfg0))
        run_plain("all_gather_async 2 links", plain_ag(2))
        run_plain("all_gather_async 1 link", plain_ag(1))
    run("prod again hifi2 fp32d 1x4")
    print("AGMM_RL_DONE")
