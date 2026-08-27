# SPDX-License-Identifier: Apache-2.0
"""LN-epilogue de-risk probe (matmul -> residual -> LayerNorm boundary, N=1024).

Route A (LN fusion) target: remove the matmul-output DRAM round-trip at the
post-attention (AttnOut, K=1024) and post-MLP (Wo, K=4096) boundaries. The
residual add is ALREADY fused into ttnn.layer_norm(residual_input_tensor=...),
so the boundary today is 2 programs: matmul(->DRAM) then layer_norm(reads DRAM).

This probe tests the CHEAP form first (no new kernel): keep the matmul output in
L1 and let LN consume it from L1. Output is only 1024-wide (32 tiles) vs the
4096-wide FFN intermediate that OOM'd, so it may fit. Gate = PCC preserved AND
wall reduced AND fits L1 (clean TT_THROW on overflow = board-safe).

Variants per boundary:
  baseline : matmul(mem=DRAM,bf8) -> layer_norm(in=DRAM, resid=DRAM) -> DRAM
  l1_act   : matmul(mem=L1 ,bf8) -> layer_norm(in=L1 , resid=DRAM) -> DRAM
  l1_both  : matmul(mem=L1 ,bf8) -> layer_norm(in=L1 , resid=L1 ) -> DRAM

Run:
  source /localdev/gtobar/bge_optimization/local_env.sh ; export TT_VISIBLE_DEVICES=0
  pytest models/demos/wormhole/bge_m3/tests/perf/ln_epilogue_probe.py -s -q
"""
import time

import pytest
import torch
from loguru import logger

import ttnn

M = 49152
DIM = 1024
N_ITERS = 20


def _mm_cfg(M_block, K_block, N_block, sbh=4, sbw=2):
    return ttnn.MinimalMatmulConfig(
        M_block_size=M_block, K_block_size=K_block, N_block_size=N_block,
        subblock_h=sbh, subblock_w=sbw,
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    )


BOUNDARIES = {
    # name : (K, mm_config)
    "attnout_k1024": (1024, _mm_cfg(16, 8, 4)),
    "wo_k4096": (4096, _mm_cfg(8, 32, 4)),
}


@pytest.mark.parametrize("device_params", [{"trace_region_size": 90_000_000, "num_command_queues": 1}], indirect=True)
def test_ln_epilogue(mesh_device):
    dev = mesh_device
    torch.manual_seed(0)
    ckc_mm = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    # LN kernel: match DP path (HiFi2, fp32_dest off)
    ckc_ln = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    eps = 1e-5

    def mk(t, dt=ttnn.bfloat8_b, mem=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem)

    # LN affine params (mirror norm.py: [1,1,dim/32,32] row-major bf16)
    gw = torch.randn(DIM) * 0.1 + 1.0
    gb = torch.randn(DIM) * 0.1
    ln_w = ttnn.from_torch(gw.reshape(1, 1, DIM // 32, 32), dtype=ttnn.bfloat16,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ln_b = ttnn.from_torch(gb.reshape(1, 1, DIM // 32, 32), dtype=ttnn.bfloat16,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    for bname, (K, mmcfg) in BOUNDARIES.items():
        x_t = torch.randn(1, 1, M, K, dtype=torch.bfloat16) * 0.1
        w_t = torch.randn(1, 1, K, DIM, dtype=torch.bfloat16) * 0.05
        r_t = torch.randn(1, 1, M, DIM, dtype=torch.bfloat16) * 0.1
        tx, tw = mk(x_t), mk(w_t)
        tr_dram = mk(r_t, dt=ttnn.bfloat16, mem=ttnn.DRAM_MEMORY_CONFIG)

        # torch reference: LN(matmul(x,w) + residual)
        mm_ref = (x_t.float().reshape(M, K) @ w_t.float().reshape(K, DIM))
        h_ref = mm_ref + r_t.float().reshape(M, DIM)
        mu = h_ref.mean(-1, keepdim=True)
        var = h_ref.var(-1, unbiased=False, keepdim=True)
        ref = ((h_ref - mu) / torch.sqrt(var + eps)) * gw + gb

        def pcc(t):
            got = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:1].float().reshape(M, DIM)
            return torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()

        def run(mm_mem, resid_mem):
            act = ttnn.experimental.minimal_matmul(
                input_tensor=tx, weight_tensor=tw, config=mmcfg,
                memory_config=mm_mem, dtype=ttnn.bfloat8_b, compute_kernel_config=ckc_mm,
            )
            resid = tr_dram if resid_mem is ttnn.DRAM_MEMORY_CONFIG else ttnn.to_memory_config(tr_dram, resid_mem)
            out = ttnn.layer_norm(
                act, epsilon=eps, weight=ln_w, bias=ln_b, residual_input_tensor=resid,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc_ln,
            )
            ttnn.deallocate(act)
            if resid_mem is not ttnn.DRAM_MEMORY_CONFIG:
                ttnn.deallocate(resid)
            return out

        def bench(name, mm_mem, resid_mem):
            try:
                out = run(mm_mem, resid_mem)
                ttnn.synchronize_device(dev)
                p = pcc(out)
                ttnn.deallocate(out)
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                out = run(mm_mem, resid_mem)
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                ttnn.synchronize_device(dev)
                for _ in range(4):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                ts = []
                for _ in range(N_ITERS):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                    ts.append((time.perf_counter() - t0) * 1e3)
                ttnn.release_trace(dev, tid)
                ttnn.deallocate(out)
                ts.sort()
                logger.info(f"  [{bname}] {name:<10} wall min={ts[0]:7.3f} med={ts[len(ts)//2]:7.3f} ms  pcc={p:.5f}")
                return ts[0], p
            except Exception as e:
                logger.error(f"  [{bname}] {name:<10} FAILED: {str(e)[:130]}")
                return None, None

        logger.info(f"===== boundary {bname} (K={K}, N={DIM}, M={M}) =====")
        base, base_pcc = bench("baseline", ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)
        a, _ = bench("l1_act", ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)
        b, _ = bench("l1_both", ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
        if base and a:
            logger.info(f"  [{bname}] l1_act delta = {a-base:+.3f} ms vs baseline {base:.3f} ms")
        if base and b:
            logger.info(f"  [{bname}] l1_both delta = {b-base:+.3f} ms vs baseline {base:.3f} ms")
