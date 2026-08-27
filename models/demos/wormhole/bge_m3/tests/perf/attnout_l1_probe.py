# SPDX-License-Identifier: Apache-2.0
"""concat->AttnOut L1-intermediate probe (the one un-checked DRAM-round-trip box).

Per-ASIC AttnOut: [M=49152, K=1024] @ [1024,1024] = [49152,1024]. The concat
output (~50MB bf8) round-trips DRAM (~13.7ms/pass). AttnOut CBs are smaller than
Wo's (K=1024 vs 4096) so an L1-resident concat output MIGHT fit. Test: feed
AttnOut an L1 input tensor (proxy for concat output staying in L1). If it runs
without the static-CB clash, the direct-layout concat->AttnOut fusion is viable.
"""
import time

import pytest
import torch
from loguru import logger

import ttnn

M, K, N = 49152, 1024, 1024
N_ITERS = 15


def _cfg(M=16, Kb=8, Nb=4, sh=4, sw=2):
    return ttnn.MinimalMatmulConfig(
        M_block_size=M,
        K_block_size=Kb,
        N_block_size=Nb,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 90_000_000, "num_command_queues": 1}], indirect=True)
def test_attnout_l1(mesh_device):
    torch.manual_seed(0)
    x = torch.randn(1, 1, M, K, dtype=torch.bfloat16) * 0.1
    w = torch.randn(1, 1, K, N, dtype=torch.bfloat16) * 0.05
    dev = mesh_device
    mk = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    tw = mk(w, ttnn.DRAM_MEMORY_CONFIG)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    ref = x.float() @ w.float()

    def pcc(t):
        got = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:1].float()
        return torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()

    def bench(name, tx_mc, out_mc, cfg):
        tx = mk(x, tx_mc)

        def body():
            return ttnn.experimental.minimal_matmul(
                input_tensor=tx,
                weight_tensor=tw,
                config=cfg,
                memory_config=out_mc,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=ckc,
            )

        try:
            out = body()
            ttnn.synchronize_device(dev)
            p = pcc(out)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            out = body()
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.synchronize_device(dev)
            ts = []
            for _ in range(N_ITERS):
                t0 = time.perf_counter()
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                ts.append((time.perf_counter() - t0) * 1e3)
            ttnn.release_trace(dev, tid)
            ts.sort()
            logger.info(f"{name:<30} wall min={ts[0]:.3f} ms  pcc={p:.5f}")
        except Exception as e:
            logger.error(f"{name} FAILED: {str(e)[:150]}")
        finally:
            ttnn.deallocate(tx)

    D, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    bench("DRAM_in stock cfg", D, D, _cfg())
    # sweep smaller CB configs to free ~10KB/core for the L1 concat tensor (bf8, precision-preserving)
    variants = {
        "L1_stock_m16k8n4": _cfg(16, 8, 4, 4, 2),
        "L1_k4": _cfg(16, 4, 4, 4, 2),
        "L1_n2": _cfg(16, 8, 2, 4, 2),
        "L1_m8": _cfg(8, 8, 4, 4, 2),
        "L1_k4n2": _cfg(16, 4, 2, 4, 2),
        "L1_sh2sw1": _cfg(16, 8, 4, 2, 1),
        "L1_m8k4n2": _cfg(8, 4, 2, 4, 2),
    }
    for nm, cfg in variants.items():
        bench(nm, L1, D, cfg)
