# bs1 SDPA [1,32,512,128] (Q/K/V bfp8, L1-resident): grid x q_chunk x k_chunk x fp32_dest_acc (streaming kernel when False)
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced

CAUSAL = os.getenv("IS_CAUSAL", "0") == "1"

B8 = ttnn.bfloat8_b
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
traced = make_traced(D)


def CK(f32):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=f32, packer_l1_acc=True
    )


try:
    L1 = ttnn.L1_MEMORY_CONFIG
    q = ttnn.from_torch(torch.randn(1, 32, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    k = ttnn.from_torch(torch.randn(1, 8, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    v = ttnn.from_torch(torch.randn(1, 8, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    res = []
    for gx, gy in ((8, 8), (8, 10), (12, 10)):
        for qc in (512, 256, 128):
            for kc in (512, 256, 128):
                for f32 in (True, False):
                    try:
                        pcfg = ttnn.SDPAProgramConfig(
                            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                            q_chunk_size=qc,
                            k_chunk_size=kc,
                            exp_approx_mode=True,
                        )
                        us = traced(
                            lambda: ttnn.transformer.scaled_dot_product_attention(
                                q,
                                k,
                                v,
                                is_causal=False,
                                scale=0.088388346,
                                program_config=pcfg,
                                compute_kernel_config=CK(f32),
                                memory_config=L1,
                            )
                        )
                        res.append((us, gx, gy, qc, kc, f32))
                    except Exception as e:
                        res.append((float("inf"), gx, gy, qc, kc, f32))
    for f in (True, False):
        b = [r for r in res if (r[1], r[2], r[3], r[4], r[5]) == (8, 8, 512, 256, f)][0][0]
        print(f"[bs1 SDPA causal={CAUSAL}] grid 8x8 q512/k256 fp32_acc={f}: {b:7.1f} us", flush=True)
    base = [r for r in res if (r[1], r[2], r[3], r[4], r[5]) == (8, 8, 512, 256, True)][0][0]
    print(
        f"[bs1 SDPA] {sum(1 for r in res if r[0]==float('inf'))} of {len(res)} failed; top 12 vs 8x8 q512/k256 fp32_acc=True:"
    )
    for us, gx, gy, qc, kc, f32 in sorted(res)[:12]:
        print(f"    grid {gx}x{gy} q{qc}/k{kc} fp32_acc={f32}: {us:7.1f} us ({100*(us/base-1):+5.1f}%)")
    print("[done]")
finally:
    ttnn.close_device(D)
