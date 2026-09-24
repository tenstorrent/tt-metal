# batched SDPA sweep (model config: LoFi, fp32 acc off, exp approx, bfp8 Q/K/V in DRAM): grid x q_chunk x k_chunk
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced

B8 = ttnn.bfloat8_b
BS = int(os.getenv("BS", "8"))
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
traced = make_traced(D)
ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)
try:
    DR = ttnn.DRAM_MEMORY_CONFIG
    q = ttnn.from_torch(torch.randn(BS, 32, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=DR)
    k = ttnn.from_torch(torch.randn(BS, 8, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=DR)
    v = ttnn.from_torch(torch.randn(BS, 8, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=DR)
    res = []
    for gx, gy in ((12, 10), (8, 10), (8, 8), (12, 8), (10, 10)):
        for qc in (512, 256, 128, 64):
            for kc in (512, 256):
                try:
                    pc = ttnn.SDPAProgramConfig(
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
                            program_config=pc,
                            compute_kernel_config=ckc,
                            memory_config=DR,
                        ),
                        n=2,
                    )
                    res.append((us, gx, gy, qc, kc))
                except Exception as e:
                    res.append((float("inf"), gx, gy, qc, kc))
    base = [r for r in res if (r[1], r[2], r[3], r[4]) == (12, 10, 256, 512)][0][0]
    print(
        f"[bs{BS} SDPA] shipped 12x10 q256/k512: {base:7.1f} us ({BS*4.29e9/base/1e6:4.0f} TFLOP/s); {sum(1 for r in res if r[0]==float('inf'))}/{len(res)} failed",
        flush=True,
    )
    for us, gx, gy, qc, kc in sorted(res)[:10]:
        print(f"    grid {gx}x{gy} q{qc}/k{kc}: {us:7.1f} us ({100*(us/base-1):+5.1f}%)")
    print("[done]")
finally:
    ttnn.close_device(D)
