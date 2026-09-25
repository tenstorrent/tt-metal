# bs16 batched SDPA at the model's config (Q [16,32,512,128], K/V [16,8,512,128] bfp8, non-causal, 12x10, q512 / k512,
# LoFi, exp approx, streaming kernel, heads-concat output), plus placement / config arms that isolate what bounds it:
# DRAM traffic (operands in L1), load balance (grids / chunks that even out the 512 units) and the softmax exp.
# Usage: TT_VISIBLE_DEVICES=<chip> bench_sdpa_bs16_ablate.py [batch=16] [sweep]  (sweep: the reuse_kv grid / q-chunk sweep)
import os
import statistics
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced

B8 = ttnn.bfloat8_b
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=128 * 1024 * 1024)
traced = make_traced(D)
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
CK = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)


def pc(grid=(12, 10), q=512, k=512, approx=True):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=q, k_chunk_size=k, exp_approx_mode=approx
    )


# (name, placement of q / kv / out, program config[, pack_gqa_heads])
ARMS = [
    ("model config (12x10 q512/k512)", (DRAM, DRAM, DRAM), pc()),
    ("reuse_kv, 12x10 q128", (DRAM, DRAM, DRAM), pc(q=128), False, True),
    ("reuse_kv, 12x10 q64 (4096 units)", (DRAM, DRAM, DRAM), pc(q=64), False, True),
    ("reuse_kv, 12x10 q96", (DRAM, DRAM, DRAM), pc(q=96), False, True),
    ("reuse_kv, 12x10 q160", (DRAM, DRAM, DRAM), pc(q=160), False, True),
    ("reuse_kv, 12x8 q128", (DRAM, DRAM, DRAM), pc(grid=(12, 8), q=128), False, True),
    ("reuse_kv, 11x10 q128", (DRAM, DRAM, DRAM), pc(grid=(11, 10), q=128), False, True),
    ("reuse_kv, 12x9 q128", (DRAM, DRAM, DRAM), pc(grid=(12, 9), q=128), False, True),
    ("reuse_kv + packed, 12x10 q128", (DRAM, DRAM, DRAM), pc(q=128), True, True),
]

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 16
if len(sys.argv) > 2 and sys.argv[2] == "sweep":
    MODEL = {8: pc(grid=(12, 8)), 16: pc(), 32: pc()}.get(BATCH, pc())
    ARMS = [("model config", (DRAM, DRAM, DRAM), MODEL)]
    for grid in ((12, 10), (12, 8)):
        for q in (512, 256, 128, 96):
            ARMS.append((f"reuse_kv, {grid[0]}x{grid[1]} q{q}", (DRAM, DRAM, DRAM), pc(grid=grid, q=q), False, True))

try:
    torch.manual_seed(0)
    q_h, k_h, v_h = (torch.randn(BATCH, n, 512, 128) for n in (32, 8, 8))
    ref = None
    for name, (qm, kvm, om), cfg, *flags in ARMS:
        pack = bool(flags and flags[0])
        reuse = bool(len(flags) > 1 and flags[1])
        try:
            q = ttnn.from_torch(q_h, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=qm)
            k = ttnn.from_torch(k_h, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=kvm)
            v = ttnn.from_torch(v_h, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=kvm)

            def run():
                return ttnn.transformer.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=False,
                    scale=0.088388346,
                    program_config=cfg,
                    compute_kernel_config=CK,
                    memory_config=om,
                    output_heads_concat=True,
                    pack_gqa_heads=pack,
                    reuse_kv=reuse,
                )

            out = ttnn.to_torch(run()).float()
            ref = out if ref is None else ref
            p = torch.corrcoef(torch.stack([out.flatten().double(), ref.flatten().double()]))[0, 1].item()
            us = statistics.median(traced(run, n=1) for _ in range(7))
            print(f"  {name:44s} {us:7.1f} us  pcc vs model config {p:.5f}", flush=True)
            for t in (q, k, v):
                ttnn.deallocate(t)
        except Exception as e:
            print(f"  {name:44s} FAILED: {str(e).splitlines()[0][:150]}", flush=True)
    print("[done]")
finally:
    ttnn.close_device(D)
