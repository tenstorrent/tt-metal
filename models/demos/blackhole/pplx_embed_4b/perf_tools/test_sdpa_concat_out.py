# SDPA output_heads_concat: bit-identical to concat_heads(SDPA) at bs1/bs8 model shapes + timing of the op alone
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced

B8 = ttnn.bfloat8_b
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
traced = make_traced(D)
ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)
try:
    for BS, mc, grid, qc, kc in (
        (1, ttnn.L1_MEMORY_CONFIG, (8, 8), 256, 256),
        (8, ttnn.DRAM_MEMORY_CONFIG, (12, 8), 512, 512),
        (32, ttnn.DRAM_MEMORY_CONFIG, (12, 10), 512, 512),
    ):
        q = ttnn.from_torch(
            torch.randn(BS, 32, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=mc
        )
        k = ttnn.from_torch(torch.randn(BS, 8, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=mc)
        v = ttnn.from_torch(torch.randn(BS, 8, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=mc)
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=True
        )
        f_std = lambda: ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=0.088388346, program_config=pc, compute_kernel_config=ckc, memory_config=mc
        )
        f_cat = lambda: ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=0.088388346,
            program_config=pc,
            compute_kernel_config=ckc,
            memory_config=mc,
            output_heads_concat=True,
        )
        o = f_std()
        ref = ttnn.experimental.nlp_concat_heads(o, memory_config=mc)
        c = f_cat()
        rt = ttnn.to_torch(ref)
        ct = ttnn.to_torch(c)
        print(
            f"[bs{BS}] std out {tuple(o.shape)} concat_heads {tuple(ref.shape)} | flag out {tuple(c.shape)} identical={torch.equal(rt,ct)} max|d|={(rt.float()-ct.float()).abs().max().item():.3g}",
            flush=True,
        )
        [ttnn.deallocate(t) for t in (o, ref, c)]
        us_std = traced(f_std, n=2)
        us_cat = traced(f_cat, n=2)

        def std_plus_concat():
            o = f_std()
            r = ttnn.experimental.nlp_concat_heads(o, memory_config=mc)
            ttnn.deallocate(o)
            return r

        us_both = traced(std_plus_concat, n=2)
        print(
            f"[bs{BS}] SDPA {us_std:8.1f} us | SDPA+concat_heads {us_both:8.1f} | SDPA(output_heads_concat) {us_cat:8.1f} us  -> saves {us_both-us_cat:7.1f} us/call",
            flush=True,
        )
        [ttnn.deallocate(t) for t in (q, k, v)]
    print("[done]")
finally:
    ttnn.close_device(D)
