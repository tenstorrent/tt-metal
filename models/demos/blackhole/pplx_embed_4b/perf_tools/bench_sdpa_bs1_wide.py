# bs1 SDPA [1,32,512,128] with pack_gqa_heads (Q/K/V bfp8, L1-resident): targeted grid x q_chunk x k_chunk configs
# toward 120 cores (NEGATIVE_RESULTS 52). Packed, Q is 8 heads x 64 row tiles; a q chunk that does not divide 64 tiles
# pads each head's last chunk. Output is the unconcatenated [1,8,2048,128] (same memory as [1,32,512,128]) so chunks
# may span Q heads; each config is checked against the 8x8 q256/k512 packed output, and q256 / q192 are also timed
# with the concat output. Usage: TT_VISIBLE_DEVICES=<chip> bench_sdpa_bs1_wide.py (opening all chips adds minutes).
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced

B8 = ttnn.bfloat8_b
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
traced = make_traced(D)
CK = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)
L1 = ttnn.L1_MEMORY_CONFIG

# (grid, q_chunk, k_chunk): the shipped config, then the unit counts that can beat 8 row tiles per core (see header)
CONFIGS = [
    ((8, 8), 256, 512),
    ((11, 8), 192, 512),  # 11 chunks per head -> 88 units, one per core, each grid row one head (row mcast kept)
    ((11, 8), 192, 256),
    ((12, 9), 160, 512),  # 13 per head -> 104 units, one per core, heads span rows (unicast chains)
    ((12, 9), 160, 256),
    ((12, 10), 128, 512),  # 128 units: 8 cores do 2 (control: no makespan gain expected)
    ((12, 10), 64, 512),  # 256 units: up to 3 per core
    ((12, 10), 64, 256),
]


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


try:
    torch.manual_seed(0)
    q = ttnn.from_torch(torch.randn(1, 32, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    k = ttnn.from_torch(torch.randn(1, 8, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    v = ttnn.from_torch(torch.randn(1, 8, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)

    def run(gx, gy, qc, kc, concat=False):
        pcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            q_chunk_size=qc,
            k_chunk_size=kc,
            exp_approx_mode=True,
        )
        return lambda: ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=0.088388346,
            program_config=pcfg,
            compute_kernel_config=CK,
            memory_config=L1,
            pack_gqa_heads=True,
            output_heads_concat=concat,
        )

    ref_fn = run(8, 8, 256, 512)
    ref = ttnn.to_torch(ref_fn()).reshape(1, 32, 512, 128)
    shipped_us = traced(run(8, 8, 256, 512, concat=True))
    print(f"[q256] 8x8 q256/k512 packed + concat: {shipped_us:6.1f} us", flush=True)
    cat192 = run(11, 8, 192, 512, concat=True)
    p192 = pcc(ttnn.to_torch(cat192()).reshape(1, 512, 32, 128).permute(0, 2, 1, 3), ref)
    print(f"[q192] 11x8 q192/k512 packed + concat: {traced(cat192):6.1f} us  pcc {p192:.5f}", flush=True)

    res = []
    for (gx, gy), qc, kc in CONFIGS:
        tag = f"{gx}x{gy} q{qc}/k{kc}"
        try:
            fn = run(gx, gy, qc, kc)
            p = pcc(ttnn.to_torch(fn()).reshape(1, 32, 512, 128), ref)
            us = traced(fn)
            res.append((us, tag, p))
            print(f"  {tag:16s} {us:6.1f} us  pcc {p:.5f}", flush=True)
        except Exception as e:
            print(f"  {tag:16s} FAILED: {str(e).splitlines()[0][:120]}", flush=True)
    base = [r for r in res if r[1] == "8x8 q256/k512"][0][0]
    print(f"[vs 8x8 q256/k512 unconcat {base:.1f} us]")
    for us, tag, p in sorted(res):
        print(f"    {tag:16s} {us:6.1f} us ({100 * (us / base - 1):+5.1f}%)  pcc {p:.5f}")
    print("[done]")
finally:
    ttnn.close_device(D)
