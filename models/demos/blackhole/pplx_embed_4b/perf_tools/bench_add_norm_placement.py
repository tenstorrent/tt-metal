# fused_add_rmsnorm_split at the model's placement per batch (a DRAM, b / normalised output L1, sum L1 at bs8 / 16 and
# DRAM at bs32): traced us per call, outputs saved to $OUT_DIR (default /tmp) as addnorm_<tag>_bs<B>.pt for bit-identity checks
# across env arms (e.g. QWEN_ADD_NORM_PART_TRID=0 / 1). Usage: bench_add_norm_placement.py <tag>
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced

from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_add_rmsnorm import (
    fused_add_rmsnorm_split,
    make_add_norm_constants,
)

tag = sys.argv[1]
out_dir = os.environ.get("OUT_DIR", "/tmp")
W, EPS, B8 = 2560, 1e-6, ttnn.bfloat8_b
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
traced = make_traced(D)
DR, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
try:
    torch.manual_seed(0)
    G, SC, EP = make_add_norm_constants(torch.rand(W) * 0.5 + 0.75, EPS, D)
    # bs: (R, sum memcfg); a (residual) DRAM, b (WO/FF2 out) L1, normalised out L1
    for bs, R, smc in ((8, 5, L1), (16, 5, L1), (32, 4, DR)):
        M = bs * 512
        a = ttnn.from_torch(torch.randn(1, 1, M, W), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=DR)
        b = ttnn.from_torch(
            torch.randn(1, 1, M, W) * 0.3, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1
        )
        f = lambda: fused_add_rmsnorm_split(a, b, G, SC, EP, R=R, memory_config=smc, out_memory_config=L1)
        s, n = f()
        torch.save((ttnn.to_torch(s), ttnn.to_torch(n)), f"{out_dir}/addnorm_{tag}_bs{bs}.pt")
        ttnn.deallocate(s)
        ttnn.deallocate(n)
        o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, W]), B8, ttnn.TILE_LAYOUT, D, L1)

        def g():
            s, n = fused_add_rmsnorm_split(a, b, G, SC, EP, R=R, memory_config=smc, out_memory_config=L1, out_tensor=o)
            return s

        us = [traced(g, n=4) for _ in range(5)]
        ttnn.deallocate(o)
        print(f"RES addnorm tag={tag} bs={bs} R={R} us_min={min(us):.1f} us_med={sorted(us)[2]:.1f}", flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
finally:
    ttnn.close_device(D)
