# QKV minimal_matmul block sweep with in0 in DRAM or ND-sharded in the fused add+RMSNorm's unit layout (shard =
# 1 tile row x 80/R tiles, round-robin over 12x10, y fastest). bs16 in0 [1,4,2048,2560], bs32 [1,8,2048,2560];
# weights bfp4 DRAM width-sharded, LoFi, 12x10 (the model's placement). Prints the fastest configs.
# Usage: TT_VISIBLE_DEVICES=<chip> sweep_qkv_sharded_in0.py <bs> <dram|R>
import itertools
import os
import statistics
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_batched_mm_sharded_io import CK, DRAM, D, cfg, mm, traced, unit_sharded, weight  # noqa: E402

bs, place = int(sys.argv[1]), sys.argv[2]
shape = [1, bs // 4, 2048, 2560]
try:
    torch.manual_seed(0)
    w = weight(2560, 6144, True)
    x_dram = ttnn.from_torch(
        torch.randn(*shape), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=DRAM
    )
    x = x_dram if place == "dram" else ttnn.to_memory_config(x_dram, unit_sharded(shape, int(place)))
    res = []
    for m, k, n in itertools.product((4, 8, 16), (4, 8, 10, 16, 20), (4, 8, 16)):
        for sw in sorted({s for s in (2, 4, 8) if n % s == 0 and s <= n}):
            c = cfg(m, k, n, sw)
            try:
                ttnn.deallocate(mm(x, w, c, DRAM))
                us = statistics.median(traced(lambda: mm(x, w, c, DRAM), n=1) for _ in range(5))
                res.append((us, m, k, n, sw))
            except Exception:
                pass
    res.sort()
    ship = [r for r in res if r[1:] == (8, 8, 8, 8)]
    print(f"[bs{bs} QKV in0 {place}] shipped 8/8/8 sb1x8: {ship[0][0]:.1f} us; {len(res)} configs ran", flush=True)
    for us, m, k, n, sw in res[:6]:
        print(f"    M{m:<2d} K{k:<2d} N{n:<2d} sb1x{sw}: {us:7.1f} us", flush=True)
finally:
    ttnn.close_device(D)
