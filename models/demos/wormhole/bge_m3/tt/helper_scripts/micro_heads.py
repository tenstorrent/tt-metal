"""Short microbenchmark: three QKV head-split variants at B8 and B16, S512.

The traced model calls bge_qkv_heads_headsplit with head_groups=4 at B8/B16.
tt-perf-report shows that op at 119.9 us/call (B8) and 152.0 us/call (B16),
which is superlinear against batch. Compare the stock ttnn op and Track A.
"""

import time

import torch

import ttnn
from models.demos.wormhole.bge_m3.tt.custom_ops.fused_qkv_heads import (
    bge_qkv_heads_headsplit,
    bge_qkv_heads_stock,
    bge_qkv_heads_tracka,
)

SEQ, HEADS, HEAD_DIM = 512, 16, 64
HIDDEN = HEADS * HEAD_DIM
ITERS = 20

d = ttnn.open_device(device_id=0)
torch.manual_seed(0)

print("variant                    B     us/call")
for batch in (8, 16):
    qkv_host = torch.randn((batch, 1, SEQ, 3 * HIDDEN), dtype=torch.bfloat16)
    qkv = ttnn.from_torch(
        qkv_host, device=d, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    cases = [
        ("stock", lambda t: bge_qkv_heads_stock(t, num_heads=HEADS)),
        ("headsplit hg=4", lambda t: bge_qkv_heads_headsplit(t, num_heads=HEADS, head_groups=4)),
        ("headsplit hg=16", lambda t: bge_qkv_heads_headsplit(t, num_heads=HEADS, head_groups=16)),
        ("tracka", lambda t: bge_qkv_heads_tracka(t, num_heads=HEADS)),
    ]
    for name, fn in cases:
        try:
            out = fn(qkv)
            ttnn.synchronize_device(d)
            for t in out:
                ttnn.deallocate(t)
            start = time.perf_counter()
            for _ in range(ITERS):
                out = fn(qkv)
                for t in out:
                    ttnn.deallocate(t)
            ttnn.synchronize_device(d)
            us = (time.perf_counter() - start) * 1e6 / ITERS
            print("  %-24s %2d %10.1f" % (name, batch, us))
        except Exception as exc:
            print("  %-24s %2d   FAILED: %s" % (name, batch, str(exc).split("\n")[0][:46]))
    ttnn.deallocate(qkv)

ttnn.close_device(d)
