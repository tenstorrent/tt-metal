"""Measure which M-block patterns expose the N=1024 gather/output CB alias bug."""

import math
import os

import torch
import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs

EMB, HIDDEN, CAPACITY = 7168, int(os.environ.get("MOE_PROBE_HIDDEN", "1024")), 1024
GRID = (11, 8)
GLOBAL_ID, LOCAL_ID, NUM_GLOBAL, NUM_LOCAL = 137, 3, 256, 8
COUNTS = (32, 128, 256, 257, 288, 320, 384, 480, 512, 513, 544, 768, 1024)


def correlation(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not finite.all() or finite.sum() < 2:
        return float("nan")
    a, b = a[finite], b[finite]
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / torch.sqrt((a @ a) * (b @ b)))


device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(173)
    x = torch.randn((1, 1, CAPACITY, EMB), dtype=torch.bfloat16)
    weights = [torch.randn(shape, dtype=torch.bfloat16) for shape in ((EMB, HIDDEN), (EMB, HIDDEN), (HIDDEN, EMB))]
    gu_mc, down_mc = weight_memory_configs(device, EMB, HIDDEN, core_grid=GRID)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_w = [
        ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        for w, mc in zip(weights, (gu_mc, gu_mc, down_mc))
    ]
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL for i in range(NUM_LOCAL)], dtype=torch.int32)
    idx[LOCAL_ID] = GLOBAL_ID
    tt_idx = ttnn.from_torch(
        idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    def run(count, dtype):
        counts = torch.zeros(NUM_GLOBAL, dtype=torch.int32)
        counts[GLOBAL_ID] = count
        tt_counts = ttnn.from_torch(
            counts,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = moe_fused_swiglu(tt_x, *tt_w, tt_counts, tt_idx, LOCAL_ID, core_grid=GRID, dtype=dtype)
        written = math.ceil(count / 32) * 32
        host = ttnn.to_torch(out)[0, 0, :written].float().clone()
        ttnn.deallocate(out)
        ttnn.deallocate(tt_counts)
        return host

    print("count mt blocks tail  bad_bfp8       max_ratio       pcc_vs_bf16", flush=True)
    for count in COUNTS:
        bf16 = run(count, ttnn.bfloat16)
        bfp8 = run(count, ttnn.bfloat8_b)
        bad = int((~torch.isfinite(bfp8)).sum())
        finite = torch.isfinite(bfp8)
        got_max = float(bfp8[finite].abs().max()) if finite.any() else float("inf")
        ref_max = float(bf16.abs().max())
        mt = math.ceil(count / 32)
        blocks = math.ceil(mt / 8)
        tail = mt - (blocks - 1) * 8
        print(
            f"{count:5d} {mt:2d} {blocks:6d} {tail:4d} {bad:9d} {got_max / ref_max:15.6e} "
            f"{correlation(bf16, bfp8):16.9f}",
            flush=True,
        )
        if blocks > 1:
            block_pcc = []
            for block in range(blocks):
                lo, hi = block * 256, min((block + 1) * 256, bf16.shape[0])
                block_pcc.append(correlation(bf16[lo:hi], bfp8[lo:hi]))
            print(f"      per-block PCC: {block_pcc}", flush=True)

    for tensor in (tt_x, *tt_w, tt_idx):
        ttnn.deallocate(tensor)
finally:
    ttnn.close_device(device)
