"""Exercise the N=3072, M_BLOCK=8 BF16-RM pressure path across block boundaries."""

import time

import torch
import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs


EMB, HIDDEN, CAPACITY = 7168, 3072, 1024
GRID = (11, 8)
GLOBAL_ID, LOCAL_ID, NUM_GLOBAL, NUM_LOCAL = 137, 3, 256, 8
COUNTS = (256, 257, 512, 513, 768, 1024)


device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(3072)
    block = torch.randn((1, 1, 256, EMB), dtype=torch.bfloat16)
    x = block.repeat(1, 1, CAPACITY // 256, 1)
    weights = [torch.randn(shape, dtype=torch.bfloat16) for shape in ((EMB, HIDDEN), (EMB, HIDDEN), (HIDDEN, EMB))]
    gu_mc, down_mc = weight_memory_configs(device, EMB, HIDDEN, core_grid=GRID)
    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

    def run(count):
        counts = torch.zeros(NUM_GLOBAL, dtype=torch.int32)
        counts[GLOBAL_ID] = count
        tt_counts = ttnn.from_torch(
            counts,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        started = time.perf_counter_ns()
        out = moe_fused_swiglu(tt_x, *tt_w, tt_counts, tt_idx, LOCAL_ID, core_grid=GRID)
        ttnn.synchronize_device(device)
        elapsed_us = (time.perf_counter_ns() - started) / 1000
        host = ttnn.to_torch(out)[0, 0, :count].clone()
        ttnn.deallocate(out)
        ttnn.deallocate(tt_counts)
        return host, elapsed_us

    reference, _ = run(256)  # compile/warmup and establish the repeated-block answer
    print("count blocks exact_repeated_rows elapsed_us", flush=True)
    for count in COUNTS:
        got, elapsed_us = run(count)
        expected = reference.repeat((count + 255) // 256, 1)[:count]
        exact = torch.equal(got, expected)
        print(f"{count:5d} {(count + 255) // 256:6d} {str(exact):>19s} {elapsed_us:10.3f}", flush=True)
        if not exact:
            diff_rows = torch.any(got != expected, dim=-1)
            raise AssertionError(f"count={count}: {int(diff_rows.sum())} repeated rows differ")

    for tensor in (tt_x, *tt_w, tt_idx):
        ttnn.deallocate(tensor)
finally:
    ttnn.close_device(device)
