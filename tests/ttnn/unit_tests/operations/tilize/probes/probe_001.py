import torch, ttnn, math
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"grid = {grid.x}x{grid.y} = {grid.x*grid.y} cores")
print(f"L1 unreserved budget = {ttnn.get_max_worker_l1_unreserved_size()} B")

SHAPES = [
    ((1, 1, 32, 32), "single_tile"),
    ((1, 1, 64, 128), "multi_tile"),
    ((1, 1, 32, 128), "non_square_wide"),
    ((1, 1, 128, 32), "non_square_tall"),
    ((2, 3, 64, 96), "multi_batch"),
    ((1, 1, 2048, 64), "tall_narrow_grid_scale / regime_full_width"),
    ((1, 1, 32, 2048), "short_wide_canonical / regime_width_chunked"),
    ((1, 1, 64, 4096), "short_wide_two_tile_rows"),
    ((1, 1, 1024, 1024), "square_large (LOOSE)"),
    ((1, 1, 2048, 2048), "square_large"),
    ((8, 1, 64, 128), "leading_fold"),
    ((1, 1, 32, 16384), "PERF FOCUS (attention)"),
    ((1, 1, 32, 32768), "short_wide LOOSE"),
    ((1, 1, 64, 12288), "short_wide LOOSE"),
    ((1, 1, 16384, 32), "tall_narrow LOOSE (transposed pair)"),
    ((8, 1, 32, 2048), "square_large LOOSE-ish"),
]

hdr = f"{'shape':22s} {'R':>6s} {'C':>6s} {'chunks':>7s} {'bw':>4s} {'rowgrp':>7s} {'blocks':>7s} {'cores':>6s} {'wrpb':>5s} {'L1/core':>9s} {'rd bytes':>9s}"
print(hdr)
print("-" * len(hdr))
for shape, name in SHAPES:
    t = torch.zeros(shape, dtype=torch.bfloat16)
    ti = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    to = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    p = pd.derive_plan(ti, to, low_l1=False, grid=grid)
    cores = len(p.assignment)
    print(
        f"{str(shape):22s} {p.tensor_row_blocks:6d} {p.tensor_col_tiles:6d} {p.num_w_chunks:7d} {p.block_width_tiles:4d} "
        f"{p.num_row_groups:7d} {p.num_blocks_total:7d} {cores:6d} {p.write_rows_per_barrier:5d} "
        f"{p.l1_per_core_bytes//1024:8d}K {p.block_row_bytes:9d}   {name}"
    )
    ttnn.deallocate(ti)
    ttnn.deallocate(to)

# low_l1 cap check on the L1-forcing width at fp32
print()
for lo in (False, True):
    shape = (1, 1, 32, 8192)
    t = torch.zeros(shape, dtype=torch.float32)
    ti = ttnn.from_torch(
        t, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    to = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    p = pd.derive_plan(ti, to, low_l1=lo, grid=grid)
    print(
        f"fp32 [1,1,32,8192] low_l1={lo}: bw={p.block_width_tiles} chunks={p.num_w_chunks} cores={len(p.assignment)} L1/core={p.l1_per_core_bytes//1024}K"
    )
    ttnn.deallocate(ti)
    ttnn.deallocate(to)

ttnn.close_device(device)
