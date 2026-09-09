import torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
budget = ttnn.get_max_worker_l1_unreserved_size()
print(f"grid={grid.x}x{grid.y}={grid.x*grid.y}  budget={budget} B  WRITE_BATCH_MIN_TILES={pd.WRITE_BATCH_MIN_TILES}")
for dt, tb in ((ttnn.bfloat16, 2048), (ttnn.float32, 4096)):
    denom = pd.INPUT_DEPTH_ROWS * tb + pd.OUTPUT_DEPTH_BATCHES * tb
    head = budget - pd.OUTPUT_DEPTH_BATCHES * pd.WRITE_BATCH_MIN_TILES * tb
    print(f"  W_FIT({dt}) = {max(1, min(head//denom, pd.FAST_TILIZE_WIDTH_CAP))}")

CASES = [
    ((1, 1, 32, 32), ttnn.bfloat16, False, "single tile"),
    ((1, 1, 2048, 2048), ttnn.bfloat16, False, "square_large"),
    ((1, 1, 1024, 1024), ttnn.bfloat16, False, "square_large 1024"),
    ((1, 1, 32, 16384), ttnn.bfloat16, False, "PERF FOCUS"),
    ((1, 1, 32, 32768), ttnn.bfloat16, False, "short_wide 32768"),
    ((1, 1, 32, 2048), ttnn.bfloat16, False, "short_wide canonical"),
    ((1, 1, 2048, 64), ttnn.bfloat16, False, "tall_narrow"),
    ((1, 1, 16384, 32), ttnn.bfloat16, False, "tall_narrow 16384"),
    ((1, 1, 32, 8192), ttnn.float32, False, "fp32 l1_forcing low_l1=False"),
    ((1, 1, 32, 8192), ttnn.float32, True, "fp32 l1_forcing low_l1=True"),
]
print(f"\n| Shape | bw | wrpb | cores | L1_per_core |")
print(f"|---|---|---|---|---|")
for shape, dt, lo, name in CASES:
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    t = torch.zeros(shape, dtype=tdt)
    ti = ttnn.from_torch(
        t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    to = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    p = pd.derive_plan(ti, to, low_l1=lo, grid=grid)
    kb = p.l1_per_core_bytes / 1024
    print(
        f"| `{list(shape)}` {name} | {p.block_width_tiles} | {p.write_rows_per_barrier} | {len(p.assignment)} / {grid.x*grid.y} | {kb:g} KB |"
    )
    ttnn.deallocate(ti)
    ttnn.deallocate(to)
ttnn.close_device(device)
