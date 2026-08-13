import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("GRID", g.x, g.y, "=", g.x * g.y, "cores")
    for shape, lay in [
        ((32, 32), ttnn.TILE_LAYOUT),
        ((64, 64), ttnn.TILE_LAYOUT),
        ((32, 128), ttnn.TILE_LAYOUT),
        ((256, 32), ttnn.TILE_LAYOUT),
        ((2, 4, 128, 256), ttnn.TILE_LAYOUT),
        ((4096, 64), ttnn.TILE_LAYOUT),
        ((2048, 128), ttnn.TILE_LAYOUT),
        ((1, 1, 32, 4096), ttnn.TILE_LAYOUT),
        ((1, 1, 32, 8192), ttnn.TILE_LAYOUT),
        ((1, 1, 64, 12288), ttnn.TILE_LAYOUT),
        ((1, 1, 32, 7168), ttnn.TILE_LAYOUT),
        ((1, 1, 8192, 1024), ttnn.TILE_LAYOUT),
        ((47, 100), ttnn.ROW_MAJOR_LAYOUT),
    ]:
        x = torch.zeros(shape, dtype=torch.bfloat16)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = {
            "in_tile": ttnn.tile_size(ttnn.bfloat16),
            "out_tile": ttnn.tile_size(ttnn.bfloat16),
            "gamma_tile": ttnn.tile_size(ttnn.bfloat16),
            "stat_tile": ttnn.tile_size(ttnn.float32),
            "bf16_tile": ttnn.tile_size(ttnn.bfloat16),
        }
        p = pd._plan(device, t, has_gamma=True, bytes_=b)
        cores = p["num_row_groups"] * p["num_hidden_slices"]
        print(
            "PLAN",
            shape,
            "Rt=%d Wt=%d" % (p["row_tiles"], p["hidden_tiles"]),
            "g=%d s=%d S=%d B=%d"
            % (p["num_row_groups"], p["num_hidden_slices"], p["slice_hidden_tiles"], p["block_rows"]),
            "rect=%dx%d" % (p["rect_w"], p["rect_h"]),
            "cores=%d" % cores,
        )
        ttnn.deallocate(t)
finally:
    ttnn.close_device(device)
