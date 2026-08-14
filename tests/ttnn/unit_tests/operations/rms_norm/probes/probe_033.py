import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    for W in (1024, 7168):
        x = ttnn.from_torch(
            torch.zeros(1, 1, 32, W, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        g = ttnn.from_torch(
            torch.zeros(1, 1, 1, W, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = {"in_tile": 2048, "out_tile": 2048, "gamma_tile": 2048, "stat_tile": 4096, "bf16_tile": 2048}
        for floor in (4, 8, 16, 32):
            pd.HIDDEN_TILES_PER_CORE_FLOOR = floor
            p = pd._plan(device, x, has_gamma=True, bytes_=b)
            print(
                f"W={W} floor={floor}: s={p['num_hidden_slices']} S={p['slice_hidden_tiles']} g={p['num_row_groups']} B={p['block_rows']} rect={p['rect_w']}x{p['rect_h']}"
            )
finally:
    ttnn.close_device(device)
