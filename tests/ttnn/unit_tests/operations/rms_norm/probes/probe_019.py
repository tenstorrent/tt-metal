import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd


def column_pinned(gx, gy):
    out = []
    for w in range(1, gx + 1):
        if gx % w == 0:
            out.append((w, 1))
    for h in range(2, gy + 1):
        out.append((gx, h))
    return [(w * h, w, h) for (w, h) in out]


device = ttnn.open_device(device_id=0)
try:
    b = {
        "in_tile": ttnn.tile_size(ttnn.bfloat16),
        "out_tile": ttnn.tile_size(ttnn.bfloat16),
        "gamma_tile": ttnn.tile_size(ttnn.bfloat16),
        "stat_tile": ttnn.tile_size(ttnn.float32),
        "bf16_tile": ttnn.tile_size(ttnn.bfloat16),
    }
    x = torch.zeros((1, 1, 32, 7168), dtype=torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    p = pd._plan(device, t, has_gamma=True, bytes_=b)
    print("CHECK grid_wide", p["num_row_groups"] * p["num_hidden_slices"])
    orig = pd._rect_candidates
    pd._rect_candidates = column_pinned
    p = pd._plan(device, t, has_gamma=True, bytes_=b)
    print("CHECK column_pinned", p["num_row_groups"] * p["num_hidden_slices"])
    pd._rect_candidates = orig
finally:
    ttnn.close_device(device)
