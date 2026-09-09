# Refinement 3: padded-path footprint rows for l1_ledger.md after the wave rule.
import torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()


def row(shape, low):
    t = torch.randn(shape, dtype=torch.float32).bfloat16()
    x = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    y = tilize(x, pad_value=0.0)
    return pd.derive_plan(x, y, low_l1=low, grid=grid, pad_value=0.0)


for shape, note, low in [
    ((1, 1, 50, 50), "hw tails", False),
    ((1, 1, 1, 2048), "single stick", False),
    ((1, 1, 32, 4090), "short_wide W tail", False),
    ((1, 1, 1, 50304), "logits row", False),
    ((8, 1, 249, 2048), "H tail through the fold", False),
    ((1, 1, 50, 50), "hw tails low_l1", True),
    ((1, 1, 1, 50304), "logits low_l1", True),
]:
    p = row(shape, low)
    print(
        f"{str(shape):18s} {note:26s} bw={p.block_width_tiles:3d} chunks={p.num_w_chunks:4d} "
        f"wrpb={p.write_rows_per_barrier} cores={len(p.assignment):2d} pad={p.pad_row_bytes}B "
        f"stream={(p.l1_per_core_bytes-p.pad_row_bytes)//1024}KB L1={p.l1_per_core_bytes//1024}KB"
    )
print("budget:", ttnn.get_max_worker_l1_unreserved_size())
ttnn.close_device(device)
