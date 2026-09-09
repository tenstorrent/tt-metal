# Refinement 3: rebuild the l1_ledger footprint tables off the BUILT plan after
# the PIPELINE_WAVES_PER_CORE / MIN_BLOCK_ROW_BYTES rule.
import torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()

UNPADDED = [
    ((1, 1, 32, 32), ttnn.bfloat16, False, "single tile"),
    ((1, 1, 2048, 2048), ttnn.bfloat16, False, "square_large"),
    ((1, 1, 1024, 1024), ttnn.bfloat16, False, "square_large"),
    ((1, 1, 32, 16384), ttnn.bfloat16, False, "perf focus"),
    ((1, 1, 32, 32768), ttnn.bfloat16, False, ""),
    ((1, 1, 32, 2048), ttnn.bfloat16, False, "short_wide"),
    ((1, 1, 2048, 64), ttnn.bfloat16, False, "tall_narrow"),
    ((1, 1, 16384, 32), ttnn.bfloat16, False, "tall_narrow"),
    ((1, 1, 32, 8192), ttnn.float32, False, "fp32 low_l1=False"),
    ((1, 1, 32, 8192), ttnn.float32, True, "fp32 low_l1=True"),
    ((1, 1, 1, 50304), ttnn.bfloat16, True, "logits low_l1=True"),
]
PADDED = [
    ((1, 1, 50, 50), "hw tails"),
    ((1, 1, 1, 2048), "single stick"),
    ((1, 1, 32, 4090), "short_wide W tail"),
    ((1, 1, 1, 50304), "logits row"),
    ((8, 1, 249, 2048), "H tail through the fold"),
]


def row(shape, dt, low, pad=None):
    t = torch.randn(shape, dtype=torch.float32)
    t = t.to(torch.float32) if dt == ttnn.float32 else t.bfloat16()
    x = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y = tilize(x, pad_value=pad) if pad is not None else tilize(x)
    p = pd.derive_plan(x, y, low_l1=low, grid=grid, pad_value=pad)
    return p


print("=== UNPADDED ===")
for shape, dt, low, note in UNPADDED:
    p = row(shape, dt, low)
    print(
        f"{str(shape):20s} {note:20s} bw={p.block_width_tiles:3d} chunks={p.num_w_chunks:4d} "
        f"wrpb={p.write_rows_per_barrier} cores={len(p.assignment):2d} "
        f"L1={p.l1_per_core_bytes//1024:4d} KB read={p.block_row_bytes}B"
    )

print("=== PADDED (pad_mode=auto) ===")
for shape, note in PADDED:
    p = row(shape, ttnn.bfloat16, False, pad=0.0)
    print(
        f"{str(shape):20s} {note:24s} bw={p.block_width_tiles:3d} wrpb={p.write_rows_per_barrier} "
        f"cores={len(p.assignment):2d} pad={p.pad_row_bytes}B L1={p.l1_per_core_bytes//1024:4d} KB"
    )
p = row((1, 1, 50, 50), ttnn.bfloat16, True, pad=0.0)
print(f"{'(1,1,50,50) low_l1=True':44s} bw={p.block_width_tiles} L1={p.l1_per_core_bytes//1024} KB")
print("budget:", ttnn.get_max_worker_l1_unreserved_size())
ttnn.close_device(device)
