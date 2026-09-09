# Refinement 3: rebuild the l1_ledger footprint tables off the BUILT plan after
# the PIPELINE_WAVES_PER_CORE / MIN_BLOCK_ROW_BYTES rule. bf16 only (fp32 is
# Refinement 5); the fp32 rows are unchanged and re-derived in the ledger text.
import torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()


def row(shape, low, pad=None):
    t = torch.randn(shape, dtype=torch.float32).bfloat16()
    x = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    y = tilize(x, pad_value=pad) if pad is not None else tilize(x)
    return pd.derive_plan(x, y, low_l1=low, grid=grid, pad_value=pad)


print("=== UNPADDED ===")
for shape, low, note in [
    ((1, 1, 32, 32), False, "single tile"),
    ((1, 1, 2048, 2048), False, "square_large"),
    ((1, 1, 1024, 1024), False, "square_mid"),
    ((1, 1, 32, 16384), False, "perf focus"),
    ((1, 1, 32, 32768), False, "short_wide_wide"),
    ((1, 1, 32, 2048), False, "short_wide"),
    ((1, 1, 2048, 64), False, "full_width"),
    ((1, 1, 16384, 32), False, "tall_narrow"),
    ((1, 1, 1, 50304), True, "logits low_l1=True"),
]:
    p = row(shape, low)
    print(
        f"{str(shape):20s} {note:20s} bw={p.block_width_tiles:3d} chunks={p.num_w_chunks:4d} "
        f"wrpb={p.write_rows_per_barrier} cores={len(p.assignment):2d} "
        f"L1={p.l1_per_core_bytes//1024:4d} KB read={p.block_row_bytes}B "
        f"waves={p.tensor_row_blocks*p.num_w_chunks/max(len(p.assignment),1):.1f}"
    )

print("=== PADDED ===")
for shape, note in [
    ((1, 1, 50, 50), "hw tails"),
    ((1, 1, 1, 2048), "single stick"),
    ((1, 1, 32, 4090), "short_wide W tail"),
    ((1, 1, 1, 50304), "logits row"),
    ((8, 1, 249, 2048), "H tail through the fold"),
]:
    p = row(shape, False, pad=0.0)
    print(
        f"{str(shape):20s} {note:24s} bw={p.block_width_tiles:3d} wrpb={p.write_rows_per_barrier} "
        f"cores={len(p.assignment):2d} pad={p.pad_row_bytes}B L1={p.l1_per_core_bytes//1024:4d} KB "
        f"stream={(p.l1_per_core_bytes-p.pad_row_bytes)//1024} KB"
    )
p = row((1, 1, 50, 50), True, pad=0.0)
print(f"(1,1,50,50) low_l1=True  bw={p.block_width_tiles} L1={p.l1_per_core_bytes//1024} KB")
print("budget:", ttnn.get_max_worker_l1_unreserved_size())
ttnn.close_device(device)
