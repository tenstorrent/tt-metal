import torch, ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize.tilize_program_descriptor import derive_plan

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    for shape, low_l1 in (
        [(1, 1, 1, 50304), False],
        [(8, 1, 249, 2048), False],
        [(1, 1, 50, 50), False],
        [(1, 1, 32, 4090), False],
        [(1, 1, 1, 2048), False],
        [(1, 1, 50, 50), True],
    ):
        x = torch.zeros(shape, dtype=torch.bfloat16)
        ti = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = tilize(ti, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, pad_value=0.0, low_l1=low_l1)
        p = derive_plan(ti, out, low_l1=low_l1, grid=g, pad_value=0.0)
        cores = sum(1 for a in p.assignment if a[2] > 0)
        print(
            f"{shape} low_l1={low_l1}: R={p.tensor_row_blocks} C={p.tensor_col_tiles} bw={p.block_width_tiles} "
            f"wchunks={p.num_w_chunks} wrpb={p.write_rows_per_barrier} cores={cores}/{g.x*g.y} "
            f"pad_active={p.pad_active} pad_row={p.pad_row_bytes}B L1={p.l1_per_core_bytes/1024:.0f}KB "
            f"(unpadded_part={(p.l1_per_core_bytes - p.pad_row_bytes)/1024:.0f}KB)"
        )
finally:
    ttnn.close_device(device)
