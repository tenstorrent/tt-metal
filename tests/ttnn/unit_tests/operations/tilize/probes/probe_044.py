import torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
shapes = [
    (1, 1, 32, 16384),
    (1, 1, 32, 32768),
    (1, 1, 1024, 1024),
    (1, 1, 16384, 32),
    (1, 1, 32, 2048),
]
for shape in shapes:
    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
    rows_per_blk = plan.tensor_row_blocks // plan.num_row_groups
    elem_size = 2
    row_bytes = plan.block_width_tiles * 32 * elem_size
    print(
        f"{shape}: R={plan.tensor_row_blocks} C={plan.tensor_col_tiles} bw={plan.block_width_tiles} "
        f"rows_per_blk={rows_per_blk} stick_reads_per_blk={rows_per_blk*32} row_bytes={row_bytes} "
        f"num_w_chunks={plan.num_w_chunks} num_row_groups={plan.num_row_groups} "
        f"blocks_total={plan.num_blocks_total} cores={len(plan.assignment)} split_reader={getattr(plan,'split_reader_rows',None)}"
    )
ttnn.close_device(device)
