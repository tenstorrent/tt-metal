import torch, ttnn
from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()


def show(shape, hi, ho):
    n = 1
    for d in shape:
        n *= d
    t = (torch.arange(n) % 4093).reshape(shape).to(torch.bfloat16)
    layout = ttnn.TILE_LAYOUT if hi else ttnn.ROW_MAJOR_LAYOUT
    kw = {"tile": ttnn.Tile([hi, 32])} if hi else {}
    tt = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw
    )
    out = tilize(tt, dtype=ttnn.bfloat16, tile=ttnn.Tile([ho, 32]))
    p = pd.derive_plan(tt, out, low_l1=False, grid=grid, pad_value=None)
    print(
        f"L1 {shape} {hi or 'RM'}->{ho} bw={p.block_width_tiles} wrpb={p.write_rows_per_barrier} "
        f"in_pages={p.input_cb_pages} out_pages={p.output_cb_pages} l1={p.l1_per_core_bytes}",
        flush=True,
    )


show([1, 1, 2048, 2048], 0, 32)
show([1, 1, 2048, 2048], 32, 16)
show([1, 1, 2048, 2048], 16, 32)
show([1, 1, 32, 2048], 0, 16)
show([1, 1, 32, 2048], 0, 1)
show([1, 1, 2048, 64], 0, 1)
