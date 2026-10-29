import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as d

dev = ttnn.open_device(device_id=0)
print("grid", dev.compute_with_storage_grid_size())
shapes = [
    (1, 1, 32, 4096),
    (1, 1, 32, 8192),
    (1, 1, 128, 4096),
    (128, 8192),
    (1, 1, 32, 16384),
    (1, 1, 64, 12288),
    (1, 1, 32, 32768),
    (1, 1, 160, 11008),
    (3104, 4064),
    (1, 1, 8192, 7168),
]
for dt in (ttnn.bfloat16, ttnn.float32):
    for s in shapes:
        for g in (False, True):
            x = ttnn.allocate_tensor_on_device(ttnn.Shape(list(s)), dt, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
            gm = (
                ttnn.from_torch(torch.ones(1, 1, 1, s[-1]), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
                if g
                else None
            )
            b = d.select_blocking(x, gm)
            print(
                dt,
                s,
                "gamma" if g else "nog",
                None
                if b is None
                else (b.block_row_tiles, b.w_chunk_tiles, b.in_depth, b.tensor_w_tiles, len(b.assignment)),
            )
ttnn.close_device(dev)
