import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
try:
    for th in (32, 16, 8, 4, 2, 1):
        for dt in (ttnn.bfloat16, ttnn.float32):
            shape = [1, 1, 32, 32]
            x = (torch.arange(32 * 32).reshape(shape).float() % 97) - 48
            xt = x.to(torch.bfloat16) if dt == ttnn.bfloat16 else x
            tt = ttnn.from_torch(
                xt, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            out = tilize(
                tt,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                use_multicore=False,
                tile=ttnn.Tile([th, 32]),
            )
            got = ttnn.to_torch(out).float()
            err = (got - x).abs().max().item()
            print(
                f"th={th:2d} {str(dt):22s} page={out.buffer_page_size():4d} maxdiff={err:8.3f} first8_got={got.flatten()[:8].tolist()} exp={x.flatten()[:8].tolist()}"
            )
finally:
    ttnn.close_device(device)
