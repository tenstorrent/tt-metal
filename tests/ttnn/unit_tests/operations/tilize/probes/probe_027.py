import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
try:
    for th in (32, 16, 8, 4, 2, 1):
        shape = [1, 1, 128, 256]
        x = torch.arange(128 * 256).reshape(shape).to(torch.bfloat16)
        tt = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        try:
            out = tilize(
                tt,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                use_multicore=True,
                tile=ttnn.Tile([th, 32]),
            )
            got = ttnn.to_torch(out)
            ok = torch.equal(got.float(), x.float())
            print(
                f"tile_h={th}: shape={list(got.shape)} equal={ok} maxdiff={(got.float()-x.float()).abs().max().item()}"
            )
        except Exception as e:
            print(f"tile_h={th}: EXC {type(e).__name__}: {str(e)[:300]}")
finally:
    ttnn.close_device(device)
