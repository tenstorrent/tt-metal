import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 128, 256)
    t = torch.arange(128 * 256, dtype=torch.float32).reshape(shape).to(torch.bfloat16)
    for th in [32, 16, 8, 4, 2, 1]:
        tt_in = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = tilize(tt_in, tile=ttnn.Tile([th, 32]))
        got = ttnn.to_torch(out)
        ok = torch.equal(got.float(), t.float())
        print(
            f"tile_h={th}: shape={list(got.shape)} tile={out.tile.tile_shape} equal={ok} maxdiff={(got.float()-t.float()).abs().max().item()}"
        )
finally:
    ttnn.close_device(device)
