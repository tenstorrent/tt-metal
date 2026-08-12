import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 128, 256)
    t = torch.arange(128 * 256, dtype=torch.float32).reshape(shape).to(torch.bfloat16)
    for th in [1, 2, 32]:
        try:
            tt_in = ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = tilize(tt_in, tile=ttnn.Tile([th, 32]))
            got = ttnn.to_torch(out)
            print(f"tile_h={th}: OK equal={torch.equal(got.float(), t.float())}")
        except Exception as e:
            print(f"tile_h={th}: FAIL {type(e).__name__}: {str(e)[:300]}")
finally:
    ttnn.close_device(device)
