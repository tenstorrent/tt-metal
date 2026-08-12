import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
DR = ttnn.DRAM_MEMORY_CONFIG
try:
    t = torch.randn((1, 1, 128, 256))
    for th in [4, 2, 1]:
        tt_in = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DR)
        out = tilize(tt_in, memory_config=DR, dtype=ttnn.bfloat8_b, tile=ttnn.Tile([th, 32]))
        print(f"case bfp8 th={th}: maxdiff={(ttnn.to_torch(out).float() - t).abs().max().item():.5f}")
    # uint8 narrow-stick staging at tiny tile heights (R7's alignment path)
    ti = torch.randint(0, 200, (1, 1, 128, 96), dtype=torch.int32)
    for th in [16, 8, 1]:
        tt_in = ttnn.from_torch(ti, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DR)
        out = tilize(tt_in, tile=ttnn.Tile([th, 32]))
        got = ttnn.to_torch(out).float()
        print(f"case uint8 W=96 th={th}: equal={torch.equal(got, ti.float())}")
finally:
    ttnn.close_device(device)
