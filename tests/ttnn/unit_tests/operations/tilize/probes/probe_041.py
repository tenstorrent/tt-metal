import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
DR = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
try:
    t = torch.randn((1, 1, 128, 256))
    # host loopback: does a tiny-tile bfp8 tensor even round-trip through from_torch?
    for th in [32, 16, 8, 1]:
        try:
            x = ttnn.from_torch(
                t,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DR,
                tile=ttnn.Tile([th, 32]),
            )
            back = ttnn.to_torch(x).float()
            print(f"host th={th}: maxdiff={(back - t).abs().max().item():.5f}")
        except Exception as e:
            print(f"host th={th}: FAIL {type(e).__name__}: {str(e)[:200]}")
    for th in [32, 16, 8, 1]:
        for mcname, mc in (("dram", DR), ("l1", L1)):
            try:
                tt_in = ttnn.from_torch(
                    t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc
                )
                out = tilize(tt_in, memory_config=mc, dtype=ttnn.bfloat8_b, tile=ttnn.Tile([th, 32]))
                got = ttnn.to_torch(out).float()
                print(
                    f"case bfp8 th={th} {mcname}: maxdiff={(got - t).abs().max().item():.5f} pagesz={out.buffer_page_size()}"
                )
            except Exception as e:
                print(f"case bfp8 th={th} {mcname}: FAIL {type(e).__name__}: {str(e)[:200]}")
finally:
    ttnn.close_device(device)
