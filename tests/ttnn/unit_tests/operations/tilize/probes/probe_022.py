import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
try:
    x = torch.randint(0, 256, [1, 1, 64, 128], dtype=torch.uint8)
    # (1) host tilize round-trip: is the READBACK of a uint8 TILE tensor sane?
    h = ttnn.from_torch(x, dtype=ttnn.uint8, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print("host TILE round-trip equal:", torch.equal(ttnn.to_torch(h).to(torch.uint8), x))
    # (2) our op with fp32 DEST forced on for 8-bit
    for mc in (True, False):
        t = ttnn.from_torch(
            x, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        o = tilize(t, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.uint8, use_multicore=mc)
        r = ttnn.to_torch(o).to(torch.uint8)
        ok = torch.equal(r, x)
        print(f"op uint8 multicore={mc}: equal={ok}")
        if not ok:
            d = r.to(torch.int32) - x.to(torch.int32)
            print("  wrong:", int((d != 0).sum()), "/", d.numel())
            for rr in range(4):
                print(f"  row{rr} got", r[0, 0, rr, :8].tolist(), "exp", x[0, 0, rr, :8].tolist())
finally:
    ttnn.close_device(dev)
