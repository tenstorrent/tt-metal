import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
try:
    for dt, tdt in [
        (ttnn.uint32, torch.int32),
        (ttnn.uint16, torch.int32),
        (ttnn.int32, torch.int32),
        (ttnn.uint8, torch.uint8),
    ]:
        for shape in ([1, 1, 64, 128], [1, 1, 32, 64]):
            try:
                if tdt == torch.uint8:
                    x = torch.randint(0, 256, shape, dtype=torch.uint8)
                else:
                    x = torch.randint(0, 100, shape, dtype=torch.int32)
                t = ttnn.from_torch(
                    x, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                o = tilize(t, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dt, use_multicore=True)
                r = ttnn.to_torch(o)
                ok = torch.equal(r.to(tdt), x)
                print(f"{dt} {shape}: equal={ok}")
                if not ok:
                    d = r.to(torch.int32) - x.to(torch.int32)
                    print("  nonzero diffs:", int((d != 0).sum()), "of", d.numel())
                    print("  got[0,0,:2,:8]", r[0, 0, :2, :8].tolist())
                    print("  exp[0,0,:2,:8]", x[0, 0, :2, :8].tolist())
                    print("  got row1 zeros?", int((r[0, 0, 1, :] == 0).sum()), "of", r.shape[-1])
            except Exception as e:
                print(f"{dt} {shape}: EXC {type(e).__name__}: {e}")
finally:
    ttnn.close_device(dev)
