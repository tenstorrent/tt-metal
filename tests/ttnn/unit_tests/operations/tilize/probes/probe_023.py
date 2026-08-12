# R7 probe A: the uint8 failure SIGNATURE. Dump one output tile as a 32x32
# matrix and compare it position-by-position with torch, so "strided /
# every-other-row-zero" is observed rather than inferred from PCC.
import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    shape = (1, 1, 32, 64)
    # position-encoding source: value = row*100 + col (mod 251, prime, so no aliasing)
    t = torch.zeros(shape, dtype=torch.int32)
    for r in range(32):
        for c in range(64):
            t[0, 0, r, c] = (r * 100 + c) % 251
    for dt in (ttnn.uint8, ttnn.uint16, ttnn.uint32):
        x = ttnn.from_torch(
            t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        y = _dispatch(x)
        out = ttnn.to_torch(y).to(torch.int32)
        bad = out != t
        print(f"=== {dt} : mismatches {int(bad.sum())} / {t.numel()}")
        if int(bad.sum()):
            rows_bad = [r for r in range(32) if int(bad[0, 0, r].sum())]
            print("   bad rows:", rows_bad[:40])
            print("   got row0[:16] ", out[0, 0, 0, :16].tolist())
            print("   exp row0[:16] ", t[0, 0, 0, :16].tolist())
            print("   got row1[:16] ", out[0, 0, 1, :16].tolist())
            print("   exp row1[:16] ", t[0, 0, 1, :16].tolist())
            print("   got row16[:16]", out[0, 0, 16, :16].tolist())
            print("   exp row16[:16]", t[0, 0, 16, :16].tolist())
        ttnn.deallocate(x)
        ttnn.deallocate(y)
finally:
    ttnn.close_device(dev)
