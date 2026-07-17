import time, importlib, torch, ttnn

tmod = importlib.import_module("ttnn.operations.tilize.tilize")
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)


def prof(dtype, shape, iters=30):
    torch_in = (
        torch.randint(0, 100, shape, dtype=torch.int32)
        if dtype in (ttnn.uint32, ttnn.int32)
        else torch.randn(shape, dtype=torch.float32)
    )
    tt_in = ttnn.from_torch(torch_in, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # warm program cache
    for _ in range(3):
        o = tilize(tt_in, use_multicore=True)
        ttnn.synchronize_device(device)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        o = tilize(tt_in, use_multicore=True)
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


try:
    shape = (1, 1, 2048, 2048)
    fp32 = prof(ttnn.float32, shape)
    u32 = prof(ttnn.uint32, shape)
    print(f"[1,1,2048,2048] multicore warmed median wall/iter (host+device):")
    print(f"  fp32  = {fp32:.1f} us")
    print(f"  uint32= {u32:.1f} us   (ratio uint32/fp32 = {u32/fp32:.3f})")
    print(
        f"  DRAM bytes moved = read {4*2048*2048/1e6:.2f} MB + write {4*2048*2048/1e6:.2f} MB (4B/elem, same as fp32)"
    )
finally:
    ttnn.close_device(device)
