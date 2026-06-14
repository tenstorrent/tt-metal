"""Minimal on-device smoke test: open the Blackhole device, run a tiny tensor op, close.
Run via:  TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/smoke_device.py
"""
import torch
import ttnn

print("[smoke] opening device 0 (the single exposed Blackhole)")
dev = ttnn.open_device(device_id=0)
try:
    for label, fn in [
        ("arch", lambda: dev.arch()),
        ("num dram ch", lambda: dev.num_dram_channels()),
        ("compute grid", lambda: (lambda g: f"{g.x} x {g.y}")(dev.compute_with_storage_grid_size())),
    ]:
        try:
            print(f"[smoke] {label:13s}: {fn()}")
        except Exception as e:
            print(f"[smoke] {label:13s}: <unavailable: {type(e).__name__}>")

    a = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev)
    tc = ttnn.matmul(ta, tb)
    c = ttnn.to_torch(tc)
    ref = (a.float() @ b.float())
    err = (c.float() - ref).abs().max().item()
    print(f"[smoke] matmul out shape {tuple(c.shape)}  max|err| vs torch fp32 = {err:.4f}")
    assert tuple(c.shape) == (1, 1, 32, 128), c.shape
    print("[smoke] OK — device compute works")
finally:
    ttnn.close_device(dev)
    print("[smoke] device closed")
