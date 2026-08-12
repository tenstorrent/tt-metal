# R7 probe D: does a 32-bit DEST fix uint8, and does Fp32Mode::Lossless make the
# fp32 identity bit-exact?  Both arms of each knob, on device.
import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)


def report(tag, exp, got):
    a = exp.to(torch.float32).flatten()
    b = got.to(torch.float32).flatten()
    bad = int((a != b).sum())
    md = float((a - b).abs().max()) if a.numel() else 0.0
    print(f"  {tag:38s} mismatches={bad:6d}/{a.numel()}  maxdiff={md:g}")


try:
    torch.manual_seed(7)
    shape = (1, 1, 64, 128)
    ti = torch.randint(0, 251, shape, dtype=torch.int32)
    tf = torch.randn(shape, dtype=torch.float32)

    print("=== uint8 identity")
    for arm, lv in (("dest_acc ON (default)", None), ("dest_acc OFF", dict(fp32_dest_acc_8bit=0))):
        x = ttnn.from_torch(
            ti, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        y = _dispatch(x, levers=lv)
        report(arm, ti, ttnn.to_torch(y).to(torch.int32))
        ttnn.deallocate(x)
        ttnn.deallocate(y)

    print("=== float32 identity")
    for arm, lv in (("lossless ON (default)", None), ("lossless OFF (Fast)", dict(fp32_lossless=0))):
        x = ttnn.from_torch(
            tf, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        y = _dispatch(x, levers=lv)
        report(arm, tf, ttnn.to_torch(y))
        ttnn.deallocate(x)
        ttnn.deallocate(y)
finally:
    ttnn.close_device(dev)
