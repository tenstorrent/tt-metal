# R7 probe E: fp32 identity, lossless ON vs OFF; plus the narrowing casts that
# must NOT pay for it.
import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)


def report(tag, exp, got):
    a = exp.to(torch.float32).flatten()
    b = got.to(torch.float32).flatten()
    print(f"  {tag:34s} mismatches={int((a != b).sum()):6d}/{a.numel()}  maxdiff={float((a-b).abs().max()):g}")


try:
    torch.manual_seed(7)
    shape = (1, 1, 64, 128)
    tf = torch.randn(shape, dtype=torch.float32)
    print("=== float32 -> float32")
    for arm, lv in (("lossless ON (default)", None), ("lossless OFF (Fast)", dict(fp32_lossless=0))):
        x = ttnn.from_torch(
            tf, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        y = _dispatch(x, levers=lv)
        report(arm, tf, ttnn.to_torch(y))
        ttnn.deallocate(x)
        ttnn.deallocate(y)
    print("=== casts")
    tb = torch.randn(shape).bfloat16()
    for src, dst, ref in (
        (ttnn.bfloat16, ttnn.float32, tb.float()),
        (ttnn.float32, ttnn.bfloat16, tf.bfloat16().float()),
        (ttnn.bfloat16, ttnn.bfloat8_b, tb.float()),
        (ttnn.float32, ttnn.bfloat8_b, tf),
    ):
        t = tb if src == ttnn.bfloat16 else tf
        x = ttnn.from_torch(
            t, dtype=src, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        y = _dispatch(x, dtype=dst)
        got = ttnn.to_torch(y).to(torch.float32)
        a, b = ref.flatten(), got.flatten()
        pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        print(
            f"  {str(src).split('.')[-1]:>9s} -> {str(dst).split('.')[-1]:<10s} pcc={pcc:.8f} maxdiff={float((a-b).abs().max()):g}"
        )
        ttnn.deallocate(x)
        ttnn.deallocate(y)
finally:
    ttnn.close_device(dev)
