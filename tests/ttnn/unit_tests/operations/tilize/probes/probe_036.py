# R7 probe H: the output-format pad rewrite. bf16 -> fp32 with a fill bf16
# cannot hold (10.2), against the golden oracle's own reference; plus the arms
# that must NOT change (same dtype, narrowing, representable fill, uint8).
import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch


def check(tag, dev, shape, padded, dtype, out_dtype, value, torch_dtype):
    torch.manual_seed(1)
    if torch_dtype in (torch.uint8, torch.int32):
        t = torch.randint(0, 100, shape, dtype=torch.int32).to(torch_dtype)
    else:
        t = torch.randn(shape).to(torch_dtype)
    x = ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y = _dispatch(x, dtype=out_dtype, output_padded_shape=padded, pad_value=value)
    got = y.cpu().to_torch_with_padded_shape().to(torch.float32)
    cmp_t = t.to(torch.float32)
    pads = tuple(j for i in reversed(range(len(shape))) for j in (0, padded[i] - shape[i]))
    exp = torch.nn.functional.pad(cmp_t, pads, value=float(value))
    print(
        f"  {tag:34s} mismatches={int((got != exp).sum()):6d}/{exp.numel()}  maxdiff={float((got-exp).abs().max()):g}"
    )
    ttnn.deallocate(x)
    ttnn.deallocate(y)


dev = ttnn.open_device(device_id=0)
try:
    print("=== widening cast, fill NOT bf16-representable (the R7 case)")
    check(
        "bf16->fp32 pad 10.2 (h tail)",
        dev,
        (3, 100, 128),
        [3, 128, 128],
        ttnn.bfloat16,
        ttnn.float32,
        10.2,
        torch.bfloat16,
    )
    check(
        "bf16->fp32 pad 10.2 (hw tail)",
        dev,
        (1, 50, 50),
        [1, 128, 128],
        ttnn.bfloat16,
        ttnn.float32,
        10.2,
        torch.bfloat16,
    )
    check(
        "bf16->fp32 pad -18.3 (w tail)",
        dev,
        (1, 64, 100),
        [1, 64, 128],
        ttnn.bfloat16,
        ttnn.float32,
        -18.3,
        torch.bfloat16,
    )
    print("=== arms that must be unchanged")
    check("bf16->bf16 pad 10.2", dev, (3, 100, 128), [3, 128, 128], ttnn.bfloat16, ttnn.bfloat16, 10.2, torch.bfloat16)
    check("fp32->fp32 pad 10.2", dev, (3, 100, 128), [3, 128, 128], ttnn.float32, ttnn.float32, 10.2, torch.float32)
    check(
        "bf16->fp32 pad 10.0 (exact)",
        dev,
        (3, 100, 128),
        [3, 128, 128],
        ttnn.bfloat16,
        ttnn.float32,
        10.0,
        torch.bfloat16,
    )
    check("uint8->uint8 pad 7", dev, (1, 30, 32), [1, 32, 32], ttnn.uint8, ttnn.uint8, 7, torch.uint8)
finally:
    ttnn.close_device(dev)
