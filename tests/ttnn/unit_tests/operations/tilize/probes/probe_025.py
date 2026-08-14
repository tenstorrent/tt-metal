import torch, ttnn, torch.nn.functional as F
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
CMP = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.uint8: torch.uint8,
    ttnn.uint32: torch.int32,
    ttnn.uint16: torch.int32,
    ttnn.int32: torch.int32,
}


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.equal(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def run(shape, dt, odt, padded, pv, mode="explicit", mc=True):
    if dt == ttnn.uint8:
        x = torch.randint(0, 200, shape, dtype=torch.uint8)
    elif dt in (ttnn.uint32, ttnn.uint16, ttnn.int32):
        x = torch.randint(0, 100, shape, dtype=torch.int32)
    elif dt == ttnn.float32:
        x = torch.randn(shape, dtype=torch.float32)
    else:
        x = torch.randn(shape).bfloat16()
    t = ttnn.from_torch(x, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kw = dict(pad_value=pv)
    if mode == "explicit":
        kw["output_padded_shape"] = padded
    o = tilize(t, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=odt, use_multicore=mc, **kw)
    cmp = CMP[odt]
    got = o.cpu().to_torch_with_padded_shape().to(cmp)
    xx = x.to(cmp)
    if xx.dim() < len(padded):
        xx = xx.reshape((1,) * (len(padded) - xx.dim()) + tuple(xx.shape))
    pads = tuple(j for i in reversed(range(xx.dim())) for j in (0, padded[i] - xx.shape[i]))
    exp = F.pad(xx, pads, value=pv)
    eq = torch.equal(got, exp)
    p = pcc(got, exp)
    tag = f"{shape}->{padded} {str(dt).split('.')[-1]}->{str(odt).split('.')[-1]} pv={pv} mc={mc}"
    print(("OK  " if eq else "DIFF"), tag, f"exact={eq} pcc={p:.6f}")
    return eq, p


try:
    run([1, 1, 50, 50], ttnn.bfloat16, ttnn.bfloat8_b, [1, 1, 64, 64], 0.0)
    run([1, 1, 50, 50], ttnn.bfloat16, ttnn.bfloat8_b, [1, 1, 64, 64], 10.2)
    run([1, 1, 50, 50], ttnn.bfloat16, ttnn.bfloat8_b, [1, 1, 128, 128], -18.0)
    run([1, 1, 50, 50], ttnn.float32, ttnn.bfloat8_b, [1, 1, 64, 64], 10.2)
    for mc in (True, False):
        run([1, 1, 50, 50], ttnn.bfloat16, ttnn.float32, [1, 1, 64, 64], 10.2, mc=mc)
    run([1, 1, 50, 50], ttnn.bfloat16, ttnn.float32, [1, 1, 128, 128], -18.3)
    run([1, 1, 32, 50], ttnn.bfloat16, ttnn.float32, [1, 1, 32, 128], 3.5)
    run([3, 100, 128], ttnn.bfloat16, ttnn.float32, [3, 128, 128], 10.2)
    run([2, 32, 50], ttnn.bfloat16, ttnn.float32, [2, 32, 64], -32.5, mode="auto")
    run([], ttnn.bfloat16, ttnn.float32, [32, 32], 42.3)
    run([1, 1, 50, 50], ttnn.bfloat16, ttnn.bfloat16, [1, 1, 64, 64], 10.2)
    run([1, 1, 50, 50], ttnn.float32, ttnn.bfloat16, [1, 1, 64, 64], 10.2)
    run([1, 1, 50, 50], ttnn.float32, ttnn.float32, [1, 1, 64, 64], 10.2)
finally:
    ttnn.close_device(dev)
