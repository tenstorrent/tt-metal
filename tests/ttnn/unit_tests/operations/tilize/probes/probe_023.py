import torch, ttnn, torch.nn.functional as F
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)


def run(shape, dt, odt, padded, pv, tdt_in, mc=True):
    if tdt_in == torch.uint8:
        x = torch.randint(0, 200, shape, dtype=torch.uint8)
    elif tdt_in == torch.int32:
        x = torch.randint(0, 100, shape, dtype=torch.int32)
    elif tdt_in == torch.float32:
        x = torch.randn(shape, dtype=torch.float32)
    else:
        x = torch.randn(shape).bfloat16()
    t = ttnn.from_torch(x, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    o = tilize(
        t, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=odt, use_multicore=mc, output_padded_shape=padded, pad_value=pv
    )
    got = o.cpu().to_torch_with_padded_shape()
    cmp = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.bfloat8_b: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.uint8: torch.uint8,
        ttnn.uint32: torch.int32,
        ttnn.int32: torch.int32,
    }[odt]
    xx = x.to(cmp)
    if xx.dim() < len(padded):
        xx = xx.reshape((1,) * (len(padded) - xx.dim()) + tuple(xx.shape))
    pads = tuple(j for i in reversed(range(xx.dim())) for j in (0, padded[i] - xx.shape[i]))
    exp = F.pad(xx, pads, value=pv)
    same = torch.equal(got, exp)
    tag = f"{shape}->{padded} {dt}->{odt} pv={pv}"
    if same:
        print("OK  ", tag)
    else:
        d = (got.float() - exp.float()).abs()
        print("FAIL", tag, "maxdiff", d.max().item(), "nwrong", int((d > 0).sum()), "/", d.numel())
    return same


try:
    # rank 0 scalar
    for dt, tdt in [
        (ttnn.bfloat16, torch.bfloat16),
        (ttnn.float32, torch.float32),
        (ttnn.uint32, torch.int32),
        (ttnn.uint8, torch.uint8),
    ]:
        try:
            run([], dt, dt, [32, 32], 42.0 if tdt != torch.uint8 else 42, tdt)
        except Exception as e:
            print("EXC rank0", dt, type(e).__name__, e)
    # uint8 / int padded
    for dt, tdt in [(ttnn.uint8, torch.uint8), (ttnn.uint32, torch.int32)]:
        try:
            run([1, 1, 50, 50], dt, dt, [1, 1, 64, 64], 10, tdt)
            run([1, 1, 50, 50], dt, dt, [1, 1, 128, 128], 7, tdt)
        except Exception as e:
            print("EXC int-pad", dt, type(e).__name__, e)
    # bf8b output + pad
    for pv in (0.0, 10.2):
        try:
            run([1, 1, 50, 50], ttnn.bfloat16, ttnn.bfloat8_b, [1, 1, 64, 64], pv, torch.bfloat16)
        except Exception as e:
            print("EXC bf8b-pad", type(e).__name__, e)
    # bf16 -> fp32 widening + inexact pad
    for pv in (10.2, -18.0):
        try:
            run([1, 1, 50, 50], ttnn.bfloat16, ttnn.float32, [1, 1, 64, 64], pv, torch.bfloat16)
        except Exception as e:
            print("EXC widen-pad", type(e).__name__, e)
finally:
    ttnn.close_device(dev)
