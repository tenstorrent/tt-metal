import torch, ttnn
import torch.nn.functional as F
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)


def check(name, shape, padded, pad_value, **kw):
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16() if len(shape) else torch.randn(()).bfloat16()
    tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = tilize(tt, pad_value=pad_value, **kw)
    got = out.cpu().to_torch_with_padded_shape().to(torch.float32)
    xe = x.to(torch.float32)
    if xe.dim() < len(padded):
        xe = xe.reshape((1,) * (len(padded) - xe.dim()) + tuple(xe.shape))
    pads = tuple(j for i in reversed(range(xe.dim())) for j in (0, padded[i] - xe.shape[i]))
    exp = F.pad(xe, pads, value=float(pad_value))
    ok = list(got.shape) == list(exp.shape) and torch.equal(got, exp)
    print(f"{name}: shape={list(got.shape)} want={list(exp.shape)} equal={ok} logical={list(out.shape)}")
    if not ok and list(got.shape) == list(exp.shape):
        d = (got - exp).abs()
        idx = (d > 0).nonzero()
        print(
            "   nmismatch",
            idx.shape[0],
            "first",
            idx[:5].tolist(),
            "got",
            got[tuple(idx[0].tolist())].item(),
            "exp",
            exp[tuple(idx[0].tolist())].item(),
        )


try:
    check("auto h_tail", (1, 1, 30, 32), [1, 1, 32, 32], 0.0)
    check("auto w_tail", (1, 1, 32, 50), [1, 1, 32, 64], 3.5)
    check("auto both -", (1, 1, 50, 50), [1, 1, 64, 64], -18.0)
    check("auto noop", (1, 1, 64, 64), [1, 1, 64, 64], 0.0)
    check("explicit tile-rounded", (1, 1, 50, 50), [1, 1, 64, 64], 10.0, output_padded_shape=[1, 1, 64, 64])
    check("explicit pad tiles hw", (1, 1, 50, 50), [1, 1, 128, 128], -18.0, output_padded_shape=[1, 1, 128, 128])
    check("explicit pad tiles w", (1, 1, 32, 50), [1, 1, 32, 128], 3.5, output_padded_shape=[1, 1, 32, 128])
    check("explicit rank3 h", (3, 100, 128), [3, 128, 128], 10.2, output_padded_shape=[3, 128, 128])
    check("explicit rank2", (50, 50), [64, 64], 0.0, output_padded_shape=[64, 64])
    check("scalar", (), [32, 32], 42.0, output_padded_shape=[32, 32])
    check("row vector", (1, 1, 1, 4096), [1, 1, 32, 4096], 0.0)
    check("rank3 auto neg", (2, 32, 50), [2, 32, 64], -32.5)
    check("wide auto", (1, 1, 50, 2048), [1, 1, 64, 2048], 0.0)
finally:
    ttnn.close_device(device)
