import torch, ttnn
import torch.nn.functional as F
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)


def check(name, shape, target, pv, in_dt, out_dt):
    t = torch.randn(shape).to(torch.bfloat16 if in_dt == ttnn.bfloat16 else torch.float32)
    src = ttnn.from_torch(t, dtype=in_dt, device=dev, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = tilize(src, dtype=out_dt, output_padded_shape=target, pad_value=pv)
    got = ttnn.to_torch(out)
    logical = torch.equal(got.to(t.dtype), t)
    padded = ttnn.to_torch(out.cpu().to_torch_with_padded_shape()) if False else out.cpu().to_torch_with_padded_shape()
    ref = t.to(torch.float32 if out_dt == ttnn.float32 else t.dtype)
    pad = [0, target[-1] - shape[-1], 0, target[-2] - shape[-2]]
    exp = F.pad(ref, pad, value=pv)
    padok = torch.equal(padded.to(exp.dtype), exp)
    print(f"{name}: logical={logical} padded={padok}")
    return logical and padok


ok = True
try:
    ok &= check("focus whole-pad-tiles", [1, 1, 1024, 2048], [1, 1, 2048, 2048], 10.2, ttnn.bfloat16, ttnn.float32)
    ok &= check("mixed tails+padtiles", [1, 1, 1000, 1000], [1, 1, 2048, 2048], 10.2, ttnn.bfloat16, ttnn.float32)
    ok &= check("ragged no pad tile", [1, 1, 50, 50], [1, 1, 64, 64], 10.2, ttnn.bfloat16, ttnn.float32)
    ok &= check("Wtail only", [1, 1, 64, 50], [1, 1, 64, 64], -3.7, ttnn.bfloat16, ttnn.float32)
    ok &= check("Htail only", [1, 1, 50, 64], [1, 1, 64, 64], 10.2, ttnn.bfloat16, ttnn.float32)
    ok &= check("no cast (out_fill off)", [1, 1, 50, 50], [1, 1, 128, 128], 10.2, ttnn.bfloat16, ttnn.bfloat16)
    print("ALL EXACT" if ok else "FAILURE")
finally:
    ttnn.close_device(dev)
