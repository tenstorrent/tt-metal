import torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize.tilize import tilize

dev = ttnn.open_device(device_id=0)


def run(x, di, do, **kw):
    t = ttnn.from_torch(x, dtype=di, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    o = tilize(t, dtype=do, **kw)
    return ttnn.to_torch(o)


base_u, base_d = pd.UNPACK_TO_DEST_DTYPES, pd.DEST32_DTYPES
for name, (u, d) in {
    "none": (base_u, base_d),
    "d32": (base_u, base_d + (ttnn.uint8,)),
    "u2d": (base_u + (ttnn.uint8,), base_d + (ttnn.uint8,)),
}.items():
    pd.UNPACK_TO_DEST_DTYPES, pd.DEST32_DTYPES = u, d
    for shape in [(1, 1, 32, 64), (1, 1, 64, 128)]:
        x = torch.randint(0, 256, shape, dtype=torch.uint8)
        try:
            y = run(x, ttnn.uint8, ttnn.uint8)
            dd = (y.int() - x.int()).abs()
            print(
                "RES u8",
                name,
                shape,
                "exact" if dd.max() == 0 else f"bad nbad={(dd>0).sum().item()}/{dd.numel()}",
                y[0, 0, 0, :8].tolist(),
                x[0, 0, 0, :8].tolist(),
                y[0, 0, 1, :8].tolist(),
            )
        except Exception as e:
            print("RES err", name, str(e)[:300])
pd.UNPACK_TO_DEST_DTYPES, pd.DEST32_DTYPES = base_u, base_d
for shape in [(1, 1, 32, 32), (1, 1, 32, 64), (1, 1, 64, 64), (1, 1, 64, 128), (2, 3, 96, 1056)]:
    xf = torch.randn(shape).to(torch.float8_e4m3fn).to(torch.float32)
    for do in [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b]:
        try:
            y = run(xf, ttnn.fp8_e4m3, do).float()
            a = xf.double().flatten()
            b = y.double().flatten()
            print(
                "RES fp8",
                shape,
                do,
                "exact"
                if torch.equal(a, b)
                else f"pcc={torch.corrcoef(torch.stack([a,b]))[0,1].item():.5f} nbad={(a!=b).sum().item()}",
            )
        except Exception as e:
            print("RES fp8 err", do, str(e)[:300])
ttnn.close_device(dev)
