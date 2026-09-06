import os, torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 32, 4064), (1, 1, 3104, 4064)]:
        for lay in [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]:
            for mode in ["none", "gamma_bias_residual"]:
                torch.manual_seed(0)
                t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
                x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=lay, device=device)
                kw = {}
                ref = {"input_tensor": t.float()}
                if mode != "none":
                    g = torch.randn(1, 1, 1, shape[-1], dtype=torch.float32).to(torch.bfloat16)
                    b = torch.randn(1, 1, 1, shape[-1], dtype=torch.float32).to(torch.bfloat16)
                    r = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
                    kw["weight"] = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=lay, device=device)
                    kw["bias"] = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=lay, device=device)
                    kw["residual_input_tensor"] = ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=lay, device=device)
                    ref.update(weight=g.float(), bias=b.float(), residual_input_tensor=r.float())
                print("CASE", shape, lay, mode, flush=True)
                out = rms_norm_ttnn(x, epsilon=1e-12, **kw)
                got = ttnn.to_torch(out)
                exp = torch_rms_norm_ttnn(**ref, epsilon=1e-12)
                print("PCCRESULT", shape, str(lay), mode, "pcc=%.6f" % pcc(got, exp), flush=True)
finally:
    ttnn.close_device(device)
