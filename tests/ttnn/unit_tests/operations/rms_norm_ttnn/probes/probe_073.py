import torch, ttnn
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
            for mode in ["none", "gamma", "gamma_bias_residual", "residual"]:
                torch.manual_seed(0)
                t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
                x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=lay, device=device)
                kw = {}
                ref = {"input_tensor": t.float()}
                if "gamma" in mode:
                    g = torch.randn(1, 1, 1, shape[-1], dtype=torch.float32).to(torch.bfloat16)
                    kw["weight"] = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=lay, device=device)
                    ref["weight"] = g.float()
                if "bias" in mode:
                    b = torch.randn(1, 1, 1, shape[-1], dtype=torch.float32).to(torch.bfloat16)
                    kw["bias"] = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=lay, device=device)
                    ref["bias"] = b.float()
                if "residual" in mode:
                    r = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
                    kw["residual_input_tensor"] = ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=lay, device=device)
                    ref["residual_input_tensor"] = r.float()
                print("CASE", shape, lay, mode, flush=True)
                out = rms_norm_ttnn(x, epsilon=1e-12, **kw)
                print(
                    "PCCRESULT %s %s %s pcc=%.6f"
                    % (shape, lay, mode, pcc(ttnn.to_torch(out), torch_rms_norm_ttnn(**ref, epsilon=1e-12))),
                    flush=True,
                )
finally:
    ttnn.close_device(device)
