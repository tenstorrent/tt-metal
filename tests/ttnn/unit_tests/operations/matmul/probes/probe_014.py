import torch, ttnn
from ttnn.operations.matmul import matmul


def rel_rms(actual, expected):
    expected_f = expected.float()
    actual_f = actual.float()
    abs_rms = torch.nn.functional.mse_loss(actual_f, expected_f).sqrt().item()
    scale = expected_f.std().item()
    return abs_rms / scale if scale > 1e-12 else abs_rms


def pcc(actual, expected):
    a = actual.float().flatten()
    e = expected.float().flatten()
    a = a - a.mean()
    e = e - e.mean()
    return (a @ e / (a.norm() * e.norm())).item()


device = ttnn.open_device(device_id=0)
try:
    M, K, N = 256, 8192, 2048
    for wdt, wname in [(ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")]:
        torch.manual_seed(0)
        A = torch.randn((M, K), dtype=torch.bfloat16)
        B = torch.randn((K, N), dtype=(torch.bfloat16 if wdt == ttnn.bfloat16 else torch.float32))
        expected = torch.matmul(A.float(), B.float()).to(torch.bfloat16)
        tA = ttnn.from_torch(A, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tB = ttnn.from_torch(B, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
        )
        out = matmul(tA, tB, compute_kernel_config=cfg)
        res = ttnn.to_torch(out).to(torch.bfloat16)
        print(
            f"act=bf16 weight={wname} acc=False : relRMS={rel_rms(res, expected):.4f}  PCC={pcc(res, expected):.5f}  (band relRMS<=0.10)"
        )
finally:
    ttnn.close_device(device)
