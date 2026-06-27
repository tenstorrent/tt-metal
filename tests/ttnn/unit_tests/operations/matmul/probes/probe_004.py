import torch, ttnn
from ttnn.operations.matmul import matmul
import ttnn.operations.matmul.matmul_program_descriptor as pd

device = ttnn.open_device(device_id=0)


def pcc(g, c):
    a = g.flatten().double()
    b = c.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def relrms(g, c):
    return float(((c - g).pow(2).mean().sqrt()) / (g.std() + 1e-12))


def run(A, B, dt, wdt, cfg):
    a = ttnn.from_torch(A, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(B, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(matmul(a, b, compute_kernel_config=cfg)).float()


try:
    torch.manual_seed(0)
    A = torch.randn(256, 2048, dtype=torch.bfloat16)
    B = torch.randn(2048, 512, dtype=torch.bfloat16)
    exp = A.float() @ B.float()

    cfg_acc = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    cfg_noacc = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
    )

    got = run(A, B, ttnn.bfloat16, ttnn.bfloat16, cfg_acc)
    print(f"[bf16 HiFi4->clamp acc=True]   PCC={pcc(exp,got):.6f} relRMS={relrms(exp,got):.5f}")

    orig = pd._effective_compute_config
    pd._effective_compute_config = lambda user, fp32_acc, low: ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_acc, math_approx_mode=False
    )
    got_h4 = run(A, B, ttnn.bfloat16, ttnn.bfloat16, cfg_acc)
    print(f"[bf16 forced-HiFi4 acc=True]   PCC={pcc(exp,got_h4):.6f} relRMS={relrms(exp,got_h4):.5f}")
    pd._effective_compute_config = orig

    got_naf = run(A, B, ttnn.bfloat16, ttnn.bfloat16, cfg_noacc)
    print(f"[bf16 HiFi4->clamp acc=False]  PCC={pcc(exp,got_naf):.6f} relRMS={relrms(exp,got_naf):.5f}")

    got_bf8 = run(A, B, ttnn.bfloat8_b, ttnn.bfloat8_b, cfg_acc)
    print(f"[bf8b HiFi4->clamp acc=True]   PCC={pcc(exp,got_bf8):.6f} relRMS={relrms(exp,got_bf8):.5f}")

    got_bf8_naf = run(A, B, ttnn.bfloat8_b, ttnn.bfloat8_b, cfg_noacc)
    print(f"[bf8b HiFi4->clamp acc=False]  PCC={pcc(exp,got_bf8_naf):.6f} relRMS={relrms(exp,got_bf8_naf):.5f}")

    got_mix = run(A, B, ttnn.bfloat16, ttnn.float32, cfg_acc)
    print(f"[bf16act/fp32wt clamp acc=True] PCC={pcc(exp,got_mix):.6f} relRMS={relrms(exp,got_mix):.5f}")
finally:
    ttnn.close_device(device)
