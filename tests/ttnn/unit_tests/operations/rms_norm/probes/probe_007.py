import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def run(shape, dtype, gamma_dtype=None, fp32_acc=True, eps=1e-6):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16 if dtype != ttnn.float32 else torch.float32)
    xf = x.float()
    g = None
    tg = None
    if gamma_dtype is not None:
        W = shape[-1]
        gt = torch.randn(W, dtype=torch.bfloat16 if gamma_dtype != ttnn.float32 else torch.float32)
        tg = gt.float()
        g = ttnn.from_torch(gt.reshape(1, 1, 1, W), dtype=gamma_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    exp = xf / rms
    if tg is not None:
        exp = exp * tg.reshape(-1)
    ti = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_acc
    cfg.math_approx_mode = False
    out = ttnn.to_torch(rms_norm(ti, gamma=g, epsilon=eps, compute_kernel_config=cfg)).float()
    pcc = torch.corrcoef(torch.stack([out.flatten(), exp.flatten()]))[0, 1].item()
    rrms = ((out - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
    print(
        f"{str(dtype):20s} g={str(gamma_dtype):14s} fp32acc={fp32_acc} shape={shape}: PCC={pcc:.5f} relRMS={rrms:.4f}"
    )


run((1, 1, 32, 64), ttnn.float32)
run((1, 1, 32, 64), ttnn.float32, gamma_dtype=ttnn.float32)
run((1, 1, 64, 128), ttnn.float32, gamma_dtype=ttnn.bfloat16)
run((1, 1, 32, 64), ttnn.bfloat16, gamma_dtype=ttnn.float32)
run((1, 1, 32, 64), ttnn.bfloat8_b)
run((1, 1, 64, 128), ttnn.bfloat8_b, gamma_dtype=ttnn.bfloat8_b)
run((1, 1, 32, 64), ttnn.bfloat16, fp32_acc=False)
run((1, 1, 32, 64), ttnn.bfloat8_b, fp32_acc=False)
# excluded:
try:
    run((1, 1, 32, 64), ttnn.float32, fp32_acc=False)
    print("ERROR: fp32+False not excluded")
except NotImplementedError as e:
    print("OK fp32+False excluded:", type(e).__name__)
