import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
try:

    def run(shape, dtype, gamma=False, fp32_acc=True, eps=1e-6):
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.bfloat16)
        xf = x.float()
        g = None
        tg = None
        if gamma:
            W = shape[-1]
            gt = torch.randn(W, dtype=torch.bfloat16)
            tg = gt.float()
            g = ttnn.from_torch(gt.reshape(1, 1, 1, W), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
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
        ninf = torch.isinf(out).sum().item()
        pcc = torch.corrcoef(torch.stack([out.flatten(), exp.flatten()]))[0, 1].item()
        rrms = ((out - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
        print(f"{str(dtype):18s} g={gamma} fp32acc={fp32_acc} {shape}: PCC={pcc:.5f} relRMS={rrms:.4f} #inf={ninf}")

    # the failing Regime B bf8b cases
    run((128, 512), ttnn.bfloat8_b, gamma=True)
    run((128, 512), ttnn.bfloat8_b, gamma=False)
    run((128, 512), ttnn.bfloat8_b, gamma=True, fp32_acc=False)
    run((32, 4096), ttnn.bfloat8_b, gamma=True)
    run((128, 8192), ttnn.bfloat8_b, gamma=False)
    run((1, 32, 8192), ttnn.bfloat8_b, gamma=False, fp32_acc=False)
    # bf16/fp32 Regime B non-regression
    run((128, 512), ttnn.bfloat16, gamma=True)
    run((32, 4096), ttnn.float32, gamma=True)
finally:
    ttnn.close_device(device)
