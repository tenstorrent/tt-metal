import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = device  # harness injects `device` at module scope


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def ref(x, g, eps):
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if g is not None:
        out = out * g.float().reshape(-1)
    return out


def run(name, dtype, fp32_acc, gamma_dtype=None, gamma_layout=ttnn.ROW_MAJOR_LAYOUT, mf=ttnn.MathFidelity.HiFi4):
    torch.manual_seed(0)
    shape = (1, 1, 64, 256)  # tile-aligned
    W = shape[-1]
    ti = torch.randn(shape, dtype=torch.bfloat16)
    xi = ttnn.from_torch(ti, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g = None
    tg = None
    if gamma_dtype is not None:
        tg = torch.randn(W, dtype=torch.bfloat16)
        g = ttnn.from_torch(
            tg.reshape(1, 1, 1, W),
            dtype=gamma_dtype,
            layout=gamma_layout,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=mf, fp32_dest_acc_en=fp32_acc)
    out = rms_norm(xi, gamma=g, epsilon=1e-6, compute_kernel_config=cfg)
    got = ttnn.to_torch(out).float()
    exp = ref(ti, tg, 1e-6)
    p = pcc(got, exp)
    rms = ((got - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
    ok = "PASS" if (p >= 0.99 and rms <= 0.10) else "FAIL"
    print(f"[{name}] PCC={p:.5f} relRMS={rms:.5f} layout_ok={out.layout==ttnn.TILE_LAYOUT} -> {ok}")


run("bf8b/fp32acc/no_gamma", ttnn.bfloat8_b, True)
run("bf8b/fp32acc/gamma_bf16_rm", ttnn.bfloat8_b, True, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
run("bf16/bf16acc(False)/no_gamma", ttnn.bfloat16, False)
run(
    "bf16/bf16acc(False)/gamma_bf16_rm/HiFi2",
    ttnn.bfloat16,
    False,
    ttnn.bfloat16,
    ttnn.ROW_MAJOR_LAYOUT,
    ttnn.MathFidelity.HiFi2,
)
run("bf8b/bf16acc(False)/no_gamma", ttnn.bfloat8_b, False)
print("ALL PROBES DONE")
