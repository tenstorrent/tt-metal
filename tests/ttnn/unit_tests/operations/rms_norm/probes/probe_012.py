import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return (a @ b / denom).item()


def ref(x, gamma=None, eps=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def cfg(fp32_acc=True):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = fp32_acc
    c.math_approx_mode = False
    return c


def run(name, shape, in_dtype, layout, gamma_dtype=None, gamma_layout=None, fp32_acc=True, min_pcc=0.99):
    torch.manual_seed(0)
    W = shape[-1]
    ti = torch.randn(shape, dtype=torch.bfloat16)
    xi = ttnn.from_torch(ti, dtype=in_dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tg = None
    xg = None
    if gamma_dtype is not None:
        tg = torch.randn(W, dtype=torch.bfloat16)
        xg = ttnn.from_torch(
            tg.reshape(1, 1, 1, W),
            dtype=gamma_dtype,
            layout=gamma_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    exp = ref(ti, gamma=tg)
    try:
        out = rms_norm(xi, gamma=xg, epsilon=1e-6, compute_kernel_config=cfg(fp32_acc))
        act = ttnn.to_torch(out).reshape(exp.shape)
        p = pcc(exp, act)
        print(f"[{name}] shape={shape} PCC={p:.5f} {'PASS' if p>=min_pcc else 'FAIL'}")
    except Exception as e:
        print(f"[{name}] shape={shape} EXCEPTION: {type(e).__name__}: {str(e)[:120]}")


device = ttnn.open_device(device_id=0)
try:
    # bf16 + fp32_dest_acc_en=False (dropped exclusion)
    run("bf16_False", (1, 1, 64, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, fp32_acc=False, min_pcc=0.995)
    # TILE gamma, bf16
    run(
        "TILEgamma_bf16",
        (1, 1, 64, 128),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        min_pcc=0.995,
    )
    # TILE gamma fp32 with bf16 input (mixed)
    run(
        "TILEgamma_fp32mix",
        (1, 1, 64, 128),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        min_pcc=0.995,
    )
    # bf8b input tile-aligned, no gamma
    run("bf8b_nogamma", (1, 1, 64, 128), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, min_pcc=0.99)
    # bf8b input + bf8b TILE gamma
    run(
        "bf8b_bf8bgamma",
        (1, 1, 64, 256),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        min_pcc=0.99,
    )
    # bf8b input + RM bf16 gamma
    run(
        "bf8b_RMbf16gamma",
        (1, 1, 64, 256),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        min_pcc=0.99,
    )
    # bf8b non-aligned W (expect FAIL -> justifies exclusion)
    run("bf8b_wnonalign", (1, 1, 64, 100), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, min_pcc=0.99)
    # bf8b non-aligned H (expect FAIL -> justifies exclusion)
    run("bf8b_hnonalign", (1, 1, 50, 128), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, min_pcc=0.99)
    # RM input + TILE gamma (cross-layout)
    run(
        "RMin_TILEgamma",
        (1, 1, 64, 128),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        min_pcc=0.995,
    )
finally:
    ttnn.close_device(device)
