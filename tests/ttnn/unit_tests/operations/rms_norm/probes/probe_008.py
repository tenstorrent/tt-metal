import torch, ttnn
from ttnn.operations.rms_norm import rms_norm, validate

device = ttnn.open_device(device_id=0)
try:
    # Exact perf-1 anchor config: bf16 / fp32_dest_acc_en=False / TILE input /
    # TILE gamma / INTERLEAVED / HiFi2 (feature_spec _PERF_BASE).
    torch.manual_seed(0)
    shape = (1, 1, 128, 2304)  # prefill-ish; Gemma 2 2B hidden
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.float32)
    g = torch.randn(W, dtype=torch.float32)

    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    gt = ttnn.from_torch(
        g.reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False

    # validate() must NOT raise for this config (it is now supported).
    validate(xt, gamma=gt, compute_kernel_config=cfg)
    print("VALIDATE: perf-1 anchor config accepted")

    out = rms_norm(xt, gamma=gt, compute_kernel_config=cfg)
    res = ttnn.to_torch(out).to(torch.float32)

    ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    ref = ref * g.reshape(-1)

    a, b = res.flatten(), ref.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    fro = (a - b).norm().item() / b.norm().item()
    print(
        f"perf-1 anchor (TILE in + TILE gamma, fp32_acc=False, HiFi2): PCC={pcc:.6f} soft-gate>=0.9995 -> {'PASS' if pcc>=0.9995 else 'FAIL'}; rel-Frob={fro:.4f}"
    )
finally:
    ttnn.close_device(device)
