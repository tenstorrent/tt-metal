import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

torch.manual_seed(0)
shape = (1, 1, 256, 512)
W = shape[-1]
x = torch.randn(shape, dtype=torch.bfloat16)
g = torch.randn(W, dtype=torch.bfloat16)

# reference (fp32)
xf = x.to(torch.float32)
rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
expected = (xf / rms) * g.to(torch.float32).reshape(-1)

device = ttnn.open_device(device_id=0)
try:
    mc = auto_shard_config(
        list(shape), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    print("SHARD memory_config:", mc)
    print("shard_spec.shape:", mc.shard_spec.shape, "grid:", mc.shard_spec.grid)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    gt = ttnn.from_torch(
        g.reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=xt.memory_config())
    print("OUT memory_config:", out.memory_config())
    res = ttnn.to_torch(out).reshape(expected.shape).to(torch.float32)
    diff = (res - expected).abs()
    # PCC
    a = res.flatten().double()
    b = expected.flatten().double()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    rms_err = (diff.pow(2).mean().sqrt() / expected.std()).item()
    print(f"max_abs_diff={diff.max().item():.5f}  PCC={pcc:.6f}  relRMS={rms_err:.5f}")
    print("PASS" if (pcc >= 0.995 and rms_err <= 0.04) else "FAIL")
finally:
    ttnn.close_device(device)
