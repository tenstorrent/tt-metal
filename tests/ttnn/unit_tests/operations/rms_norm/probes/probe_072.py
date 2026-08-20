import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
torch.manual_seed(42)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False
# Regime C shapes (masked, >1 row-block/core) plus a Regime B control.
for shape in [(1, 1, 8192, 4095), (1, 1, 8192, 6143), (1, 1, 4096, 4095), (1, 1, 32, 4095)]:
    x = torch.rand(shape, dtype=torch.float32)
    g = torch.rand((1, 1, 1, shape[-1]), dtype=torch.float32)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gt = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    t = ttnn.fill_implicit_tile_padding(t, 1000.0)
    gt = ttnn.fill_implicit_tile_padding(gt, 1000.0)
    o = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, t.memory_config()
    )
    p = pd.blocking_plan(t, gt, o, device, cfg)
    out = ttnn.to_torch(rms_norm(t, gamma=gt, compute_kernel_config=cfg)).to(torch.float32)
    bias = ((out.abs().mean() / ref.abs().mean()) - 1).item() * 100
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    print(f"POISON {shape} regime={p.regime} pcc={pcc:.6f} row-scale-bias={bias:+.4f}%")
ttnn.close_device(device)
