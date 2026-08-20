import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
torch.manual_seed(42)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False
cases = [
    ((1, 1, 8192, 4095), ttnn.bfloat16),
    ((1, 1, 32, 4095), ttnn.bfloat16),
    ((1, 1, 8192, 6143), ttnn.bfloat16),
    ((1, 1, 32, 7168), ttnn.bfloat16),
    ((1, 1, 8192, 1024), ttnn.bfloat16),
    ((1, 1, 100, 736), ttnn.bfloat16),
]
for shape, dt in cases:
    x = torch.rand(shape, dtype=torch.float32)
    g = torch.rand((1, 1, 1, shape[-1]), dtype=torch.float32)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g
    t = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    gt = ttnn.from_torch(g, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, device, t.memory_config())
    p = pd.blocking_plan(t, gt, o, device, cfg)
    out = ttnn.to_torch(rms_norm(t, gamma=gt, compute_kernel_config=cfg)).to(torch.float32)
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    print(
        f"PLAN {shape} regime={p.regime} bht={p.BLOCK_HT} ws={p.WT_SCALE_BLOCK} rg={p.RESIDENT_GAMMA} G={p.group_size} pcc={pcc:.6f} scale={(out.abs().mean()/ref.abs().mean()).item():.5f}"
    )
ttnn.close_device(device)
