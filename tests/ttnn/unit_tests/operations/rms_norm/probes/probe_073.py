import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi4
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False
shape = (5, 3, 928, 544)
for lay, glay in ((ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT), (ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT)):
    x = torch.rand(shape, dtype=torch.float32)
    g = torch.rand((1, 1, 1, shape[-1]), dtype=torch.float32)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=lay, device=device)
    gt = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=glay, device=device)
    o = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, lay, device, t.memory_config())
    p = pd.blocking_plan(t, gt, o, device, cfg)
    print(
        "PLAN",
        lay,
        "regime",
        p.regime,
        "bht",
        p.BLOCK_HT,
        "rowblk",
        p.num_row_blocks,
        "ws",
        p.WT_SCALE_BLOCK,
        "rg",
        p.RESIDENT_GAMMA,
        "G",
        p.group_size,
    )
    out = ttnn.to_torch(rms_norm(t, gamma=gt, compute_kernel_config=cfg)).to(torch.float32)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g
    print("OK", lay, "pcc", torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item())
ttnn.close_device(device)
