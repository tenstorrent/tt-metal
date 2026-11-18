import torch, ttnn
from ttnn.operations.rms_norm.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
for rows, W in [(32, 5120)]:
    for gamma_on in (False, True):
        torch.manual_seed(0)
        x = torch.randn(1, 1, rows, W).bfloat16()
        g = torch.randn(W).bfloat16()
        ref = x.float() / torch.sqrt((x.float() ** 2).mean(-1, keepdim=True) + 1e-6)
        if gamma_on:
            ref = ref * g.float()
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tg = (
            ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            if gamma_on
            else None
        )
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
        )
        o = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg)).float()
        d = o - ref
        print(
            rows,
            W,
            gamma_on,
            "rel rms",
            (d.pow(2).mean().sqrt() / ref.std()).item(),
            "mean ratio",
            (o.abs().mean() / ref.abs().mean()).item(),
        )
ttnn.close_device(device)
