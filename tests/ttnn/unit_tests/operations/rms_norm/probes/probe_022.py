import torch, ttnn
from ttnn.operations.rms_norm import rms_norm, default_compute_kernel_config
from ttnn.operations.rms_norm.rms_norm_program_descriptor import blocking_plan


def stats(e, a):
    rms = (a - e).pow(2).mean().sqrt().item() / e.std().item()
    pcc = torch.corrcoef(torch.stack([e.flatten().double(), a.flatten().double()]))[0, 1].item()
    return pcc, rms


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    for W in [17, 50, 64, 512]:  # 17/50 -> Regime B (w_non_aligned); 64/512 -> Regime A
        shape = (1, 1, 64, W)
        tx = torch.randn(shape)
        tg = torch.randn(W).reshape(1, 1, 1, W)
        for in_dt, dn in [(ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")]:
            x = ttnn.from_torch(
                tx.to(torch.float32 if in_dt == ttnn.float32 else torch.bfloat16),
                dtype=in_dt,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
            )
            g8 = ttnn.from_torch(tg.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
            # the TRUE gamma the device holds, read back through bf8b
            g_true = ttnn.to_torch(g8).float()[..., :W]
            g_harness = tg.to(torch.bfloat16).float()  # what the golden reference uses
            x32 = ttnn.to_torch(x).float()
            base = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
            p = blocking_plan(x, g8, None, dev, default_compute_kernel_config(), None)
            out = ttnn.to_torch(rms_norm(x, gamma=g8, compute_kernel_config=default_compute_kernel_config())).float()
            ph, rh = stats(base * g_harness, out)
            pt, rt = stats(base * g_true, out)
            print(
                f"RESULT W={W:>4} in={dn} reg={p.regime} | vs bf16-gamma ref (harness): PCC={ph:.6f} rms={rh:.5f}"
                f" | vs TRUE bf8b-gamma ref: PCC={pt:.6f} rms={rt:.5f}"
            )
finally:
    ttnn.close_device(dev)
