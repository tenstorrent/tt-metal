import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def stats(e, a):
    return (a - e).pow(2).mean().sqrt().item() / e.std().item()


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    for W in [7168, 11008]:
        shape = (1, 1, 32, W)
        tx = torch.randn(shape).to(torch.bfloat16)
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        x32 = tx.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
        gammas = {
            "none": (None, None),
            "ones": (torch.ones(W).to(torch.bfloat16),) * 1 + (None,),
            "randn": (torch.randn(W).to(torch.bfloat16),) * 1 + (None,),
        }
        for gname, (tg, _) in gammas.items():
            if tg is None:
                g, e = None, normed
            else:
                g = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                e = normed * tg.float()
            for acc_en in [True, False]:
                cfg = ttnn.ComputeConfigDescriptor()
                cfg.math_fidelity = ttnn.MathFidelity.HiFi2
                cfg.fp32_dest_acc_en = acc_en
                cfg.math_approx_mode = False
                out = ttnn.to_torch(rms_norm(x, gamma=g, compute_kernel_config=cfg)).float()
                print(f"RESULT W={W:>5} gamma={gname:<5} fp32acc={acc_en!s:<5} rms={stats(e,out):.5f}")
finally:
    ttnn.close_device(dev)
