import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def stats(e, a):
    return (a - e).pow(2).mean().sqrt().item() / e.std().item()


def cfg_of(acc):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    # --- 1. DEST_BLOCK sensitivity of the gamma multiply (gamma == exactly 1.0)
    W = 7168
    tx = torch.randn((1, 1, 32, W)).to(torch.bfloat16)
    x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    x32 = tx.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
    g = ttnn.from_torch(
        torch.ones(1, 1, 1, W).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
    )
    for acc in [True, False]:
        for db in [0, 8, 4, 2, 1]:
            lv = dict(dest_block=db) if db else None
            out = ttnn.to_torch(rms_norm(x, gamma=g, compute_kernel_config=cfg_of(acc), _levers=lv)).float()
            print(f"RESULT destblk W={W} fp32acc={acc!s:<5} dest_block={db or 'solver'} rms={stats(normed,out):.5f}")
    # --- 2. accumulator-magnitude test: rms_norm is scale invariant, so any
    #        dependence on the input scale is the bf16 accumulator's magnitude
    for W2 in [7168, 11008]:
        base = torch.randn((1, 1, 32, W2))
        for scale in [0.25, 1.0, 4.0]:
            t = (base * scale).to(torch.bfloat16)
            xx = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            t32 = t.float()
            e = t32 * torch.rsqrt(t32.pow(2).mean(-1, keepdim=True) + 1e-6)
            sumsq = t32.pow(2).sum(-1).mean().item()
            out = ttnn.to_torch(rms_norm(xx, compute_kernel_config=cfg_of(False))).float()
            print(f"RESULT scale W={W2} in_scale={scale} sumsq~{sumsq:.0f} rms={stats(e,out):.5f}")
finally:
    ttnn.close_device(dev)
