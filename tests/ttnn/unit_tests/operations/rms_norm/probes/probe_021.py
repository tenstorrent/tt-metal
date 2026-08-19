import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import blocking_plan


def cfg_of(acc):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    for W in [1016, 2040, 4088, 5120, 7168, 11008]:
        shape = (1, 1, 32, W)
        tx = torch.randn(shape).to(torch.bfloat16)
        tg = torch.randn(W).to(torch.bfloat16).reshape(1, 1, 1, W)
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        g = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        x32 = tx.float()
        e = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6) * tg.float()
        for acc in [True, False]:
            for via in [1, 0]:
                lv = dict(reduce_via_add=via)
                p = blocking_plan(x, g, None, dev, cfg_of(acc), lv)
                out = ttnn.to_torch(rms_norm(x, gamma=g, compute_kernel_config=cfg_of(acc), _levers=lv)).float()
                rms = (out - e).pow(2).mean().sqrt().item() / e.std().item()
                pcc = torch.corrcoef(torch.stack([e.flatten().double(), out.flatten().double()]))[0, 1].item()
                print(
                    f"RESULT W={W:>5} Wt={p.Wt_core:>3} reg={p.regime} acc={acc!s:<5} via_add={p.reduce_via_add} "
                    f"PCC={pcc:.6f} rms={rms:.5f} {'FAIL' if rms > 0.04 else 'ok'}"
                )
finally:
    ttnn.close_device(dev)
