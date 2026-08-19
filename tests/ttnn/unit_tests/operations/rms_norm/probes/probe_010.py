import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import blocking_plan


def ref(x, g, eps=1e-6):
    x32 = x.to(torch.float32)
    o = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return o * g.to(torch.float32)


def stats(e, a):
    err = (a - e).abs()
    rms = err.pow(2).mean().sqrt().item() / e.std().item()
    ec = e.flatten().double()
    ac = a.flatten().double()
    pcc = torch.corrcoef(torch.stack([ec, ac]))[0, 1].item()
    return pcc, rms


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    for shape in [(1, 1, 32, 5120), (1, 1, 32, 7168), (1, 1, 160, 11008)]:
        tx = torch.randn(shape).to(torch.bfloat16)
        tg = torch.randn(shape[-1]).to(torch.bfloat16).reshape(1, 1, 1, shape[-1])
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        g = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        e = ref(tx, tg)
        for acc_en in [True, False]:
            cfg = ttnn.ComputeConfigDescriptor()
            cfg.math_fidelity = ttnn.MathFidelity.HiFi2
            cfg.fp32_dest_acc_en = acc_en
            cfg.math_approx_mode = False
            for narrow in [1, 0]:
                lv = dict(acc_narrow=narrow)
                p = blocking_plan(x, g, None, dev, cfg, lv)
                out = ttnn.to_torch(rms_norm(x, gamma=g, compute_kernel_config=cfg, _levers=lv)).float()
                pcc, rms = stats(e, out)
                print(
                    f"RESULT {shape} fp32acc={acc_en} acc_narrow={narrow} regime={p.regime} "
                    f"wr={p.WT_REDUCE_BLOCK} bht={p.BLOCK_HT} dest={p.DEST_BLOCK} acc_dt={p.acc_dtype} "
                    f"PCC={pcc:.6f} rms={rms:.5f}"
                )
finally:
    ttnn.close_device(dev)
