import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as opd


def cfg_loose():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


dev = ttnn.open_device(device_id=0)
try:
    for shape, dt, lay in [
        ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 3071), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 4095), ttnn.float32, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 4095), ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ]:
        torch.manual_seed(0)
        xt = torch.randn(shape, dtype=torch.float32)
        gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
        x = ttnn.from_torch(xt, dtype=dt, layout=lay, device=dev)
        g = ttnn.from_torch(gt, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        c = cfg_loose()
        p = opd.blocking_plan(x, g, x, dev, c, None)
        out = rms_norm(x, gamma=g, compute_kernel_config=cfg_loose())
        ref = xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + 1e-6) * gt.reshape(-1)
        got = ttnn.to_torch(out).float()
        print(
            f"{str(shape):20s} {str(dt):18s} G={p.group_size} reg={p.regime} Wtc={p.Wt_core} wr={p.WT_REDUCE_BLOCK} "
            f"bht={p.BLOCK_HT} rva={p.reduce_via_add} L1={p.working_set_bytes()} pcc={pcc(got, ref):.6f}"
        )
        ttnn.deallocate(out)
        ttnn.deallocate(x)
        ttnn.deallocate(g)
finally:
    ttnn.close_device(dev)
