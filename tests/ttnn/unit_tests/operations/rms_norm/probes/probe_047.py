import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod
from ttnn.operations.rms_norm import rms_norm

captured = []
_orig = pdmod.ttnn.KernelDescriptor


def spy(**kw):
    if "compute" in str(kw.get("kernel_source", "")):
        captured.append(list(kw.get("compile_time_args", [])))
    return _orig(**kw)


pdmod.ttnn.KernelDescriptor = spy


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


def torch_rms(x, g, eps=1e-6):
    xf = x.to(torch.float32)
    r = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    return (xf / r) * g.to(torch.float32).reshape(-1)


device = ttnn.open_device(device_id=0)
try:
    for shape, lay, glay in [
        ((1, 1, 8192, 5120), ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
        ((1, 1, 8192, 7168), ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 5120), ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 96, 6144), ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
        ((1, 1, 64, 4064), ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
        ((1, 1, 64, 5120), ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 64, 8192), ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
    ]:
        torch.manual_seed(0)
        W = shape[-1]
        tx = torch.randn(shape, dtype=torch.bfloat16)
        tg = torch.randn(W, dtype=torch.bfloat16)
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=lay, device=device)
        g = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=glay, device=device)
        captured.clear()
        out = rms_norm(x, gamma=g, compute_kernel_config=cfg())
        ct = captured[-1]
        wtc, nwc, br, xres = ct[1], ct[2], ct[3], ct[15]
        regime = "RESIDENT" if nwc == 1 else ("ROW_RESIDENT" if xres else "STREAM")
        got = ttnn.to_torch(out).float()
        ref = torch_rms(tx, tg)
        err = got - ref
        rms = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        num = (got * ref).sum()
        pcc = (num / (got.norm() * ref.norm())).item()
        print(
            f"SHAPE {shape} lay={'T' if lay==ttnn.TILE_LAYOUT else 'RM'} g={'T' if glay==ttnn.TILE_LAYOUT else 'RM'} Wt={(W+31)//32} {regime} wtc={wtc} nwc={nwc} br={br} pcc={pcc:.6f} rms={rms:.5f}"
        )
        ttnn.deallocate(x)
        ttnn.deallocate(g)
        ttnn.deallocate(out)
finally:
    ttnn.close_device(device)
