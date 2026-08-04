import torch, ttnn
from eval.sharding import auto_shard_config
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
_ML = ttnn.TensorMemoryLayout


def cfg(acc=True):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


for shape in [(1, 1, 64, 512), (1, 1, 256, 512)]:
    for gam in (False, True):
        for xname, x in [
            ("ones", torch.ones(*shape, dtype=torch.bfloat16)),
            ("randn", torch.randn(*shape, dtype=torch.bfloat16)),
        ]:
            torch.manual_seed(0)
            mc = auto_shard_config(
                list(shape), _ML.WIDTH_SHARDED, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=dev
            )
            xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
            gd = g = None
            if gam:
                g = torch.randn(shape[-1], dtype=torch.bfloat16)
                gd = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            out = ttnn.to_torch(
                rms_norm(xd, gamma=gd, compute_kernel_config=cfg(), memory_config=xd.memory_config())
            ).float()
            xf = x.float()
            e = xf / torch.sqrt(torch.mean(xf**2, -1, True) + 1e-6)
            if g is not None:
                e = e * g.float()
            a = out.flatten()
            ee = e.flatten()
            pcc = torch.corrcoef(torch.stack([a, ee]))[0, 1].item()
            rms = ((a - ee).pow(2).mean().sqrt() / ee.pow(2).mean().sqrt()).item()
            r = out / e
            print(
                f"RES {shape} gamma={gam} {xname}: pcc={pcc:.5f} rms={rms:.5f} ratio[row0,:4]={r[0,0,0,:4].tolist()} ratio[row0,64:68]={r[0,0,0,64:68].tolist()}",
                flush=True,
            )
ttnn.close_device(dev)
