import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
_ML = ttnn.TensorMemoryLayout


def cfg(acc=True):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


for shape in [(1, 1, 64, 512), (1, 1, 256, 512), (1, 1, 224, 3072), (1, 1, 32, 50), (1, 1, 224, 1000)]:
    for ml in (_ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED):
        for gl in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, None):
            torch.manual_seed(0)
            x = torch.randn(*shape, dtype=torch.bfloat16)
            mc = auto_shard_config(list(shape), ml, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=dev)
            xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
            gd = g = None
            if gl is not None:
                g = torch.randn(shape[-1], dtype=torch.bfloat16)
                gt = g.reshape(1, 1, 1, -1) if gl == ttnn.TILE_LAYOUT else g
                gd = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=gl, device=dev)
            try:
                out = ttnn.to_torch(
                    rms_norm(xd, gamma=gd, compute_kernel_config=cfg(), memory_config=xd.memory_config())
                ).float()
            except Exception as ex:
                print(f"RES {shape} {ml} g={gl}: EXC {type(ex).__name__}: {str(ex)[:120]}", flush=True)
                continue
            xf = x.float()
            e = xf / torch.sqrt(torch.mean(xf**2, -1, True) + 1e-6)
            if g is not None:
                e = e * g.float()
            a = out.flatten()
            ee = e.flatten()
            pcc = torch.corrcoef(torch.stack([a, ee]))[0, 1].item()
            rms = ((a - ee).pow(2).mean().sqrt() / ee.pow(2).mean().sqrt()).item()
            print(
                f"RES {shape} {str(ml).split('.')[-1]:14s} g={str(gl).split('.')[-1]:18s}: pcc={pcc:.6f} rms={rms:.5f}",
                flush=True,
            )
ttnn.close_device(dev)
