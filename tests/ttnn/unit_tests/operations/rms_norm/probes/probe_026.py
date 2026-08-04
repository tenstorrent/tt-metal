import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

ML = ttnn.TensorMemoryLayout
dev = ttnn.open_device(device_id=0)


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c


def ref(x, g, eps=1e-6):
    xf = x.to(torch.float32)
    return (xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)) * g.to(torch.float32).reshape(-1)


torch.manual_seed(0)
for lay in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
    for ml in (ML.INTERLEAVED, ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED):
        shape = (1, 1, 224, 3072)
        try:
            x = torch.randn(*shape, dtype=torch.bfloat16)
            g = torch.randn(shape[-1], dtype=torch.bfloat16)
            kw = {}
            if ml != ML.INTERLEAVED:
                mc = auto_shard_config(list(shape), ml, layout=lay, dtype=ttnn.bfloat16, device=dev)
                kw["memory_config"] = mc
            xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=lay, device=dev, **kw)
            gd = ttnn.from_torch(
                g if lay == ttnn.ROW_MAJOR_LAYOUT else g.reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=lay,
                device=dev,
            )
            out = rms_norm(xd, gamma=gd, compute_kernel_config=cfg(), memory_config=xd.memory_config())
            a = ttnn.to_torch(out).float()
            e = ref(x, g)
            pcc = torch.corrcoef(torch.stack([a.flatten(), e.flatten()]))[0, 1].item()
            rms = ((a - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
            sh = list(kw["memory_config"].shard_spec.shape) if kw else None
            print(
                f"OK  {'RM ' if lay==ttnn.ROW_MAJOR_LAYOUT else 'TIL'} {str(ml).split('.')[-1]:15s} pcc={pcc:.6f} rms={rms:.5f} shard={sh}"
            )
        except Exception as ex:
            print(
                f"ERR {'RM ' if lay==ttnn.ROW_MAJOR_LAYOUT else 'TIL'} {str(ml).split('.')[-1]:15s} {type(ex).__name__}: {str(ex)[:120]}"
            )

ttnn.close_device(dev)
