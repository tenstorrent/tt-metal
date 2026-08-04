import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

ML = ttnn.TensorMemoryLayout
dev = ttnn.open_device(device_id=0)


def cfg(acc=True, fid=ttnn.MathFidelity.HiFi4):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


def ref(x, g=None, eps=1e-6):
    xf = x.to(torch.float32)
    out = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    if g is not None:
        out = out * g.to(torch.float32).reshape(-1)
    return out


CASES = [
    ((1, 1, 256, 512), ML.BLOCK_SHARDED),
    ((1, 1, 224, 72), ML.BLOCK_SHARDED),
    ((1, 1, 8192, 1024), ML.BLOCK_SHARDED),
    ((1, 1, 32, 4064), ML.WIDTH_SHARDED),
    ((1, 1, 32, 7168), ML.WIDTH_SHARDED),
    ((1, 1, 3232, 96), ML.WIDTH_SHARDED),
    ((1, 1, 17, 50), ML.BLOCK_SHARDED),
    ((1, 1, 40, 40), ML.BLOCK_SHARDED),
]
torch.manual_seed(0)
for shape, ml in CASES:
    try:
        x = torch.randn(*shape, dtype=torch.bfloat16)
        g = torch.randn(shape[-1], dtype=torch.bfloat16)
        mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
        xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        gd = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        out = rms_norm(xd, gamma=gd, compute_kernel_config=cfg(), memory_config=xd.memory_config())
        a = ttnn.to_torch(out).float()
        e = ref(x, g)
        d = a - e
        rms = (d.pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
        pcc = torch.corrcoef(torch.stack([a.flatten(), e.flatten()]))[0, 1].item()
        print(
            f"OK  {shape} {str(ml).split('.')[-1]:15s} relRMS={rms:.5f} pcc={pcc:.7f} shard={list(mc.shard_spec.shape)} nc={mc.shard_spec.grid.num_cores()}"
        )
    except Exception as ex:
        import traceback

        print(f"ERR {shape} {ml}: {type(ex).__name__}: {ex}")
        traceback.print_exc()

ttnn.close_device(dev)
