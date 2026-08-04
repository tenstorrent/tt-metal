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


def ref(x, g=None):
    xf = x.to(torch.float32)
    o = xf / torch.sqrt(torch.mean(xf**2, -1, True) + 1e-6)
    return o * g.to(torch.float32).reshape(-1) if g is not None else o


for shape, ml in [((1, 1, 256, 512), _ML.WIDTH_SHARDED)]:
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    g = torch.randn(shape[-1], dtype=torch.bfloat16)
    mc = auto_shard_config(list(shape), ml, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
    gd = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    out = rms_norm(xd, gamma=gd, compute_kernel_config=cfg(), memory_config=xd.memory_config())
    a = ttnn.to_torch(out).float().flatten()
    e = ref(x, g).flatten()
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
    rms = ((a - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    print(f"RESULT {shape} {ml} pcc={pcc:.6f} rms={rms:.5f}", flush=True)
ttnn.close_device(dev)
