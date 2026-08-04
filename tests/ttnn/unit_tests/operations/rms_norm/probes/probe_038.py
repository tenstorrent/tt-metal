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


shape = (1, 1, 64, 512)
# all-ones, no gamma: out should be ~1.0 everywhere.  Any deviation is a scale
# error => the cross-core sum is wrong; a per-position deviation => staging.
for name, x in [
    ("ones", torch.ones(*shape, dtype=torch.bfloat16)),
    ("arange", (torch.arange(64 * 512, dtype=torch.float32).reshape(shape) % 7).to(torch.bfloat16)),
]:
    mc = auto_shard_config(
        list(shape), _ML.WIDTH_SHARDED, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=dev
    )
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
    out = ttnn.to_torch(rms_norm(xd, compute_kernel_config=cfg(), memory_config=xd.memory_config())).float()
    xf = x.float()
    e = xf / torch.sqrt(torch.mean(xf**2, -1, True) + 1e-6)
    print(f"RES {name}: got[0,0,0,:16]={out[0,0,0,:16].tolist()}")
    print(f"RES {name}: exp[0,0,0,:16]={e[0,0,0,:16].tolist()}")
    print(f"RES {name}: got[0,0,0,504:]={out[0,0,0,504:].tolist()}")
    print(f"RES {name}: ratio row0 = {(out[0,0,0]/e[0,0,0])[:16].tolist()}")
    print(f"RES {name}: ratio row1 = {(out[0,0,1]/e[0,0,1])[:8].tolist()}")
    print(f"RES {name}: ratio row32 = {(out[0,0,32]/e[0,0,32])[:8].tolist()}")
ttnn.close_device(dev)
