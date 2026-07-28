import torch, ttnn, traceback
from ttnn.operations.rms_norm import rms_norm
import sys

sys.path.insert(0, "eval")
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)


def ref(x, g, eps=1e-6):
    xf = x.float()
    r = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    out = xf / r
    if g is not None:
        out = out * g.float().reshape(-1)
    return out


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


ML = ttnn.TensorMemoryLayout
cases = [
    ((1, 1, 32, 2048), ML.WIDTH_SHARDED),
    ((1, 1, 32, 64), ML.WIDTH_SHARDED),
    ((1, 1, 256, 512), ML.BLOCK_SHARDED),
    ((1, 1, 64, 128), ML.BLOCK_SHARDED),
]
for shape, ml in cases:
    try:
        mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        print(f"--- {shape} {ml} shard={mc.shard_spec.shape} grid={mc.shard_spec.grid}")
        t = torch.randn(*shape, dtype=torch.bfloat16)
        g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
        ti = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        gi = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        o = ttnn.to_torch(rms_norm(ti, gamma=gi, memory_config=ti.memory_config())).float()
        e = ref(t, g)
        print(f"    pcc={pcc(o,e):.6f} maxdiff={(o-e).abs().max().item():.4f}")
    except Exception as ex:
        print("    FAIL:", type(ex).__name__, str(ex)[:400])
        traceback.print_exc()
ttnn.close_device(device)
