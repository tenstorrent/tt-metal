import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
ML = ttnn.TensorMemoryLayout


def ref(x, g=None, eps=1e-6):
    x = x.float()
    o = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return o * g.float().reshape(-1) if g is not None else o


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


cases = []
for shape in [(1, 1, 32, 4096), (1, 1, 32, 8192), (128, 8192), (1, 1, 160, 11008)]:
    for ml in (ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED):
        for layout in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
            cases.append((shape, ml, layout))
try:
    for shape, ml, layout in cases:
        tag = f"RESULT {str(shape):18s} {str(ml).split('.')[-1]:16s} {str(layout).split('.')[-1]:10s}"
        try:
            x = torch.randn(shape, dtype=torch.bfloat16)
            g = torch.randn(shape[-1], dtype=torch.bfloat16)
            mc = auto_shard_config(list(shape), ml, layout=layout, dtype=ttnn.bfloat16, device=device)
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc)
            tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=layout, device=device)
            out = rms_norm(tx, gamma=tg, memory_config=tx.memory_config())
            got = ttnn.to_torch(out)
            exp = ref(x, g)
            print(
                f"{tag} PCC={pcc(got, exp):.6f} shard={list(mc.shard_spec.shape)} ncores={mc.shard_spec.grid.num_cores()}"
            )
        except Exception as e:
            print(f"{tag} FAILED {type(e).__name__}: {str(e)[:170]}")
finally:
    ttnn.close_device(device)
