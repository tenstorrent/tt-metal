import torch, ttnn, sys, traceback

sys.path.insert(0, ".")
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def ref(x, g=None, eps=1e-6):
    x = x.float()
    o = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return o * g.float().reshape(-1) if g is not None else o


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


cases = [
    ((1, 1, 256, 512), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ((1, 1, 32, 2048), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ((1, 1, 256, 512), ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ((1, 1, 64, 128), ttnn.TensorMemoryLayout.INTERLEAVED),
]
try:
    for shape, ml in cases:
        try:
            x = torch.randn(shape, dtype=torch.bfloat16)
            g = torch.randn(shape[-1], dtype=torch.bfloat16)
            mc = (
                ttnn.DRAM_MEMORY_CONFIG
                if ml == ttnn.TensorMemoryLayout.INTERLEAVED
                else auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
            )
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            kw = {} if ml == ttnn.TensorMemoryLayout.INTERLEAVED else {"memory_config": tx.memory_config()}
            out = rms_norm(tx, gamma=tg, **kw)
            got = ttnn.to_torch(out)
            exp = ref(x, g)
            print(
                f"RESULT {shape} {str(ml).split('.')[-1]:16s} PCC={pcc(got, exp):.6f} maxdiff={(got.float()-exp).abs().max():.4f}"
            )
        except Exception as e:
            print(f"RESULT {shape} {str(ml).split('.')[-1]:16s} FAILED: {type(e).__name__}: {str(e)[:600]}")
finally:
    ttnn.close_device(device)
