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


# The previously-failing HEIGHT RM shapes + RM regression (mixed gamma dtype -> no alias) + interleaved RM
cases = [
    ((2047, 2047), ML.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((1, 1, 224, 3072), ML.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((1, 1, 352, 2560), ML.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((1, 1, 32, 2848), ML.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((1, 1, 992, 3000), ML.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((7, 224, 3072), ML.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((1, 1, 100, 736), ML.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.float32),
    ((1, 1, 224, 1000), ML.WIDTH_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((1, 1, 256, 512), ML.BLOCK_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((1, 1, 64, 17), None, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((4, 128, 47), None, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
]
try:
    for shape, ml, layout, dt in cases:
        tag = f"RESULT {str(shape):16s} {(str(ml).split('.')[-1] if ml else 'INTERLEAVED'):16s} {str(dt).split('.')[-1]:9s}"
        try:
            tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
            x = torch.randn(shape, dtype=tdt)
            g = torch.randn(shape[-1], dtype=tdt)
            if ml is None:
                mc, kw = ttnn.DRAM_MEMORY_CONFIG, {}
            else:
                mc = auto_shard_config(list(shape), ml, layout=layout, dtype=dt, device=device)
                kw = None
            tx = ttnn.from_torch(x, dtype=dt, layout=layout, device=device, memory_config=mc)
            tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=dt, layout=layout, device=device)
            kw = {} if ml is None else {"memory_config": tx.memory_config()}
            out = rms_norm(tx, gamma=tg, **kw)
            got = ttnn.to_torch(out)
            exp = ref(x, g)
            print(f"{tag} PCC={pcc(got, exp):.6f}")
        except Exception as e:
            print(f"{tag} FAILED {type(e).__name__}: {str(e)[:130]}")
finally:
    ttnn.close_device(device)
