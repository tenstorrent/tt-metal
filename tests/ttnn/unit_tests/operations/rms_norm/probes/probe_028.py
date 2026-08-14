import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config


def ref(x, g=None, eps=1e-6):
    xf = x.to(torch.float32)
    o = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if g is not None:
        o = o * g.to(torch.float32).reshape(-1)
    return o


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


_ML = ttnn.TensorMemoryLayout
CASES = [
    ((1, 1, 32, 4064), _ML.HEIGHT_SHARDED),
    ((13, 777, 1023), _ML.BLOCK_SHARDED),
    ((1, 1, 96, 6144), _ML.HEIGHT_SHARDED),
    ((1, 1, 160, 11008), _ML.HEIGHT_SHARDED),
    ((1, 224, 11008), _ML.HEIGHT_SHARDED),
    ((1, 1, 32, 4064), _ML.WIDTH_SHARDED),
    ((1, 1, 256, 512), _ML.HEIGHT_SHARDED),
    ((1, 1, 3232, 96), _ML.BLOCK_SHARDED),
]
res = []
device = ttnn.open_device(device_id=0)
try:
    for shape, ml in CASES:
        x = torch.randn(shape, dtype=torch.bfloat16)
        gm = torch.randn(shape[-1], dtype=torch.bfloat16)
        try:
            mc = auto_shard_config(list(shape), ml, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
            tg = ttnn.from_torch(
                gm.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            out = rms_norm(tx, gamma=tg, memory_config=tx.memory_config())
            p = pcc(ttnn.to_torch(out), ref(x, gm))
            res.append(f"RM {ml} {shape} shard={list(mc.shard_spec.shape)} -> PCC {p:.6f} {'OK' if p>0.99 else 'FAIL'}")
        except Exception as e:
            res.append(f"RM {ml} {shape} -> EXC {type(e).__name__}: {str(e)[:250]}")
        print(res[-1], flush=True)
finally:
    ttnn.close_device(device)
print("=== SUMMARY ===")
for r in res:
    print(r)
