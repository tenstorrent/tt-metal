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


MLS = [
    ("H", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ("W", ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ("B", ttnn.TensorMemoryLayout.BLOCK_SHARDED),
]
SHAPES = [
    (1, 1, 256, 512),
    (1, 1, 224, 3072),
    (1, 1, 416, 1184),
    (1, 1, 3232, 96),
    (1, 1, 32, 4064),
    (1, 1, 224, 1000),
    (1, 1, 333, 1000),
    (7136, 736),
]

device = ttnn.open_device(device_id=0)
try:
    for lay_name, lay in (("TILE", ttnn.TILE_LAYOUT), ("RM", ttnn.ROW_MAJOR_LAYOUT)):
        for shape in SHAPES:
            for tag, ml in MLS:
                x = torch.randn(shape, dtype=torch.bfloat16)
                gm = torch.randn(shape[-1], dtype=torch.bfloat16)
                try:
                    mc = auto_shard_config(list(shape), ml, layout=lay, dtype=ttnn.bfloat16, device=device)
                    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=mc)
                    tg = ttnn.from_torch(gm.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=lay, device=device)
                    out = rms_norm(tx, gamma=tg, memory_config=tx.memory_config())
                    got = ttnn.to_torch(out)
                    p = pcc(got, ref(x, gm))
                    print(
                        f"CASE {lay_name} {tag} {shape} shard={list(mc.shard_spec.shape)} -> PCC {p:.6f} {'OK' if p>0.99 else 'FAIL'}"
                    )
                except Exception as e:
                    print(f"CASE {lay_name} {tag} {shape} -> EXC {type(e).__name__}: {str(e)[:200]}")
finally:
    ttnn.close_device(device)
