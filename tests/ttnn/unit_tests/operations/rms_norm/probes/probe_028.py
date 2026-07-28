import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd
from eval.sharding import auto_shard_config


def ref(x, g=None, eps=1e-6):
    xf = x.to(torch.float32)
    o = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    return o * g.to(torch.float32).reshape(-1) if g is not None else o


def pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
shapes = [
    (1, 1, 256, 512),
    (1, 1, 64, 128),
    (4, 8, 32, 256),
    (1, 1, 17, 64),
    (1, 1, 32, 50),
    (2, 512, 1024),
    (1024, 1024),
]
for shape in shapes:
    try:
        mc = auto_shard_config(list(shape), HS, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        x = torch.randn(shape, dtype=torch.bfloat16)
        g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ht, wt = pd._tile_geometry(tx)
        p = pd._select_placement(device, device.compute_with_storage_grid_size(), tx, ht, wt, True)
        out = ttnn.to_torch(rms_norm(tx, gamma=tg, memory_config=mc)).to(torch.float32)
        print(
            f"{shape} shard={list(mc.shard_spec.shape)} cores={p.num_cores} cw={p.cw} rows_max={p.rows_core_max} PCC={pcc(out, ref(x,g)):.6f}"
        )
    except Exception as e:
        print(f"{shape} FAILED: {type(e).__name__}: {str(e)[:300]}")
