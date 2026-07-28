import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)


def ref(x, g=None, eps=1e-6):
    xf = x.to(torch.float32)
    o = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    return o * g.to(torch.float32).reshape(-1) if g is not None else o


def pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
# RM activation + RM gamma (the live cells: gamma_layout TILE is INVALID-skipped)
for shape, has_g in [((1, 1, 64, 128), True), ((1, 1, 32, 64), False), ((1, 1, 17, 64), True)]:
    try:
        mc = auto_shard_config(list(shape), HS, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
        print(f"{shape} shard={list(mc.shard_spec.shape)} grid={mc.shard_spec.grid}", flush=True)
        x = torch.randn(shape, dtype=torch.bfloat16)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
        g = tg = None
        if has_g:
            g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
            tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out = ttnn.to_torch(rms_norm(tx, gamma=tg, memory_config=mc)).to(torch.float32)
        print(f"   PCC={pcc(out, ref(x,g)):.6f}", flush=True)
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
ttnn.close_device(device)
