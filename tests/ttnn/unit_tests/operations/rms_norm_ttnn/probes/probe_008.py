import torch, ttnn
from dataclasses import dataclass
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
_ML = ttnn.TensorMemoryLayout
TD = torch.bfloat16


def check(lab, out, exp):
    a, e = out.float().flatten(), exp.float().flatten()
    if a.numel() < 2:
        print(f"{lab}  OK  scalar act={a.tolist()} exp={e.tolist()}", flush=True)
        return
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
    rms = ((a - e).pow(2).mean().sqrt() / e.std()).item()
    print(f"{lab}  {'OK ' if pcc>0.995 and rms<0.05 else 'BAD'} pcc={pcc:.6f} rms={rms:.4f}", flush=True)


# ---------- 1. blocked (Wt, 32) ROW_MAJOR per-channel form ----------
def blocked(shape, ml, with_bias):
    torch.manual_seed(0)
    W = shape[-1]
    Wt = (W + 31) // 32
    x = torch.randn(shape, dtype=torch.float32).to(TD)
    mc = (
        ttnn.DRAM_MEMORY_CONFIG
        if ml == _ML.INTERLEAVED
        else auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    )
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    kw, tkw = {}, {}
    for name, key in (("weight", "weight"), ("bias", "bias") if with_bias else ("weight", "weight")):
        pass
    names = ["weight"] + (["bias"] if with_bias else [])
    for i, name in enumerate(names):
        torch.manual_seed(10 + i)
        v = torch.randn(W, dtype=torch.float32).to(TD)
        pad = torch.zeros(Wt * 32, dtype=torch.float32).to(TD)
        pad[:W] = v
        kw[name] = ttnn.from_torch(
            pad.reshape(Wt, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        tkw[name] = v
    if ml != _ML.INTERLEAVED:
        kw["memory_config"] = tx.memory_config()
    lab = f"BLOCKED {shape} {str(ml).split('.')[-1]:16s} bias={int(with_bias)}"
    try:
        out = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12, **kw))
    except Exception as ex:
        print(f"{lab}  EXC {type(ex).__name__}: {str(ex)[:200]}", flush=True)
        return
    check(lab, out, torch_rms_norm_ttnn(x, epsilon=1e-12, **tkw))


for shape in [(1, 1, 32, 128), (1, 1, 256, 512), (1, 1, 32, 72), (1, 1, 32, 4096)]:
    for ml in (_ML.INTERLEAVED, _ML.HEIGHT_SHARDED, _ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED):
        for wb in (False, True):
            blocked(shape, ml, wb)


# ---------- 2. degenerate ranks with every operand ----------
def degen(shape):
    torch.manual_seed(0)
    W = shape[-1] if len(shape) else 1
    x = torch.randn(shape, dtype=torch.float32).to(TD)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tw = torch.randn(W, dtype=torch.float32).to(TD)
    tb = torch.randn(W, dtype=torch.float32).to(TD)
    tr = torch.randn(shape, dtype=torch.float32).to(TD)
    kw = dict(
        weight=ttnn.from_torch(
            tw.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        ),
        bias=ttnn.from_torch(tb.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
        residual_input_tensor=ttnn.from_torch(tr, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
    )
    lab = f"DEGEN {shape}"
    try:
        out = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12, **kw))
    except Exception as ex:
        print(f"{lab}  EXC {type(ex).__name__}: {str(ex)[:200]}", flush=True)
        return
    check(lab, out, torch_rms_norm_ttnn(x, epsilon=1e-12, weight=tw, bias=tb, residual_input_tensor=tr))


for shape in [(), (64,), (50,), (2, 1, 1, 32, 64), (2, 1, 1, 47, 50)]:
    degen(shape)


# ---------- 3. pad poison with all operands ----------
def poison(shape, ml, val=1000.0):
    torch.manual_seed(0)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.float32).to(TD)
    tw = torch.randn(W, dtype=torch.float32).to(TD)
    tb = torch.randn(W, dtype=torch.float32).to(TD)
    tr = torch.randn(shape, dtype=torch.float32).to(TD)
    mc = (
        ttnn.DRAM_MEMORY_CONFIG
        if ml == _ML.INTERLEAVED
        else auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    )
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tres = ttnn.from_torch(
        tr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=tx.memory_config()
    )
    tx = ttnn.fill_implicit_tile_padding(tx, val)
    tres = ttnn.fill_implicit_tile_padding(tres, val)
    tW = ttnn.fill_implicit_tile_padding(
        ttnn.from_torch(tw.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device), val
    )
    tB = ttnn.fill_implicit_tile_padding(
        ttnn.from_torch(tb.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device), val
    )
    kw = dict(weight=tW, bias=tB, residual_input_tensor=tres)
    if ml != _ML.INTERLEAVED:
        kw["memory_config"] = tx.memory_config()
    lab = f"POISON {shape} {str(ml).split('.')[-1]:16s}"
    try:
        out = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12, **kw))
    except Exception as ex:
        print(f"{lab}  EXC {type(ex).__name__}: {str(ex)[:200]}", flush=True)
        return
    check(lab, out, torch_rms_norm_ttnn(x, epsilon=1e-12, weight=tw, bias=tb, residual_input_tensor=tr))


for shape in [(1, 1, 32, 40), (1, 1, 32, 72), (1, 1, 32, 200), (1, 1, 40, 40), (1, 1, 224, 72)]:
    for ml in (_ML.INTERLEAVED, _ML.HEIGHT_SHARDED, _ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED):
        poison(shape, ml)
ttnn.close_device(device)
