import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
_ML = ttnn.TensorMemoryLayout


def run(shape, layout, ml, w=False, b=False, r=False, dt=ttnn.bfloat16, tag=""):
    torch.manual_seed(0)
    td = torch.bfloat16
    x = torch.randn(shape, dtype=torch.float32).to(td)
    W = shape[-1]
    mc = (
        ttnn.DRAM_MEMORY_CONFIG
        if ml == _ML.INTERLEAVED
        else auto_shard_config(list(shape), ml, layout=layout, dtype=dt, device=device)
    )
    tx = ttnn.from_torch(x, dtype=dt, layout=layout, device=device, memory_config=mc)
    kw, tkw = {}, {}
    if w:
        tw = torch.randn(W, dtype=torch.float32).to(td)
        kw["weight"] = ttnn.from_torch(tw.reshape(1, 1, 1, W), dtype=dt, layout=layout, device=device)
        tkw["weight"] = tw
    if b:
        tb = torch.randn(W, dtype=torch.float32).to(td)
        kw["bias"] = ttnn.from_torch(tb.reshape(1, 1, 1, W), dtype=dt, layout=layout, device=device)
        tkw["bias"] = tb
    if r:
        tr = torch.randn(shape, dtype=torch.float32).to(td)
        kw["residual_input_tensor"] = ttnn.from_torch(
            tr, dtype=dt, layout=layout, device=device, memory_config=tx.memory_config()
        )
        tkw["residual_input_tensor"] = tr
    if ml != _ML.INTERLEAVED:
        kw["memory_config"] = tx.memory_config()
    lab = f"{tag}{shape} {'TILE' if layout==ttnn.TILE_LAYOUT else 'RM  '} {str(ml).split('.')[-1]:16s} w={int(w)} b={int(b)} r={int(r)}"
    try:
        out = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12, **kw))
    except Exception as e:
        print(f"{lab}  EXC {type(e).__name__}: {str(e)[:180]}", flush=True)
        return
    exp = torch_rms_norm_ttnn(x, epsilon=1e-12, **tkw)
    a, e = out.float().flatten(), exp.float().flatten()
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
    rms = ((a - e).pow(2).mean().sqrt() / e.std()).item()
    flag = "OK " if (pcc > 0.995 and rms < 0.04) else "BAD"
    print(f"{lab}  {flag} pcc={pcc:.6f} rms={rms:.4f}", flush=True)


# --- wide: forces the width split / ROW_RESIDENT / STREAM with a residual present ---
for shape in [
    (1, 1, 32, 16384),
    (1, 1, 32, 7168),
    (1, 1, 64, 12288),
    (1, 1, 160, 11008),
    (1, 1, 32, 4064),
    (1, 1, 32, 2848),
]:
    run(shape, ttnn.TILE_LAYOUT, _ML.INTERLEAVED, True, True, True, tag="WIDE ")
for shape in [(1, 1, 8192, 5120)]:
    run(shape, ttnn.TILE_LAYOUT, _ML.INTERLEAVED, True, True, True, tag="PREFILL ")
# --- tall/prime resilience with all operands ---
for shape in [(1, 1, 3232, 96), (1, 1, 4064, 160), (1, 1, 544, 2047), (99991, 64)]:
    run(shape, ttnn.TILE_LAYOUT, _ML.INTERLEAVED, True, True, True, tag="RES ")
ttnn.close_device(device)
