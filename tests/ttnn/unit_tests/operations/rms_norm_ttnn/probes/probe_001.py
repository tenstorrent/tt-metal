import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn


def run(shape, layout=ttnn.TILE_LAYOUT, w=False, b=False, r=False, dt=ttnn.bfloat16):
    torch.manual_seed(42)
    td = torch.bfloat16 if dt != ttnn.float32 else torch.float32
    x = torch.randn(shape, dtype=torch.float32).to(td)
    W = shape[-1] if len(shape) else 1
    kw, tkw = {}, {}
    tx = ttnn.from_torch(x, dtype=dt, layout=layout, device=device)
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
        kw["residual_input_tensor"] = ttnn.from_torch(tr, dtype=dt, layout=layout, device=device)
        tkw["residual_input_tensor"] = tr
    out = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12, **kw))
    exp = torch_rms_norm_ttnn(x, epsilon=1e-12, **tkw)
    a, e = out.float().flatten(), exp.float().flatten()
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item() if a.numel() > 1 else 1.0
    print(
        f"{shape} {('TILE' if layout==ttnn.TILE_LAYOUT else 'RM')} w={int(w)} b={int(b)} r={int(r)}  pcc={pcc:.6f} maxdiff={(a-e).abs().max().item():.4g}"
    )


run((1, 1, 64, 128))
run((1, 1, 64, 128), w=True)
