import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

device = ttnn.open_device(device_id=0)


def run(shape, layout=ttnn.TILE_LAYOUT, w=False, b=False, r=False, dt=ttnn.bfloat16, olayout=None, odt=None, eps=1e-12):
    torch.manual_seed(42)
    td = torch.bfloat16 if dt != ttnn.float32 else torch.float32
    olayout = olayout or layout
    odt = odt or dt
    otd = torch.bfloat16 if odt != ttnn.float32 else torch.float32
    x = torch.randn(shape, dtype=torch.float32).to(td)
    W = shape[-1] if len(shape) else 1
    kw, tkw = {}, {}
    tx = ttnn.from_torch(x, dtype=dt, layout=layout, device=device)
    if w:
        tw = torch.randn(W, dtype=torch.float32).to(otd)
        kw["weight"] = ttnn.from_torch(tw.reshape(1, 1, 1, W), dtype=odt, layout=olayout, device=device)
        tkw["weight"] = tw
    if b:
        tb = torch.randn(W, dtype=torch.float32).to(otd)
        kw["bias"] = ttnn.from_torch(tb.reshape(1, 1, 1, W), dtype=odt, layout=olayout, device=device)
        tkw["bias"] = tb
    if r:
        tr = torch.randn(shape, dtype=torch.float32).to(td)
        kw["residual_input_tensor"] = ttnn.from_torch(tr, dtype=dt, layout=layout, device=device)
        tkw["residual_input_tensor"] = tr
    out = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=eps, **kw))
    exp = torch_rms_norm_ttnn(x, epsilon=eps, **tkw)
    a, e = out.float().flatten(), exp.float().flatten()
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item() if a.numel() > 1 else 1.0
    print(
        f"{shape} {('TILE' if layout==ttnn.TILE_LAYOUT else 'RM')} w={int(w)} b={int(b)} r={int(r)}  pcc={pcc:.6f} maxdiff={(a-e).abs().max().item():.4g}",
        flush=True,
    )


run((1, 1, 64, 128))
run((1, 1, 64, 128), w=True)
run((1, 1, 64, 128), layout=ttnn.ROW_MAJOR_LAYOUT)
run((1, 1, 64, 128), layout=ttnn.ROW_MAJOR_LAYOUT, w=True)
ttnn.close_device(device)
