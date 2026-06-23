import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    print("NASTY_BEGIN")
    # Wt=224 (mult of 8), Wt=288 (mult of 8), Wt=220 (even, NOT mult of 8), Wt=219 (odd)
    for W, label in [
        (7168, "Wt=224 mult8"),
        (9216, "Wt=288 mult8"),
        (7040, "Wt=220 even-not-mult8"),
        (7008, "Wt=219 ODD"),
    ]:
        shp = (1, 1, 32, W)
        x = torch.randn(shp, dtype=torch.float32)
        try:
            ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            o = ttnn.to_torch(rms_norm(ti)).float()
            ref = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)
            pcc = torch.corrcoef(torch.stack([o.flatten(), ref.flatten()]))[0, 1].item()
            print(f"  {label:24s} W={W}: OK   PCC={pcc:.5f}")
        except Exception as e:
            print(f"  {label:24s} W={W}: {type(e).__name__}: {str(e)[:90]}")
    print("NASTY_END")
finally:
    ttnn.close_device(dev)
