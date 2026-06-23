import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    print("NASTY_BEGIN")
    # large widths: even vs odd, beyond single-core budget
    for Wt, label in [
        (330, "even (2*165)"),
        (329, "ODD (7*47)"),
        (331, "ODD prime"),
        (512, "even big"),
        (513, "ODD big"),
    ]:
        W = Wt * 32
        shp = (1, 1, 32, W)
        x = torch.randn(shp, dtype=torch.float32)
        try:
            ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            o = ttnn.to_torch(rms_norm(ti)).float()
            ref = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)
            pcc = torch.corrcoef(torch.stack([o.flatten(), ref.flatten()]))[0, 1].item()
            print(f"  Wt={Wt:4d} {label:16s} W={W}: OK   PCC={pcc:.5f}")
        except Exception as e:
            print(f"  Wt={Wt:4d} {label:16s} W={W}: {type(e).__name__}: {str(e)[:80]}")
    print("NASTY_END")
finally:
    ttnn.close_device(dev)
