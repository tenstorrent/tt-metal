import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    B, H, S, D = 1, 1, 128, 1024
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32)
    v = torch.randn(B, H, S, D, dtype=torch.float32)

    qt = ttnn.from_torch(q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    try:
        out = scaled_dot_product_attention(qt, kt, vt)
        print("PASSED unexpectedly")
    except Exception as e:
        msg = str(e)
        import re
        m = re.search(r"grow to (\d+) B which is beyond max L1 size of (\d+) B", msg)
        if m:
            print(f"FAIL: CBs={m.group(1)} B vs L1={m.group(2)} B (over by {int(m.group(1)) - int(m.group(2))} B)")
        else:
            print(f"FAIL (first 600): {msg[:600]}")
finally:
    ttnn.close_device(device)
