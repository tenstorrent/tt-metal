import torch, ttnn, math
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
        result = ttnn.to_torch(out)
        # Reference
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        ref = torch.matmul(attn, v.float())
        diff = (result.float() - ref.float()).abs()
        # PCC
        rf, ff = ref.float().flatten(), result.float().flatten()
        pcc = torch.nn.functional.cosine_similarity(rf - rf.mean(), ff - ff.mean(), dim=0).item()
        rms = (diff.pow(2).mean().sqrt() / ref.float().abs().max()).item()
        print(f"PASS: PCC={pcc:.6f}, RMS_rel={rms:.6f}, max_abs={diff.max().item():.6f}")
    except Exception as e:
        msg = str(e)
        import re
        m = re.search(r"grow to (\d+) B which is beyond max L1 size of (\d+) B", msg)
        if m:
            print(f"FAIL: CBs={m.group(1)} B vs L1={m.group(2)} B (over by {int(m.group(1)) - int(m.group(2))} B)")
        else:
            print(f"FAIL: {msg[:600]}")
finally:
    ttnn.close_device(device)
