import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
results = []
try:
    torch.manual_seed(0)
    B, H, S, D = 1, 1, 128, 1024
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32)
    v = torch.randn(B, H, S, D, dtype=torch.float32)

    qt = ttnn.from_torch(q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # causal mask (additive: 0 on lower triangle, -inf on upper)
    causal = torch.zeros(B, 1, S, S, dtype=torch.float32)
    causal.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))
    causal_tt = ttnn.from_torch(causal, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # 4 cells: auto/explicit × none/causal
    cases = [
        ("auto",     None),
        ("explicit", 1.0/math.sqrt(D)),
    ]
    for scale_label, scale_val in cases:
        for mask_label, mask_tt, mask_pt in [
            ("none",   None,      None),
            ("causal", causal_tt, causal),
        ]:
            try:
                kwargs = {}
                if scale_val is not None:
                    kwargs["scale"] = scale_val
                if mask_tt is not None:
                    kwargs["attention_mask"] = mask_tt
                out = scaled_dot_product_attention(qt, kt, vt, **kwargs)
                result = ttnn.to_torch(out)

                s = scale_val if scale_val is not None else 1.0/math.sqrt(D)
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * s
                if mask_pt is not None:
                    scores = scores + mask_pt
                attn = torch.softmax(scores, dim=-1)
                ref = torch.matmul(attn, v.float())

                rf, ff = ref.float().flatten(), result.float().flatten()
                pcc = torch.nn.functional.cosine_similarity(rf - rf.mean(), ff - ff.mean(), dim=0).item()
                rms = ((ref.float() - result.float()).pow(2).mean().sqrt() / ref.float().abs().max()).item()
                results.append(f"PASS  scale={scale_label}, mask={mask_label}: PCC={pcc:.6f}, RMS_rel={rms:.6f}")
            except Exception as e:
                results.append(f"FAIL  scale={scale_label}, mask={mask_label}: {str(e)[:200]}")
finally:
    ttnn.close_device(device)

for r in results:
    print(r)
