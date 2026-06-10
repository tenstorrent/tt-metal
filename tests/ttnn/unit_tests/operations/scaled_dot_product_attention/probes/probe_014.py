"""R6 target: verify Q1x1x8192x64 fp32 self mha none cells under target."""
import math
import torch
import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

torch.manual_seed(0)
B, H, S, D = 1, 1, 8192, 64

q_pt = torch.randn(B, H, S, D, dtype=torch.float32)
k_pt = torch.randn(B, H, S, D, dtype=torch.float32)
v_pt = torch.randn(B, H, S, D, dtype=torch.float32)

# Torch reference
qf, kf, vf = q_pt.float(), k_pt.float(), v_pt.float()
s = 1.0 / math.sqrt(D)
scores = torch.matmul(qf, kf.transpose(-2, -1)) * s
attn = torch.softmax(scores, dim=-1)
reference = torch.matmul(attn, vf)

device = ttnn.open_device(device_id=0)
try:
    qt = ttnn.from_torch(q_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    print("Running scaled_dot_product_attention auto-scale...")
    out_auto = scaled_dot_product_attention(qt, kt, vt)
    res_auto = ttnn.to_torch(out_auto)

    print("Running scaled_dot_product_attention explicit-scale...")
    out_exp = scaled_dot_product_attention(qt, kt, vt, scale=s)
    res_exp = ttnn.to_torch(out_exp)

    for name, res in [("auto", res_auto), ("explicit", res_exp)]:
        rf = reference.float().flatten()
        af = res.float().flatten()
        pcc = torch.nn.functional.cosine_similarity(rf - rf.mean(), af - af.mean(), dim=0).item()
        rms = ((reference - res).pow(2).mean().sqrt() / reference.float().abs().clamp_min(1e-12).max()).item()
        target_pcc, target_rms = 0.999, 0.02
        status = "PASS" if (pcc >= target_pcc and rms <= target_rms) else "FAIL"
        print(f"S=8192 fp32 {name}: PCC={pcc:.6f} RMS={rms:.6f} target=({target_pcc},{target_rms}) {status}")
finally:
    ttnn.close_device(device)
