import torch, ttnn, math
from eval.golden_tests.scaled_dot_product_attention.helpers import pytorch_scaled_dot_product_attention
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def rms_metric(ttnn_out, ref):
    out = ttnn.to_torch(ttnn_out).to(torch.float32)
    ref = ref.to(torch.float32)
    rmse = torch.sqrt(torch.mean((out - ref) ** 2))
    rms = (rmse / ref.std()).item()
    a = out.flatten()
    b = ref.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return pcc, rms


dev = ttnn.open_device(device_id=0)
try:
    for S in (2048, 8192):
        torch.manual_seed(0)
        Q = torch.randn((1, 1, S, 64), dtype=torch.float32)
        K = torch.randn((1, 1, S, 64), dtype=torch.float32)
        V = torch.randn((1, 1, S, 64), dtype=torch.float32)
        ref = pytorch_scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=False, scale=None)
        qt = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        kt = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        vt = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        out = scaled_dot_product_attention(qt, kt, vt, attn_mask=None, is_causal=False, scale=None)
        pcc, rms = rms_metric(out, ref)
        print(f"DEFAULT(HiFi4) S={S}: pcc={pcc:.6f} rms={rms:.6f}")
finally:
    ttnn.close_device(dev)
