import torch, ttnn, math
from eval.golden_tests.scaled_dot_product_attention.helpers import pytorch_scaled_dot_product_attention
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def m(o, r):
    o = ttnn.to_torch(o).float()
    r = r.float()
    rms = (torch.sqrt(torch.mean((o - r) ** 2)) / r.std()).item()
    a = o.flatten()
    b = r.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return pcc, rms


dev = ttnn.open_device(device_id=0)
try:
    for S in (1024, 2048, 4096, 8192):
        torch.manual_seed(0)
        Q = torch.randn((1, 1, S, 64), dtype=torch.float32)
        K = torch.randn((1, 1, S, 64), dtype=torch.float32)
        V = torch.randn((1, 1, S, 64), dtype=torch.float32)
        ref = pytorch_scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=False, scale=None)
        qt = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        kt = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        vt = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        configs = {
            "HiFi4_acc_exact": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
            ),
            "HiFi4_acc_APPROX": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=True
            ),
        }
        nblk = S // 32
        for name, cfg in configs.items():
            out = scaled_dot_product_attention(
                qt, kt, vt, attn_mask=None, is_causal=False, scale=None, compute_kernel_config=cfg
            )
            pcc, rms = m(out, ref)
            print(f"S={S} ({nblk} blk) {name}: pcc={pcc:.6f} rms={rms:.6f}")
finally:
    ttnn.close_device(dev)
