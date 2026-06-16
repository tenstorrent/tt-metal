import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
from eval.golden_tests.scaled_dot_product_attention.helpers import (
    pytorch_scaled_dot_product_attention as ref,
    TOLERANCES,
)


def relrms(o, e):
    return ((o.float() - e.float()).pow(2).mean().sqrt() / e.float().std()).item()


def pcc(o, e):
    a = o.float().flatten()
    b = e.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)
dev = ttnn.open_device(device_id=0)
try:
    for dt, tdt in [(ttnn.bfloat16, torch.bfloat16), (ttnn.bfloat8_b, torch.bfloat16)]:
        tol = TOLERANCES[(dt, False)]
        for S in [
            128,
            160,
            4096,
            8192,
        ]:  # 160 -> Skv_t=5, not div by 8 -> Bkv_t=1; tests qc%Bkv alignment via 4096/8192
            shp = (1, 2, S, 64)
            torch.manual_seed(0)
            Q = torch.randn(shp, dtype=tdt)
            K = torch.randn(shp, dtype=tdt)
            V = torch.randn(shp, dtype=tdt)
            e = ref(Q, K, V, is_causal=True)
            tq = ttnn.from_torch(Q, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            tk = ttnn.from_torch(K, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            tv = ttnn.from_torch(V, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            o = scaled_dot_product_attention(tq, tk, tv, is_causal=True, compute_kernel_config=cfg)
            ot = ttnn.to_torch(o)
            rr = relrms(ot, e)
            pc = pcc(ot, e)
            fail = (pc < tol[0]) or (rr > tol[1])
            print(
                f"{str(dt).split('.')[-1]:9s} causal S={S:5d} relrms={rr:.4f} pcc={pc:.5f} {'FAIL' if fail else 'ok'}"
            )
finally:
    ttnn.close_device(dev)
