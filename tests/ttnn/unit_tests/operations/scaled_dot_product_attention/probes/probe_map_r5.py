import math, torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
from eval.golden_tests.scaled_dot_product_attention.helpers import (
    pytorch_scaled_dot_product_attention as ref,
    make_causal_mask,
    TOLERANCES,
)


def relrms(o, e):
    return ((o.float() - e.float()).pow(2).mean().sqrt() / e.float().std()).item()


def pcc(o, e):
    a = o.float().flatten()
    b = e.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


shapes = [
    ((1, 1, 4096, 64),) * 3,
    ((1, 1, 8192, 64),) * 3,
    ((1, 4, 4096, 64),) * 3,
    ((1, 4, 4096, 64), (1, 1, 4096, 64), (1, 1, 4096, 64)),
    ((1, 8, 4096, 128), (1, 2, 4096, 128), (1, 2, 4096, 128)),
]
cfg_noacc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)
combos = [
    (ttnn.bfloat16, torch.bfloat16, cfg_noacc, False),
    (ttnn.bfloat8_b, torch.bfloat16, cfg_noacc, False),
    (ttnn.float32, torch.float32, None, True),
]

dev = ttnn.open_device(device_id=0)
try:
    for dt, tdt, cfg, acc in combos:
        tol = TOLERANCES[(dt, acc)]
        for sh in shapes:
            q, k, v = sh
            for mm in ["none", "custom", "causal"]:
                torch.manual_seed(0)
                Q = torch.randn(q, dtype=tdt)
                K = torch.randn(k, dtype=tdt)
                V = torch.randn(v, dtype=tdt)
                is_c = mm == "causal"
                tmask = None
                if mm == "custom":
                    tmask = make_causal_mask(q[0], q[2], k[2], torch_dtype=tdt)
                e = ref(Q, K, V, attn_mask=tmask, is_causal=is_c)
                tq = ttnn.from_torch(Q, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
                tk = ttnn.from_torch(K, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
                tv = ttnn.from_torch(V, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
                tmt = (
                    ttnn.from_torch(tmask, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev) if tmask is not None else None
                )
                o = scaled_dot_product_attention(tq, tk, tv, attn_mask=tmt, is_causal=is_c, compute_kernel_config=cfg)
                ot = ttnn.to_torch(o)
                rr = relrms(ot, e)
                pc = pcc(ot, e)
                fail = (pc < tol[0]) or (rr > tol[1])
                print(
                    f"{str(dt).split('.')[-1]:9s} acc={acc} {mm:7s} {str(q):18s} relrms={rr:.4f} pcc={pc:.5f} tol={tol} {'FAIL' if fail else 'ok'}"
                )
finally:
    ttnn.close_device(dev)
