import math, torch, ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def run(B, H, Hkv, S, D, mask_mode, fp32_dest, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16)
    tm = None
    torch_mask = None
    if mask_mode == "custom":
        torch_mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
        torch_mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))
        tm = ttnn.from_torch(torch_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    scale = 1.0 / math.sqrt(D)
    kk = k.repeat_interleave(H // Hkv, dim=1)
    vv = v.repeat_interleave(H // Hkv, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        kk.float(),
        vv.float(),
        attn_mask=torch_mask.float() if torch_mask is not None else None,
        is_causal=False,
        scale=scale,
    )
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest, math_approx_mode=False
    )
    out = scaled_dot_product_attention(tq, tk, tv, attn_mask=tm, scale=scale, compute_kernel_config=cfg)
    res = ttnn.to_torch(out).float()
    import numpy as np
    from models.utility_functions import comp_pcc

    ok, pcc = comp_pcc(ref, res, 0.99)
    rms = ((res - ref).pow(2).mean().sqrt() / ref.std()).item()
    print(
        f"B{B}H{H}Hkv{Hkv}S{S}D{D} mask={mask_mode} fp32_dest={fp32_dest}: pcc={pcc} rms={rms:.4f} maxabs={(res-ref).abs().max():.3f}"
    )


# repro the failing regime with a smaller multi-work-unit shape
run(1, 71, 1, 512, 64, "custom", False)  # total_work=142>110, fused, custom -> expect FAIL
run(1, 71, 1, 512, 64, "none", False)  # same but mask=none -> isolate custom
run(1, 71, 1, 512, 64, "custom", True)  # same but fp32_dest=True (non-fused) -> isolate fused
run(1, 71, 1, 256, 64, "custom", False)  # total_work=71<110 (1 wu/core), custom, fused -> isolate multi-wu
