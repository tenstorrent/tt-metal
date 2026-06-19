import math, torch, ttnn
from eval.golden_tests.scaled_dot_product_attention.helpers import (
    pytorch_scaled_dot_product_attention,
    create_ttnn_input_tensor,
    make_causal_mask,
    TOLERANCES,
    EXPLICIT_SCALE,
    _TORCH_DTYPE,
)
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    fin = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[fin], b[fin]
    if torch.allclose(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def run(qkv, dtype, mask_mode, scale_mode, fid):
    q_shape, k_shape, v_shape = qkv
    td = _TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    Q = torch.randn(q_shape, dtype=td)
    K = torch.randn(k_shape, dtype=td)
    V = torch.randn(v_shape, dtype=td)
    B, _H, S_q, _D = q_shape
    S_kv = k_shape[-2]
    tmask = make_causal_mask(B, S_q, S_kv, torch_dtype=td) if mask_mode == "causal" else None
    scale = EXPLICIT_SCALE if scale_mode == "explicit" else None
    exp = pytorch_scaled_dot_product_attention(Q, K, V, attention_mask=tmask, scale=scale)
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, fp32_dest_acc_en=True, math_approx_mode=False)
    tq = create_ttnn_input_tensor(Q, device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tk = create_ttnn_input_tensor(K, device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tv = create_ttnn_input_tensor(V, device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tm = create_ttnn_input_tensor(tmask, device, dtype=dtype, layout=ttnn.TILE_LAYOUT) if tmask is not None else None
    out = scaled_dot_product_attention(tq, tk, tv, attention_mask=tm, scale=scale, compute_kernel_config=cfg)
    got = ttnn.to_torch(out).float()
    e = exp.float()
    p = pcc(e, got)
    rms = torch.sqrt(torch.mean((got - e) ** 2)).item() / (e.std().item() + 1e-12)
    pcc_t, rms_t = TOLERANCES[dtype]
    ok = (p >= pcc_t) and (rms <= rms_t)
    print(
        f"  {str(fid).split('.')[-1]:6s} {str(dtype).split('.')[-1]:9s} {mask_mode:6s} Q{q_shape} KV{k_shape[-2]}: PCC={p:.5f} relRMS={rms:.4f} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


S = lambda *d: tuple(d)
HiFi4 = ttnn.MathFidelity.HiFi4
try:
    print("=== #38306 canary: bf16 multi-KV with HiFi4+fp32_acc (SUM reduce every KV block) ===")
    run((S(1, 8, 256, 64),) * 3, ttnn.bfloat16, "none", "auto", HiFi4)
    run((S(1, 2, 512, 64),) * 3, ttnn.bfloat16, "causal", "explicit", HiFi4)
    run((S(1, 1, 32, 64),) * 3, ttnn.bfloat16, "none", "explicit", HiFi4)
    print("=== fp32 with HiFi4 ===")
    for qkv in [
        (S(1, 1, 128, 64),) * 3,
        (S(1, 8, 256, 64),) * 3,
        (S(1, 1, 128, 1024),) * 3,
        (S(1, 8, 1024, 128), S(1, 2, 1024, 128), S(1, 2, 1024, 128)),
    ]:
        run(qkv, ttnn.float32, "none", "auto", HiFi4)
    run((S(1, 1, 128, 1024),) * 3, ttnn.float32, "causal", "explicit", HiFi4)
finally:
    ttnn.close_device(device)
