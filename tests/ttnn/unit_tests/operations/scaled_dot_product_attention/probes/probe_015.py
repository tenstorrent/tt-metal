import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
from ttnn.operations._op_contract import ExcludedCell
from tests.ttnn.utils_for_testing import assert_with_pcc

device = ttnn.open_device(device_id=0)
try:

    def run(B, H, S, D, dtype, scale=None):
        torch.manual_seed(0)
        Q = torch.randn(B, H, S, D)
        K = torch.randn(B, H, S, D)
        V = torch.randn(B, H, S, D)
        expected = torch.nn.functional.scaled_dot_product_attention(
            Q.float(), K.float(), V.float(), is_causal=True, scale=scale
        )
        to = lambda t: ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        out = scaled_dot_product_attention(to(Q), to(K), to(V), is_causal=True, scale=scale, compute_kernel_config=cfg)
        got = ttnn.to_torch(out).float()
        assert_with_pcc(expected, got, 0.99)
        print(f"CAUSAL {dtype} scale={scale}: PASS")

    run(1, 2, 128, 64, ttnn.float32)
    run(1, 2, 128, 64, ttnn.bfloat8_b)
    run(1, 2, 128, 64, ttnn.bfloat16, scale=0.125)  # explicit scale

    # {causal, cross} EXCLUSION -> ExcludedCell
    to = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    Q = torch.randn(1, 4, 64, 64)
    K = torch.randn(1, 4, 128, 64)
    V = torch.randn(1, 4, 128, 64)
    try:
        scaled_dot_product_attention(to(Q), to(K), to(V), is_causal=True)
        print("EXCLUSION cross: FAIL (no raise)")
    except ExcludedCell as e:
        print(f"EXCLUSION {{causal,cross}}: PASS ({type(e).__name__})")

    # is_causal + attn_mask -> ValueError
    Q = torch.randn(1, 1, 128, 64)
    K = torch.randn(1, 1, 128, 64)
    V = torch.randn(1, 1, 128, 64)
    M = torch.zeros(1, 1, 128, 128)
    try:
        scaled_dot_product_attention(to(Q), to(K), to(V), is_causal=True, attn_mask=to(M))
        print("mutual-excl: FAIL (no raise)")
    except ValueError as e:
        print(f"is_causal+attn_mask ValueError: PASS")
    print("ALL EDGE PROBES DONE")
finally:
    ttnn.close_device(device)
