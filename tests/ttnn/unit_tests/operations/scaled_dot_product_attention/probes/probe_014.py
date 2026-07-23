import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
from tests.ttnn.utils_for_testing import assert_with_pcc

device = ttnn.open_device(device_id=0)
try:

    def run(B, H, S, D, Hkv=None, dtype=ttnn.bfloat16):
        Hkv = Hkv or H
        torch.manual_seed(0)
        Q = torch.randn(B, H, S, D)
        K = torch.randn(B, Hkv, S, D)
        V = torch.randn(B, Hkv, S, D)
        Kf, Vf = K, V
        if H != Hkv:
            r = H // Hkv
            Kf = K.repeat_interleave(r, dim=1)
            Vf = V.repeat_interleave(r, dim=1)
        expected = torch.nn.functional.scaled_dot_product_attention(Q.float(), Kf.float(), Vf.float(), is_causal=True)
        to = lambda t: ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = scaled_dot_product_attention(to(Q), to(K), to(V), is_causal=True)
        got = ttnn.to_torch(out).float()
        assert_with_pcc(expected, got, 0.99)
        print(f"CAUSAL B{B}H{H}S{S}D{D} Hkv{Hkv} {dtype}: shape={list(got.shape)} PASS")

    run(1, 1, 32, 32)
    run(1, 1, 128, 64)
    run(1, 4, 256, 64)
    run(1, 8, 128, 64, Hkv=2)
    run(1, 8, 128, 64, Hkv=1)
    run(2, 4, 128, 64)
    run(1, 1, 512, 64)
    print("ALL CAUSAL PROBES PASS")
finally:
    ttnn.close_device(device)
