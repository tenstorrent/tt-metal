import torch, ttnn
from tests.ttnn.utils_for_testing import check_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
try:

    def ref(Q, K, V):
        return torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float())

    shape = (1, 4, 256, 64)
    torch.manual_seed(0)
    Q = torch.randn(shape)
    K = torch.randn(shape)
    V = torch.randn(shape)
    expected = ref(Q, K, V)

    def run(dtype, fp32_acc, mf):
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=mf, fp32_dest_acc_en=fp32_acc, math_approx_mode=False)
        dev = lambda t: ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = ttnn.to_torch(scaled_dot_product_attention(dev(Q), dev(K), dev(V), compute_kernel_config=cfg)).float()
        p, msg = check_with_pcc(expected, out, 0.99)
        rms = (torch.sqrt(((out - expected) ** 2).mean()) / expected.std()).item()
        print(f"dtype={dtype} acc={fp32_acc} mf={mf}: {msg} relRMS={rms:.4f}")

    for dtype in (ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat16):
        for acc in (True, False):
            if dtype == ttnn.float32 and acc is False:
                continue
            run(dtype, acc, ttnn.MathFidelity.HiFi4)
finally:
    ttnn.close_device(device)
