import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def run(dtype, fp32_acc, shape=(1, 1, 128, 64)):
    torch.manual_seed(0)
    tdt = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    Q = torch.randn(shape, dtype=tdt)
    K = torch.randn(shape, dtype=tdt)
    V = torch.randn(shape, dtype=tdt)
    exp = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float())
    to = lambda t: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_acc, math_approx_mode=False
    )
    out = ttnn.to_torch(scaled_dot_product_attention(to(Q), to(K), to(V), compute_kernel_config=cfg)).float()
    diff = (exp - out).abs()
    # PCC
    e = exp.flatten().float()
    o = out.flatten().float()
    pcc = torch.corrcoef(torch.stack([e, o]))[0, 1].item()
    print(f"dtype={dtype} fp32_acc={fp32_acc} shape={shape}: PCC={pcc:.6f} max_abs={diff.max():.5f}")


device = ttnn.open_device(device_id=0)
try:
    run(ttnn.float32, True)
    run(ttnn.bfloat8_b, True)
    run(ttnn.bfloat8_b, False)
    run(ttnn.bfloat16, True)
finally:
    ttnn.close_device(device)
