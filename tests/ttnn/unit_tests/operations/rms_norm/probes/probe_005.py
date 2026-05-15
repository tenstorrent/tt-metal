import math, torch, ttnn
from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor

dev = ttnn.open_device(device_id=0)
try:
    H, W = 32, 64
    shape = (1, 1, H, W)

    def run(x_val, eps, input_dt, gamma_dt, label):
        x = torch.full(shape, float(x_val), dtype=torch.float32)
        g = torch.ones((1, 1, 1, W), dtype=torch.float32)
        x_t = ttnn.from_torch(
            x, dtype=input_dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        g_t = ttnn.from_torch(
            g, dtype=gamma_dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out_t = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), input_dt, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        pd = create_program_descriptor(x_t, g_t, out_t, epsilon=eps)
        result = ttnn.generic_op([x_t, g_t, out_t], pd)
        y = ttnn.to_torch(result).to(torch.float32)

        true_mean = x_val**2
        # Hypothesis A: mean reads as 0.5× true → output = x / sqrt(0.5*x² + eps)
        hyp_A = x_val / math.sqrt(0.5 * true_mean + eps)
        # Hypothesis B: rsqrt result × √2 → output = √2 · x / sqrt(x² + eps)
        hyp_B = math.sqrt(2.0) * x_val / math.sqrt(true_mean + eps)
        # Hypothesis C: just √2 × correct → output = √2 · x / sqrt(x² + eps)
        # (same as B, kept for clarity)
        no_bug = x_val / math.sqrt(true_mean + eps)
        print(
            f"{label}: x={x_val} eps={eps:.0e}  actual={y.mean().item():.5f}  hypA(half_mean)={hyp_A:.5f}  hypB(rsqrt_sqrt2)={hyp_B:.5f}  nobug={no_bug:.5f}"
        )

    print("--- bf16 input + fp32 gamma: eps sweep with x=1.0 ---")
    for e in [1e-6, 1e-2, 1.0, 4.0]:
        run(1.0, e, ttnn.bfloat16, ttnn.float32, "bf16+fp32")
    print("--- bf16 input + fp32 gamma: eps sweep with x=2.0 ---")
    for e in [1e-6, 1e-2, 1.0, 4.0]:
        run(2.0, e, ttnn.bfloat16, ttnn.float32, "bf16+fp32")
finally:
    ttnn.close_device(dev)
