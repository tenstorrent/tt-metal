import torch, ttnn
from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor

dev = ttnn.open_device(device_id=0)
try:
    H, W = 32, 64
    shape = (1, 1, H, W)

    def run(x_val, g_val, input_dt, gamma_dt, label):
        x = torch.full(shape, float(x_val), dtype=torch.float32)
        g = torch.full((1, 1, 1, W), float(g_val), dtype=torch.float32)
        # Expected: x · rsqrt(x² + eps) · g  ≈ sign(x)·g  (because mean(x²) = x²)
        expected = (x_val / abs(x_val)) * g_val if x_val != 0 else 0.0
        x_t = ttnn.from_torch(
            x, dtype=input_dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        g_t = ttnn.from_torch(
            g, dtype=gamma_dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out_t = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), input_dt, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        pd = create_program_descriptor(x_t, g_t, out_t, epsilon=1e-6)
        result = ttnn.generic_op([x_t, g_t, out_t], pd)
        y = ttnn.to_torch(result).to(torch.float32)
        actual = y.mean().item()
        ratio = actual / expected if expected != 0 else float("inf")
        print(
            f"{label}: x={x_val:5.2f} g={g_val:5.2f}  actual_mean={actual:.5f}  expected={expected:.5f}  ratio={ratio:.5f}"
        )

    # bf16+fp32 — vary input magnitude
    print("--- bf16 input + fp32 gamma: vary x ---")
    for v in [1.0, 2.0, 0.5, 4.0, 8.0]:
        run(v, 1.0, ttnn.bfloat16, ttnn.float32, "bf16+fp32")
    # And vary gamma magnitude
    print("--- bf16 input + fp32 gamma: vary g ---")
    for v in [0.5, 1.0, 2.0, 4.0]:
        run(1.0, v, ttnn.bfloat16, ttnn.float32, "bf16+fp32")
finally:
    ttnn.close_device(dev)
