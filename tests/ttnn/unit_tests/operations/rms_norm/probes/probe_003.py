import torch, ttnn
from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor

dev = ttnn.open_device(device_id=0)
try:

    def run(W, input_dt, gamma_dt, label):
        H = 32
        shape = (1, 1, H, W)
        x = torch.ones(shape, dtype=torch.float32)
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
        pd = create_program_descriptor(x_t, g_t, out_t, epsilon=1e-6)
        result = ttnn.generic_op([x_t, g_t, out_t], pd)
        y = ttnn.to_torch(result).to(torch.float32)
        print(
            f"W={W:5d} Wt={W//32:2d} {label}: mean={y.mean().item():.5f} std={y.std().item():.5f} min={y.min().item():.5f} max={y.max().item():.5f}"
        )

    # bf16+fp32 — uniform √2; check vs Wt
    print("--- bf16 input + fp32 gamma (Wt varies) ---")
    for W in [32, 64, 128, 256, 512]:
        run(W, ttnn.bfloat16, ttnn.float32, "bf16+fp32")

    # fp32+bf16 — varying
    print("--- fp32 input + bf16 gamma (Wt varies) ---")
    for W in [32, 64, 128, 256, 512]:
        run(W, ttnn.float32, ttnn.bfloat16, "fp32+bf16")
finally:
    ttnn.close_device(dev)
