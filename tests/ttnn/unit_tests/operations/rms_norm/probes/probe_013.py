import torch, ttnn
from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor

dev = ttnn.open_device(device_id=0)
try:
    H, W = 32, 64
    shape = (1, 1, H, W)

    def run(input_dt, gamma_dt, label):
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
        print(f"{label}: mean={y.mean().item():.5f}", flush=True)

    print("--- 1024 TTI_NOPs on all three threads after Phase 0 ---")
    run(ttnn.bfloat16, ttnn.bfloat16, "matched bf16+bf16")
    run(ttnn.bfloat16, ttnn.float32, "BUGGY bf16+fp32")
    run(ttnn.float32, ttnn.bfloat16, "BUGGY fp32+bf16")
finally:
    ttnn.close_device(dev)
