import torch
import ttnn

device = ttnn.open_device(device_id=0)

torch.manual_seed(42)
combine_output_torch = torch.randn(1, 3200, 8, 7168, dtype=torch.bfloat16)
weights_torch = torch.randn(1, 3200, 8, 1, dtype=torch.bfloat16)

combine_tt = ttnn.from_torch(
    combine_output_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
weights_tt = ttnn.from_torch(
    weights_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

# Warmup
for _ in range(3):
    result = ttnn.experimental.deepseek_moe_post_combine_reduce(
        combine_tt, weights_tt, expert_dim=2, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
ttnn.synchronize_device(device)

# Benchmark
for _ in range(10):
    result = ttnn.experimental.deepseek_moe_post_combine_reduce(
        combine_tt, weights_tt, expert_dim=2, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
ttnn.synchronize_device(device)

ttnn.close_device(device)
