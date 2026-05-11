import torch, ttnn, math

shape = (1, 1, 32, 32)
device = ttnn.open_device(device_id=0)

# Direct ttnn.lgamma test on 0.000358
torch_input = torch.full(shape, 0.000358, dtype=torch.float32)
ttnn_input = ttnn.from_torch(
    torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
val_direct = ttnn.to_torch(ttnn.lgamma(ttnn_input)).float()[0, 0, 0, 0].item()

# Then test ttnn.lgamma after a previous lgamma call that processed 0.500358
torch_input2 = torch.full(shape, 0.500358, dtype=torch.float32)
ttnn_input2 = ttnn.from_torch(
    torch_input2, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
_ = ttnn.lgamma(ttnn_input2)  # warmup with 0.500358 first
val_after = ttnn.to_torch(ttnn.lgamma(ttnn_input)).float()[0, 0, 0, 0].item()

print(f"RESULT ttnn.lgamma(0.000358) cold={val_direct} after-warmup={val_after} torch={math.lgamma(0.000358):.6f}")
ttnn.close_device(device)
