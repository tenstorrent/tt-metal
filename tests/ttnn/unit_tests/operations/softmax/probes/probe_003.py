import torch, ttnn

device = ttnn.open_device(device_id=0)

# Test: non-tile-aligned W, dim=-1 (W reduction)
# Shape (1,1,32,50) → W=50 not aligned, dim=-1 reduces along W
torch_input = torch.randn(1, 1, 32, 50, dtype=torch.float32)
expected = torch.softmax(torch_input, dim=-1)

ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
result = ttnn.to_torch(ttnn_output)

max_diff = (result.float() - expected.float()).abs().max().item()
pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
print(f"dim=-1, W=50 (non-aligned): max_diff={max_diff:.6f}, PCC={pcc:.6f}")
print(f"  result shape: {result.shape}, expected shape: {expected.shape}")

# Test: non-tile-aligned H, dim=-2 (H reduction)
torch_input2 = torch.randn(1, 1, 50, 64, dtype=torch.float32)
expected2 = torch.softmax(torch_input2, dim=-2)

ttnn_input2 = ttnn.from_torch(torch_input2, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_output2 = ttnn.softmax(ttnn_input2, dim=-2)
result2 = ttnn.to_torch(ttnn_output2)

max_diff2 = (result2.float() - expected2.float()).abs().max().item()
pcc2 = torch.corrcoef(torch.stack([result2.float().flatten(), expected2.float().flatten()]))[0, 1].item()
print(f"dim=-2, H=50 (non-aligned): max_diff={max_diff2:.6f}, PCC={pcc2:.6f}")

# Test: aligned baseline still works
torch_input3 = torch.randn(1, 1, 32, 64, dtype=torch.float32)
expected3 = torch.softmax(torch_input3, dim=-1)
ttnn_input3 = ttnn.from_torch(torch_input3, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_output3 = ttnn.softmax(ttnn_input3, dim=-1)
result3 = ttnn.to_torch(ttnn_output3)
max_diff3 = (result3.float() - expected3.float()).abs().max().item()
print(f"dim=-1, W=64 (aligned): max_diff={max_diff3:.6f}")

ttnn.close_device(device)
