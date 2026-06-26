import torch
import ttnn

# Test: RM layout, wide W (triggers V2), dim=-1 (attention use case)
# Shape (1,1,32,4096) — V1 footprint exceeds 256 KiB for fp32
torch.manual_seed(42)
x = torch.randn(1, 1, 32, 4096, dtype=torch.float32)
expected = torch.softmax(x, dim=-1)

device = ttnn.open_device(0)
ttnn_input = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
output = ttnn.operations.softmax.softmax(ttnn_input, dim=-1)
result = ttnn.to_torch(output)
ttnn.close_device(device)

# Check correctness
max_diff = (result.float() - expected.float()).abs().max().item()
pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
print(f"Shape: (1,1,32,4096) RM dim=-1 fp32")
print(f"  max_diff: {max_diff}")
print(f"  PCC: {pcc}")
print(f"  PASS: {pcc >= 0.999 and max_diff < 0.01}")
