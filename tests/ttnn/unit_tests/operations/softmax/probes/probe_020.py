import torch
import ttnn

# Deterministic input: all ones to make hand-calculation easy
x = torch.ones(1, 1, 32, 512, dtype=torch.float32)
expected = torch.softmax(x, dim=-1)

device = ttnn.open_device(device_id=0)
ttnn_input = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
output = ttnn.operations.softmax.softmax(ttnn_input, dim=-1)
result = ttnn.to_torch(output)
ttnn.close_device(device)

max_diff = (result.float() - expected.float()).abs().max().item()
print(f"Shape: (1,1,32,512) RM dim=-1 fp32 all-ones")
print(f"  max_diff: {max_diff}")
print(f"  result[0,0,0,:5]: {result[0,0,0,:5]}")
print(f"  expected[0,0,0,:5]: {expected[0,0,0,:5]}")
