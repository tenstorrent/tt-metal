import torch
import ttnn

# Same shape but TILE layout to compare
x = torch.ones(1, 1, 32, 512, dtype=torch.float32)
expected = torch.softmax(x, dim=-1)

device = ttnn.open_device(device_id=0)
ttnn_input = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
output = ttnn.operations.softmax.softmax(ttnn_input, dim=-1)
result = ttnn.to_torch(output)
ttnn.close_device(device)

print(f"TILE result[0,0,0,:5]: {result[0,0,0,:5]}")
print(f"expected[0,0,0,:5]: {expected[0,0,0,:5]}")
