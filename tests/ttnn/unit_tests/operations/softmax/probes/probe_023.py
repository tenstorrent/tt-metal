import torch
import ttnn

x = torch.ones(1, 1, 32, 512, dtype=torch.float32)

device = ttnn.open_device(device_id=0)
# TILE path
ttnn_input_t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
output_t = ttnn.operations.softmax.softmax(ttnn_input_t, dim=-1)
result_t = ttnn.to_torch(output_t)
# RM path
ttnn_input_r = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
output_r = ttnn.operations.softmax.softmax(ttnn_input_r, dim=-1)
result_r = ttnn.to_torch(output_r)
ttnn.close_device(device)

print(f"TILE result[0,0,0,:3]: {result_t[0,0,0,:3]}")
print(f"RM   result[0,0,0,:3]: {result_r[0,0,0,:3]}")
