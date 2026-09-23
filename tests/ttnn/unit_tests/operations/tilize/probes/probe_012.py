import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
x = torch.randn([2, 3, 40, 40]).bfloat16()
t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
out = tilize(t, output_padded_shape=[3, 4, 64, 96], pad_value=-3)
rb = out.cpu().to_torch_with_padded_shape()
exp = torch.nn.functional.pad(x, [0, 56, 0, 24, 0, 1, 0, 1], value=-3.0)
print("RES padded_eq", torch.equal(rb, exp))
# host-only check of TTNN's own logical extraction for inner leading-dim padding
h = ttnn.from_torch(exp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
v = ttnn.reshape(h, ttnn.Shape([2, 3, 40, 40]), ttnn.Shape([3, 4, 64, 96]))
print("RES host_view_logical_eq", torch.equal(ttnn.to_torch(v), x))
ttnn.close_device(device)
