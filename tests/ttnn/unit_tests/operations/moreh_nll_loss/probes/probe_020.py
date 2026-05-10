import torch
import ttnn

device = ttnn.open_device(device_id=0)

for shape in [[32], [16], [5, 32], [32, 32], [32, 16], [16, 16]]:
    fake = torch.ones(shape, dtype=torch.float32)
    tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    expected = torch.prod(torch.tensor(shape)).item()
    out = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    r = ttnn.operations.moreh.sum(tt_in, dim=None, keepdim=False, output=out)
    val = ttnn.to_torch(r).reshape([-1])[0].item()
    print(f"shape={shape} padded={tt_in.padded_shape} sum={val} expected={expected}")

ttnn.close_device(device)
