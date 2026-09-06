import os, torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn

device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 32, 4064), (1, 1, 3104, 4064)]:
        for lay in [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]:
            t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
            x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=lay, device=device)
            print("SHAPE", shape, lay)
            out = rms_norm_ttnn(x, epsilon=1e-12)
            ttnn.deallocate(out)
            ttnn.deallocate(x)
finally:
    ttnn.close_device(device)
