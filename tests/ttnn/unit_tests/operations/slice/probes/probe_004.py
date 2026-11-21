# Packet-limit boundary: 8,256-B entries (just over the 8 KiB overlay packet), block 2
import torch, ttnn
from ttnn.operations.slice.slice import _slice
dev = ttnn.open_device(device_id=0)
W, Ho = 4128, 96
t = torch.randn((1, Ho + 64, W), dtype=torch.bfloat16)
x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
print("RES start entry", x.buffer_aligned_page_size(), flush=True)
out = ttnn.to_torch(_slice(x, (0, 64, 0), (1, Ho + 64, W)))
print("RES ok", torch.equal(out, t[:, 64:]), flush=True)
ttnn.close_device(dev)
