# Entry-size sweep at the failing geometry (80 rows: 16 cores x 3 pages + 16 x 2, block 2, ring 4)
import torch, ttnn, sys
from ttnn.operations.slice.slice import _slice
from ttnn.operations.slice import slice_program_artifacts as A
dev = ttnn.open_device(device_id=0)
W = int(__import__("os").environ.get("PW", "0"))
for W in (4096, 8192, 8208, 8400):
    for Ho in (96, 80):
        t = torch.randn((1, Ho + 64, W), dtype=torch.bfloat16)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        print("RES start", W, Ho, "entry", x.buffer_aligned_page_size(), flush=True)
        out = ttnn.to_torch(_slice(x, (0, 64, 0), (1, Ho + 64, W)))
        print("RES ok", W, Ho, torch.equal(out, t[:, 64:]), flush=True)
ttnn.close_device(dev)
