# Ring-size sweep: does a DM->DM implicit-sync ring over 64 KiB fail? (one process: stops at first abort)
import torch, ttnn, sys
from ttnn.operations.slice.slice import _slice
from ttnn.operations.slice import slice_program_artifacts as A
dev = ttnn.open_device(device_id=0)
cases = [(4096, 256), (2048, 512), (2048, 544), (4096, 288)]
for W, Ho in cases:
    t = torch.randn((1, Ho + 32, W), dtype=torch.bfloat16)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    per_core = -(-Ho // 32)
    blk = A.block_pages_for(per_core, x.buffer_aligned_page_size(), (1, 1))
    print("RES start", W, Ho, "entry", x.buffer_aligned_page_size(), "block", blk, "ring", 2 * blk * x.buffer_aligned_page_size(), flush=True)
    out = ttnn.to_torch(_slice(x, (0, 32, 0), (1, Ho + 32, W)))
    print("RES ok", W, Ho, torch.equal(out, t[:, 32:]), flush=True)
ttnn.close_device(dev)
