#!/usr/bin/env python3
"""Tracy bench, final. Cells in call order (parse: composite=TTC triplet, else T):
  A. rows=160 k=2048 plain  -> composite  (TTC)
  B. rows=130 k=2048 plain  -> RP         (T)
  C. rows=30  k=2048 plain  -> RP (auto multi-row disabled)  (T)
  D. rows=160 k=2048 P=2    -> explicit single-program rect  (T)
  E. rows=160 k=512  plain  -> composite (DS-V4 shape)       (TTC)
  F. rows=185 k=2048 plain  -> composite (r2=55, P=2, softened margin) (TTC)
2 warmup + 5 measured iters per cell. Run under: python -m tracy -r -v
"""

import torch
import ttnn

W = 65536
WARMUP, ITERS = 2, 5

device = ttnn.open_device(device_id=0)
device.enable_program_cache()

torch.manual_seed(1234)
tensors = {}
for rows in (160, 130, 30, 185):
    x = torch.randn(1, 1, rows, W, dtype=torch.float32).to(torch.bfloat16)
    tensors[rows] = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def run(rows, k, p):
    kwargs = {"num_slices": p} if p else {}
    out = ttnn.experimental.topk_large_indices(tensors[rows], k=k, **kwargs)
    ttnn.deallocate(out)


CELLS = ((160, 2048, None), (130, 2048, None), (30, 2048, None), (160, 2048, 2), (160, 512, None), (185, 2048, None))
for rows, k, p in CELLS:
    for _ in range(WARMUP + ITERS):
        run(rows, k, p)
    ttnn.synchronize_device(device)

ttnn.close_device(device)
print("BENCH DONE")
