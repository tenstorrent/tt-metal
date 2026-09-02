#!/usr/bin/env python3
"""Bit-exact capture of ttnn.relu_max / ttnn.relu_min over the whole bf16 domain.

The tt-llk harness routes MathOperation.ReluMin/ReluMax to tt-llk's own
_relu_min_/_relu_max_, not to the metal relu_min/relu_max that the branch
rewrites, so these two have to be driven from ttnn.

Run once per header variant (ttnn JIT-compiles the kernel from $TT_METAL_HOME,
so only the kernel cache needs clearing between variants, not the host build):

    TT_METAL_HOME=/localdev/ldjurovic/tt-metal python ttnn_relu_equiv.py <tag>

Dumps int16 bit patterns of the bf16 output, so NaN payloads and -0 are compared
exactly rather than through float equality.
"""
import os
import sys

import numpy as np
import torch
import ttnn

TAG = sys.argv[1]
OUT = os.environ.get("SFPU_EQUIV_OUT", "/tmp/sfpu_equiv")
os.makedirs(OUT, exist_ok=True)

# Every one of the 65,536 bf16 bit patterns, in bit order, padded to a tile grid.
bits = torch.arange(0, 1 << 16, dtype=torch.int32).to(torch.int16)
vals = bits.view(torch.bfloat16)  # includes both zeros, all inf, all NaN
N = vals.numel()
ROWS, COLS = 256, 256
assert ROWS * COLS == N

# Thresholds worth pinning: the ordinary positive case, zero, a negative
# threshold (where clamp() and max(min()) disagree), and a NaN threshold.
THRESHOLDS = [6.0, 1.0, 0.0, -2.5, 3.5]

device = ttnn.open_device(device_id=0)
try:
    x = vals.reshape(ROWS, COLS)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for opname, fn in (("relu_max", ttnn.relu_max), ("relu_min", ttnn.relu_min)):
        for t in THRESHOLDS:
            y = ttnn.to_torch(fn(tx, t))
            ybits = y.to(torch.bfloat16).view(torch.int16).numpy().astype(np.int64)
            xbits = x.view(torch.int16).numpy().astype(np.int64)
            name = f"{opname}_t{t}"
            np.savez_compressed(
                f"{OUT}/{TAG}__{name}.npz", x=xbits.ravel(), y=ybits.ravel()
            )
            print(f"wrote {TAG}__{name}.npz ({ybits.size} points)")
finally:
    ttnn.close_device(device)
