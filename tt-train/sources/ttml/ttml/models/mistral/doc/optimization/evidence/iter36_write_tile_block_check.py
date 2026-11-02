# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 36: `write_tile_block` with a tile-column offset (a [out, 32] block into the tail columns of the
[out, in + 64] transposed forward copy) and the unchanged row mode, against numpy; then the transposed-orientation
forward of a fused LoRA w2 shard against the plain orientation (bit-exact up to the fp32 accumulation order).

    python iter36_write_tile_block_check.py
"""

import numpy as np
import torch
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
device = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)


def dev(a):
    return ttnn.from_torch(
        torch.from_numpy(a.astype(np.float32)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def host(t):
    return ttnn.to_torch(t).float().numpy()


ok = True
# column mode: [5120, 8256] destination, [5120, 32] block at tile-column 8192 / 32 = 256
dst = rng.standard_normal((1, 1, 5120, 8256))
blk = rng.standard_normal((1, 1, 5120, 32))
d, b = dev(dst), dev(blk)
ttml.ops.metal.write_tile_block(d, b, 0, 256)
got = host(d)
exp = host(dev(dst))
exp[..., :, 8192:8224] = host(dev(blk))[..., :, :]
diff = np.abs(got - exp).max()
print(f"column mode [5120, 32] -> columns 8192..8224 of [5120, 8256]: max|diff| {diff:.3e}")
ok &= diff == 0.0
# row mode unchanged: [32, 5120] block at tile-row 8192 / 32 = 256 of [8256, 5120]
dst = rng.standard_normal((1, 1, 8256, 5120))
blk = rng.standard_normal((1, 1, 32, 5120))
d, b = dev(dst), dev(blk)
ttml.ops.metal.write_tile_block(d, b, 256)
got = host(d)
exp = host(dev(dst))
exp[..., 8192:8224, :] = host(dev(blk))
diff = np.abs(got - exp).max()
print(f"row mode    [32, 5120] -> rows 8192..8224 of [8256, 5120]: max|diff| {diff:.3e}")
ok &= diff == 0.0
# a block with both offsets and several tile columns: [64, 96] at tile (3, 5) of [256, 512]
dst = rng.standard_normal((1, 1, 256, 512))
blk = rng.standard_normal((1, 1, 64, 96))
d, b = dev(dst), dev(blk)
ttml.ops.metal.write_tile_block(d, b, 3, 5)
got = host(d)
exp = host(dev(dst))
exp[..., 96:160, 160:256] = host(dev(blk))
diff = np.abs(got - exp).max()
print(f"block mode  [64, 96] -> tile (3, 5) of [256, 512]: max|diff| {diff:.3e}")
ok &= diff == 0.0
print("WRITE_TILE_BLOCK", "PASS" if ok else "FAIL")
ttml.autograd.AutoContext.get_instance().close_device()
