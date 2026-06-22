# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Debug harness for TRANSPORT_REDUCE_BCAST (mode 2) numerical mismatch.
# all-ones (1,1,32,2048), K=8 -> Wt_s=8 per shard.
#   local Sigma x^2 per shard = 8 tiles * 32 cols = 256 (per stick)
#   global Sigma x^2 = 8 * 256 = 2048 = W
#   out = 1 * rsqrt(2048 * (1/2048) + 1e-6) = rsqrt(1.000001) ~= 1.0
# So cb_partial_sumsq col0 should print 2048.0 on EVERY core (root and peers).
import torch

import ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc


def test_mode2_all_ones(device):
    desc._FORCE_REGIME = "B"
    desc._FORCE_TRANSPORT = 2
    desc._FORCE_K = 8

    shape = (1, 1, 32, 2048)
    W = shape[-1]
    x = torch.ones(*shape)
    t = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(t)).float()

    # Per-shard first-col (one column per K-shard of width W/K).
    shard_w = W // 8
    first_cols = [out[0, 0, 0, k * shard_w].item() for k in range(8)]
    print("PER_SHARD_FIRST_COL:", first_cols)
    maxerr = (out - 1.0).abs().max().item()
    print("MAXERR:", maxerr)
    assert maxerr < 0.1, f"all-ones not exact, first_cols={first_cols}"
