# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.utility_functions import print_diff_argmax
from models.common.utility_functions import is_wormhole_b0


def golden_fill_rm(n, c, h, w, fill_h, fill_w, val_hi, val_lo):
    xp = torch.full((n, c, h, w), val_lo)
    xp[:, :, :fill_h, :fill_w] = val_hi
    return xp


def test_fill_rm_program_cache(device):
    # Program-cache-hit correctness after Metal 2.0 TensorBinding on the output.
    # On a cache hit the factory is not rebuilt; the output buffer address is
    # patched via tensor_args. Run fill_rm twice with freshly allocated outputs
    # (prior tensors kept alive in `held` so the allocator hands out different
    # addresses) and verify both results and that the second call reuses the
    # cached program. A stale output binding would fail the second check.
    N = 2
    C = 3
    H = 64
    W = 96
    fillH = 33
    fillW = 31
    val_hi = 2.0
    val_lo = -1.0

    if is_wormhole_b0():
        N, C, H, W = [1, 1, 32, 32]
        fillH = 31
        fillW = 31

    device.clear_program_cache()
    golden = golden_fill_rm(N, C, H, W, fillH, fillW, val_hi, val_lo)

    held = []
    entries = None
    for i in range(2):
        x = torch.zeros((N, C, H, W))
        xt = (
            ttnn.Tensor(
                x.reshape(-1).tolist(),
                x.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        output = ttnn.fill_rm(N, C, H, W, fillH, fillW, xt, val_hi, val_lo)
        tt_got_back = output.cpu().to_torch()
        assert torch.equal(tt_got_back, golden)
        held.extend([xt, output])

        if i == 0:
            entries = device.num_program_cache_entries()
            assert entries == 1, f"fill_rm should cache exactly one program, got {entries}"
        else:
            assert (
                device.num_program_cache_entries() == entries
            ), "cache-hit run created a new program entry instead of reusing the cached one"


def test_fill_rm(device):
    N = 2
    C = 3
    H = 64
    W = 96

    fillH = 33
    fillW = 31

    if is_wormhole_b0():
        N, C, H, W = [1, 1, 32, 32]
        fillH = 31
        fillW = 31

    x = torch.zeros((N, C, H, W))
    xp = torch.clone(x)
    xp[:, :, :fillH, :fillW] = 1.0

    xt = (
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(device)
    )
    xtt = ttnn.fill_ones_rm(N, C, H, W, fillH, fillW, xt)
    assert list(xtt.padded_shape) == [N, C, H, W]

    tt_got_back = xtt.cpu().to_torch()

    # x[1,1,2,2] = 2.0
    print("reshape() max absdiff=")
    print_diff_argmax(tt_got_back, xp)
    eq = torch.equal(tt_got_back, xp)
    assert eq
