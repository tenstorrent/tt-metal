# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression test for issue 56707.

`binary_ng`'s no-broadcast interleaved reader advanced each operand's per-row
page id by the *output's* tile-row width (`Wt`) instead of the operand's own.
When an operand is padded wider than the output -- e.g. via
`ttnn.tilize_with_val_padding` with a caller-supplied padded shape -- every
tile-row past the first landed on the operand's own zero padding instead of
its real data, producing a silent wrong answer on the very first (cold-cache)
call.

The fix passes each operand's own tile-row width to the reader and uses it
for the per-row advance and the c-dimension shift. This test reproduces the
exact width-over-padding shape from the issue (`[64, 64]` padded to
`[64, 128]`, `aWt = 4` vs `cWt = 2`) and checks that rows past the first tile
row match the torch reference instead of the operand's padding.
"""

import torch
import ttnn


def test_binary_ng_width_over_padded_operand(device):
    torch.manual_seed(0)
    ta = torch.rand([64, 64], dtype=torch.bfloat16)
    tb = torch.rand([64, 64], dtype=torch.bfloat16)

    # a: logical [64, 64] but padded to [64, 128] -> aWt = 4, wider than the output's cWt = 2.
    rm = ttnn.from_torch(
        ta, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    a = ttnn.tilize_with_val_padding(rm, [64, 128], 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    b = ttnn.from_torch(
        tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out = ttnn.to_torch(ttnn.add(a, b))
    expected = ta + tb

    # bf16 rounding only; a wrong tile-row read (the pre-fix bug) shows up as the operand's
    # own zero padding leaking in, i.e. rows 32-63 becoming exactly `0 + b`.
    torch.testing.assert_close(out, expected, atol=0.01, rtol=0.01)
