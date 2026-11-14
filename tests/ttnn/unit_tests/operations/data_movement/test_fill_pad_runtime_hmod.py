# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


# Many different H % 32 values inside a few padded heights, plus heights that need no bottom fill.
LENGTHS = [33, 37, 45, 50, 63, 64, 1, 7, 31, 993, 1000, 1013, 1022, 1024, 8191, 8200, 8319]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("width", [1344, 100])
def test_fill_pad_bottom_rows_one_program_per_padded_height(device, width, dtype):
    torch.manual_seed(0)
    new_programs = 0
    buckets = set()
    for length in LENGTHS:
        x = ttnn.from_torch(torch.randn(1, length, width), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y = ttnn.add(x, 1.0)  # eltwise covers whole tiles, so the padding becomes nonzero
        if length % 32:
            assert torch.count_nonzero(y.cpu().to_torch_with_padded_shape()[:, length:]) > 0, "precondition"
        before = device.num_program_cache_entries()
        z = ttnn.fill_implicit_tile_padding(y, 0.0)
        new_programs += device.num_program_cache_entries() - before
        buckets.add((z.padded_shape[1], length % 32 == 0))
        full = z.cpu().to_torch_with_padded_shape()
        expected = y.cpu().to_torch_with_padded_shape()
        assert torch.equal(full[:, :length, :width], expected[:, :length, :width]), f"length {length}"
        assert torch.count_nonzero(full[:, length:]) == 0, f"length {length}: bottom padding not filled"
        assert torch.count_nonzero(full[:, :, width:]) == 0, f"length {length}: right padding not filled"
    # Heights that are a multiple of 32 with no right pad need no fill and launch no program.
    expected_programs = len([b for b in buckets if not (b[1] and width % 32 == 0)])
    assert new_programs == expected_programs, (new_programs, sorted(buckets))
