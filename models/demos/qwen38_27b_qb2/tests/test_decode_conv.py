# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact retained-row equivalence and independent-user packing on device."""

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tt.decode_conv import make_actual_start, packed_decode_conv


@pytest.mark.parametrize("batch", [8, 16, 24, 32])
@pytest.mark.parametrize("seed", [0, 17])
def test_packed_decode_conv_matches_independent_users(device, batch, seed):
    torch.manual_seed(seed)
    widths = (512, 512, 1536)
    channels = sum(widths)

    def upload(tensor, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(tensor.bfloat16(), device=device, layout=layout)

    row = upload(torch.randn(batch, 32, channels))
    history = upload(torch.randn(batch, 3, channels))
    taps = [upload(torch.randn(1, 1, channels), ttnn.TILE_LAYOUT) for _ in range(4)]
    actual_start = make_actual_start(device)
    reference = [
        ttnn.experimental.kda.qkv_causal_conv1d_silu(
            row[i : i + 1],
            history[i : i + 1],
            *taps,
            *widths,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
            actual_start=actual_start,
            predecessor_carry=history[i : i + 1],
        )
        for i in range(batch)
    ]
    result = packed_decode_conv(row, history, taps, widths, actual_start)
    for component, output in enumerate(result):
        expected = torch.cat([ttnn.to_torch(parts[component])[:, :1] for parts in reference])
        actual = ttnn.to_torch(output)
        assert torch.equal(actual[:, :1], expected)
        assert torch.count_nonzero(actual[:, 1:]) == 0

    # Changing one user's state must not leak into any other retained row.
    changed_history = ttnn.to_torch(history).clone()
    changed_history[batch // 2] *= -3
    changed = packed_decode_conv(row, upload(changed_history), taps, widths, actual_start)
    keep = [i for i in range(batch) if i != batch // 2]
    for original, modified in zip(result, changed):
        assert torch.equal(ttnn.to_torch(original)[keep], ttnn.to_torch(modified)[keep])
