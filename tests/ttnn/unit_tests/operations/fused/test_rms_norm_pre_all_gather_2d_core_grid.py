# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #64187.

``LayerNormPreAllGather2DProgramFactory`` only supports RMSNorm: its compute kernel
computes ``sum(x**2)`` and the factory hardcodes one output tile per row, while
``compute_output_specs`` makes the LayerNorm output two tiles per row. A LayerNorm call
that reached the 2D core grid path would therefore write the statistics to the wrong
output tiles and leave half of the output unwritten. The device operation now rejects
that combination.

The public LayerNorm API never requests the 2D core grid (``layer_norm_pre_all_gather``
always passes ``std::nullopt`` for ``use_2d_core_grid``), so the rejected combination is
only reachable from internal C++ callers of the prim. These tests therefore cover the
reachable contract: the RMSNorm 2D core grid path still runs and still produces
``sum(x**2)``, and it agrees with the 1D path.
"""

import pytest
import torch
import ttnn

# The 2D core grid path is exercised with the same shape family as the upstream
# distributed RMSNorm tests: 128 rows (4 tile rows) and a hidden size that is a whole
# number of 32-wide tiles.
SEQ_LEN = 128
HIDDEN_DIM = 1024
TILE_WIDTH = 32


def _make_input(device):
    # A constant input keeps the expected statistic exactly representable in bfloat16:
    # sum(x**2) = HIDDEN_DIM * 4 = 4096.
    x_torch = torch.full((1, 1, SEQ_LEN, HIDDEN_DIM), 2.0, dtype=torch.bfloat16)
    x_ttnn = ttnn.from_torch(x_torch, device=device, layout=ttnn.TILE_LAYOUT)
    return x_torch, x_ttnn


@pytest.mark.parametrize("use_2d_core_grid", [False, True])
def test_rms_norm_pre_all_gather_sum_of_squares(use_2d_core_grid, device):
    """The statistics are sum(x**2) over the last dimension, in the left most column."""
    x_torch, x_ttnn = _make_input(device)

    stats = ttnn.rms_norm_pre_all_gather(x_ttnn, use_2d_core_grid=use_2d_core_grid)
    stats_torch = ttnn.to_torch(stats)

    # RMSNorm computes a single statistic, so the output is one tile wide.
    assert stats_torch.shape == (1, 1, SEQ_LEN, TILE_WIDTH)

    expected = (x_torch.float() ** 2).sum(dim=-1)
    torch.testing.assert_close(stats_torch[..., 0].float(), expected, rtol=1e-2, atol=1e-2)


def test_rms_norm_pre_all_gather_2d_core_grid_matches_1d(device):
    """The 2D core grid path must produce the same statistics as the 1D path."""
    _, x_ttnn = _make_input(device)

    stats_1d = ttnn.to_torch(ttnn.rms_norm_pre_all_gather(x_ttnn, use_2d_core_grid=False))
    stats_2d = ttnn.to_torch(ttnn.rms_norm_pre_all_gather(x_ttnn, use_2d_core_grid=True))

    assert stats_2d.shape == stats_1d.shape
    torch.testing.assert_close(stats_2d.float(), stats_1d.float(), rtol=1e-2, atol=1e-2)
