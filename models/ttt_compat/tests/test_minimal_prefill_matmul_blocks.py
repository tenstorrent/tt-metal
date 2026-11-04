# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.ttt_compat.tt import model_config


@pytest.mark.parametrize(
    "wormhole, num_devices, grid, expected",
    [
        (True, 1, (8, 8), (4, 4)),
        (True, 1, (8, 7), (8, 1)),
        (True, 1, (7, 8), (8, 1)),
        (True, 2, (8, 8), (8, 1)),
        (True, 8, (8, 7), (8, 1)),
        (True, 1, None, (8, 1)),
        (False, 1, (8, 8), (8, 1)),
        (False, 1, (11, 10), (8, 1)),
    ],
)
def test_minimal_prefill_matmul_blocks(monkeypatch, wormhole, num_devices, grid, expected):
    monkeypatch.setattr(model_config, "is_wormhole_b0", lambda: wormhole)
    core_grid = None if grid is None else ttnn.CoreGrid(x=grid[0], y=grid[1])
    assert model_config.minimal_prefill_matmul_blocks(num_devices, core_grid) == expected
