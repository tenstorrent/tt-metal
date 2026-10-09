# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the fused decode gate (tt/fused_decode/config.py).

fused_decode_layout_supported decides which decoder layers use the fused decode path. It must accept exactly the
validated envelope (gpt-oss-20b, Blackhole 1x4, TP=4, one token per device) and keep every other configuration on
the original decode path. No device is needed.
"""

from unittest.mock import MagicMock

import pytest

from models.demos.gpt_oss.tt.fused_decode.config import (
    FUSED_DECODE_DRAM_BANKS,
    FUSED_DECODE_MIN_GRID,
    fused_decode_layout_supported,
    fused_decode_supported,
)

# gpt-oss-20b on a QuietBox 2: Blackhole, mesh 1x4, TP=4, EP=1, 32 experts, batch 1, 11x10 grid, 8 DRAM banks.
SUPPORTED = dict(
    is_blackhole=True,
    mesh_shape=(1, 4),
    tp=4,
    ep=1,
    num_experts=32,
    use_throughput_experts=False,
    tokens=1,
    grid=(11, 10),
    dram_banks=8,
)


def test_supported_envelope():
    assert fused_decode_layout_supported(**SUPPORTED)


def test_minimum_grid_is_supported():
    assert fused_decode_layout_supported(**{**SUPPORTED, "grid": FUSED_DECODE_MIN_GRID})


@pytest.mark.parametrize(
    "change",
    [
        {"is_blackhole": False},
        {"mesh_shape": (1, 1), "tp": 1},
        {"mesh_shape": (1, 2), "tp": 2},
        {"mesh_shape": (1, 8), "tp": 8},
        {"mesh_shape": (4, 8), "tp": 8, "ep": 4},
        {"tp": 2},
        {"ep": 4},
        {"num_experts": 128},
        {"use_throughput_experts": True},
        {"tokens": 32},
        {"grid": (FUSED_DECODE_MIN_GRID[0] - 1, 10)},
        {"grid": (11, FUSED_DECODE_MIN_GRID[1] - 1)},
        {"dram_banks": FUSED_DECODE_DRAM_BANKS - 1},
    ],
    ids=[
        "wormhole",
        "mesh_1x1",
        "mesh_1x2",
        "mesh_1x8",
        "mesh_4x8",
        "tp_2",
        "ep_4",
        "gpt_oss_120b_experts",
        "throughput_experts",
        "batch_32",
        "grid_too_narrow",
        "grid_too_short",
        "seven_dram_banks",
    ],
)
def test_outside_envelope_keeps_original_path(change):
    assert not fused_decode_layout_supported(**{**SUPPORTED, **change})


def test_no_mesh_config_keeps_original_path():
    # Without a mesh config the device is never queried.
    mesh_device = MagicMock()
    assert not fused_decode_supported(mesh_device, None, MagicMock(), False, 1)
    mesh_device.compute_with_storage_grid_size.assert_not_called()
