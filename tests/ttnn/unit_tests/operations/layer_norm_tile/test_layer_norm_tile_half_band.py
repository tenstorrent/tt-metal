# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regime R3 (16-row half-tile bands, Refinement 4): the gate's branches and the band pairings.

Pairings covered: whole pairs, a lone last band, a lone first band (odd start), split pairs that straddle
tile-rows and images, all-padding bands skipped (H % 32 in [1, 16]), one band per core, several blocks
per core with the residual on either RISC.
"""

import importlib

import pytest
import torch
import ttnn

descriptor = importlib.import_module("ttnn.operations.layer_norm_tile.layer_norm_tile_program_descriptor")
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_layer_norm_tile import (
    DRAM,
    L1,
    W,
    compute_config,
    run_and_check,
    to_device,
)

CASES = [
    # 128 tile-rows on 110 cores: 2 / 3 bands, even-first with the DRAM residual, heavy-first without.
    pytest.param((8, 1, 512, W), DRAM, True, True, id="8x512_dram_residual"),
    pytest.param((8, 1, 512, W), L1, False, True, id="8x512_l1_no_residual"),
    # 1000 % 32 = 8: each image's last bottom band is skipped; runs straddle images.
    pytest.param((4, 1, 1000, W), DRAM, True, True, id="4x1000_skip_band_residual"),
    # 225 bands: 5 three-band cores first, so the two-band runs start on a bottom band (split pairs).
    pytest.param((3, 1, 1200, W), DRAM, False, True, id="3x1200_skip_band_odd_starts"),
    # One band per core.
    pytest.param((2, 1, 500, W), L1, True, True, id="2x500_one_band_per_core"),
    pytest.param((5, 1, 16, W), DRAM, True, True, id="5x16_single_band_images"),
    pytest.param((30, 1, 40, W), DRAM, False, True, id="30x40_skip_band_per_image"),
    # Several blocks per core (12 / 13 bands); 3 / 4 bands with skips in L1.
    pytest.param((7, 1, 3200, W), DRAM, True, True, id="7x3200_multi_block_residual"),
    pytest.param((5, 1, 1100, W), L1, False, True, id="5x1100_skip_band_l1"),
    # R1 kept: a balanced R1 split that R3 would unbalance, and a multiple of the grid.
    pytest.param((8, 1, 384, W), DRAM, True, False, id="8x384_r1"),
    pytest.param((11, 1, 640, W), DRAM, True, False, id="11x640_r1"),
]


def _selected_half_band(device, shape, buf, with_residual):
    x = to_device(torch.zeros(shape, dtype=torch.bfloat16), device, buf)
    residual = x if with_residual else None
    blocking = descriptor.select_blocking(x, residual=residual, compute_kernel_config=compute_config())
    return blocking.half_band


@pytest.mark.parametrize("shape,buf,with_residual,half_band", CASES)
def test_half_band(device, shape, buf, with_residual, half_band):
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y != 110:
        pytest.skip("regime expectations are for the 110-core BH p100a grid")
    assert _selected_half_band(device, shape, buf, with_residual) == half_band
    run_and_check(device, shape, with_residual=with_residual, input_buf=buf, output_buf=buf)


KNOBS = {
    "heavy_first": dict(HALF_BAND_EVEN_FIRST=False),
    "even_first": dict(HALF_BAND_EVEN_FIRST=True),
    "stream": dict(HALF_BAND_STREAM=True),
    "noc_plan_off": dict(HALF_BAND_NOC_PLAN=False),
    "residual_on_reader": dict(RESIDUAL_ON_WRITER=False),
}


@pytest.mark.parametrize("knobs", list(KNOBS), ids=list(KNOBS))
@pytest.mark.parametrize(
    "shape,buf,with_residual",
    [
        pytest.param((8, 1, 512, W), DRAM, True, id="8x512_dram_residual"),
        pytest.param((3, 1, 1200, W), L1, True, id="3x1200_skip_band"),
    ],
)
def test_half_band_knobs(device, monkeypatch, knobs, shape, buf, with_residual):
    for name, value in KNOBS[knobs].items():
        monkeypatch.setattr(descriptor, name, value)
    run_and_check(device, shape, with_residual=with_residual, input_buf=buf, output_buf=buf)


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32_dest", "bf16_dest"])
def test_half_band_dest_modes(device, fp32_dest_acc_en):
    run_and_check(
        device,
        (4, 1, 1000, W),
        with_residual=True,
        input_buf=L1,
        residual_buf=DRAM,
        output_buf=DRAM,
        config=compute_config(fp32_dest_acc_en=fp32_dest_acc_en),
    )
