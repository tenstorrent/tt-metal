# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TtModelConfig's device check, on stand-in devices: it needs no hardware, so it runs on every machine."""

import pytest

import ttnn

from models.experimental.nomic_embed_text_v2_moe.tt.model_config import (
    GROUPED_WEIGHT_BANKS,
    TtModelConfig,
    UnsupportedDeviceError,
    check_device_supported,
)


class StandInDevice:
    """Answers only the DRAM grid query: the check must come before anything else reads the device."""

    def __init__(self, x: int, y: int):
        self.grid = ttnn.CoreCoord(x, y)

    def dram_grid_size(self) -> ttnn.CoreCoord:
        return self.grid


@pytest.mark.parametrize("x, y", [(7, 1), (12, 1), (4, 1)])
def test_a_device_without_eight_dram_banks_is_refused_before_anything_loads(x, y, expect_error):
    """A p100 has 7 banks; the chunked weights spread over exactly GROUPED_WEIGHT_BANKS."""
    with expect_error(UnsupportedDeviceError, f"needs a device with {GROUPED_WEIGHT_BANKS} DRAM banks"):
        TtModelConfig.from_device(StandInDevice(x, y))


def test_eight_dram_banks_pass_the_check():
    check_device_supported(StandInDevice(GROUPED_WEIGHT_BANKS, 1))
