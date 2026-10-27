# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import ttnn
from models.common.device_utils import get_device_name


@pytest.mark.parametrize(
    "num_devices, expected",
    [(1, "P150"), (2, "P300"), (4, "P150x4"), (8, "P150x8"), (32, "BHGLX")],
)
def test_quasar_takes_the_blackhole_device_names(num_devices, expected):
    mesh_device = SimpleNamespace(
        arch=lambda: ttnn.device.Arch.QUASAR,
        get_num_devices=lambda: num_devices,
        dram_grid_size=lambda: SimpleNamespace(x=2),
    )
    assert get_device_name(mesh_device) == expected
