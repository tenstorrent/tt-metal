# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.experimental.SSD512.tests.pcc.test_ssd import test_ssd512_network


@pytest.mark.parametrize("device_params", [{"l1_small_size": 98304}], indirect=True)
@pytest.mark.parametrize("pcc", [(0.97)])
@pytest.mark.parametrize("size", [(512)])
def test_ssd512(device, pcc, size, reset_seeds):
    test_ssd512_network(device=device, pcc=pcc, size=size, reset_seeds=reset_seeds)
