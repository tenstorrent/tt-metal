# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.mark.parametrize("num_devices", [(8)])
def test_all_devices(
    all_devices,
    num_devices,
):
    print("Testing All Devices")
