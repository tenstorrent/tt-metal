# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers import HardwareController


@pytest.fixture(scope="function", autouse=True)
def manage_hardware_controller():
    # Setup: initialize the hardware controller
    controller = HardwareController()
    controller.reset_card()
    yield controller

    # Teardown:
