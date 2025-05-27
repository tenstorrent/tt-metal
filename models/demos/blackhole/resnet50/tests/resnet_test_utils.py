# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.utility_functions import is_blackhole


def skip_resnet_if_blackhole_p100(device):
    is_p100 = (
        is_blackhole() and device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y != 130
    )
    if is_p100:
        pytest.skip("Expected to run only on blackhole devices with 130 cores (unharvested grid), see #21319")
