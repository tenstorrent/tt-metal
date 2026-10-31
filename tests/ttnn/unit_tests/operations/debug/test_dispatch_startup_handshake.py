# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.common.utility_functions import is_blackhole


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole worker-core dispatch")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "device_params", [{"num_command_queues": 1}, {"num_command_queues": 2}], indirect=True, ids=["cq1", "cq2"]
)
def test_dispatch_startup_handshake(device):
    """The device fixture must finish both open and close; inject the documented delay to reproduce the race."""
    ttnn.synchronize_device(device)
