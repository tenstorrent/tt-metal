# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from ....pipelines.minimax_h3.pipeline_minimax_h3 import minimax_h3_worker_l1_size


@pytest.fixture(scope="function")
def device_params(request):
    """The root `device_params`, plus the `worker_l1_size` every MiniMax-H3 deployment opens with.

    Filled in here rather than in each parametrize list so no H3 test opens a mesh with a layout that is
    not deployed (see `MINIMAX_H3_KERNEL_CONFIG_RING_BYTES`), and at fixture time because the value comes
    from the HAL, which collection must not touch. A test that sets its own `worker_l1_size` keeps it.
    """
    params = dict(getattr(request, "param", {}))
    params.setdefault("worker_l1_size", minimax_h3_worker_l1_size())
    return params
