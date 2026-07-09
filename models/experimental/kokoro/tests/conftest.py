# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device fixtures for Kokoro TTNN tests under ``models/experimental/kokoro/tests``."""

from __future__ import annotations

import pytest

import ttnn


@pytest.fixture(scope="function")
def device(request):
    """Single-device fixture.

    When parametrized with ``indirect=True`` (e.g. to set ``num_command_queues``),
    the parametrize value is a dict merged with the defaults below::

        @pytest.mark.parametrize("device", [{"num_command_queues": 2}], indirect=True)

    Unset keys fall back to: ``l1_small_size=24576``, ``num_command_queues=1``,
    ``trace_region_size=0`` (0 = tracing disabled; set it to reserve a DRAM trace
    region for ``ttnn.begin_trace_capture``).
    """
    params: dict = getattr(request, "param", {})
    l1_small_size = params.get("l1_small_size", 24576)
    num_command_queues = params.get("num_command_queues", 1)
    trace_region_size = params.get("trace_region_size", 0)
    dev = ttnn.open_device(
        device_id=0,
        l1_small_size=l1_small_size,
        num_command_queues=num_command_queues,
        trace_region_size=trace_region_size,
    )
    yield dev
    ttnn.close_device(dev)
