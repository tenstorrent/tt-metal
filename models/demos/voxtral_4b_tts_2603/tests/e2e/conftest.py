# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device and reference fixtures for the end-to-end suite.

The fixture is the SOLE opener of the device. `tt/` must never open one -- it runs on the device
handed to `build_pipeline(device, ...)`, and a second ad-hoc open inside the importable pipeline
path creates a competing device with a different command-queue count (the
`id < mesh_command_queues_.size()` fatal that breaks trace).
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common

# Matches device_session.py, which is what the standalone selftest entrypoints open, so the suite
# and the observers are testing the same configuration.
L1_SMALL_SIZE = 24576
TRACE_REGION_SIZE = 200 * 1024 * 1024


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(
        device_id=int(os.environ.get("VOXTRAL_DEVICE_ID", "0")),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=TRACE_REGION_SIZE,
        num_command_queues=1,
    )
    try:
        yield dev
    finally:
        ttnn.close_device(dev)


@pytest.fixture(scope="session")
def hf_model():
    """Source A's reference, rebuilt from the native checkpoint by Source B's reference loader."""
    common.use_all_cpu_threads()
    torch.manual_seed(0)
    return common.load_reference_model()
