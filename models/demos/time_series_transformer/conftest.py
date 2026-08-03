# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Time Series Transformer tests."""

import pytest
import torch

import ttnn
from models.demos.time_series_transformer.reference.torch_reference import (
    capture_goldens,
    embedder_weights,
    load_hf_model,
)
from models.demos.time_series_transformer.tt.config import config_from_hf

TRACE_REGION_SIZE = 64 * 1024 * 1024


@pytest.fixture(scope="session")
def device():
    """One shared TTNN device for the whole test session, with room for decode traces.

    ``throw_exception_on_fallback`` makes the runtime raise rather than quietly running an op
    on the host. Without it a silent fallback would still produce correct numbers, so the parity
    tests would pass while the work had left the device -- and the perf numbers would be
    measuring torch. Enabling it here means every test in this suite carries that guarantee.
    """
    ttnn.CONFIG.throw_exception_on_fallback = True
    dev = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="session")
def hf_model():
    return load_hf_model()


@pytest.fixture(scope="session")
def hf_state(hf_model):
    return hf_model.state_dict()


@pytest.fixture(scope="session")
def hf_embedder_weights(hf_model):
    return embedder_weights(hf_model)


@pytest.fixture(scope="session")
def config(hf_model):
    """Runtime config matching the reference checkpoint, on the float32 accuracy path."""
    return config_from_hf(hf_model.config, dtype="float32")


@pytest.fixture(scope="session")
def goldens():
    return capture_goldens()


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
