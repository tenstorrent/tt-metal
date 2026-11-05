# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest

from models.demos.minimax_m3.tt.model_config import ModelArgs


def pytest_addoption(parser):
    parser.addoption("--skip-model-load", action="store_true", default=False, help="Skip loading the model state dict")


@pytest.fixture(scope="session")
def state_dict(request):
    load_model = not request.config.getoption("--skip-model-load")
    model_path = os.getenv("HF_MODEL", None)
    if model_path is None or not load_model:
        return {}
    else:
        return ModelArgs.load_state_dict(model_path, dummy_weights=False)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Remove the MoE overlap sub-device managers before the test's fixtures close the mesh."""
    overlap = sys.modules.get("models.demos.minimax_m3.tt.moe.shared_overlap")
    if overlap is not None:
        overlap.release_all()
    return (yield)
