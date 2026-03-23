# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest

from models.demos.gpt_oss.tt.model_config import ModelArgs


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


@pytest.fixture
def test_thresholds(request):
    with open("models/demos/gpt_oss/unit_test_thresholds.json", "r") as f:
        thresholds = json.load(f)
    return thresholds
