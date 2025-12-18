# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import json

from models.demos.gpt_oss.tt.model_config import ModelArgs


@pytest.fixture(scope="session")
def state_dict():
    model_path = os.getenv("HF_MODEL", None)
    if model_path is None:
        return None
    else:
        return ModelArgs.load_state_dict(model_path, dummy_weights=False)


@pytest.fixture
def test_thresholds(request):
    return json.load(open("models/demos/gpt_oss/unit_test_thresholds.json", "r"))