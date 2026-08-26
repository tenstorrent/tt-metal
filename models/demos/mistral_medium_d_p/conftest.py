# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Package-level fixtures for mistral_medium_d_p. Mirrors ``minimax_m3/conftest.py``.

The block tests run on RANDOM weights (nothing here needs the 125 GB checkpoint), so this only
exists for the real-weight path: point ``HF_MODEL`` at a checkpoint to load it once per session.
"""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--skip-model-load", action="store_true", default=False, help="Skip loading the model state dict")


@pytest.fixture(scope="session")
def state_dict(request):
    """Real Mistral-Medium-3.5 weights, loaded once per session. Empty dict when HF_MODEL is unset
    or --skip-model-load is passed — which is the normal path for the block tests."""
    from models.demos.mistral_medium_d_p.tt.model_config import ModelArgs

    load_model = not request.config.getoption("--skip-model-load")
    model_path = os.getenv("HF_MODEL", None)
    if model_path is None or not load_model:
        return {}
    return ModelArgs.load_state_dict(model_path, dummy_weights=False)
