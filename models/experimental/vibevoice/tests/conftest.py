# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration and shared fixtures for VibeVoice-1.5B tests.

Prepends the tt-metal root so ``models.experimental.vibevoice`` imports resolve when
pytest is invoked from outside the repo root.
"""

import sys
from pathlib import Path

import pytest

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]

if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))


@pytest.fixture(scope="session", autouse=True)
def vibevoice_demo_resources():
    """Download demo text/voice assets once per session from upstream GitHub."""
    from models.experimental.vibevoice.common.resource_utils import ensure_demo_resources

    try:
        ensure_demo_resources()
    except Exception as exc:
        pytest.skip(
            f"VibeVoice demo resources unavailable: {exc}. "
            "Ensure network access to github.com or pre-populate "
            "models/experimental/vibevoice/resources/."
        )


@pytest.fixture(scope="session", autouse=True)
def vibevoice_model_weights(model_location_generator):
    """Download weights once per session and expose them via config.MODEL_PATH."""
    from models.experimental.vibevoice.common import config
    from models.experimental.vibevoice.common.model_utils import ensure_model_weights

    try:
        model_path = ensure_model_weights(model_location_generator=model_location_generator)
    except Exception as exc:
        pytest.skip(
            f"VibeVoice weights unavailable: {exc}. "
            f"Set {config.MODEL_PATH_ENV_VAR}, install huggingface_hub, "
            "or ensure network access for auto-download."
        )

    config.MODEL_PATH = str(model_path)
    return model_path


@pytest.fixture(scope="session")
def model_path(vibevoice_model_weights):
    return vibevoice_model_weights


@pytest.fixture(scope="module")
def vv_config(model_path):
    """VibeVoice model config loaded from session weights path."""
    from models.experimental.vibevoice.tt.vibevoice_config import load_vibevoice_model_config

    return load_vibevoice_model_config(model_path)


@pytest.fixture(scope="module")
def lm_state(model_path):
    """LM submodule state dict remapped for TT transformers."""
    from models.experimental.vibevoice.tt.load_weights import (
        load_vibevoice_state_dict,
        remap_lm_keys_to_tt_transformers,
        split_submodule_weights,
    )

    full_sd = load_vibevoice_state_dict(model_path)
    sub = split_submodule_weights(full_sd)
    return remap_lm_keys_to_tt_transformers(sub["lm"])
