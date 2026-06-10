# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for VibeVoice-1.5B reference tests.

Prepends reference/ so vendored `vibevoice` imports resolve, and tt-metal root
so `models.experimental.vibevoice` imports work when running from repo root.
"""

import sys
from pathlib import Path

import pytest

_VIBEVOICE_ROOT = Path(__file__).resolve().parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
_TT_METAL_ROOT = _VIBEVOICE_ROOT.parent.parent.parent

for path in (_REFERENCE_DIR, _TT_METAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


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
