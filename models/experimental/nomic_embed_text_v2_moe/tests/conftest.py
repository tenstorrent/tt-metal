# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the nomic-embed-text-v2-moe reference tests.

Loading the checkpoint costs ~1.8 GB of host RAM and several seconds, and the upstream HF
model costs that again, so both are session-scoped. Tests that need neither are kept
strictly free of these fixtures -- `test_reference_modules.py` runs with no network and no
weights at all, which is what keeps the structural suite usable in CI.
"""

import pytest
import torch

from models.experimental.nomic_embed_text_v2_moe.common import checkpoint_is_cached, resolve_checkpoint
from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import load_vendored_config
from models.experimental.nomic_embed_text_v2_moe.reference.loader import (
    load_reference_model,
    load_state_dict_from_safetensors,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_weights: requires the pinned checkpoint (~1.8 GB)")


@pytest.fixture(scope="session")
def config():
    """The pinned config, validated through `from_hf_config`. No network."""
    return load_vendored_config()


@pytest.fixture(scope="session")
def checkpoint_path():
    if not checkpoint_is_cached():
        pytest.skip("pinned checkpoint is not in the local HF cache")
    return resolve_checkpoint(allow_download=False)


@pytest.fixture(scope="session")
def state_dict(checkpoint_path):
    return load_state_dict_from_safetensors(checkpoint_path)


@pytest.fixture(scope="session")
def reference_model(config, state_dict):
    """The vendored reference with real weights, loaded `strict=True`."""
    return load_reference_model(config, state_dict)


@pytest.fixture(scope="session")
def hf_model():
    """The upstream model, guaranteed to come from remote code rather than native."""
    from models.experimental.nomic_embed_text_v2_moe.reference.hf_reference import load_hf_model

    try:
        return load_hf_model()
    except Exception as exc:  # network down, or hub unreachable
        pytest.skip(f"could not load the upstream HF model: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="session")
def tokenizer():
    from models.experimental.nomic_embed_text_v2_moe.common import load_tokenizer

    try:
        return load_tokenizer()
    except Exception as exc:
        pytest.skip(f"could not load the tokenizer: {type(exc).__name__}: {exc}")


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    yield
