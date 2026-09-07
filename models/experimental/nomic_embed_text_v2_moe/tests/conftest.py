# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the nomic-embed-text-v2-moe reference tests.

The checkpoint and the upstream model each cost ~1.8 GB of host RAM, so both are
session-scoped. test_reference_modules.py uses none of these fixtures, which is what keeps the
structural suite runnable with no network and no weights.
"""

import pytest
import torch

from models.experimental.nomic_embed_text_v2_moe.common import checkpoint_is_cached, resolve_checkpoint
from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import load_vendored_config
from models.experimental.nomic_embed_text_v2_moe.reference.loader import (
    load_pretrained_reference_model,
    load_state_dict_from_safetensors,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_weights: requires the pinned checkpoint (~1.8 GB)")


@pytest.fixture(scope="session")
def config():
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
def reference_model(checkpoint_path):
    # Depends on checkpoint_path only for its skip-when-uncached guard; the helper resolves the
    # checkpoint itself.
    return load_pretrained_reference_model(allow_download=False)


@pytest.fixture(scope="session")
def hf_model():
    from models.experimental.nomic_embed_text_v2_moe.reference.hf_reference import load_hf_model

    try:
        return load_hf_model()
    except Exception as exc:
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
