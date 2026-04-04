# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig


@pytest.fixture
def model_config():
    """Default Qwen3-Coder-Next config with standard parameters."""
    return Qwen3CoderNextConfig()


@pytest.fixture
def hf_model_name():
    """HuggingFace model name, overridable via HF_MODEL env var."""
    return os.environ.get("HF_MODEL", "Qwen/Qwen3-Coder-Next")


@pytest.fixture
def reset_seeds():
    """Reset random seeds for reproducibility."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


@pytest.fixture(autouse=True)
def ensure_gc():
    """Force garbage collection between tests."""
    import gc

    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
