# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for Qwen3.5-27B module tests."""

import pytest

# Re-export helpers from the monolithic test module
from models.experimental.tt_symbiote.tests.test_qwen3_5_27b_modules import (
    load_model,
)

# ──────────────────────────────────────────────────────────────────────
# Optional imports with try/except
# ──────────────────────────────────────────────────────────────────────

try:
    pass

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

try:
    import transformers

    TRANSFORMERS_5 = transformers.__version__.startswith("5.")
except ImportError:
    TRANSFORMERS_5 = False

try:
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear
    from models.experimental.tt_symbiote.modules.normalization import TTNNRMSNorm
    from models.experimental.tt_symbiote.utils.device_management import set_device

    TT_SYMBIOTE_AVAILABLE = True
except ImportError:
    TT_SYMBIOTE_AVAILABLE = False
    TorchTTNNTensor = None
    TTNNLinear = None
    TTNNRMSNorm = None
    set_device = None


# ──────────────────────────────────────────────────────────────────────
# Skip decorators
# ──────────────────────────────────────────────────────────────────────

skip_no_ttnn = pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
skip_no_transformers = pytest.mark.skipif(not TRANSFORMERS_5, reason="Requires transformers 5.0+ for Qwen3.5")
skip_no_symbiote = pytest.mark.skipif(not TT_SYMBIOTE_AVAILABLE, reason="tt_symbiote modules not available")


# ──────────────────────────────────────────────────────────────────────
# Module-scoped model fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model_1_layer():
    """Load Qwen3.5-27B-FP8 with 1 hidden layer (module-scoped for reuse)."""
    model, config = load_model(num_hidden_layers=1)
    return model, config


@pytest.fixture(scope="module")
def model_4_layers():
    """Load Qwen3.5-27B-FP8 with 4 hidden layers (module-scoped for reuse)."""
    model, config = load_model(num_hidden_layers=4)
    return model, config
