# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Some Mistral Small 4 tests import ``transformers.models.mistral4`` (see ``_HF_MISTRAL4_TEST_FILES``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Tests that import HF Mistral4 modules; others (e.g. linear) only need torch + ttnn.
_HF_MISTRAL4_TEST_FILES = frozenset(
    {
        "test_rms_norm.py",
        "test_dense_mlp.py",
        "test_router.py",
        "test_routing.py",
        "test_attention_slice.py",
        "test_attention_full.py",
        "test_attention_device_sdpa.py",
        "test_decoder_layer_dense.py",
        "test_decoder_layer_moe.py",
        "test_model_backbone.py",
        "test_causal_lm_prefill.py",
        "test_causal_lm_prefill_hf_causal.py",
        "test_model_backbone_hf_causal.py",
        "test_hf_padding_mask.py",
        "test_attention_kv_cache.py",
        "test_causal_lm_incremental.py",
    }
)


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("transformers.models.mistral4") is not None:
        return
    skip = pytest.mark.skip(
        reason=(
            "This test imports ``transformers.models.mistral4``. Install: `pip install -U transformers` "
            "(see `requirements.txt` in this folder)."
        )
    )
    for item in items:
        path = Path(item.path) if hasattr(item, "path") else Path(str(item.fspath))
        if path.name not in _HF_MISTRAL4_TEST_FILES:
            continue
        item.add_marker(skip)
