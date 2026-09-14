# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import gc

import pytest

from models.tt_transformers.tt.model_config import parse_optimizations

# transformers 5.x removed Cache.get_usable_length, but some trust_remote_code reference models
# still call it -- e.g. microsoft/Phi-3-mini-128k-instruct's modeling_phi3.py does
# `kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)`, which raises
# AttributeError under transformers 5.10.2. For an unbounded cache (DynamicCache) the old method
# simply returned get_seq_length(layer_idx), so restore it as that alias to keep those reference
# models working. Scoped to DynamicCache only -- bounded caches had different (max-length) logic.
try:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length"):

        def _get_usable_length(self, new_seq_length=0, layer_idx=0):
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = _get_usable_length
except Exception:  # defensive: transformers cache internals may move
    pass


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()


def pytest_addoption(parser):
    parser.addoption(
        "--optimizations",
        action="store",
        default=None,
        type=parse_optimizations,
        help="Precision and fidelity configuration diffs over default (i.e., accuracy)",
    )

    parser.addoption(
        "--decoder_config_file",
        action="store",
        default=None,
        type=str,
        help="Provide a JSON file defining per-decoder precision and fidelity settings",
    )
    parser.addoption(
        "--use_hf_rope",
        action="store_true",
        default=False,
        help="Whether to use HF-style rope, if not passed, the default mllama will be used",
    )
