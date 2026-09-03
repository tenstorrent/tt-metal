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
        "--accuracy",
        action="store_true",
        default=False,
        help="Run test_long_context as a token-accuracy test instead of a performance test. "
        "Teacher-forces the model through a precomputed full-precision reference and reports "
        "top-1/top-5 agreement instead of tokens per second. Requires a reference sized for the "
        "context under test (see --accuracy_ref); disables Metal trace, which teacher forcing "
        "is incompatible with.",
    )
    parser.addoption(
        "--accuracy_ref",
        action="store",
        default=None,
        help="Path to the .refpt reference for --accuracy. Defaults to "
        "models/tt_transformers/tests/reference_outputs/<model>_<ctx label>.refpt, e.g. "
        "Qwen3-8B_32k.refpt, so each context length gets its own reference.",
    )
    parser.addoption(
        "--tracy_decode",
        action="store_true",
        default=False,
        help="Configure the run for Tracy profiling of the decode phase: disables Metal "
        "trace and caps generation at 2 tokens. Both are required -- trace replay under "
        "the profiler raises 'Device data mismatch', and a long decode loop overflows the "
        "device marker buffer. The `decode` signpost means tt-perf-report defaults to the "
        "decode phase. Not for measurement: the decode average degrades to one sample.",
    )
    parser.addoption(
        "--use_hf_rope",
        action="store_true",
        default=False,
        help="Whether to use HF-style rope, if not passed, the default mllama will be used",
    )
