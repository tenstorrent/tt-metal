# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``Gemma4ForCausalLM`` (vLLM wrapper).

Tests the layer→group routing helper that maps Gemma4's 5:1
sliding-window:full-attention layer pattern onto upstream's
kv_cache_groups numbering. Real model loading + prefill/decode forward
integration is a follow-up; this test pins the structural pieces.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Stub ttnn so importing tt-metal modules doesn't blow up on the local
# C++ extension.
sys.modules.setdefault("ttnn", MagicMock(name="ttnn-test-mock"))
sys.modules.setdefault("ttnn._ttnn", MagicMock(name="ttnn._ttnn-test-mock"))


def _make_instance(layer_types):
    """Build a Gemma4ForCausalLM bypassing __init__ for routing tests."""
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    instance = Gemma4ForCausalLM.__new__(Gemma4ForCausalLM)
    text_config = SimpleNamespace(layer_types=layer_types, sliding_window=1024)
    hf_config = SimpleNamespace(text_config=text_config)
    instance.model_args = [hf_config]
    return instance


def test_layer_to_group_gemma4_31b_pattern():
    """Gemma4-31B: 50 sliding + 10 full, 5:1 ratio → group_size = 10
    via the min-rule. Expect 1 full group + 5 sliding groups = 6 groups."""
    pattern = ["sliding_attention"] * 5 + ["full_attention"]
    layers = pattern * 10  # 60 layers
    instance = _make_instance(layers)

    layer_to_group = instance._build_layer_to_group(num_groups=6)

    assert len(layer_to_group) == 60
    # All full-attention layers (indices 5, 11, 17, ...) land in one group.
    full_indices = [i for i, lt in enumerate(layers) if lt == "full_attention"]
    full_groups = {layer_to_group[i] for i in full_indices}
    assert len(full_groups) == 1, "all 10 full-attn layers should share one group"
    # Sliding layers land in 5 different groups (50 / 10 = 5).
    sliding_indices = [i for i, lt in enumerate(layers) if lt == "sliding_attention"]
    sliding_groups = {layer_to_group[i] for i in sliding_indices}
    assert len(sliding_groups) == 5


def test_layer_to_group_gpt_oss_alternating_pattern():
    """GPT-OSS-style alternating 1:1 pattern: 12 sliding + 12 full.
    Counts are equal so max_count < min_count * 1.25 fires, group_size = 12,
    yielding 1 full group + 1 sliding group = 2 groups."""
    layers = ["sliding_attention", "full_attention"] * 12
    instance = _make_instance(layers)

    layer_to_group = instance._build_layer_to_group(num_groups=2)

    assert len(layer_to_group) == 24
    sliding_indices = [i for i, lt in enumerate(layers) if lt == "sliding_attention"]
    full_indices = [i for i, lt in enumerate(layers) if lt == "full_attention"]
    assert len({layer_to_group[i] for i in sliding_indices}) == 1
    assert len({layer_to_group[i] for i in full_indices}) == 1


def test_layer_to_group_uniform_full_attention():
    """All-full attention (no hybrid): one group, all layers → 0."""
    layers = ["full_attention"] * 16
    instance = _make_instance(layers)

    layer_to_group = instance._build_layer_to_group(num_groups=1)

    assert layer_to_group == [0] * 16


def test_layer_to_group_count_mismatch_raises():
    """If the wrapper-side grouping computes a different number of groups
    than vLLM reports, fail loudly so the divergence is immediately
    visible (rather than silently mis-routing layers)."""
    pattern = ["sliding_attention"] * 5 + ["full_attention"]
    layers = pattern * 10
    instance = _make_instance(layers)

    with pytest.raises(ValueError, match="out of sync"):
        instance._build_layer_to_group(num_groups=99)


def test_layer_to_group_missing_layer_types_raises():
    instance = _make_instance([])

    with pytest.raises(ValueError, match="layer_types"):
        instance._build_layer_to_group(num_groups=1)


def test_layer_to_group_caches_result():
    """Helper caches the routing per (instance, num_groups); verify the
    cache short-circuits the second call."""
    pattern = ["sliding_attention"] * 5 + ["full_attention"]
    layers = pattern * 10
    instance = _make_instance(layers)

    first = instance._build_layer_to_group(num_groups=6)
    second = instance._build_layer_to_group(num_groups=6)

    # Same object → cache hit (cheap to verify with `is`).
    assert first is second
