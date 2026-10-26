# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``HybridAttentionForCausalLM``.

The class is the vLLM wrapper base for hybrid attention models (Gemma3,
Gemma4, GPT-OSS, ...). The bulk of its responsibility is the
``get_kv_cache_spec`` classmethod that translates ``layer_types`` from
HF config into per-layer KVCacheSpecs that upstream's hybrid kv cache
manager groups by attention type. This test pins that translation
across the typical patterns we'll see on real models.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# Stub ttnn so importing generator_vllm doesn't blow up on the local
# tt-metal C++ extension. We don't exercise any real ttnn behaviour here.
sys.modules.setdefault("ttnn", MagicMock(name="ttnn-test-mock"))
sys.modules.setdefault("ttnn._ttnn", MagicMock(name="ttnn._ttnn-test-mock"))


def _make_vllm_config(layer_types, sliding_window=1024, num_kv_heads=8, head_size=128):
    text_config = SimpleNamespace(layer_types=layer_types, sliding_window=sliding_window)
    hf_config = SimpleNamespace(text_config=text_config)
    cfg = MagicMock()
    cfg.model_config.hf_config = hf_config
    cfg.model_config.dtype = torch.bfloat16
    cfg.model_config.get_num_kv_heads.return_value = num_kv_heads
    cfg.model_config.get_head_size.return_value = head_size
    cfg.cache_config.cache_dtype = "auto"
    cfg.cache_config.block_size = 64
    return cfg


def test_spec_emits_one_entry_per_layer():
    """KV cache groups temporarily disabled: every layer is FullAttentionSpec
    regardless of layer_types entry. Reverts to one uniform spec until the
    bounded-sliding-cache decode bug is fixed."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM

    layers = ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5
    spec = HybridAttentionForCausalLM.get_kv_cache_spec(_make_vllm_config(layers))

    assert len(spec) == len(layers)
    for i in range(len(layers)):
        name = f"model.layers.{i}.self_attn"
        assert name in spec
        assert isinstance(spec[name], FullAttentionSpec)
        assert not isinstance(spec[name], SlidingWindowSpec)


def test_spec_gemma3_27b_pattern():
    """All layers are FullAttentionSpec while kv cache groups are disabled."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM

    pattern = ["sliding_attention"] * 5 + ["full_attention"]
    layers = pattern * 10  # 60 layers
    spec = HybridAttentionForCausalLM.get_kv_cache_spec(_make_vllm_config(layers))

    full_count = sum(isinstance(v, FullAttentionSpec) for v in spec.values())
    sliding_count = sum(isinstance(v, SlidingWindowSpec) for v in spec.values())
    assert full_count == 60
    assert sliding_count == 0


def test_spec_gpt_oss_alternating_pattern():
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM

    layers = ["sliding_attention", "full_attention"] * 12  # 24 layers
    spec = HybridAttentionForCausalLM.get_kv_cache_spec(_make_vllm_config(layers))

    full_count = sum(isinstance(v, FullAttentionSpec) for v in spec.values())
    sliding_count = sum(isinstance(v, SlidingWindowSpec) for v in spec.values())
    assert full_count == 24
    assert sliding_count == 0


def test_spec_uniform_full_attention_still_works():
    """All-full layer_types → single-type config; spec generation still succeeds."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM

    spec = HybridAttentionForCausalLM.get_kv_cache_spec(_make_vllm_config(["full_attention"] * 4))
    assert all(isinstance(v, FullAttentionSpec) for v in spec.values())
    assert not any(isinstance(v, SlidingWindowSpec) for v in spec.values())


def test_spec_propagates_kv_heads_and_head_size():
    from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM

    spec = HybridAttentionForCausalLM.get_kv_cache_spec(
        _make_vllm_config(["full_attention", "sliding_attention"], num_kv_heads=4, head_size=256)
    )

    for layer_spec in spec.values():
        assert layer_spec.num_kv_heads == 4
        assert layer_spec.head_size == 256
        assert layer_spec.block_size == 64
        assert layer_spec.dtype == torch.bfloat16


def test_spec_missing_layer_types_raises():
    from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM

    cfg = _make_vllm_config(["full_attention"])
    cfg.model_config.hf_config.text_config.layer_types = None

    with pytest.raises(ValueError, match="layer_types"):
        HybridAttentionForCausalLM.get_kv_cache_spec(cfg)


def test_spec_unknown_layer_type_raises():
    from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM

    cfg = _make_vllm_config(["full_attention", "rotary_chunked_xyz"])

    with pytest.raises(ValueError, match="Unsupported layer_type"):
        HybridAttentionForCausalLM.get_kv_cache_spec(cfg)


def test_subclass_must_override_prefill_and_decode():
    """The base class's prefill_forward / decode_forward are explicit
    NotImplementedError stubs — subclasses must provide model-specific
    routing that consumes ``page_tables_per_group``."""
    from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM

    instance = HybridAttentionForCausalLM.__new__(HybridAttentionForCausalLM)

    with pytest.raises(NotImplementedError, match="prefill_forward"):
        instance.prefill_forward()
    with pytest.raises(NotImplementedError, match="decode_forward"):
        instance.decode_forward()
