# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests of the prefill runner's weight-cache helpers and chunk-size env (no device, no checkpoint)."""

import os

import pytest

from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc


def test_format_missing_summarises_layer_ranges():
    assert (
        wc.format_missing({wc.EMBEDDING: [-1], wc.MOE_GATE: [3, 4, 5]}) == "embedding, moe_gate (layers 3..5, 3 total)"
    )


def test_chunk_size_env_selects_chunk_and_default_is_one_place(monkeypatch):
    from models.demos.deepseek_v3_d_p.utils import chunk_config

    monkeypatch.delenv("PREFILL_CHUNK_SIZE", raising=False)
    assert chunk_config.prefill_chunk_tokens() == chunk_config.DEFAULT_PREFILL_CHUNK_TOKENS
    monkeypatch.setenv("PREFILL_CHUNK_SIZE", "2560")
    assert chunk_config.prefill_chunk_tokens() == 2560
    monkeypatch.setenv("PREFILL_CHUNK_SIZE", "")
    assert chunk_config.prefill_chunk_tokens() == chunk_config.DEFAULT_PREFILL_CHUNK_TOKENS


def test_checkpoint_available_needs_a_safetensors_index(tmp_path):
    assert not wc.checkpoint_available("")
    assert not wc.checkpoint_available(tmp_path)  # config-only directory
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    assert wc.checkpoint_available(tmp_path)


_OPT_OUT = ("deepseek_v4_pro", "deepseek_v4_flash", "kimi_k3", "mistral_small_4")


def _mla_adapters():
    pytest.importorskip("ttnn")
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter

    def walk(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from walk(sub)

    import importlib
    import pkgutil

    import models.demos.deepseek_v3_d_p.tt.runners.adapters as pkg

    for module in pkgutil.iter_modules(pkg.__path__):  # the package __init__ imports only some of them
        importlib.import_module(f"{pkg.__name__}.{module.name}")

    return [c for c in walk(MLAPrefillAdapter) if getattr(c, "name", None)]


def test_mla_adapters_build_missing_cache_by_default_except_the_known_opt_outs():
    adapters = _mla_adapters()
    assert adapters
    for cls in adapters:
        assert cls.builds_missing_weight_cache == (cls.name not in _OPT_OUT), cls.name


def test_checkpoint_layout_matches_the_reader_for_enabled_adapters():
    """The reader hard-codes the DeepSeek-V3 HF key layout; check it against every enabled adapter's checkpoint that is
    readable on this host (skipped elsewhere)."""
    checked = 0
    for cls in _mla_adapters():
        path = os.environ.get("PREFILL_HF_MODEL") or cls.hf_model_default
        if not cls.builds_missing_weight_cache or not wc.checkpoint_available(path):
            continue
        weight_map = wc._Checkpoint(path).weight_map
        for key in (
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ):
            assert key in weight_map, f"{cls.name}: {key} not in {path}"
        assert any(k.startswith("model.layers.0.self_attn.") for k in weight_map), cls.name
        checked += 1
    if not checked:
        pytest.skip("no enabled adapter's checkpoint is readable on this host")
