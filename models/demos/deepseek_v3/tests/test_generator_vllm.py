# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from models.demos.deepseek_v3.tt import generator_vllm as generator_vllm_module

pytestmark = pytest.mark.t3k_compat


def test_initialize_vllm_model_passes_cache_dir_without_enabling_weight_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setenv("DEEPSEEK_V3_HF_MODEL", str(tmp_path / "model"))
    monkeypatch.setenv("DEEPSEEK_V3_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(generator_vllm_module, "load_tokenizer", lambda model_path: object())
    monkeypatch.setattr(generator_vllm_module.DeepseekV3ForCausalLM, "__init__", fake_init)

    generator_vllm_module.DeepseekV3ForCausalLM.initialize_vllm_model(
        hf_config=object(),
        mesh_device=object(),
        max_batch_size=1,
        max_seq_len=128,
    )

    assert captured["model_path"] == Path(tmp_path / "model")
    assert captured["cache_dir"] == Path(tmp_path / "cache")
    assert "use_weight_cache" not in captured
