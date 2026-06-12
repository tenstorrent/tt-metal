# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from models.common.model_capabilities import FALLBACK_MAX_TOKENS_ALL_USERS
from models.demos.deepseek_v3.tt import generator_vllm as generator_vllm_module

pytestmark = pytest.mark.t3k_compat


@pytest.mark.skip(reason="Disabled: see #45677")
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


def test_deepseek_generator_exposes_max_tokens_all_users_capability() -> None:
    assert generator_vllm_module.DeepseekGenerator.get_max_tokens_all_users() == FALLBACK_MAX_TOKENS_ALL_USERS


def test_decode_forward_passes_slot_remap_to_device_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeModel:
        model_run_config_decode = object()
        mesh_device = object()
        batch_size_per_row = 8
        batch_size = 16

        def set_kv_cache(self, kv_cache):
            captured["kv_cache"] = kv_cache

        def _validate_and_initialize_sampling(self, *args, **kwargs):
            captured["validate_sampling"] = (args, kwargs)

        def _sample_tokens_device(self, logits, **kwargs):
            captured["sample_call"] = (logits, kwargs)
            return "sampled"

        def sample_decode_on_device(self, logits, **kwargs):
            captured["sample_decode_on_device"] = (logits, kwargs)
            return self._sample_tokens_device(logits, **kwargs)

        def _tokens_from_device(self, tokens, mesh_device, batch_size_per_row):
            captured["tokens_from_device"] = (tokens, mesh_device, batch_size_per_row)
            return "host_tokens"

    def fake_super_decode_forward(self, *args, **kwargs):
        captured["super_decode_forward"] = kwargs
        return "decode_logits"

    monkeypatch.setattr(generator_vllm_module.DeepseekGenerator, "decode_forward", fake_super_decode_forward)

    fake_model = _FakeModel()
    output = generator_vllm_module.DeepseekV3ForCausalLM.decode_forward(
        fake_model,
        tokens=generator_vllm_module.torch.tensor([[11], [22]]),
        start_pos=generator_vllm_module.torch.tensor([3, 4]),
        sampling_params=object(),
        slot_remap=[7, 6, 5, 4],
        read_from_device=True,
    )

    assert captured["super_decode_forward"]["tokens"].tolist() == [11, 22]
    assert captured["super_decode_forward"]["sample_on_device"] is True
    assert captured["sample_call"][0] == "decode_logits"
    assert captured["sample_call"][1]["slot_remap"] == [7, 6, 5, 4]
    assert captured["sample_call"][1]["skip_precompile"] is True
    assert output == "host_tokens"


@pytest.mark.skip(reason="Disabled: see #45677")
def test_get_max_tokens_all_users_overrides_deepseek_r1_on_wormhole(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator_vllm_module, "is_wormhole_b0", lambda: True)

    assert (
        generator_vllm_module.DeepseekV3ForCausalLM.get_max_tokens_all_users(
            model_name="/models/DeepSeek-R1-0528",
            num_devices=32,
            tt_data_parallel=1,
        )
        == 32_768
    )


@pytest.mark.skip(reason="Disabled: see #45677")
def test_get_max_tokens_all_users_uses_fallback_for_other_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator_vllm_module, "is_wormhole_b0", lambda: False)

    assert (
        generator_vllm_module.DeepseekV3ForCausalLM.get_max_tokens_all_users(
            model_name="/models/DeepSeek-R1-0528",
            num_devices=32,
            tt_data_parallel=1,
        )
        == FALLBACK_MAX_TOKENS_ALL_USERS
    )
