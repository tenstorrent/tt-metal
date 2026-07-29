# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from models.demos.deepseek_v3.tt import generator_vllm as generator_vllm_module

pytestmark = pytest.mark.t3k_compat


def test_initialize_vllm_model_passes_cache_dir_without_enabling_weight_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mesh_device
):
    captured: dict[str, object] = {}

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs)

    # initialize_vllm_model reads mesh_device.shape and validates that
    # tt_data_parallel == mesh_rows * mesh_cols and max_batch_size % tt_data_parallel == 0
    # before constructing the model (see PR #45284). Derive both from the actual mesh
    # so this test works on any MESH_DEVICE (N150/N300/T3K/TG/...).
    mesh_rows, mesh_cols = mesh_device.shape[0], mesh_device.shape[1]
    tt_data_parallel = mesh_rows * mesh_cols
    max_batch_size = tt_data_parallel

    monkeypatch.setenv("DEEPSEEK_V3_HF_MODEL", str(tmp_path / "model"))
    monkeypatch.setenv("DEEPSEEK_V3_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(generator_vllm_module, "load_tokenizer", lambda model_path: object())
    monkeypatch.setattr(generator_vllm_module.DeepseekV3ForCausalLM, "__init__", fake_init)

    generator_vllm_module.DeepseekV3ForCausalLM.initialize_vllm_model(
        hf_config=object(),
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=128,
        tt_data_parallel=tt_data_parallel,
    )

    assert captured["mesh_device"] is mesh_device
    assert captured["model_path"] == Path(tmp_path / "model")
    assert captured["cache_dir"] == Path(tmp_path / "cache")
    assert "use_weight_cache" not in captured


def test_get_max_tokens_all_users_returns_max_model_len_times_max_num_seqs_on_wormhole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wormhole + DeepSeek-R1-0528 derives the budget from vLLM's max_model_len * max_num_seqs (see #45284)."""
    monkeypatch.setattr(generator_vllm_module, "is_wormhole_b0", lambda: True)

    assert (
        generator_vllm_module.DeepseekV3ForCausalLM.get_max_tokens_all_users(
            model_name="/models/DeepSeek-R1-0528",
            num_devices=32,
            tt_data_parallel=1,
            max_model_len=32_768,
            max_num_seqs=1,
        )
        == 32_768
    )


def test_get_max_tokens_all_users_requires_max_lens_on_wormhole_deepseek_r1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without max_model_len/max_num_seqs the Wormhole path raises (vllm plugin must supply them)."""
    monkeypatch.setattr(generator_vllm_module, "is_wormhole_b0", lambda: True)

    with pytest.raises(ValueError, match="max_model_len and max_num_seqs"):
        generator_vllm_module.DeepseekV3ForCausalLM.get_max_tokens_all_users(
            model_name="/models/DeepSeek-R1-0528",
            num_devices=32,
            tt_data_parallel=1,
        )


def test_get_max_tokens_all_users_rejects_non_wormhole_deepseek_r1(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSeek-R1-0528 is only supported on Wormhole; non-Wormhole must raise."""
    monkeypatch.setattr(generator_vllm_module, "is_wormhole_b0", lambda: False)

    with pytest.raises(ValueError, match="not supported on non-Wormhole"):
        generator_vllm_module.DeepseekV3ForCausalLM.get_max_tokens_all_users(
            model_name="/models/DeepSeek-R1-0528",
            num_devices=32,
            tt_data_parallel=1,
        )
