# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import argparse

import pytest

from models.tt_transformers.demo.conftest import _cli_bool
from models.tt_transformers.demo.simple_text_demo import (
    get_parametrized_mesh_device,
    is_greedy_sampling_request,
    normalize_greedy_sampling_params,
    resolve_paged_attention_mode,
    should_disable_device_sampling,
)
from models.tt_transformers.tt.model_config import (
    MathFidelitySetting,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
    should_use_phi1_single_split_lm_head,
)


def test_phi1_accuracy_uses_bf16_attention_tensors_without_broad_hifi4_override():
    opt = ModelOptimizations.accuracy("phi-1")

    assert opt.tensor_dtype_settings[TensorGroup.WQKV] == PrecisionSetting.BF16
    assert opt.tensor_dtype_settings[TensorGroup.KV_CACHE] == PrecisionSetting.BF16
    assert opt.tensor_dtype_settings[TensorGroup.WO] == PrecisionSetting.BF16

    # Keep the fix narrow: preserve the standard attention kernel fidelities and
    # only retain the existing FP16-accumulate prefill QKV override.
    assert opt.op_fidelity_settings[OpGroup.LI_QKV_DECODE] == MathFidelitySetting.HIFI2
    assert opt.op_fidelity_settings[OpGroup.LI_QKV_PREFILL] == MathFidelitySetting.HIFI2_FP16
    assert opt.op_fidelity_settings[OpGroup.SDPA_DECODE] == MathFidelitySetting.HIFI2
    assert opt.op_fidelity_settings[OpGroup.SDPA_PREFILL] == MathFidelitySetting.HIFI4


def test_phi1_uses_host_sampling_only_in_accuracy_mode():
    greedy_sampling = {"temperature": 0, "top_p": 0.08, "top_k": 32}
    non_greedy_sampling = {"temperature": 0.8, "top_p": 0.9, "top_k": 32}

    assert should_disable_device_sampling("phi-1", "accuracy", greedy_sampling, num_devices=1)
    assert should_disable_device_sampling("phi-1", "accuracy", greedy_sampling, num_devices=2)
    assert not should_disable_device_sampling("phi-1", "performance", greedy_sampling, num_devices=1)
    assert not should_disable_device_sampling("phi-1", "performance", greedy_sampling, num_devices=2)
    assert should_disable_device_sampling("phi-1", "performance", non_greedy_sampling, num_devices=1)

    assert should_disable_device_sampling("Mistral-7B", "accuracy", greedy_sampling, num_devices=1)
    assert should_disable_device_sampling("Mistral-7B", "performance", greedy_sampling, num_devices=2)


def test_greedy_sampling_normalization_matches_host_argmax_semantics():
    sampling_params = {"temperature": 0, "top_p": 0.08, "top_k": 32}

    assert is_greedy_sampling_request(sampling_params)
    assert normalize_greedy_sampling_params(sampling_params) == {"temperature": 0, "top_p": 1.0, "top_k": 1}


def test_parametrized_mesh_device_uses_env_mapping_without_hardware_probe(monkeypatch):
    monkeypatch.setenv("MESH_DEVICE", "N150")

    assert get_parametrized_mesh_device() == (1, 1)


def test_phi1_defaults_to_non_paged_attention_for_single_user_short_context():
    assert not resolve_paged_attention_mode(
        "microsoft/phi-1",
        paged_attention=True,
        batch_size=1,
        data_parallel=1,
        max_seq_len=1024,
    )


def test_phi1_keeps_explicit_paged_attention_override():
    assert resolve_paged_attention_mode(
        "microsoft/phi-1",
        paged_attention=True,
        batch_size=1,
        data_parallel=1,
        max_seq_len=1024,
        paged_attention_arg=True,
    )


def test_non_phi_models_keep_default_paged_attention():
    assert resolve_paged_attention_mode(
        "meta-llama/Llama-3.2-1B",
        paged_attention=True,
        batch_size=1,
        data_parallel=1,
        max_seq_len=1024,
    )


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("False", False),
        ("no", False),
        ("off", False),
    ],
)
def test_demo_cli_bool_accepts_existing_true_false_spellings(raw_value, expected):
    assert _cli_bool(raw_value) is expected


def test_demo_cli_bool_rejects_invalid_values():
    with pytest.raises(argparse.ArgumentTypeError):
        _cli_bool("maybe")


def test_phi1_single_device_lm_head_budget_allows_one_split():
    assert should_use_phi1_single_split_lm_head(
        "phi-1",
        hidden_dim=2048,
        padded_vocab_size=51200,
        num_devices=1,
        num_cores=64,
    )

    assert not should_use_phi1_single_split_lm_head(
        "Llama-3.1-8B",
        hidden_dim=4096,
        padded_vocab_size=128256,
        num_devices=4,
        num_cores=48,
    )
