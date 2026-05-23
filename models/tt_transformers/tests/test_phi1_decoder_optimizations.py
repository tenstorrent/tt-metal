# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from models.tt_transformers.demo.simple_text_demo import should_disable_device_sampling
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
    assert should_disable_device_sampling("phi-1", "accuracy", num_devices=1)
    assert should_disable_device_sampling("phi-1", "accuracy", num_devices=2)
    assert should_disable_device_sampling("phi-1", "performance", num_devices=1)
    assert not should_disable_device_sampling("phi-1", "performance", num_devices=2)

    assert should_disable_device_sampling("Mistral-7B", "accuracy", num_devices=1)
    assert should_disable_device_sampling("Mistral-7B", "performance", num_devices=2)


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
