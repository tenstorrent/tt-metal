# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.tt_transformers.tt.model_config import (
    MathFidelitySetting,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)
from models.tt_transformers.tt.phi_mlp import phi_activation_to_fused_linear_activation


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


def test_phi1_mlp_activation_maps_to_fused_linear_activation():
    assert phi_activation_to_fused_linear_activation("gelu") == "gelu"
    assert phi_activation_to_fused_linear_activation("gelu_new") == "gelu_approx"
    assert phi_activation_to_fused_linear_activation("gelu_pytorch_tanh") == "gelu_approx"
    assert phi_activation_to_fused_linear_activation("relu") == "relu"
    assert phi_activation_to_fused_linear_activation("swish") == "silu"
