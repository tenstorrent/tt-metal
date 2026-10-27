# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.tt_transformers.tt.model_config import ModelOptimizations, PrecisionSetting, TensorGroup


def test_krikri_accuracy_uses_the_llama3_settings():
    krikri = ModelOptimizations.accuracy("Llama-Krikri-8B-Base")
    llama = ModelOptimizations.accuracy("Llama-3.1-8B")
    assert krikri.tensor_dtype_settings == llama.tensor_dtype_settings
    assert krikri.op_fidelity_settings == llama.op_fidelity_settings
    assert krikri.tensor_dtype_settings[TensorGroup.KV_CACHE] == PrecisionSetting.BFP8
