# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decoder precision profile for Qwen3 caption-encoder PCC vs HuggingFace."""

from __future__ import annotations

from models.tt_transformers.tt.model_config import DecodersPrecision, ModelOptimizations, PrecisionSetting, TensorGroup


def qwen3_encoder_pcc_optimizations(num_layers: int, model_name: str):
    """All weight groups in BF16; HiFi4 matmul/SDPA (accuracy defaults for small Qwen)."""
    conf = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BF16,
                TensorGroup.FF2: PrecisionSetting.BF16,
                TensorGroup.WQKV: PrecisionSetting.BF16,
                TensorGroup.WO: PrecisionSetting.BF16,
                TensorGroup.KV_CACHE: PrecisionSetting.BF16,
            },
        }
    )
    inst = DecodersPrecision(num_layers, model_name, decoder_conf=conf)
    inst.__name__ = "qwen3_encoder_pcc"
    return inst
