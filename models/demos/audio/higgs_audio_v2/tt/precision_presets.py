# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""Precision presets for Higgs Audio v2 decode experiments.

The stock tt_transformers presets are:
  - accuracy    : bf8 attention (WQKV/WO/KV), bf16 MLP accumulation (Phase-1 gate)
  - performance : default bf8 everywhere + FF1/FF3 -> bf4 (current Stage-1 perf)

This module adds ``bf4_all`` for the Stage-3 weight-bandwidth experiment:
push every decode matmul weight (FF1/FF3/FF2/WQKV/WO) to bf4. KV cache stays
bf8 (kv reads are tiny at short seq and bf4 kv is lossy for little gain). This
is the only single-stream lever left that attacks the ~7ms weight-DRAM half of
the ~15.45ms device floor; the cost is accuracy, which the gate test measures.
"""
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    ModelOptimizations,
    TensorGroup,
    PrecisionSetting,
    OpGroup,
    MathFidelitySetting,
)


def bf4_everything(num_decoders, model_name):
    conf = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BFP4,
                TensorGroup.FF2: PrecisionSetting.BFP4,
                TensorGroup.WQKV: PrecisionSetting.BFP4,
                TensorGroup.WO: PrecisionSetting.BFP4,
                TensorGroup.KV_CACHE: PrecisionSetting.BFP8,
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI,
                OpGroup.LI_FF2: MathFidelitySetting.LOFI,
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.LOFI,
                OpGroup.LI_O_DECODE: MathFidelitySetting.LOFI,
            },
        }
    )
    inst = DecodersPrecision(num_decoders, model_name, decoder_conf=conf)
    inst.__name__ = "bf4_all"
    return inst


def mlp_bf4(num_decoders, model_name):
    """Middle ground: whole MLP (FF1/FF3/FF2) -> bf4, attention stays bf8.

    bf4_everything craters accuracy (0.62) because attention is precision-
    sensitive (stock presets keep WQKV/WO at bf8 even in accuracy mode). The
    MLP is the largest param group and the least sensitive, so this is the only
    candidate that might beat the 63.5 tok/s perf preset while holding >=0.95.
    """
    conf = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BFP4,
                TensorGroup.FF2: PrecisionSetting.BFP4,
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI,
                OpGroup.LI_FF2: MathFidelitySetting.LOFI,
            },
        }
    )
    inst = DecodersPrecision(num_decoders, model_name, decoder_conf=conf)
    inst.__name__ = "mlp_bf4"
    return inst


def build_precision(name, num_decoders, model_name):
    """Resolve a HIGGS_PRECISION env value to a DecodersPrecision instance."""
    if name == "accuracy":
        return DecodersPrecision.accuracy(num_decoders, model_name)
    if name == "performance":
        return DecodersPrecision.performance(num_decoders, model_name)
    if name == "mlp_bf4":
        return mlp_bf4(num_decoders, model_name)
    if name == "bf4_all":
        return bf4_everything(num_decoders, model_name)
    raise ValueError(f"unknown HIGGS_PRECISION={name!r} (accuracy|performance|mlp_bf4|bf4_all)")
