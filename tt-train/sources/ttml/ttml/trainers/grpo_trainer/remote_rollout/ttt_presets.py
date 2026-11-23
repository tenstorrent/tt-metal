# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llama and Qwen3 presets (optimizations + stop/pad IDs) for the family-agnostic
:mod:`ttt_rollout_sampler`. Consulted by the launcher, not the sampler."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

from transformers import AutoTokenizer

from ttml.common.utils import resolve_pad_token_id, resolve_stop_token_ids
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)


def bf16_attn_bfp8_mlp_optimizations(num_decoders: int, model_name: str) -> Any:
    """Single-chip preset: bf16 attention (Q/K/V/O + KV cache) at HIFI4,
    BFP8 MLP (FF1/FF2/FF3) at HIFI2_FP16. Returns a ``DecodersPrecision``."""
    conf = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BFP8,
                TensorGroup.FF2: PrecisionSetting.BFP8,
                TensorGroup.WQKV: PrecisionSetting.BF16,
                TensorGroup.WO: PrecisionSetting.BF16,
                TensorGroup.KV_CACHE: PrecisionSetting.BF16,
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2_FP16,
                OpGroup.LI_FF2: MathFidelitySetting.HIFI2_FP16,
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI4,
            },
        }
    )
    conf.__name__ = "bf16_attn_bfp8_mlp"
    inst = DecodersPrecision(num_decoders, model_name, decoder_conf=conf)
    inst.__name__ = "bf16_attn_bfp8_mlp"
    return inst


def stop_and_pad(model_id: str) -> Tuple[Sequence[int], int]:
    """Load the HF tokenizer for ``model_id`` to derive ``(stop_token_ids,
    pad_token_id)`` for :class:`TTTRolloutSampler`.

    Call with the same ``model_id`` as the ttml side so the IDs stay consistent
    with the peer tokenizer.
    """
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return sorted(resolve_stop_token_ids(tok)), resolve_pad_token_id(tok)
