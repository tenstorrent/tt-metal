# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-family presets (optimizations + stop/pad IDs) for the family-agnostic
:mod:`ttt_generation_worker`. Consulted by the launcher, not the worker."""

from __future__ import annotations

from typing import Any, Sequence, Tuple


# Stop-token strings (stable across Llama-3 base/instruct variants; the IDs are
# not). Resolved to IDs by :func:`llama_stop_and_pad`.
LLAMA_STOP_TOKEN_STRS: tuple[str, ...] = (
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|eom_id|>",
)


def bf16_attn_bfp8_mlp_optimizations(num_decoders: int, model_name: str) -> Any:
    """Llama-3 single-chip preset: bf16 attention (Q/K/V/O + KV cache) at HIFI4,
    BFP8 MLP (FF1/FF2/FF3) at HIFI2_FP16. Returns a ``DecodersPrecision``."""
    from models.tt_transformers.tt.model_config import (
        DecodersPrecision,
        MathFidelitySetting,
        ModelOptimizations,
        OpGroup,
        PrecisionSetting,
        TensorGroup,
    )

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


def llama_stop_and_pad(model_id: str) -> Tuple[Sequence[int], int]:
    """Load the HF tokenizer for ``model_id`` to derive ``(stop_token_ids,
    pad_token_id)`` for :class:`TttGenerationWorker`.

    Call with the same ``model_id`` as the ttml side so the IDs stay consistent
    with the peer tokenizer.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    ids: set[int] = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for s in LLAMA_STOP_TOKEN_STRS:
        tid = tok.convert_tokens_to_ids(s)
        if tid is not None and tid >= 0 and tid != tok.unk_token_id:
            ids.add(int(tid))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:
        raise RuntimeError(
            f"llama_stop_and_pad({model_id!r}): tokenizer exposes neither "
            "pad_token_id nor eos_token_id; cannot derive a filler id."
        )
    return sorted(ids), int(pad)
