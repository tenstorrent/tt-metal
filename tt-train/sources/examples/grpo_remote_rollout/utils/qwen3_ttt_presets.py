# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-family presets (optimizations + stop/pad IDs) for the family-agnostic
:mod:`ttt_generation_worker`. Consulted by the launcher, not the worker."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

from transformers import AutoTokenizer

from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)


# Stop-token strings for Qwen3 chat templates. Resolved to IDs by
# :func:`qwen3_stop_and_pad` using the actual model tokenizer.
QWEN3_STOP_TOKEN_STRS: tuple[str, ...] = (
    "<|im_end|>",
    "<|endoftext|>",
)


def bf16_attn_bfp8_mlp_optimizations(num_decoders: int, model_name: str) -> Any:
    """Qwen3 single-chip preset: bf16 attention (Q/K/V/O + KV cache) at HIFI4,
    BFP8 MLP (gate/up/down) at HIFI2_FP16. Returns a ``DecodersPrecision``.

    Same shape as the Llama preset -- the tt-transformers precision knobs are
    architecture-agnostic -- but exposed under a Qwen3 alias so the launcher's
    preset choice is explicit about the model family.
    """
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


def qwen3_stop_and_pad(model_id: str) -> Tuple[Sequence[int], int]:
    """Load the HF tokenizer for ``model_id`` to derive ``(stop_token_ids,
    pad_token_id)`` for :class:`TttGenerationWorker`.

    Falls back to ``eos_token_id`` for the pad slot when the tokenizer has no
    dedicated pad token (Qwen3 tokenizers typically expose one, but the
    fallback mirrors :func:`llama_stop_and_pad` so both families follow the
    same contract).
    """
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    ids: set[int] = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for s in QWEN3_STOP_TOKEN_STRS:
        tid = tok.convert_tokens_to_ids(s)
        if tid is not None and tid >= 0 and tid != tok.unk_token_id:
            ids.add(int(tid))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:
        raise RuntimeError(
            f"qwen3_stop_and_pad({model_id!r}): tokenizer exposes neither "
            "pad_token_id nor eos_token_id; cannot derive a filler id."
        )
    return sorted(ids), int(pad)
