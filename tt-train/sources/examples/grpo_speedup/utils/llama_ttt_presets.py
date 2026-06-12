# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-family presets for the generic :mod:`ttt_generation_worker`.

The worker itself is family-agnostic: it accepts an ``optimizations``
callable and explicit ``stop_token_ids``/``pad_token_id`` arguments and
holds no tokenizer. This module collects the Llama-specific defaults we
plumb into those constructor arguments, plus a small helper that loads
an HF tokenizer briefly at launcher startup to derive the IDs without
hard-coding them.

Nothing here is imported by the worker -- the launcher (or any other
caller) decides which preset module to consult.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple


# Family-specific extra stop-token strings. Every Llama-3 instruct
# tokenizer registers these as added tokens, but the strings (not the
# IDs) are stable across base/instruct variants and across releases. The
# launcher resolves them to integer IDs once via
# :func:`llama_stop_and_pad`.
LLAMA_STOP_TOKEN_STRS: tuple[str, ...] = (
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|eom_id|>",
)


def bf16_attn_bfp8_mlp_optimizations(num_decoders: int, model_name: str) -> Any:
    """tt-transformers ``ModelOptimizations`` preset used on a single chip
    for Llama-3 family models: bf16 attention (Q/K/V/O + KV cache), BFP8
    MLP (FF1/FF2/FF3), HIFI4 for the attention path, HIFI2_FP16 for the
    MLP path.

    Returned object is what ``ModelArgs(optimizations=...)`` expects:
    a ``DecodersPrecision`` instance (per-layer config aggregator).
    """
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
    """Briefly load the HF tokenizer for ``model_id`` to extract stop/pad IDs.

    Returns ``(stop_token_ids, pad_token_id)`` ready to feed into
    :class:`TttGenerationWorker`. The tokenizer object is dropped before
    returning; the worker never holds one.

    Stop IDs comprise the tokenizer's ``eos_token_id`` plus the IDs of
    every string in :data:`LLAMA_STOP_TOKEN_STRS` that the tokenizer
    knows about (unknown strings collapse to ``unk_token_id`` and are
    skipped). ``pad_token_id`` falls back to ``eos_token_id`` if the
    tokenizer has no dedicated pad token.

    The launcher is expected to call this once at startup with the same
    ``model_id`` it would use on the ttml side, so the derived IDs are
    guaranteed consistent with the peer tokenizer.
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
