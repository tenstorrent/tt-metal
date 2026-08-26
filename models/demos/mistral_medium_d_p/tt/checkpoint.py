# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5-128B checkpoint ingest — PURE TORCH, no ttnn import.

This is the only genuinely new host code in the bring-up; everything else is a reshaping of an
existing recipe. Three things the checkpoint does that no model already in the repo does the same way:

1. **`model.language_model.` key prefix.** The repo ships as ``Mistral3ForConditionalGeneration``, so
   every text-backbone tensor is ``model.language_model.layers.N....`` The existing
   ``standardize_hf_keys`` / ``map_hf_to_meta_keys`` helpers expect ``model.layers.N....``.
2. **Per-tensor fp8.** ``quantization_config.weight_block_size`` is ``null``, and the checkpoint's
   ``*.weight_scale_inv`` tensors are **scalars** (shape ``[]``) — NOT DeepSeek's
   ``[ceil(N/128), ceil(K/128)]`` block grid, so ``deepseek_v3_d_p``'s block dequant does not apply.
   Only q/k/v/o and gate/up/down are quantized; ``lm_head``, ``embed_tokens`` and every norm ship bf16
   (``modules_to_not_convert`` also excludes the vision tower and the projector).
3. **`activation_scale`.** Static activation quantization metadata. Meaningless for a bf16/bf8 TT
   path — dropped, deliberately and loudly.

Verified against the real ``model.safetensors.index.json`` + shard headers of
``mistralai/Mistral-Medium-3.5-128B``: 2465 tensors / 3 shards, layer tensors
``F8_E4M3`` with BF16 scalar ``weight_scale_inv`` and ``activation_scale``.
"""

from __future__ import annotations

import json
import os

import torch
from loguru import logger

# `model.language_model.` -> `model.`; everything below these prefixes is out of scope (vision).
TEXT_PREFIX = "model.language_model."
DROP_PREFIXES = ("model.vision_tower.", "model.multi_modal_projector.")
# fp8 sidecars: consumed by the dequant, never forwarded to the TT stack.
SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale", ".activation_scale", ".input_scale")


def strip_multimodal_wrapper(state_dict: dict) -> dict:
    """`model.language_model.X` -> `model.X`, and drop the vision tower / projector.

    ``lm_head.weight`` has no wrapper prefix and passes through untouched.
    """
    out = {}
    dropped = 0
    for k, v in state_dict.items():
        if k.startswith(DROP_PREFIXES):
            dropped += 1
            continue
        out["model." + k[len(TEXT_PREFIX) :] if k.startswith(TEXT_PREFIX) else k] = v
    if dropped:
        logger.info(f"Dropped {dropped} vision-tower / projector tensors (text backbone only)")
    return out


def _dequant_one(weight: torch.Tensor, scale: torch.Tensor, key: str) -> torch.Tensor:
    """Dequantize one per-tensor fp8 weight: ``w_bf16 = fp8.float() * weight_scale_inv``.

    ``weight_scale_inv`` is the DEQUANT multiplier (the inverse of the quantization scale) — the same
    convention DeepSeek uses, just with a scalar instead of a block grid. Fails loud on a non-scalar
    scale rather than silently broadcasting a block grid the wrong way.
    """
    if scale.numel() != 1:
        raise NotImplementedError(
            f"{key}: expected a per-tensor (scalar) fp8 scale, got shape {tuple(scale.shape)}. "
            "Block-wise fp8 needs the deepseek_v3_d_p blockwise dequant, not this one."
        )
    s = scale.reshape(()).to(torch.float32)
    if not torch.isfinite(s) or s <= 0:
        raise ValueError(f"{key}: non-positive / non-finite fp8 scale {s.item()}")
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def dequantize_fp8(state_dict: dict) -> dict:
    """Dequantize every per-tensor fp8 weight to bf16 and drop the fp8/activation sidecars.

    Tensors that are already bf16 (lm_head, embeddings, norms) pass through untouched.
    """
    out = {}
    n_dequant = 0
    for k, v in state_dict.items():
        if k.endswith(SCALE_SUFFIXES):
            continue  # consumed below / discarded (activation scales are not used on a bf16 path)
        if getattr(v, "dtype", None) in (torch.float8_e4m3fn, getattr(torch, "float8_e4m3fnuz", None)):
            scale = state_dict.get(k + "_scale_inv", state_dict.get(k + "_scale"))
            if scale is None:
                raise KeyError(f"{k} is fp8 but has no sibling weight_scale_inv / weight_scale")
            out[k] = _dequant_one(v, scale, k)
            n_dequant += 1
        else:
            out[k] = v
    logger.info(f"Dequantized {n_dequant} per-tensor fp8 weights to bf16")
    return out


def assert_supported(hf_config) -> None:
    """Refuse to build on a checkpoint whose mechanisms this implementation does not cover.

    Each of these is a mechanism that would otherwise degrade accuracy silently rather than crash.
    """

    def _get(name, default=None):
        return getattr(hf_config, name, default) if not isinstance(hf_config, dict) else hf_config.get(name, default)

    rope = _get("rope_parameters") or _get("rope_scaling") or {}
    rope_type = (rope.get("rope_type") or rope.get("type") or "default").lower()
    checks = [
        (rope_type == "yarn", f"rope_type must be 'yarn', got {rope_type!r}"),
        (
            float(rope.get("llama_4_scaling_beta", 0.0) or 0.0) == 0.0,
            "llama_4_scaling_beta != 0: Ministral3Attention applies a position-dependent Q "
            "temperature after RoPE that this bring-up does not implement",
        ),
        (_get("sliding_window") is None, "sliding_window is set: this implementation is dense causal only"),
        (_get("hidden_act", "silu") == "silu", "hidden_act must be 'silu' (plain SwiGLU)"),
        (not _get("attention_bias", False), "attention_bias is set, but Mistral projections are bias-free"),
        (not _get("tie_word_embeddings", False), "tie_word_embeddings is set, but this checkpoint unties them"),
        (
            _get("num_attention_heads", 0) * _get("head_dim", 0) == _get("hidden_size", -1),
            "n_q * head_dim != hidden_size; the fused-QKV / o_proj sharding assumes they are equal",
        ),
    ]
    for ok, msg in checks:
        if not ok:
            raise NotImplementedError(f"mistral_medium_d_p: {msg}")


def load_state_dict(weights_path: str, *, convert_to_meta_format: bool = True, head_dim: int = 128) -> dict:
    """Read the HF safetensors, strip the multimodal wrapper, dequantize fp8, and (by default)
    swizzle q/k into Meta interleaved RoPE order.

    The Meta swizzle pairs with the interleaved cos/sin from ``tt/rope_tables.build_yarn_cos_sin``
    and the ``rotary_embedding_llama`` / ``rotary_embedding_indexed`` ops. Skip it only if you intend
    to run HF-convention (``rotate_half``) RoPE, which this stack does not.
    """
    from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format, load_hf_state_dict

    sd = load_hf_state_dict(weights_path)
    sd = strip_multimodal_wrapper(sd)
    sd = dequantize_fp8(sd)
    if convert_to_meta_format:
        logger.info("Converting q/k projections HF -> Meta interleaved RoPE order")
        sd = convert_hf_qkv_to_meta_format(sd, head_dim)
    return sd


def load_hf_config_dict(model_path: str) -> dict:
    """Read the text-backbone config as a plain dict, unwrapping ``text_config`` if present.

    Used by tooling that must not pull in ``transformers``; the model stack itself uses AutoConfig.
    """
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    text = dict(cfg.get("text_config") or cfg)
    for k in ("quantization_config", "architectures", "tie_word_embeddings"):
        text.setdefault(k, cfg.get(k))
    return text
