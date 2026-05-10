# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Static architecture contract for Mistral Small 4 119B.

Values are aligned with the public Hugging Face model card and with
``mistral-small-4-119B_mins_depth.txt`` / ``mistral-small-4-119B_max_depth.txt``
from ``mistral_extractor.py``. Update here if the upstream checkpoint changes.
"""

HF_MODEL_ID = "mistralai/Mistral-Small-4-119B-2603"

# Hub key for ``Mistral4Model.embed_tokens`` (multimodal wrapper: ``language_model`` text trunk).
TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY = "language_model.model.embed_tokens.weight"
TEXT_MODEL_NORM_WEIGHT_KEY = "language_model.model.norm.weight"
TEXT_MODEL_LM_HEAD_WEIGHT_KEY = "language_model.lm_head.weight"

# Safetensors / HF state_dict prefix for decoder block ``i`` inside the multimodal wrapper (Phase A keymap).
TEXT_DECODER_LAYER_STATE_DICT_PREFIX_FMT = "language_model.model.layers.{layer_idx}."


def text_decoder_layer_state_dict_prefix(layer_idx: int) -> str:
    return TEXT_DECODER_LAYER_STATE_DICT_PREFIX_FMT.format(layer_idx=layer_idx)


# Text (Mistral4Model)
EXPECTED_HIDDEN_SIZE = 4096
EXPECTED_NUM_LAYERS = 36
EXPECTED_VOCAB_SIZE = 131072
EXPECTED_RMS_NORM_EPS = 1e-6

# MoE (per decoder layer)
EXPECTED_NUM_EXPERTS = 128
EXPECTED_NUM_EXPERTS_PER_TOK = 4


def text_decoder_layer_inner_state_dict(full_hf_sd: dict, layer_idx: int) -> dict:
    """Strip ``language_model.model.layers.{i}.`` so keys match ``Mistral4DecoderLayer.state_dict()``."""
    prefix = text_decoder_layer_state_dict_prefix(layer_idx)
    return {k[len(prefix) :]: v for k, v in full_hf_sd.items() if k.startswith(prefix)}


def strip_fp8_aux_tensors_from_decoder_inner(inner: dict) -> dict:
    """
    Drop FP8 / fine-grained scaling tensors from a decoder ``inner`` map that are not
    registered on stock ``nn.Linear`` / attention modules (hub-only aux keys).
    """
    drop_markers = ("activation_scale", "weight_scale_inv", "_scale_inv")
    return {k: v for k, v in inner.items() if not any(m in k for m in drop_markers)}


def text_decoder_self_attn_state_dict_prefix(layer_idx: int) -> str:
    return f"{text_decoder_layer_state_dict_prefix(layer_idx)}self_attn."


def text_decoder_self_attn_weight_slice(full_hf_sd: dict, layer_idx: int) -> dict:
    """``Mistral4Attention``-local keys (``q_a_proj.weight``, …) for :class:`TtMistral4SelfAttentionPrefill`."""
    p = text_decoder_self_attn_state_dict_prefix(layer_idx)
    return {k[len(p) :]: v for k, v in full_hf_sd.items() if k.startswith(p)}


def text_decoder_mlp_state_dict_prefix(layer_idx: int) -> str:
    return f"{text_decoder_layer_state_dict_prefix(layer_idx)}mlp."


def expert_index_ranges_per_mesh_device(
    num_experts: int = EXPECTED_NUM_EXPERTS, num_mesh_devices: int = 4
) -> list[tuple[int, int]]:
    """
    Contiguous expert ID ranges when sharding ``Mistral4NaiveMoe`` experts across devices.

    With 128 experts and 4 devices (e.g. P150x4), each device holds 32 experts' fused
    ``gate_up_proj`` / ``down_proj`` slices for dispatch + all-reduce (design target).
    """
    if num_experts % num_mesh_devices != 0:
        raise ValueError(f"{num_experts=} must be divisible by {num_mesh_devices=}")
    chunk = num_experts // num_mesh_devices
    return [(i * chunk, (i + 1) * chunk) for i in range(num_mesh_devices)]


# Vision (Pixtral + Mistral3MultiModalProjector) — same family as Mistral-Small-3.1-24B
EXPECTED_VISION_HIDDEN_SIZE = 1024
