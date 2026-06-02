# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading for the SeamlessM4Tv2 S2TT TT port.

Loads the HF safetensors checkpoint into a flat CPU state dict and provides
prefix-based slicing so each TT submodule (conformer conv / attention / layer /
encoder / decoder) can pull just the torch tensors it needs during bring-up.

S2TT-relevant top-level prefixes (keys in the HF checkpoint):
  speech_encoder.feature_projection.*
  speech_encoder.encoder.layers.{i}.*
  speech_encoder.encoder.layer_norm.*
  speech_encoder.intermediate_ffn.*
  speech_encoder.adapter.layers.0.*
  speech_encoder.inner_layer_norm.*
  text_decoder.*            (embed_tokens, layers.{i}, layer_norm)
  lm_head.weight            (tied to shared / text_decoder.embed_tokens)
"""

from __future__ import annotations

from typing import Optional

import torch

DEFAULT_MODEL_ID = "facebook/seamless-m4t-v2-large"

# Submodule prefixes used by the per-phase TT modules.
PREFIX_FEATURE_PROJECTION = "speech_encoder.feature_projection."
PREFIX_ENCODER_LAYER = "speech_encoder.encoder.layers.{i}."
PREFIX_ENCODER_NORM = "speech_encoder.encoder.layer_norm."
PREFIX_INTERMEDIATE_FFN = "speech_encoder.intermediate_ffn."
PREFIX_ADAPTER_LAYER = "speech_encoder.adapter.layers.{i}."
PREFIX_INNER_NORM = "speech_encoder.inner_layer_norm."
PREFIX_TEXT_DECODER = "text_decoder."
KEY_LM_HEAD = "lm_head.weight"


def load_state_dict(model_id: str = DEFAULT_MODEL_ID, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load the full HF checkpoint as a flat {name: tensor} dict (CPU, fp32)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import glob
    import os

    path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards found in {path}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        state_dict.update(load_file(shard, device=device))
    return state_dict


def filter_prefix(
    state_dict: dict[str, torch.Tensor], prefix: str, strip: bool = True
) -> dict[str, torch.Tensor]:
    """Return the sub-dict whose keys start with `prefix` (prefix stripped by default)."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :] if strip else k] = v
    return out


def encoder_layer_weights(state_dict: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:
    return filter_prefix(state_dict, PREFIX_ENCODER_LAYER.format(i=layer_idx))


def adapter_layer_weights(state_dict: dict[str, torch.Tensor], layer_idx: int = 0) -> dict[str, torch.Tensor]:
    return filter_prefix(state_dict, PREFIX_ADAPTER_LAYER.format(i=layer_idx))


def lm_head_weight(state_dict: dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """lm_head is tied to the shared embedding; fall back if the explicit key is absent."""
    if KEY_LM_HEAD in state_dict:
        return state_dict[KEY_LM_HEAD]
    for cand in ("shared.weight", "text_decoder.embed_tokens.weight"):
        if cand in state_dict:
            return state_dict[cand]
    return None
