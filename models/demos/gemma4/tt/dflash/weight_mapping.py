# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-weight loading for the Gemma4-31B DFlash drafter checkpoint
(z-lab/gemma-4-31B-it-DFlash) -- reads the checkpoint's single safetensors shard
directly (mirrors tests/test_factory.py's load_real_substate pattern used
elsewhere in this repo for the target model's own checkpoint), and slices the
flat state dict into per-component dicts shaped the way each ttnn loader wants
them (models/demos/gemma4/tt/attention/weights.py::load_attention_weights
expects "q_proj.weight" etc. with no "self_attn." prefix; RMSNorm just wants
{"weight": tensor}).

No embed_tokens / lm_head tensors exist in this checkpoint (tie_word_embeddings
in the drafter's own config + the model shares the TARGET's embedding/lm_head,
matching Qwen3.6 MTP's `mtp_use_dedicated_embeddings=false` convention) --
callers must wire in the target's own embedding/lm_head separately.
"""

from __future__ import annotations

import torch

DEFAULT_DFLASH_MODEL = "z-lab/gemma-4-31B-it-DFlash"


def _resolve_safetensors_path(model_path: str) -> str:
    """Local path if ``model_path`` already points at a directory/file; otherwise
    downloads the single ``model.safetensors`` shard from the HF Hub repo id."""
    import os

    if os.path.isdir(model_path):
        return os.path.join(model_path, "model.safetensors")
    if os.path.isfile(model_path):
        return model_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_path, "model.safetensors")


def load_dflash_flat_state_dict(model_path: str = DEFAULT_DFLASH_MODEL) -> dict[str, torch.Tensor]:
    """All 58 real tensors, keyed exactly as in the checkpoint
    (``fc.weight``, ``hidden_norm.weight``, ``norm.weight``,
    ``layers.{i}.{self_attn,mlp,input_layernorm,post_attention_layernorm}...``)."""
    from safetensors import safe_open

    path = _resolve_safetensors_path(model_path)
    out = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    return out


def layer_attention_state_dict(flat: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:
    """{"q_proj.weight": ..., "k_norm.weight": ..., ...} for one drafter layer --
    the shape models.demos.gemma4.tt.attention.weights.load_attention_weights wants."""
    prefix = f"layers.{layer_idx}.self_attn."
    return {k[len(prefix) :]: v for k, v in flat.items() if k.startswith(prefix)}


def layer_mlp_state_dict(flat: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:
    """{"gate_proj.weight": ..., "up_proj.weight": ..., "down_proj.weight": ...}."""
    prefix = f"layers.{layer_idx}.mlp."
    return {k[len(prefix) :]: v for k, v in flat.items() if k.startswith(prefix)}


def layer_norm_state_dict(flat: dict[str, torch.Tensor], layer_idx: int, name: str) -> dict[str, torch.Tensor]:
    """{"weight": ...} for ``name`` in {"input_layernorm", "post_attention_layernorm"}."""
    return {"weight": flat[f"layers.{layer_idx}.{name}.weight"]}


def top_level_norm_state_dict(flat: dict[str, torch.Tensor], name: str) -> dict[str, torch.Tensor]:
    """{"weight": ...} for ``name`` in {"hidden_norm", "norm"}."""
    return {"weight": flat[f"{name}.weight"]}


def fc_state_dict(flat: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """{"weight": [hidden_size, 6*hidden_size]} -- the context-tap projection."""
    return {"weight": flat["fc.weight"]}
