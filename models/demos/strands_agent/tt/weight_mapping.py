# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open

from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

STRANDS_DECIDER_HF_MODEL = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
STRANDS_BASE_HF_MODEL = "Qwen/Qwen3.5-2B-Base"
ADAPTER_PREFIX = "base_model.model."
LM_PREFIX = "model.language_model."
# Layout tag of the TT weight cache: the full-attention q/k head dims are stored rope-pair interleaved.
WEIGHT_LAYOUT = "rope_pairs"


def resolve_checkpoint(repo_or_dir=None) -> Path:
    """Local dir as-is, else the HF snapshot. Default: the decider repo (env STRANDS_DECIDER_MODEL)."""
    repo_or_dir = repo_or_dir or os.environ.get("STRANDS_DECIDER_MODEL", STRANDS_DECIDER_HF_MODEL)
    if os.path.isdir(repo_or_dir):
        return Path(repo_or_dir)
    from huggingface_hub import snapshot_download

    offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
    return Path(snapshot_download(repo_or_dir, local_files_only=offline))


def merge_lora(state_dict: Dict[str, torch.Tensor], adapter_dir) -> int:
    """Merge the PEFT LoRA adapter into state_dict in place: W += (alpha / r) * B @ A (fp32 math, back to W dtype)."""
    adapter_dir = Path(adapter_dir)
    with open(adapter_dir / "adapter_config.json") as f:
        cfg = json.load(f)
    assert cfg["peft_type"] == "LORA", cfg["peft_type"]
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), "rslora/dora not supported"
    scale = cfg["lora_alpha"] / cfg["r"]

    mats: Dict[str, Dict[str, torch.Tensor]] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as sf:
        for key in sf.keys():
            assert key.startswith(ADAPTER_PREFIX), key
            stem, ab, name = key[len(ADAPTER_PREFIX) :].rsplit(".", 2)
            assert ab in ("lora_A", "lora_B") and name == "weight", key
            mats.setdefault(stem, {})[ab] = sf.get_tensor(key)

    for stem, ab in mats.items():
        assert set(ab) == {"lora_A", "lora_B"}, f"{stem}: {sorted(ab)}"
        key = f"{LM_PREFIX}{stem}.weight"
        assert key in state_dict, f"adapter target missing in base: {key}"
        w = state_dict[key]
        delta = scale * (ab["lora_B"].float() @ ab["lora_A"].float())
        assert delta.shape == w.shape, f"{key}: {tuple(delta.shape)} != {tuple(w.shape)}"
        state_dict[key] = (w.float() + delta).to(w.dtype)
    return len(mats)


def load_base_state_dict(base_dir) -> Dict[str, torch.Tensor]:
    """Text tower tensors of the base checkpoint (visual.* and mtp.* skipped), with the tied lm_head."""
    base_dir = Path(base_dir)
    with open(base_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    file_to_keys: Dict[str, list] = {}
    for key, filename in weight_map.items():
        if "visual" in key or key.startswith("mtp."):
            continue
        file_to_keys.setdefault(filename, []).append(key)
    state_dict: Dict[str, torch.Tensor] = {}
    for filename, keys in file_to_keys.items():
        with safe_open(str(base_dir / filename), framework="pt") as sf:
            for key in keys:
                state_dict[key] = sf.get_tensor(key)
    if "lm_head.weight" not in state_dict:  # tie_word_embeddings
        state_dict["lm_head.weight"] = state_dict[f"{LM_PREFIX}embed_tokens.weight"]
    return state_dict


def rope_pair_perm(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """perm[p] = original dim stored at position p: (j, j + rd/2) become adjacent (2j, 2j + 1); the tail is unchanged."""
    half = rotary_dim // 2
    perm = torch.arange(head_dim)
    perm[0:rotary_dim:2] = torch.arange(half)
    perm[1:rotary_dim:2] = torch.arange(half) + half
    return perm


def permute_rope_pairs(state_dict: Dict[str, torch.Tensor], head_dim: int = 256, rotary_dim: int = 64) -> None:
    """Reorder the q/k head dims of every full-attention layer in place so that RoPE pairs are adjacent.

    HF partial RoPE pairs dims (j, j + rd/2) of each head; the stateless prefill RoPE op rotates adjacent pairs
    (2j, 2j + 1). Permuting the q_proj (q rows only, not the gate rows), k_proj and q_norm/k_norm outputs the same way
    leaves q . k per head unchanged (the permutation is orthogonal and applied to both) and lets the op run without
    slices. The TT weight cache key carries this layout (WEIGHT_LAYOUT)."""
    perm = rope_pair_perm(head_dim, rotary_dim)
    D = head_dim
    for key in [k for k in state_dict if k.endswith(".self_attn.q_proj.weight")]:
        prefix = key[: -len("q_proj.weight")]
        q = state_dict[key]  # [Hq * 2D, dim]: per head D q rows, then D gate rows
        Hq = q.shape[0] // (2 * D)
        q = q.reshape(Hq, 2, D, -1).clone()
        q[:, 0] = q[:, 0][:, perm]
        state_dict[key] = q.reshape(Hq * 2 * D, -1)
        k = state_dict[prefix + "k_proj.weight"]  # [Hkv * D, dim]
        state_dict[prefix + "k_proj.weight"] = k.reshape(-1, D, k.shape[-1])[:, perm].reshape(k.shape)
        for norm in ("q_norm", "k_norm"):
            state_dict[prefix + norm + ".weight"] = state_dict[prefix + norm + ".weight"][perm]
        for bias, per_head in (("q_proj.bias", 2 * D), ("k_proj.bias", D)):  # none in the current checkpoints
            if prefix + bias in state_dict:
                b = state_dict[prefix + bias].reshape(-1, per_head // D, D).clone()
                b[:, 0] = b[:, 0][:, perm]
                state_dict[prefix + bias] = b.reshape(-1)


def load_strands_state_dict(base_dir, adapter_dir) -> Dict[str, torch.Tensor]:
    state_dict = load_base_state_dict(base_dir)
    merge_lora(state_dict, adapter_dir)
    state_dict = remap_qwen36_state_dict(state_dict)
    permute_rope_pairs(state_dict)
    return state_dict
