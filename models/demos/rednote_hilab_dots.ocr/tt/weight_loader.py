# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Real-weights loader for rednote-hilab/dots.ocr.

Reads the actual HuggingFace safetensors shards from the local snapshot and
maps HF parameter keys onto the (flat) state_dict shape each TTNN block
expects. This replaces the seed-0 synthetic goldens used during bring-up with
the production checkpoint weights so blocks can be re-validated against the
real HF modules.

The model dir name (``rednote_hilab_dots.ocr``) contains a dot, so this module
is loaded by file path via importlib rather than the dotted package path.

Loading is sharded: ``model.safetensors.index.json`` maps each parameter name
to its shard file. We open only the shard(s) that hold the requested keys and
slice out individual tensors (safetensors supports per-tensor reads without
materializing the whole shard).
"""
import json
import os
from typing import Dict, List

import torch
from safetensors import safe_open


def _read_index(checkpoint_path: str) -> Dict[str, str]:
    """Return the HF weight_map: param name -> shard filename."""
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    return index["weight_map"]


def load_hf_tensors(checkpoint_path: str, keys: List[str]) -> Dict[str, torch.Tensor]:
    """Load a set of HF parameters by their fully-qualified keys.

    Args:
        checkpoint_path: path to the HF snapshot dir (contains the safetensors
            shards and ``model.safetensors.index.json``).
        keys: list of fully-qualified HF parameter names, e.g.
            ``["vision_tower.blocks.0.norm1.weight"]``.

    Returns:
        dict mapping each requested key to its torch.Tensor (fp32).
    """
    weight_map = _read_index(checkpoint_path)
    # Group the requested keys by the shard file that holds them so each shard
    # is opened at most once.
    by_shard: Dict[str, List[str]] = {}
    for key in keys:
        if key not in weight_map:
            raise KeyError(f"{key!r} not present in checkpoint index ({checkpoint_path})")
        by_shard.setdefault(weight_map[key], []).append(key)

    out: Dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        shard_path = os.path.join(checkpoint_path, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in shard_keys:
                out[key] = f.get_tensor(key).to(torch.float32)
    return out


def load_vision_rmsnorm_weight(
    checkpoint_path: str, hf_key: str = "vision_tower.blocks.0.norm1.weight"
) -> torch.Tensor:
    """Load a single real vision-tower RMSNorm gamma weight [embed_dim].

    The dots vision tower uses ``RMSNorm`` (modeling_dots_vision.RMSNorm) for
    every per-block ``norm1``/``norm2``, the patch-embed ``norm``, and the
    ``post_trunk_norm`` -- all share the same shape ([embed_dim]) and eps
    (config.rms_norm_eps = 1e-5). Any of these is a valid real RMSNorm gamma
    for validating :class:`TtVisionRMSNorm`; default picks block 0's norm1.

    Args:
        checkpoint_path: HF snapshot dir.
        hf_key: which RMSNorm weight to pull (default the first block's norm1).

    Returns:
        torch.Tensor of shape [embed_dim] (fp32).
    """
    tensors = load_hf_tensors(checkpoint_path, [hf_key])
    return tensors[hf_key]


def load_vision_attention_weights(checkpoint_path: str, block_idx: int = 0) -> Dict[str, torch.Tensor]:
    """Load a vision-tower attention block's real QKV + output-proj weights.

    The dots vision attention (modeling_dots_vision.VisionAttention) uses a
    fused QKV projection and an output proj, both unbiased
    (config.use_bias = False). HF keys:
        vision_tower.blocks.{i}.attn.qkv.weight   [3*embed_dim, embed_dim]
        vision_tower.blocks.{i}.attn.proj.weight  [embed_dim, embed_dim]

    Returns a flat state_dict in the shape :class:`TtVisionAttention` (and the
    eager reference vision_attention_forward) expects:
        {"qkv.weight": ..., "proj.weight": ...}  (fp32, no bias).
    """
    qkv_key = f"vision_tower.blocks.{block_idx}.attn.qkv.weight"
    proj_key = f"vision_tower.blocks.{block_idx}.attn.proj.weight"
    tensors = load_hf_tensors(checkpoint_path, [qkv_key, proj_key])
    return {
        "qkv.weight": tensors[qkv_key],
        "proj.weight": tensors[proj_key],
    }


def load_vision_mlp_weights(checkpoint_path: str, block_idx: int = 0) -> Dict[str, torch.Tensor]:
    """Load a vision-tower MLP block's real SwiGLU weights (no bias).

    The dots vision MLP (modeling_dots_vision.DotsSwiGLUFFN) is an unbiased
    SwiGLU FFN: ``fc2(silu(fc1(x)) * fc3(x))`` where fc1 is the gate, fc3 the
    up, and fc2 the down projection (config.use_bias = False). HF keys:
        vision_tower.blocks.{i}.mlp.fc1.weight  [intermediate, embed_dim]  (gate)
        vision_tower.blocks.{i}.mlp.fc3.weight  [intermediate, embed_dim]  (up)
        vision_tower.blocks.{i}.mlp.fc2.weight  [embed_dim, intermediate]  (down)
    embed_dim 1536, intermediate_size 4224.

    Returns a flat state_dict in the shape :class:`TtVisionMLP` and the eager
    reference vision_mlp_forward expect:
        {"fc1.weight": ..., "fc2.weight": ..., "fc3.weight": ...}  (fp32, no bias).
    """
    fc1_key = f"vision_tower.blocks.{block_idx}.mlp.fc1.weight"
    fc2_key = f"vision_tower.blocks.{block_idx}.mlp.fc2.weight"
    fc3_key = f"vision_tower.blocks.{block_idx}.mlp.fc3.weight"
    tensors = load_hf_tensors(checkpoint_path, [fc1_key, fc2_key, fc3_key])
    return {
        "fc1.weight": tensors[fc1_key],
        "fc2.weight": tensors[fc2_key],
        "fc3.weight": tensors[fc3_key],
    }
