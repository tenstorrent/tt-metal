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


def load_vision_block_weights(checkpoint_path: str, block_idx: int = 0) -> Dict[str, torch.Tensor]:
    """Load one full vision-tower block's real weights (the first composite).

    A dots vision block (modeling_dots_vision.DotsVisionBlock) is a pre-norm
    residual layer: ``h = h + attn(norm1(h)); h = h + mlp(norm2(h))``. This
    COMPOSES the already-verified per-leaf loaders -- the two RMSNorm gammas,
    the fused-QKV + output-proj attention weights, and the SwiGLU fc1/fc2/fc3
    -- into a single flat state_dict keyed exactly as the eager reference
    :func:`reference.functional.vision_block_forward` and :class:`TtVisionBlock`
    expect. HF keys (block ``i``, all unbiased; config.use_bias = False):
        vision_tower.blocks.{i}.norm1.weight     [embed_dim]
        vision_tower.blocks.{i}.attn.qkv.weight  [3*embed_dim, embed_dim]
        vision_tower.blocks.{i}.attn.proj.weight [embed_dim, embed_dim]
        vision_tower.blocks.{i}.norm2.weight     [embed_dim]
        vision_tower.blocks.{i}.mlp.fc1.weight   [intermediate, embed_dim]  (gate)
        vision_tower.blocks.{i}.mlp.fc3.weight   [intermediate, embed_dim]  (up)
        vision_tower.blocks.{i}.mlp.fc2.weight   [embed_dim, intermediate]  (down)

    Returns the flat block state_dict (fp32):
        {"norm1.weight", "attn.qkv.weight", "attn.proj.weight", "norm2.weight",
         "mlp.fc1.weight", "mlp.fc2.weight", "mlp.fc3.weight"}.
    """
    norm1 = load_vision_rmsnorm_weight(checkpoint_path, f"vision_tower.blocks.{block_idx}.norm1.weight")
    norm2 = load_vision_rmsnorm_weight(checkpoint_path, f"vision_tower.blocks.{block_idx}.norm2.weight")
    attn = load_vision_attention_weights(checkpoint_path, block_idx=block_idx)
    mlp = load_vision_mlp_weights(checkpoint_path, block_idx=block_idx)
    return {
        "norm1.weight": norm1,
        "attn.qkv.weight": attn["qkv.weight"],
        "attn.proj.weight": attn["proj.weight"],
        "norm2.weight": norm2,
        "mlp.fc1.weight": mlp["fc1.weight"],
        "mlp.fc2.weight": mlp["fc2.weight"],
        "mlp.fc3.weight": mlp["fc3.weight"],
    }


def load_vision_patch_merger_weights(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load the real vision PatchMerger weights (LayerNorm + 2-linear GELU MLP).

    The dots vision PatchMerger (modeling_dots_vision.DotsPatchMerger,
    pre_norm='layernorm') is, UNLIKE the rest of the vision tower (which is
    RMSNorm + unbiased Linears), a true biased block: a LayerNorm with both
    gamma AND beta (eps 1e-6) followed by a 2-linear GELU MLP where BOTH Linears
    carry bias. HF keys (single merger, lives in shard 2):
        vision_tower.merger.ln_q.weight    [context_dim]        (LN gamma)
        vision_tower.merger.ln_q.bias      [context_dim]        (LN beta)
        vision_tower.merger.mlp.0.weight   [hidden, hidden]     (fc1, biased)
        vision_tower.merger.mlp.0.bias     [hidden]
        vision_tower.merger.mlp.2.weight   [out_dim, hidden]    (fc2, biased)
        vision_tower.merger.mlp.2.bias     [out_dim]
    context_dim 1536, spatial_merge_size 2 -> hidden = 1536*4 = 6144,
    out_dim = context_dim = 1536.

    Returns the flat state_dict (fp32) keyed exactly as the eager reference
    :func:`reference.functional.vision_patch_merger_forward` and
    :class:`TtVisionPatchMerger` expect:
        {"ln_q.weight", "ln_q.bias", "mlp.0.weight", "mlp.0.bias",
         "mlp.2.weight", "mlp.2.bias"}.
    """
    keys = [
        "vision_tower.merger.ln_q.weight",
        "vision_tower.merger.ln_q.bias",
        "vision_tower.merger.mlp.0.weight",
        "vision_tower.merger.mlp.0.bias",
        "vision_tower.merger.mlp.2.weight",
        "vision_tower.merger.mlp.2.bias",
    ]
    tensors = load_hf_tensors(checkpoint_path, keys)
    return {
        "ln_q.weight": tensors["vision_tower.merger.ln_q.weight"],
        "ln_q.bias": tensors["vision_tower.merger.ln_q.bias"],
        "mlp.0.weight": tensors["vision_tower.merger.mlp.0.weight"],
        "mlp.0.bias": tensors["vision_tower.merger.mlp.0.bias"],
        "mlp.2.weight": tensors["vision_tower.merger.mlp.2.weight"],
        "mlp.2.bias": tensors["vision_tower.merger.mlp.2.bias"],
    }


def load_vision_patch_embed_weights(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load the real vision patch-embed weights (Conv2d patchify + RMSNorm).

    DotsPatchEmbed wraps a ``patchifier`` submodule: a Conv2d(3,1536,k=14,s=14)
    over packed/patchified pixels followed by an RMSNorm (eps 1e-5). The real HF
    keys carry the extra ``.patchifier.`` segment that the flat reference /
    TTNN tower keys drop; this maps them onto the flat shape:
        vision_tower.patch_embed.patchifier.proj.weight  -> proj.weight  [1536,3,14,14]
        vision_tower.patch_embed.patchifier.proj.bias    -> proj.bias    [1536]
        vision_tower.patch_embed.patchifier.norm.weight  -> norm.weight  [1536]

    The patch_embed is the documented host-resident boundary; these weights are
    consumed on host by :func:`tt.vision_tower.host_patch_embed`.

    Returns the flat state_dict (fp32):
        {"proj.weight", "proj.bias", "norm.weight"}.
    """
    proj_w = "vision_tower.patch_embed.patchifier.proj.weight"
    proj_b = "vision_tower.patch_embed.patchifier.proj.bias"
    norm_w = "vision_tower.patch_embed.patchifier.norm.weight"
    tensors = load_hf_tensors(checkpoint_path, [proj_w, proj_b, norm_w])
    return {
        "proj.weight": tensors[proj_w],
        "proj.bias": tensors[proj_b],
        "norm.weight": tensors[norm_w],
    }


def load_embedding_weight(checkpoint_path: str, hf_key: str = "model.embed_tokens.weight") -> torch.Tensor:
    """Load the real LM token-embedding table [vocab_size, hidden_size].

    The dots.ocr language model (a Qwen2 trunk) holds its input token-embedding
    table under the HF key ``model.embed_tokens.weight`` -- shape
    [151936, 1536] (vocab_size x hidden_size). The checkpoint unties input and
    output embeddings (config.tie_word_embeddings = false), so this is the input
    lookup table only; the separate ``lm_head.weight`` is a distinct parameter.

    This is the [vocab, hidden] gather table that :class:`tt.embedding.TtEmbedding`
    consumes directly (no transpose / reshape -- ttnn.embedding indexes rows).

    Args:
        checkpoint_path: HF snapshot dir.
        hf_key: which embedding table to pull (default the LM input embedding).

    Returns:
        torch.Tensor of shape [vocab_size, hidden_size] (fp32).
    """
    tensors = load_hf_tensors(checkpoint_path, [hf_key])
    return tensors[hf_key]


def load_lm_rmsnorm_weight(checkpoint_path: str, hf_key: str = "model.layers.0.input_layernorm.weight") -> torch.Tensor:
    """Load a single real LM RMSNorm gamma weight [hidden_size].

    The dots.ocr language model is a Qwen2 trunk that uses Qwen2RMSNorm (eps =
    config.rms_norm_eps = 1e-6, distinct from the vision tower's 1e-5) for every
    decoder layer's ``input_layernorm`` / ``post_attention_layernorm`` and the
    final ``model.norm``. All share the same shape ([hidden_size] = [1536]) and
    eps, so any is a valid real LM RMSNorm gamma for validating
    :class:`tt.rmsnorm.TtRMSNorm`; the default picks layer 0's input_layernorm.
    The final-norm gamma is reachable via ``hf_key='model.norm.weight'``.

    Args:
        checkpoint_path: HF snapshot dir.
        hf_key: which RMSNorm weight to pull (default layer 0's input_layernorm).

    Returns:
        torch.Tensor of shape [hidden_size] (fp32).
    """
    tensors = load_hf_tensors(checkpoint_path, [hf_key])
    return tensors[hf_key]


def load_vision_tower_weights(checkpoint_path: str, num_layers: int) -> Dict[str, torch.Tensor]:
    """Load the full DotsVisionTransformer (vision tower) real weights.

    COMPOSES the already-verified per-component loaders into the single flat
    state_dict that the eager reference
    :func:`reference.functional.vision_tower_forward` and the TTNN
    :class:`tt.vision_tower.TtVisionTower` both consume. The tower is
    ``patch_embed -> N x vision_block -> [post_trunk_norm] -> patch_merger``:

    - patch_embed: ``load_vision_patch_embed_weights`` (Conv2d + RMSNorm), emitted
      under the flat ``patch_embed.proj.weight/bias`` + ``patch_embed.norm.weight``
      keys (HF carries an extra ``.patchifier.`` segment).
    - each of the ``num_layers`` blocks: ``load_vision_block_weights(ckpt, i)``,
      re-prefixed ``blocks.{i}.``.
    - post-trunk RMSNorm: HF key ``vision_tower.post_trunk_norm.weight`` ->
      ``post_trunk_norm.weight``.
    - patch merger: ``load_vision_patch_merger_weights``, re-prefixed ``merger.``.

    ``num_layers`` MUST match the layer count of the golden being validated (the
    bring-up golden runs the REDUCED depth of 2 vs the production 42) so the
    composed state_dict and the reference run agree.

    Returns the flat tower state_dict (fp32) with keys:
        patch_embed.proj.weight, patch_embed.proj.bias, patch_embed.norm.weight,
        blocks.{i}.{norm1.weight, attn.qkv.weight, attn.proj.weight, norm2.weight,
                    mlp.fc1.weight, mlp.fc2.weight, mlp.fc3.weight} for i in [0, num_layers),
        post_trunk_norm.weight,
        merger.{ln_q.weight, ln_q.bias, mlp.0.weight, mlp.0.bias,
                mlp.2.weight, mlp.2.bias}.
    """
    sd: Dict[str, torch.Tensor] = {}

    # patch_embed (host-resident boundary): Conv2d + RMSNorm.
    pe = load_vision_patch_embed_weights(checkpoint_path)
    sd["patch_embed.proj.weight"] = pe["proj.weight"]
    sd["patch_embed.proj.bias"] = pe["proj.bias"]
    sd["patch_embed.norm.weight"] = pe["norm.weight"]

    # N transformer blocks.
    for i in range(num_layers):
        blk = load_vision_block_weights(checkpoint_path, block_idx=i)
        for k, v in blk.items():
            sd[f"blocks.{i}.{k}"] = v

    # post-trunk RMSNorm (eps 1e-5).
    sd["post_trunk_norm.weight"] = load_vision_rmsnorm_weight(checkpoint_path, "vision_tower.post_trunk_norm.weight")

    # patch merger (LayerNorm + GELU MLP, both biased).
    merger = load_vision_patch_merger_weights(checkpoint_path)
    for k, v in merger.items():
        sd[f"merger.{k}"] = v

    return sd
