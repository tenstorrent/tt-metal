# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Seeded random weights of the vision tower and projector, and a small checkpoint of them.

The checkpoint has the MiniMax-M3 checkpoint's layout. No ttnn, so host tests can use this module.
"""

from __future__ import annotations

import json
from typing import Dict, Mapping

import torch
from safetensors.torch import save_file

from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig

# write_checkpoint's layout, like the MiniMax-M3 checkpoint's:
#   - transformers 4.52.4 names (vision_tower.vision_model.*, patch_merge_mlp.*), which the loader renames;
#   - the vision tensors in two of three shards, with one bias in another shard than its weight, and a text weight
#     next to them. The third shard, text only, is in the index but not on disk, so a loader that reads it fails;
#   - the patch embedding as given (fp32 from random_vision_weights), the other weights in bf16.
TEXT_SHARD = "model-00001-of-00003.safetensors"  # listed in the index, not written
PROJECTOR_SHARD = "model-00002-of-00003.safetensors"
TOWER_SHARD = "model-00003-of-00003.safetensors"

# random_vision_weights: the seed of its generator and the standard deviation of its biases and LayerNorm terms.
SEED = 0
STD = 0.25


def random_vision_weights(config: MiniMaxM3VisionConfig) -> Dict[str, torch.Tensor]:
    """Seeded random fp32 weights of config's vision tower and projector, under transformers 5.12's names.

    The patch embedding is in its Conv3d layout. Linear and Conv3d weights are N(0, 1 / fan_in), so activations stay
    near unit scale. Biases and LayerNorm shifts are N(0, STD**2) and LayerNorm scales 1 + N(0, STD**2), so a dropped
    bias or affine term shows in the output. The device tests' bounds were measured with these exact values: another
    order of draws or another distribution needs new bounds.
    """
    c = config
    generator = torch.Generator().manual_seed(SEED)
    weights = {}

    def draw(name: str, shape: tuple, mean: float, scale: float):
        weights[name] = torch.empty(shape).normal_(mean, scale, generator=generator)

    def linear(name: str, out_features: int, in_features: int):
        draw(f"{name}.weight", (out_features, in_features), 0.0, in_features**-0.5)
        draw(f"{name}.bias", (out_features,), 0.0, STD)

    def layer_norm(name: str):
        draw(f"{name}.weight", (c.hidden_size,), 1.0, STD)
        draw(f"{name}.bias", (c.hidden_size,), 0.0, STD)

    kernel = (c.temporal_patch_size, c.patch_size, c.patch_size)
    draw("vision_tower.embeddings.proj.weight", (c.hidden_size, c.num_channels, *kernel), 0.0, c.patch_dim**-0.5)
    layer_norm("vision_tower.pre_layrnorm")
    for i in range(c.num_hidden_layers):
        layer = f"vision_tower.layers.{i}."
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(f"{layer}self_attn.{projection}", c.hidden_size, c.hidden_size)
        layer_norm(layer + "layer_norm1")
        linear(layer + "mlp.fc1", c.intermediate_size, c.hidden_size)
        linear(layer + "mlp.fc2", c.hidden_size, c.intermediate_size)
        layer_norm(layer + "layer_norm2")
    linear("multi_modal_projector.linear_1", c.projector_hidden_size, c.hidden_size)
    linear("multi_modal_projector.linear_2", c.text_hidden_size, c.projector_hidden_size)
    linear("multi_modal_projector.merge_linear_1", c.projector_hidden_size, c.merged_hidden_size)
    linear("multi_modal_projector.merge_linear_2", c.text_hidden_size, c.projector_hidden_size)
    return weights


def checkpoint_name(name: str) -> str:
    """Rename a transformers 5.12 name (random_vision_weights) to the checkpoint's transformers 4.52.4 name."""
    if name.startswith("multi_modal_projector.merge_"):
        return "patch_merge_mlp." + name[len("multi_modal_projector.merge_") :]
    if name.startswith("vision_tower."):
        rest = name[len("vision_tower.") :].replace("embeddings.proj.", "embeddings.patch_embedding.")
        return "vision_tower.vision_model." + ("encoder." + rest if rest.startswith("layers.") else rest)
    return name


def write_checkpoint(path, weights: Mapping[str, torch.Tensor], config_json: dict) -> None:
    """Write weights, under transformers 5.12 names, to path (a pathlib.Path) in the layout above, with an index and
    config_json as config.json."""
    shards = {PROJECTOR_SHARD: {}, TOWER_SHARD: {}}
    shards[PROJECTOR_SHARD]["language_model.model.layers.0.block_sparse_moe.gate.weight"] = torch.zeros(4, 4).bfloat16()
    for name, value in weights.items():
        # The projector's weights in one shard, but for its first bias, which is with the tower's weights.
        in_projector_shard = name.startswith("multi_modal_projector.") and name != "multi_modal_projector.linear_1.bias"
        value = value if name == "vision_tower.embeddings.proj.weight" else value.bfloat16()
        shards[PROJECTOR_SHARD if in_projector_shard else TOWER_SHARD][checkpoint_name(name)] = value
    weight_map = {"language_model.model.embed_tokens.weight": TEXT_SHARD}
    for shard, tensors in shards.items():
        save_file(tensors, str(path / shard))
        weight_map.update(dict.fromkeys(tensors, shard))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps(config_json))
