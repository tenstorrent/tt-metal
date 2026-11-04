# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A small checkpoint in the layout of the MiniMax-M3 checkpoint, written by the vision tests that read one from disk
(the loader's and the encoder's host tests). No ttnn, so host tests can use it."""

from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionEncoder

# A checkpoint in the layout of the MiniMax-M3 checkpoint, for the tests that read one from disk (write_checkpoint):
#   - the names are transformers 4.52.4's (vision_tower.vision_model.*, patch_merge_mlp.*), which the loader renames;
#   - the vision tensors are in two of three shards, with a bias in another shard than its weight and a text weight next
#     to them; the third shard holds text weights only and is not on disk, so a loader that reads it fails;
#   - the patch embed weight is as the source has it (fp32) and the other weights bf16.
TEXT_SHARD = "model-00001-of-00003.safetensors"  # listed in the index, not written
PROJECTOR_SHARD = "model-00002-of-00003.safetensors"
TOWER_SHARD = "model-00003-of-00003.safetensors"


def checkpoint_name(name: str) -> str:
    """Hugging Face 5.12 name (MiniMaxM3VisionEncoder.hf_state_dict) -> the checkpoint's transformers 4.52.4 name."""
    if name.startswith("multi_modal_projector.merge_"):
        return "patch_merge_mlp." + name[len("multi_modal_projector.merge_") :]
    if name.startswith("vision_tower."):
        rest = name[len("vision_tower.") :].replace("embeddings.proj.", "embeddings.patch_embedding.")
        return "vision_tower.vision_model." + ("encoder." + rest if rest.startswith("layers.") else rest)
    return name


def write_checkpoint(path, source: MiniMaxM3VisionEncoder, config_json: dict) -> None:
    """source's weights as the checkpoint stores them (above), with the index and config_json as config.json, in
    path (a pathlib.Path)."""
    shards = {PROJECTOR_SHARD: {}, TOWER_SHARD: {}}
    shards[PROJECTOR_SHARD]["language_model.model.layers.0.block_sparse_moe.gate.weight"] = torch.zeros(4, 4).bfloat16()
    for name, value in source.hf_state_dict().items():
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
