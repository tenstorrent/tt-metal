# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-side checks of the vision weight loader (tt/vision/loader.py) on a small checkpoint that the test writes in the
layout of the MiniMax-M3 checkpoint, with a config.json whose vision dimensions all differ from MiniMaxM3VisionConfig's
defaults. As in the checkpoint:
  - the names are transformers 4.52.4's (vision_tower.vision_model.*, patch_merge_mlp.*), which the loader renames;
  - the vision tensors are in two of three shards, with a bias in another shard than its weight and a text weight next
    to them; the third shard holds text weights only and is not on disk, so the test fails if the loader reads it;
  - the patch embed weight is fp32 and the other weights bf16;
  - config.json has the merge and temporal sizes under img_token_compression_config.
No device and no checkpoint.

These tests have no mesh id; the 1-card CI leg selects them by name (its -k includes "test_vision_").

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_loader.py
"""
from __future__ import annotations

import dataclasses
import json

import torch
from safetensors.torch import save_file

from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig, MiniMaxM3VisionEncoder
from models.demos.minimax_m3.tt.vision.loader import build_reference, load_vision_state_dict

COMPRESSION = {"image_token_compression_method": "patch_merge", "spatial_merge_size": 3, "temporal_patch_size": 1}
CONFIG_JSON = {
    "vision_config": {
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_channels": 1,
        "patch_size": 4,
        "layer_norm_eps": 1e-6,
        "rope_theta": 5000.0,
        "img_token_compression_config": COMPRESSION,
    },
    "img_token_compression_config": COMPRESSION,
    "projector_hidden_size": 80,
    "text_config": {"hidden_size": 48},
}
# The config that build_reference must read from CONFIG_JSON.
CONFIG = MiniMaxM3VisionConfig(
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_channels=1,
    patch_size=4,
    temporal_patch_size=1,
    spatial_merge_size=3,
    layer_norm_eps=1e-6,
    rope_theta=5000.0,
    projector_hidden_size=80,
    text_hidden_size=48,
)
TEXT_SHARD = "model-00001-of-00003.safetensors"  # listed in the index, not written
PROJECTOR_SHARD = "model-00002-of-00003.safetensors"
TOWER_SHARD = "model-00003-of-00003.safetensors"


def _checkpoint_name(name: str) -> str:
    """Hugging Face 5.12 name (MiniMaxM3VisionEncoder.hf_state_dict) -> the checkpoint's transformers 4.52.4 name."""
    if name.startswith("multi_modal_projector.merge_"):
        return "patch_merge_mlp." + name[len("multi_modal_projector.merge_") :]
    if name.startswith("vision_tower."):
        rest = name[len("vision_tower.") :].replace("embeddings.proj.", "embeddings.patch_embedding.")
        return "vision_tower.vision_model." + ("encoder." + rest if rest.startswith("layers.") else rest)
    return name


def _write_checkpoint(path, source: MiniMaxM3VisionEncoder):
    """source's weights as the checkpoint stores them (module docstring), with the index and CONFIG_JSON, in path."""
    shards = {PROJECTOR_SHARD: {}, TOWER_SHARD: {}}
    shards[PROJECTOR_SHARD]["language_model.model.layers.0.block_sparse_moe.gate.weight"] = torch.zeros(4, 4).bfloat16()
    for name, value in source.hf_state_dict().items():
        # The projector's weights in one shard, but for its first bias, which is with the tower's weights.
        in_projector_shard = name.startswith("multi_modal_projector.") and name != "multi_modal_projector.linear_1.bias"
        value = value if name == "vision_tower.embeddings.proj.weight" else value.bfloat16()
        shards[PROJECTOR_SHARD if in_projector_shard else TOWER_SHARD][_checkpoint_name(name)] = value
    weight_map = {"language_model.model.embed_tokens.weight": TEXT_SHARD}
    for shard, tensors in shards.items():
        save_file(tensors, str(path / shard))
        weight_map.update(dict.fromkeys(tensors, shard))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps(CONFIG_JSON))


def test_vision_build_reference(tmp_path):
    """load_vision_state_dict reads the vision weights under their Hugging Face 5.12 names, and build_reference builds
    the reference with the config of config.json and puts every one of them into it, in fp32 at the precision the
    checkpoint stores them."""
    default = MiniMaxM3VisionConfig()
    assert all(getattr(CONFIG, f.name) != getattr(default, f.name) for f in dataclasses.fields(CONFIG))
    source = MiniMaxM3VisionEncoder(CONFIG).init_random_weights(seed=0)
    _write_checkpoint(tmp_path, source)

    assert load_vision_state_dict(tmp_path).keys() == source.hf_state_dict().keys()
    reference = build_reference(tmp_path)
    assert reference.config == CONFIG and not reference.training
    loaded = reference.state_dict()
    assert loaded.keys() == source.state_dict().keys()
    for name, value in source.state_dict().items():
        want = value if name == "patch_embed.weight" else value.bfloat16().float()
        assert loaded[name].dtype == torch.float32 and torch.equal(loaded[name], want), name


def test_vision_build_reference_reads_config_first(tmp_path, expect_error):
    """build_reference opens config.json before the index and the shards, so in a directory without a checkpoint it
    raises for config.json; for an empty snapshot_dir it raises ValueError instead of reading the current directory."""
    with expect_error(FileNotFoundError, "config.json"):
        build_reference(tmp_path)
    with expect_error(ValueError, "snapshot_dir must be a MiniMax-M3 checkpoint directory"):
        build_reference("")
