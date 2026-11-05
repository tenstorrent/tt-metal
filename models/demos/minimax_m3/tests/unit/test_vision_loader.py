# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-side checks of the vision weight loader (tt/vision/loader.py) on a small checkpoint that the test writes in the
layout of the MiniMax-M3 checkpoint, with a config.json whose vision dimensions all differ from MiniMaxM3VisionConfig's
defaults. As in the checkpoint:
  - the names, shards and dtypes of vision_checkpoint.write_checkpoint;
  - config.json has the merge and temporal sizes under img_token_compression_config.
It also checks that the loader rejects an index with no vision tensor and that the vision config rejects an activation
other than the exact GELU. No device and no checkpoint.

These tests have no mesh id; the 1-card CI leg selects them by name (its -k includes "test_vision_").

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_loader.py
"""
from __future__ import annotations

import dataclasses
import json

import torch

from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionEncoder
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.loader import (
    CHECKPOINT_PREFIXES,
    _remap,
    build_reference,
    load_vision_state_dict,
)

from .vision_checkpoint import TEXT_SHARD, write_checkpoint

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


def test_vision_build_reference(tmp_path):
    """load_vision_state_dict reads the vision weights under their Hugging Face 5.12 names, and build_reference builds
    the reference with the config of config.json and puts every one of them into it, in fp32 at the precision the
    checkpoint stores them."""
    default = MiniMaxM3VisionConfig()
    assert all(getattr(CONFIG, f.name) != getattr(default, f.name) for f in dataclasses.fields(CONFIG))
    source = MiniMaxM3VisionEncoder(CONFIG).init_random_weights(seed=0)
    write_checkpoint(tmp_path, source, CONFIG_JSON)

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


def test_vision_load_rejects_index_without_vision_tensors(tmp_path, expect_error):
    """load_vision_state_dict raises ValueError for a checkpoint whose index lists no vision tensor, such as a text-only
    one, instead of returning no weights."""
    weight_map = {"language_model.model.embed_tokens.weight": TEXT_SHARD}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    with expect_error(ValueError, r"lists no tensor starting with vision_tower\., multi_modal_projector\."):
        load_vision_state_dict(tmp_path)


def test_vision_config_rejects_other_activation(expect_error):
    """from_hf_config takes a vision hidden_act of "gelu", the exact (erf) GELU that the reference and the TT MLP run,
    and raises ValueError for any other: Hugging Face's vision MLP reads it, so such a checkpoint would load and give
    other outputs than Hugging Face's."""

    def config_json(hidden_act):
        return {**CONFIG_JSON, "vision_config": {**CONFIG_JSON["vision_config"], "hidden_act": hidden_act}}

    assert MiniMaxM3VisionConfig.from_hf_config(config_json("gelu")) == CONFIG
    for hidden_act in ("quick_gelu", "gelu_pytorch_tanh"):
        with expect_error(ValueError, f"hidden_act must be 'gelu', the exact \\(erf\\) GELU, got '{hidden_act}'"):
            MiniMaxM3VisionConfig.from_hf_config(config_json(hidden_act))


def test_vision_checkpoint_prefixes_are_the_renamed_ones():
    """CHECKPOINT_PREFIXES, under which the encoder rejects a state_dict's weights, are the prefixes that _remap
    renames: a key under them changes name, and a key in the v5.12 layout keeps its name."""
    checkpoint_keys = [
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
        "patch_merge_mlp.linear_1.weight",
    ]
    model_keys = [_remap(key) for key in checkpoint_keys] + ["multi_modal_projector.linear_1.weight"]
    for key in checkpoint_keys:
        assert key.startswith(CHECKPOINT_PREFIXES) and _remap(key) != key, key
    for key in model_keys:
        assert not key.startswith(CHECKPOINT_PREFIXES) and _remap(key) == key, key
