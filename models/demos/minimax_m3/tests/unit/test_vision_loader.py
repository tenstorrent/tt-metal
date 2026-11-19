# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host tests of the vision weight loader (tt/vision/loader.py) and of the vision config (tt/vision/config.py).

The loader reads a small checkpoint that write_checkpoint writes in the MiniMax-M3 checkpoint's layout, so no device
and no real checkpoint are needed. test_vision_config_defaults_match_hf needs transformers with minimax_m3_vl.
"""
from __future__ import annotations

import dataclasses
import json
import os

import pytest
import torch

from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.loader import CHECKPOINT_PREFIXES, _remap, load_vision_state_dict

from .vision_checkpoint import TEXT_SHARD, random_vision_weights, write_checkpoint

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
# The config that MiniMaxM3VisionConfig.from_hf_config must read from CONFIG_JSON.
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_vision_load_state_dict_dtype(tmp_path, dtype):
    """load_vision_state_dict returns the vision weights as dtype.

    fp32 keeps the stored values. bf16 also rounds the patch embedding, which is stored in fp32.
    """
    weights = random_vision_weights(CONFIG)
    write_checkpoint(tmp_path, weights, CONFIG_JSON)
    loaded = load_vision_state_dict(tmp_path, dtype=dtype)
    assert loaded.keys() == weights.keys()
    for name, value in weights.items():
        stored = value if name == "vision_tower.embeddings.proj.weight" else value.bfloat16()
        assert loaded[name].dtype == dtype and torch.equal(loaded[name], stored.to(dtype)), name


def test_vision_load_rejects_index_without_vision_tensors(tmp_path, expect_error):
    """load_vision_state_dict raises ValueError for an index that lists no vision tensor, such as a text-only
    checkpoint's, and for an empty snapshot_dir."""
    weight_map = {"language_model.model.embed_tokens.weight": TEXT_SHARD}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    with expect_error(ValueError, r"lists no tensor starting with vision_tower\., multi_modal_projector\."):
        load_vision_state_dict(tmp_path)
    with expect_error(ValueError, "snapshot_dir must be a MiniMax-M3 checkpoint directory"):
        load_vision_state_dict("")


def test_vision_config_reads_checkpoint_json():
    """from_hf_config reads every field of a config.json in the checkpoint's layout, with the merge and temporal sizes
    under img_token_compression_config. Each value differs from the default, so a field left at it fails the test."""
    default = MiniMaxM3VisionConfig()
    assert all(getattr(CONFIG, f.name) != getattr(default, f.name) for f in dataclasses.fields(CONFIG))
    assert MiniMaxM3VisionConfig.from_hf_config(CONFIG_JSON) == CONFIG


def test_vision_config_rejects_other_activation(expect_error):
    """from_hf_config takes a vision hidden_act of "gelu", the exact (erf) GELU that the TT MLP runs, and raises
    ValueError for any other."""

    def config_json(hidden_act):
        return {**CONFIG_JSON, "vision_config": {**CONFIG_JSON["vision_config"], "hidden_act": hidden_act}}

    assert MiniMaxM3VisionConfig.from_hf_config(config_json("gelu")) == CONFIG
    for hidden_act in ("quick_gelu", "gelu_pytorch_tanh"):
        with expect_error(ValueError, f"hidden_act must be 'gelu', the exact \\(erf\\) GELU, got '{hidden_act}'"):
            MiniMaxM3VisionConfig.from_hf_config(config_json(hidden_act))


def test_vision_config_defaults_match_hf():
    """MiniMaxM3VisionConfig's defaults are Hugging Face's MiniMax-M3 defaults (MiniMaxM3VLConfig), and the
    checkpoint's when HF_MODEL is set."""
    from transformers import MiniMaxM3VLConfig

    assert MiniMaxM3VisionConfig.from_hf_config(MiniMaxM3VLConfig().to_dict()) == MiniMaxM3VisionConfig()
    checkpoint = os.getenv("HF_MODEL")
    if checkpoint and os.path.isfile(os.path.join(checkpoint, "config.json")):
        with open(os.path.join(checkpoint, "config.json")) as f:
            assert MiniMaxM3VisionConfig.from_hf_config(json.load(f)) == MiniMaxM3VisionConfig()


def test_vision_checkpoint_prefixes_are_the_renamed_ones():
    """CHECKPOINT_PREFIXES, whose keys the encoder rejects, are the prefixes that _remap renames: keys under them change
    name, and model keys keep theirs."""
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
