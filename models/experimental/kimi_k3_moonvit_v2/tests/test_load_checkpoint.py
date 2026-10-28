# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU: the checkpoint loader reads only the vision and projector shards and matches the reference's keys."""

from __future__ import annotations

import json

import pytest
import torch

from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import (
    CHECKPOINT_ENV,
    checkpoint_path,
    drop_layers,
    load_vision_state_dict,
)
from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision import KimiK3VisionModel, load_vision_config


def test_fake_checkpoint(tmp_path, shallow_config):
    """Laid out like Kimi-K3: vision keys over two shards, text keys in a third shard that is never
    written, so opening it would fail."""
    from safetensors.torch import save_file

    deeper_config = load_vision_config(num_hidden_layers=shallow_config.num_hidden_layers + 1)
    deeper = KimiK3VisionModel(deeper_config).to(torch.bfloat16).state_dict()
    keys = sorted(deeper)
    shards = {"model-2.safetensors": keys[: len(keys) // 2], "model-3.safetensors": keys[len(keys) // 2 :]}
    weight_map = {"language_model.model.embed_tokens.weight": "model-1.safetensors"}
    for shard, shard_keys in shards.items():
        save_file({k: deeper[k].contiguous() for k in shard_keys}, tmp_path / shard)
        weight_map.update({k: shard for k in shard_keys})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    state = load_vision_state_dict(tmp_path, shallow_config.num_hidden_layers)
    expected = drop_layers(deeper, shallow_config.num_hidden_layers)
    assert state.keys() == expected.keys()
    assert all(torch.equal(state[k], expected[k]) for k in state)


def test_real_checkpoint(shallow_config):
    path = checkpoint_path()
    if path is None:
        pytest.skip(f"set ${CHECKPOINT_ENV} to a Kimi-K3 checkpoint folder")
    expected = KimiK3VisionModel(
        load_vision_config(path, num_hidden_layers=shallow_config.num_hidden_layers)
    ).state_dict()
    state = load_vision_state_dict(path, shallow_config.num_hidden_layers)
    assert state.keys() == expected.keys()
    for key, value in state.items():
        assert value.shape == expected[key].shape, key
        assert value.dtype == torch.bfloat16, key
