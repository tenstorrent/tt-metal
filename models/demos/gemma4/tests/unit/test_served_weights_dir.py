# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The loaders read the served checkpoint (MODEL_WEIGHTS_DIR / HF_MODEL) when the path vLLM
handed over holds no safetensors: tt-inference-server starts vLLM with the hub id while the
container's hub cache carries only the config and tokenizer. Pure Python (tmp safetensors)."""

import pytest
import torch
from safetensors.torch import save_file

from models.demos.gemma4.tt.generator import VISION_KEY_PREFIXES, load_vision_state_dict
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs


@pytest.fixture
def served_checkpoint(tmp_path, monkeypatch):
    weights = tmp_path / "weights" / "gemma-4-test"
    weights.mkdir(parents=True)
    vision_key = f"{VISION_KEY_PREFIXES[0]}probe.weight"
    save_file(
        {"model.language_model.probe.weight": torch.ones(2, 2), vision_key: torch.full((2,), 3.0)},
        str(weights / "model-00001-of-00001.safetensors"),
    )
    empty = tmp_path / "hub_snapshot"  # what vLLM leaves behind: no weight files
    empty.mkdir()
    for env in ("MODEL_WEIGHTS_DIR", "HF_MODEL", "GEMMA4_MODEL_PATH"):
        monkeypatch.delenv(env, raising=False)
    return weights, empty, vision_key


def test_served_weights_dir_wants_a_directory_with_safetensors(served_checkpoint, monkeypatch):
    weights, empty, _ = served_checkpoint
    assert Gemma4ModelArgs.served_weights_dir() is None
    monkeypatch.setenv("HF_MODEL", "google/gemma-4-test")  # a repo id, not a directory
    assert Gemma4ModelArgs.served_weights_dir() is None
    monkeypatch.setenv("HF_MODEL", str(empty))
    assert Gemma4ModelArgs.served_weights_dir() is None
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(weights))
    assert Gemma4ModelArgs.served_weights_dir() == str(weights)


def test_loaders_fall_back_to_the_served_checkpoint(served_checkpoint, monkeypatch):
    weights, empty, vision_key = served_checkpoint
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(weights))
    state_dict = Gemma4ModelArgs.load_state_dict(str(empty))
    assert torch.equal(state_dict["model.language_model.probe.weight"], torch.ones(2, 2))
    vision = load_vision_state_dict(str(empty))
    assert list(vision) == [vision_key]


def test_vision_loader_still_fails_loudly_without_a_checkpoint(served_checkpoint, expect_error):
    _, empty, _ = served_checkpoint
    with expect_error(FileNotFoundError, "no vision tensors"):
        load_vision_state_dict(str(empty))
