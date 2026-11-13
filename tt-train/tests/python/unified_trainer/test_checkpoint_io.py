# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint directories: staged writes, PEFT adapter files and the trainer state. Device-free."""

import json

import numpy as np
import pytest

from ttml.unified_trainer.checkpoint_io import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    is_adapter_dir,
    is_model_dir,
    read_adapter,
    read_trainer_state,
    staged_directory,
    write_adapter,
    write_trainer_state,
)
from ttml.unified_trainer.providers.base import AdapterState

pytest.importorskip("safetensors")


def test_a_staged_directory_replaces_the_old_one_when_complete(tmp_path):
    target = tmp_path / "ckpt"
    target.mkdir()
    (target / "old.txt").write_text("old")
    with staged_directory(target) as directory:
        (directory / "new.txt").write_text("new")
        assert (target / "old.txt").exists()
    assert sorted(p.name for p in target.iterdir()) == ["new.txt"]
    assert not (tmp_path / "ckpt.tmp").exists()


def test_a_failed_save_leaves_the_old_checkpoint(tmp_path, expect_error):
    target = tmp_path / "ckpt"
    target.mkdir()
    (target / "old.txt").write_text("old")
    with expect_error(RuntimeError, "disk full"):
        with staged_directory(target) as directory:
            (directory / "partial.txt").write_text("partial")
            raise RuntimeError("disk full")
    assert sorted(p.name for p in target.iterdir()) == ["old.txt"]
    assert not (tmp_path / "ckpt.tmp").exists()


def test_a_staged_directory_replaces_a_file_and_creates_parents(tmp_path):
    (tmp_path / "ckpt").write_text("a file")
    with staged_directory(tmp_path / "ckpt") as directory:
        (directory / "x").write_text("x")
    assert (tmp_path / "ckpt" / "x").exists()
    with staged_directory(tmp_path / "a" / "b") as directory:
        (directory / "x").write_text("x")
    assert (tmp_path / "a" / "b" / "x").exists()


def test_adapter_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": rng.standard_normal((4, 8)),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": rng.standard_normal((8, 4)),
    }
    config = {"peft_type": "LORA", "r": 4, "lora_alpha": 8, "target_modules": ["q_proj"]}
    write_adapter(tmp_path, AdapterState(tensors, config))

    assert is_adapter_dir(tmp_path) and not is_model_dir(tmp_path)
    assert json.loads((tmp_path / ADAPTER_CONFIG).read_text()) == config
    state = read_adapter(tmp_path)
    assert state.config == config
    for name, value in tensors.items():
        assert state.tensors[name].dtype == np.float32
        np.testing.assert_array_equal(state.tensors[name], value.astype(np.float32))


def test_read_adapter_errors(tmp_path, expect_error):
    with expect_error(FileNotFoundError, ADAPTER_CONFIG):
        read_adapter(tmp_path)
    (tmp_path / ADAPTER_CONFIG).write_text("{}")
    with expect_error(FileNotFoundError, ADAPTER_WEIGHTS):
        read_adapter(tmp_path)


def test_trainer_state(tmp_path):
    assert read_trainer_state(tmp_path) == {}
    state = {"step": 10, "epoch": 0, "metrics": {"val/loss": 1.5}}
    write_trainer_state(tmp_path, state)
    assert read_trainer_state(tmp_path) == state


def test_is_model_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert not is_model_dir(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert is_model_dir(tmp_path)


def trainer_for(lora):
    """An SFTTrainer as setup leaves it, short of the model: a checkpoint of the wrong kind fails before it is read."""
    from ttml.unified_trainer import SFTTrainer, SFTTrainerConfig
    from ttml.unified_trainer.providers import LlamaProvider

    trainer = SFTTrainer()
    trainer.config = SFTTrainerConfig(
        dataset_id="sst2", model={"name": "m", "lora": lora}, optimizer={"lr": 1e-4}, batch_size=1, max_length=32
    )
    trainer.provider, trainer.model = LlamaProvider(), None
    return trainer


def test_a_checkpoint_of_the_other_kind_is_rejected(tmp_path, expect_error):
    from ttml.unified_trainer import UnsupportedTrainingConfig

    model, adapter = tmp_path / "model", tmp_path / "adapter"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"")
    adapter.mkdir()
    write_adapter(adapter, AdapterState({}, {"peft_type": "LORA"}))

    lora = {"r": 4, "alpha": 8, "target_modules": ["o_proj"]}
    with expect_error(UnsupportedTrainingConfig, "is a full model, not a LoRA adapter; .* make it the model.name"):
        trainer_for(lora).load_checkpoint(str(model))
    with expect_error(UnsupportedTrainingConfig, "is a LoRA adapter, and this run trains every weight"):
        trainer_for(None).load_checkpoint(str(adapter))
    with expect_error(FileNotFoundError, "is not an HF model directory"):
        trainer_for(None).load_checkpoint(str(tmp_path / "missing"))
    with expect_error(FileNotFoundError, "is not a PEFT adapter directory"):
        trainer_for(lora).load_checkpoint(str(tmp_path / "missing"))
