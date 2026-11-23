# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Saving checkpoint directories: a failed or interrupted save never leaves a partial checkpoint, a run never saves
over another run's, the model provider writes the weights and the trainer ``trainer_state.json``, and a checkpoint
loads only onto the model it records. Device-free."""

import importlib.metadata
import json
import os
from datetime import datetime
from types import SimpleNamespace

import pytest

import ttml.experimental.trainer.implementations.sft as sft
import ttml.experimental.trainer.managers.checkpoint as checkpoint_manager
from ttml.experimental.errors import UnsupportedTrainingConfig
from ttml.experimental.models.provider.base import ModelProvider
from ttml.experimental.trainer import SFTTrainer, SFTTrainerConfig
from ttml.experimental.trainer.configs import CheckpointConfig
from ttml.experimental.trainer.managers.checkpoint import CheckpointManager
from ttml.experimental.trainer.utils import (
    TRAINER_STATE,
    checkpoint_exists,
    read_trainer_state,
    staged_directory,
    tt_metal_version,
)


def _write_checkpoint(path, content):
    path.mkdir()
    (path / "weights").write_text(content)


def _save(path, content):
    with staged_directory(path) as directory:
        (directory / "weights").write_text(content)


def _entries(directory):
    return sorted(os.listdir(directory))


def test_save_replaces_previous_checkpoint(tmp_path):
    final = tmp_path / "checkpoint"
    _write_checkpoint(final, "old")
    _save(final, "new")
    assert (final / "weights").read_text() == "new"
    assert _entries(tmp_path) == ["checkpoint"]


def test_failed_save_keeps_previous_checkpoint(tmp_path, expect_error):
    final = tmp_path / "checkpoint"
    _write_checkpoint(final, "old")
    with expect_error(RuntimeError, "export failed"):
        with staged_directory(final) as directory:
            (directory / "weights").write_text("partial")
            raise RuntimeError("export failed")
    assert (final / "weights").read_text() == "old"
    assert _entries(tmp_path) == ["checkpoint"]


def test_failed_publish_restores_previous_checkpoint(tmp_path, monkeypatch, expect_error):
    final = tmp_path / "checkpoint"
    _write_checkpoint(final, "old")
    replace = os.replace

    def fail_publish(src, dst):
        if str(src).endswith(".tmp"):
            raise OSError("rename failed")
        replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_publish)
    with expect_error(OSError, "rename failed"):
        _save(final, "new")
    assert (final / "weights").read_text() == "old"
    assert _entries(tmp_path) == ["checkpoint"]


def test_interrupted_publish_is_restored_then_replaced(tmp_path):
    # What a crash between the two renames leaves: the previous checkpoint aside, and nothing at its name.
    final = tmp_path / "checkpoint"
    _write_checkpoint(tmp_path / "checkpoint.old", "old")
    assert checkpoint_exists(final)
    _save(final, "new")
    assert (final / "weights").read_text() == "new"
    assert _entries(tmp_path) == ["checkpoint"]


def test_failed_save_after_interrupted_publish_restores_previous_checkpoint(tmp_path, expect_error):
    final = tmp_path / "checkpoint"
    _write_checkpoint(tmp_path / "checkpoint.old", "old")
    with expect_error(RuntimeError, "export failed"):
        with staged_directory(final):
            raise RuntimeError("export failed")
    assert (final / "weights").read_text() == "old"
    assert _entries(tmp_path) == ["checkpoint"]


@pytest.mark.parametrize("taken", ["{name}", "{name}.old"])
def test_manager_never_saves_over_another_runs_checkpoint(tmp_path, monkeypatch, expect_error, taken):
    now = datetime(2026, 1, 2, 3, 4, 5)

    class FixedTime:
        @staticmethod
        def now():
            return now

    monkeypatch.setattr(checkpoint_manager, "datetime", FixedTime)
    config = CheckpointConfig(save_strategy={"type": "step", "project_dir": str(tmp_path)})
    saved = []
    manager = CheckpointManager(config, save=lambda path, metadata: saved.append(path))
    name = f"checkpoint_step10_epoch0_{now:%Y%m%d_%H%M%S}"
    _write_checkpoint(tmp_path / "checkpoints" / taken.format(name=name), "other run")

    with expect_error(FileExistsError, "is another run's checkpoint"):
        manager.save_checkpoint(step=10, epoch=0)
    assert saved == []
    assert manager.history == []
    assert not os.path.exists(manager.history_file)


def test_tt_metal_version_is_the_ttnn_packages(monkeypatch):
    versions = {"ttnn": "0.64.0"}

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert tt_metal_version() == "0.64.0"
    versions.clear()
    assert tt_metal_version() is None


class _Provider(ModelProvider):
    """Writes ``weights`` from ``export_weights`` and nothing else; records the directories ``load_weights`` gets.
    Its model's files are at ``acme-7b/``."""

    def __init__(self, config):
        super().__init__(config)
        self.loaded = []

    def model_source(self):
        return "acme-7b/"

    def check(self):
        pass

    def build(self):
        return object()

    def load_weights(self, model, directory):
        self.loaded.append(directory)

    def export_weights(self, model, directory):
        (directory / "weights").write_text("new")


def _trainer(init=None):
    """A trainer of ``acme/acme-7b`` with ``_Provider``, starting from the checkpoint at ``init``, if given."""
    config = SFTTrainerConfig(
        model={"name": "acme/acme-7b"},
        dataset={"id": "alpaca"},
        optimizer={"lr": 1e-3},
        batch_size=1,
        max_length=32,
        checkpoint={"init_strategy": {"path": str(init)}} if init else {},
    )
    trainer = SFTTrainer()
    trainer.config, trainer.provider = config, _Provider(config)
    trainer.logger = SimpleNamespace(info=lambda message: None)
    return trainer


def test_save_checkpoint_has_the_provider_export_the_weights_into_a_staged_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sft, "tt_metal_version", lambda: "0.64.0")
    trainer = _trainer()
    trainer.model = object()
    final = tmp_path / "checkpoint"
    _write_checkpoint(final, "old")
    trainer.save_checkpoint(str(final), {"step": 3, "epoch": 0, "metrics": {"train/loss": 1.5}})

    assert (final / "weights").read_text() == "new"
    assert read_trainer_state(final) == {
        "format": {"type": "hf"},
        "config": trainer.config.model_dump(mode="json"),
        "tt_metal_version": "0.64.0",
        "base_model": {"name_or_path": "acme/acme-7b"},
        "step": 3,
        "epoch": 0,
        "metrics": {"train/loss": 1.5},
    }
    assert _entries(tmp_path) == ["checkpoint"]


def test_metadata_adds_to_the_run_fields(monkeypatch):
    monkeypatch.setattr(sft, "tt_metal_version", lambda: None)
    state = _trainer().trainer_state({"image": "host"})
    assert state["image"] == "host"
    assert state["tt_metal_version"] is None


@pytest.mark.parametrize("key", ["format", "config", "tt_metal_version", "base_model"])
def test_metadata_may_not_replace_the_run_fields(expect_error, key):
    with expect_error(ValueError, key):
        _trainer().trainer_state({key: "host"})


@pytest.mark.parametrize(
    "recorded, loads",
    [(None, True), ("acme/acme-7b", True), ("acme-7b/", True), ("meta-llama/Llama-3.2-1B", False)],
    ids=["no-trainer-state", "model-name", "model-source", "other-model"],
)
def test_a_checkpoint_is_loaded_only_onto_the_model_it_records(tmp_path, expect_error, recorded, loads):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    if recorded is not None:
        (checkpoint / TRAINER_STATE).write_text(json.dumps({"base_model": {"name_or_path": recorded}}))
    trainer = _trainer(init=checkpoint)

    if loads:
        trainer._load_model()
        assert trainer.provider.loaded == [str(checkpoint)]
    else:
        with expect_error(UnsupportedTrainingConfig, r"^checkpoint: .* is for 'meta-llama/Llama-3.2-1B'"):
            trainer._load_model()
        assert trainer.provider.loaded == []
