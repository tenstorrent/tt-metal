# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CheckpointManager: when to save, which checkpoints to keep, and which one to start from. Device-free."""

import json
import os

import pytest

from ttml.unified_trainer.configs import CheckpointConfig
from ttml.unified_trainer.managers import CheckpointManager


class Store:
    """Stands in for the trainer's save_checkpoint / load_checkpoint."""

    def __init__(self) -> None:
        self.loaded = []

    def save(self, path, metadata):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "state.json"), "w") as f:
            json.dump(metadata, f)

    def load(self, path):
        self.loaded.append(path)
        with open(os.path.join(path, "state.json")) as f:
            return json.load(f)


def manager(tmp_path, store=None, **overrides):
    store = store or Store()
    return CheckpointManager(CheckpointConfig(project_dir=str(tmp_path), **overrides), store.save, store.load)


def saved_names(tmp_path):
    return sorted(name for name in os.listdir(tmp_path / "checkpoints") if name != "checkpoint_history.json")


def test_optimizer_state_is_not_supported(tmp_path, expect_error):
    with expect_error(NotImplementedError, "save_optim"):
        manager(tmp_path, save_optim=True)


@pytest.mark.parametrize(
    "strategy, step, epoch, expected",
    [
        ("step", 4, None, True),
        ("step", 3, None, False),
        ("step", 4, 0, False),
        ("epoch", 4, None, False),
        ("epoch", 4, 1, True),
        ("epoch", 4, 0, False),
        ("none", 4, None, False),
        ("none", 4, 1, False),
    ],
)
def test_should_save_checkpoint(tmp_path, strategy, step, epoch, expected):
    checkpoints = manager(tmp_path, save_strategy=strategy, steps_freq=2, epoch_freq=2)
    assert checkpoints.should_save_checkpoint(step, epoch) is expected


def test_keeps_the_last_n(tmp_path):
    checkpoints = manager(tmp_path, keep_last_n=2)
    for step in (1, 2, 3):
        checkpoints.save_checkpoint(step, 0, name=f"step{step}")
    assert saved_names(tmp_path) == ["step2", "step3"]
    history = json.loads((tmp_path / "checkpoints" / "checkpoint_history.json").read_text())
    assert [entry["step"] for entry in history["checkpoints"]] == [2, 3]


def test_keep_all_checkpoints(tmp_path):
    checkpoints = manager(tmp_path, keep_last_n=1, keep_all_checkpoints=True)
    for step in (1, 2, 3):
        checkpoints.save_checkpoint(step, 0, name=f"step{step}")
    assert saved_names(tmp_path) == ["step1", "step2", "step3"]


def test_a_best_checkpoint_outlives_last_n_until_it_is_no_longer_best(tmp_path):
    checkpoints = manager(tmp_path, keep_last_n=1, keep_best_n=1)
    checkpoints.save_checkpoint(1, 0, metrics={"val/loss": 1.0}, name="step1")
    checkpoints.save_checkpoint(2, 0, metrics={"val/loss": 2.0}, name="step2")
    assert saved_names(tmp_path) == ["step1", "step2"]
    checkpoints.save_checkpoint(3, 0, metrics={"val/loss": 0.5}, name="step3")
    assert saved_names(tmp_path) == ["step3"]


def test_max_mode_keeps_the_highest(tmp_path):
    checkpoints = manager(tmp_path, keep_last_n=1, keep_best_n=1, checkpoint_metric_mode="max")
    checkpoints.save_checkpoint(1, 0, metrics={"val/loss": 2.0}, name="step1")
    checkpoints.save_checkpoint(2, 0, metrics={"val/loss": 1.0}, name="step2")
    assert saved_names(tmp_path) == ["step1", "step2"]


def test_saving_under_a_tracked_name_replaces_its_entry(tmp_path):
    checkpoints = manager(tmp_path, keep_last_n=3)
    checkpoints.save_checkpoint(1, 0, name="final")
    checkpoints.save_checkpoint(2, 0, name="final")
    assert [entry["step"] for entry in checkpoints.history["checkpoints"]] == [2]
    assert saved_names(tmp_path) == ["final"]


def test_default_names_carry_the_step_and_epoch(tmp_path):
    path = manager(tmp_path).save_checkpoint(7, 1)
    assert os.path.basename(path).startswith("checkpoint_step7_epoch1_")


def test_init_from_the_previous_run_s_last_or_best(tmp_path):
    store = Store()
    first = manager(tmp_path, store, keep_last_n=3, keep_best_n=1)
    first.save_checkpoint(1, 0, metrics={"val/loss": 0.5}, name="best")
    first.save_checkpoint(2, 0, metrics={"val/loss": 0.9}, name="last")

    assert manager(tmp_path, store, init_option="last").load_checkpoint()["step"] == 2
    assert manager(tmp_path, store, init_option="best").load_checkpoint()["step"] == 1
    path = str(tmp_path / "checkpoints" / "best")
    assert manager(tmp_path, store, init_option="path", checkpoint_path=path).load_checkpoint()["step"] == 1


def test_init_errors(tmp_path, expect_error):
    with expect_error(FileNotFoundError, "No last checkpoint"):
        manager(tmp_path, init_option="last").load_checkpoint()
    with expect_error(ValueError, "checkpoint_path must be set"):
        manager(tmp_path, init_option="path").load_checkpoint()


def test_a_new_run_starts_a_new_history(tmp_path):
    first = manager(tmp_path, keep_last_n=3)
    first.save_checkpoint(1, 0, name="old")
    second = manager(tmp_path, keep_last_n=3)
    assert second.history == {"checkpoints": [], "best_checkpoints": []}
    second.save_checkpoint(1, 0, name="new")
    history = json.loads((tmp_path / "checkpoints" / "checkpoint_history.json").read_text())
    assert [entry["name"] for entry in history["checkpoints"]] == ["new"]
