# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CheckpointManager: when to save, and which checkpoints to keep. Device-free."""

import json
import os

import pytest

from ttml.unified_trainer.configs import CheckpointConfig
from ttml.unified_trainer.managers import CheckpointManager


class Store:
    """Stands in for the trainer's save_checkpoint."""

    def __init__(self) -> None:
        self.optimizer = []

    def save(self, path, metadata, optimizer=False):
        os.makedirs(path, exist_ok=True)
        self.optimizer.append(optimizer)


def manager(tmp_path, store=None, **overrides):
    store = store or Store()
    return CheckpointManager(CheckpointConfig(project_dir=str(tmp_path), **overrides), store.save)


def saved_names(tmp_path):
    return sorted(name for name in os.listdir(tmp_path / "checkpoints") if name != "checkpoint_history.json")


def history(tmp_path):
    return json.loads((tmp_path / "checkpoints" / "checkpoint_history.json").read_text())["checkpoints"]


@pytest.mark.parametrize(
    "strategy, step, epoch, expected",
    [
        ({"type": "step", "frequency": 2}, 4, None, True),
        ({"type": "step", "frequency": 2}, 3, None, False),
        ({"type": "step", "frequency": 2}, 4, 0, False),
        ({"type": "epoch", "frequency": 2}, 4, None, False),
        ({"type": "epoch", "frequency": 2}, 4, 1, True),
        ({"type": "epoch", "frequency": 2}, 4, 0, False),
        ({"type": "none"}, 4, None, False),
        ({"type": "none"}, 4, 1, False),
    ],
)
def test_should_save_checkpoint(tmp_path, strategy, step, epoch, expected):
    assert manager(tmp_path, save_strategy=strategy).should_save_checkpoint(step, epoch) is expected


@pytest.mark.parametrize("save_optim", [False, True])
def test_save_optim_saves_the_optimizer_state(tmp_path, save_optim):
    store = Store()
    manager(tmp_path, store, save_optim=save_optim).save_checkpoint(1, 0)
    assert store.optimizer == [save_optim]


def test_keeps_the_last_n(tmp_path):
    checkpoints = manager(tmp_path, rotation_strategy={"type": "last", "keep_n": 2})
    for step in (1, 2, 3):
        checkpoints.save_checkpoint(step, 0, name=f"step{step}")
    assert saved_names(tmp_path) == ["step2", "step3"]
    assert [entry["step"] for entry in history(tmp_path)] == [2, 3]


def test_rotation_none_keeps_every_checkpoint(tmp_path):
    checkpoints = manager(tmp_path, rotation_strategy={"type": "none"})
    for step in (1, 2, 3):
        checkpoints.save_checkpoint(step, 0, name=f"step{step}")
    assert saved_names(tmp_path) == ["step1", "step2", "step3"]


def test_keeps_the_best_n_and_the_latest(tmp_path):
    checkpoints = manager(tmp_path, rotation_strategy={"type": "best", "keep_n": 1})
    for step, loss in ((1, 1.0), (2, 2.0), (3, 0.5), (4, 3.0)):
        checkpoints.save_checkpoint(step, 0, metrics={"val/loss": loss}, name=f"step{step}")
        if step == 2:
            # Not the best, but the latest.
            assert saved_names(tmp_path) == ["step1", "step2"]
    assert saved_names(tmp_path) == ["step3", "step4"]
    assert [entry["step"] for entry in history(tmp_path)] == [3, 4]


def test_max_mode_keeps_the_highest(tmp_path):
    checkpoints = manager(tmp_path, rotation_strategy={"type": "best", "keep_n": 1, "mode": "max"})
    for step, loss in ((1, 2.0), (2, 1.0), (3, 0.5)):
        checkpoints.save_checkpoint(step, 0, metrics={"val/loss": loss}, name=f"step{step}")
    assert saved_names(tmp_path) == ["step1", "step3"]


def test_best_rotation_needs_the_metric(tmp_path, expect_error):
    checkpoints = manager(tmp_path, rotation_strategy={"type": "best"})
    with expect_error(ValueError, "best ranks checkpoints by val/loss"):
        checkpoints.save_checkpoint(1, 0, metrics={"train/loss": 1.0}, name="step1")
    assert saved_names(tmp_path) == []


def test_saving_under_a_tracked_name_replaces_its_entry(tmp_path):
    checkpoints = manager(tmp_path, rotation_strategy={"type": "last", "keep_n": 3})
    checkpoints.save_checkpoint(1, 0, name="final")
    checkpoints.save_checkpoint(2, 0, name="final")
    assert [entry["step"] for entry in checkpoints.history] == [2]
    assert saved_names(tmp_path) == ["final"]


def test_default_names_carry_the_step_and_epoch(tmp_path):
    path = manager(tmp_path).save_checkpoint(7, 1)
    assert os.path.basename(path).startswith("checkpoint_step7_epoch1_")


def test_a_new_run_starts_a_new_history_and_keeps_the_old_checkpoints(tmp_path):
    first = manager(tmp_path, rotation_strategy={"type": "last", "keep_n": 1})
    first.save_checkpoint(1, 0, name="old")
    second = manager(tmp_path, rotation_strategy={"type": "last", "keep_n": 1})
    assert second.history == []
    second.save_checkpoint(1, 0, name="new")
    second.save_checkpoint(2, 0, name="newer")
    assert [entry["name"] for entry in history(tmp_path)] == ["newer"]
    assert saved_names(tmp_path) == ["newer", "old"]
