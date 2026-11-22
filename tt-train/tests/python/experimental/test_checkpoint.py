# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Saving checkpoint directories: a failed or interrupted save never leaves a partial checkpoint, and a run never
saves over another run's. Device-free."""

import os
from datetime import datetime

import pytest

import ttml.experimental.trainer.managers.checkpoint as checkpoint_manager
from ttml.experimental.trainer.configs import CheckpointConfig
from ttml.experimental.trainer.managers.checkpoint import CheckpointManager
from ttml.experimental.trainer.utils import checkpoint_exists, staged_directory


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
