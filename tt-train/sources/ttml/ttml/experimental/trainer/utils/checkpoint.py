# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Writing checkpoint directories.

A checkpoint holds the model's weights, which the model provider writes, and a ``trainer_state.json`` with the step,
epoch and metrics, the format, and the base model.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Union

TRAINER_STATE = "trainer_state.json"

PathLike = Union[str, os.PathLike]


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _previous(path: Path) -> Path:
    return path.with_name(path.name + ".old")


def checkpoint_exists(path: PathLike) -> bool:
    """Whether ``path`` holds a checkpoint, counting the one an interrupted ``staged_directory`` save left aside, which
    the next save to ``path`` restores and then replaces."""
    path = Path(path)
    return path.exists() or _previous(path).exists()


@contextmanager
def staged_directory(path: PathLike) -> Iterator[Path]:
    """A directory to write a checkpoint into, which replaces ``path`` only once it is complete.

    A failed or interrupted save leaves what was at ``path`` before, never a partial checkpoint.
    The directory already at ``path`` is renamed aside and deleted only after the new one is in its place.
    """
    final = Path(path)
    staging = final.with_name(final.name + ".tmp")
    previous = _previous(final)
    # A crash after the previous checkpoint was renamed aside, and before the new one took its name,
    # leaves the last complete checkpoint at ``previous``.
    if not final.exists() and previous.exists():
        os.replace(previous, final)

    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        yield staging
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # Rename the previous checkpoint aside before publishing. Deleting it first would leave only the
    # staging directory if the process died before the new one was moved into place.
    if previous.exists():
        _remove(previous)
    if final.exists():
        os.replace(final, previous)
    try:
        os.replace(staging, final)
    except BaseException:
        if not final.exists() and previous.exists():
            os.replace(previous, final)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if previous.exists():
        _remove(previous)


def write_trainer_state(directory: PathLike, state: dict[str, Any]) -> None:
    (Path(directory) / TRAINER_STATE).write_text(json.dumps(state, indent=2) + "\n")
