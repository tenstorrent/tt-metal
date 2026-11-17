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


@contextmanager
def staged_directory(path: PathLike) -> Iterator[Path]:
    """A directory to write a checkpoint into, which replaces ``path`` only once it is complete.

    A failed or interrupted save leaves what was at ``path`` before, never a partial checkpoint.
    """
    final = Path(path)
    staging = final.with_name(final.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        yield staging
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if final.is_dir():
        shutil.rmtree(final)
    elif final.exists():
        final.unlink()
    os.replace(staging, final)


def write_trainer_state(directory: PathLike, state: dict[str, Any]) -> None:
    (Path(directory) / TRAINER_STATE).write_text(json.dumps(state, indent=2) + "\n")
