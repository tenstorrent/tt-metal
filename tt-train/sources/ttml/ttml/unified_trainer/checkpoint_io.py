# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint directories in HF's formats, plus the trainer's metadata.

A LoRA run's checkpoint is a PEFT adapter directory (``PeftModel.from_pretrained`` loads it); a full
fine-tune's is an HF model directory (``AutoModelForCausalLM.from_pretrained`` loads it). Both hold a
``trainer_state.json`` with the step, epoch and metrics, and may hold ``optimizer_state.pkl``, which the trainer
writes with ``ttml.checkpointing`` for a run to resume from.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Union

import numpy as np

from .providers.base import AdapterState

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
MODEL_CONFIG = "config.json"
OPTIMIZER_STATE = "optimizer_state.pkl"
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


def write_adapter(directory: PathLike, state: AdapterState) -> None:
    """Write ``adapter_model.safetensors`` (float32: lossless for bf16 weights) and ``adapter_config.json``."""
    from safetensors.numpy import save_file

    directory = Path(directory)
    tensors = {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in state.tensors.items()}
    save_file(tensors, str(directory / ADAPTER_WEIGHTS), metadata={"format": "pt"})
    (directory / ADAPTER_CONFIG).write_text(json.dumps(state.config, indent=2) + "\n")


def read_adapter(directory: PathLike) -> AdapterState:
    from safetensors.numpy import load_file

    directory = Path(directory)
    if not is_adapter_dir(directory):
        raise FileNotFoundError(f"{directory} is not a PEFT adapter directory: no {ADAPTER_CONFIG}")
    weights = directory / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"{directory} has no {ADAPTER_WEIGHTS}")
    return AdapterState(tensors=load_file(str(weights)), config=json.loads((directory / ADAPTER_CONFIG).read_text()))


def write_trainer_state(directory: PathLike, state: dict[str, Any]) -> None:
    (Path(directory) / TRAINER_STATE).write_text(json.dumps(state, indent=2) + "\n")


def read_trainer_state(directory: PathLike) -> dict[str, Any]:
    """The ``trainer_state.json`` of a checkpoint; ``{}`` for one written by another tool."""
    path = Path(directory) / TRAINER_STATE
    return json.loads(path.read_text()) if path.is_file() else {}


def is_adapter_dir(path: PathLike) -> bool:
    return (Path(path) / ADAPTER_CONFIG).is_file()


def is_model_dir(path: PathLike) -> bool:
    path = Path(path)
    return (path / MODEL_CONFIG).is_file() and any(path.glob("*.safetensors"))
