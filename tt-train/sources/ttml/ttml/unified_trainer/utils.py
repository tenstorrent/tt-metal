# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""What the trainers share: reading and writing checkpoint directories, and several optimizers stepped as one.

A checkpoint is a PEFT adapter directory (``PeftModel.from_pretrained`` loads it). It holds a
``trainer_state.json`` with the step, epoch and metrics, and may hold ``optimizer_state.pkl``, which the trainer
writes with ``ttml.checkpointing`` for a run to resume from.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, Union

import numpy as np

from ttml.models.provider import AdapterState, UnsupportedTrainingConfig

from .configs import OptimizerConfig

# ------------------------------------------------------------------ checkpoint directories

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
    """Whether ``path`` is an HF model directory: what a run trains adapters on, not a checkpoint."""
    path = Path(path)
    return (path / MODEL_CONFIG).is_file() and any(path.glob("*.safetensors"))


# ------------------------------------------------------------------ optimizers


def assign_parameters(names: Iterable[str], optimizers: Sequence[OptimizerConfig]) -> list[list[str]]:
    """The trainable parameter names each optimizer trains, in the order of ``optimizers``.

    A parameter goes to the first optimizer whose ``modules`` lists a ``/``-separated part of its name, else to the
    one optimizer without ``modules``. Raises if a parameter gets no optimizer, or an optimizer no parameter.
    """
    rest = next((index for index, optimizer in enumerate(optimizers) if optimizer.modules is None), None)
    assigned: list[list[str]] = [[] for _ in optimizers]
    untrained = []
    for name in names:
        parts = set(name.split("/"))
        index = next(
            (
                i
                for i, optimizer in enumerate(optimizers)
                if optimizer.modules and parts.intersection(optimizer.modules)
            ),
            rest,
        )
        if index is None:
            untrained.append(name)
        else:
            assigned[index].append(name)

    if untrained:
        shown = ", ".join(untrained[:3]) + (f" and {len(untrained) - 3} more" if len(untrained) > 3 else "")
        raise UnsupportedTrainingConfig(
            f"optimizer: no optimizer trains {shown}; list their modules, or leave one optimizer's modules unset to "
            "train the remaining weights"
        )
    for index, (optimizer, parameters) in enumerate(zip(optimizers, assigned)):
        if not parameters:
            if optimizer.modules:
                problem = f"modules {optimizer.modules} match no trainable weight"
            else:
                problem = "the other optimizers train every weight, so this one, without modules, trains none"
            raise UnsupportedTrainingConfig(f"optimizer[{index}]: {problem}")
    return assigned


class CombinedOptimizer:
    """The optimizers of a run, each over its own weights, stepped and zeroed together.

    ``get_group_lrs`` lists every optimizer's groups in order, so its first is the first optimizer's. The state dict
    nests each optimizer's under its index, as ``ttml.checkpointing`` saves and loads composite optimizers.
    """

    def __init__(self, optimizers: Sequence[Any]) -> None:
        self.optimizers = list(optimizers)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def get_group_lrs(self) -> list[float]:
        return [lr for optimizer in self.optimizers for lr in optimizer.get_group_lrs()]

    def get_state_dict(self) -> dict:
        return {str(index): optimizer.get_state_dict() for index, optimizer in enumerate(self.optimizers)}

    def set_state_dict(self, state: dict) -> None:
        for index, optimizer in enumerate(self.optimizers):
            optimizer.set_state_dict(state[str(index)])
