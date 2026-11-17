# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The files that hold a model's weights in a checkpoint directory, in each format ``checkpoint.format`` picks
(``ModelProvider.export_weights``):

- ``hf``: a LoRA run's checkpoint is a PEFT adapter directory (``PeftModel.from_pretrained`` loads it), which holds
  its frozen base model in ``base_model/`` if every weight is saved; full fine-tuning's is an HF model directory.
- ``tt``: ``checkpoint.pkl``, every parameter as ``ttml.checkpointing`` writes it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
BASE_MODEL_DIR = "base_model"
MODEL_CONFIG = "config.json"
TT_CHECKPOINT = "checkpoint.pkl"

PathLike = Union[str, os.PathLike]


@dataclass
class AdapterState:
    """A LoRA adapter in PEFT's form: tensors by PEFT name, and the contents of ``adapter_config.json``."""

    tensors: dict[str, np.ndarray]
    config: dict[str, Any]


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


def is_adapter_dir(path: PathLike) -> bool:
    return (Path(path) / ADAPTER_CONFIG).is_file()


def is_model_dir(path: PathLike) -> bool:
    """Whether ``path`` is an HF model directory, as ``save_pretrained`` writes one."""
    path = Path(path)
    return (path / MODEL_CONFIG).is_file() and any(path.glob("*.safetensors"))


def is_tt_checkpoint(path: PathLike) -> bool:
    return (Path(path) / TT_CHECKPOINT).is_file()
