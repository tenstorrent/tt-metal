# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The files that hold a model's weights in a checkpoint directory, which providers write and read:

- a PEFT adapter directory (``PeftModel.from_pretrained`` loads it): ``adapter_config.json`` and
  ``adapter_model.safetensors``;
- ``checkpoint.pkl``: parameters as ``ttml.checkpointing`` writes them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ...errors import UnsupportedTrainingConfig

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
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
    config, weights = directory / ADAPTER_CONFIG, directory / ADAPTER_WEIGHTS
    if not config.is_file():
        raise FileNotFoundError(f"{directory} is not a PEFT adapter directory: no {ADAPTER_CONFIG}")
    if not weights.is_file():
        raise FileNotFoundError(f"{directory} has no {ADAPTER_WEIGHTS}")
    return AdapterState(tensors=load_file(str(weights)), config=json.loads(config.read_text()))


def check_base_model(directory: PathLike, base: Optional[str], name: str, source: str) -> None:
    """Raise ``UnsupportedTrainingConfig`` if the checkpoint at ``directory``, which records ``base`` as its base
    model, is for a model other than the one ``model.name`` (``name``) names, or than its ``model_source()``
    (``source``). A checkpoint that records no base model is taken as this model's."""
    if base and base not in (name, source):
        raise UnsupportedTrainingConfig(f"checkpoint: {directory} is for {base!r}, this model is {name!r}")
