# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint save/load + discovery helpers for the training entry point."""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Callable

import ml_dtypes

import ttnn
import ttml
from ttml.common.data import CharTokenizer
from ttml.trainers import SFTTrainer

from model_builders import Model, ModelConfig, create_model


def _serialize_tensor(tensor: "ttml.autograd.Tensor") -> dict:
    """Serialize one autograd Tensor to a pickle-friendly dict: float32 array + layout + dtype."""
    val = tensor.get_value()
    if val.dtype == ttnn.DataType.FLOAT32:
        dtype = "FLOAT32"
    elif val.dtype == ttnn.DataType.BFLOAT16:
        dtype = "BFLOAT16"
    else:
        raise ValueError(f"checkpointing: unsupported tensor dtype {val.dtype} (only FLOAT32 and BFLOAT16)")
    return {
        "data": tensor.to_numpy(ttnn.DataType.FLOAT32),
        "layout": "ROW_MAJOR" if val.get_layout() == ttnn.Layout.ROW_MAJOR else "TILE",
        "dtype": dtype,
    }


def _deserialize_tensor(serialized: dict) -> "ttml.autograd.Tensor":
    """Rebuild a Tensor from a `_serialize_tensor` dict."""
    layout = ttnn.Layout.ROW_MAJOR if serialized["layout"] == "ROW_MAJOR" else ttnn.Layout.TILE
    if serialized["dtype"] == "FLOAT32":
        arr, new_type = serialized["data"], ttnn.DataType.FLOAT32
    else:
        arr, new_type = serialized["data"].astype(ml_dtypes.bfloat16), ttnn.DataType.BFLOAT16
    return ttml.autograd.Tensor.from_numpy(arr, layout=layout, new_type=new_type)


def _serialize_params(model: Model) -> dict:
    """Serialize a model's parameters into a pickle-friendly dict: float32 array + layout + dtype per param."""
    return {name: _serialize_tensor(param.tensor) for name, param in model.parameters().items()}


def _deserialize_params(model: Model, serialized: dict) -> None:
    """Restore a `_serialize_params` dict into a model's parameters in place."""
    params = model.parameters()
    restored = set()
    for name, entry in serialized.items():
        if name not in params:
            continue
        params[name].assign(_deserialize_tensor(entry))
        restored.add(name)

    missing = set(params) - restored  # in model, not restored → left at init (dangerous)
    unexpected = set(serialized) - set(params)  # in checkpoint, not in model → ignored
    if missing or unexpected:
        print(
            f"  [warn] checkpoint restore mismatch: "
            f"{len(missing)} model param(s) left at init "
            f"({sorted(missing)[:3]}{'...' if len(missing) > 3 else ''}), "
            f"{len(unexpected)} checkpoint param(s) ignored"
        )


def _serialize_optimizer(optimizer: "ttml.optimizers.OptimizerBase") -> dict:
    """Serialize an optimizer's state_dict: scalars, moment maps, and any nested composite sub-states."""

    def encode(value):  # one state_dict value: scalar, NamedParameters, or nested sub-dict
        if isinstance(value, ttml.NamedParameters):
            return {"named_parameters": {name: _serialize_tensor(tensor) for name, tensor in value.items()}}
        if isinstance(value, dict):
            return {key: encode(v) for key, v in value.items()}
        if isinstance(value, (bool, int, float)):
            return value
        raise ValueError(f"checkpointing: unsupported optimizer state of type {type(value).__name__}")

    return {key: encode(value) for key, value in optimizer.get_state_dict().items()}


def _deserialize_optimizer(optimizer: "ttml.optimizers.OptimizerBase", serialized: dict) -> None:
    """Restore a `_serialize_optimizer` blob into an optimizer in place via set_state_dict."""

    def decode(node):
        if not isinstance(node, dict):
            return node  # scalar
        if "named_parameters" in node:
            named = ttml.NamedParameters()
            for name, entry in node["named_parameters"].items():
                named[name] = _deserialize_tensor(entry)
            return named
        return {key: decode(v) for key, v in node.items()}  # nested sub-state

    optimizer.set_state_dict({key: decode(value) for key, value in serialized.items()})


def build_checkpoint_io(
    tokenizer: CharTokenizer | None,
    model_cfg: ModelConfig,
) -> tuple[Callable[[SFTTrainer, str], None], Callable[[SFTTrainer, str], int]]:
    """Return `(saver, loader)` closures. Saver writes step+params+optimizer+tokenizer+model_config; loader restores params+optimizer and returns step."""

    def saver(trainer: SFTTrainer, path: str) -> None:
        if not path.endswith(".pkl"):
            path = path + ".pkl"
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(
                {
                    "step": trainer.step,
                    "model_state": _serialize_params(trainer.model),
                    "optimizer_state": _serialize_optimizer(trainer.optimizer),
                    "tokenizer": tokenizer,
                    "model_config": model_cfg,
                },
                f,
            )
        os.replace(tmp_path, path)  # atomic rename: a crash mid-write leaves the previous checkpoint intact
        print(f"  Saved checkpoint to {path}")

    def loader(trainer: SFTTrainer, path: str) -> int:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        _deserialize_params(trainer.model, ckpt["model_state"])
        _deserialize_optimizer(trainer.optimizer, ckpt["optimizer_state"])
        return int(ckpt["step"])

    return saver, loader


def prefixed(prefix: str, suffix: str) -> str:
    """Join with `_` when prefix is non-empty; bare `suffix` otherwise. Matches SFTTrainer's convention."""
    return f"{prefix}_{suffix}" if prefix else suffix


def find_latest_checkpoint(checkpoint_dir: str, checkpoint_prefix: str = "") -> str | None:
    """Highest-step `<prefix>_step_*.pkl` (or `step_*.pkl` if prefix empty) under `checkpoint_dir`. `<prefix>_final.pkl` wins ties."""
    directory = Path(checkpoint_dir)
    step_glob = prefixed(checkpoint_prefix, "step_*.pkl")
    final_name = prefixed(checkpoint_prefix, "final.pkl")
    files = list(directory.glob(step_glob))
    final_path = directory / final_name
    if final_path.exists():
        files.append(final_path)
    if not files:
        return None

    def step_of(p: Path) -> float:
        if p.name == final_name:
            return float("inf")
        m = re.search(r"step_(\d+)\.pkl$", p.name)
        return int(m.group(1)) if m else -1

    return str(max(files, key=step_of))


def peek_checkpoint(path: str) -> tuple[int, ModelConfig]:
    """Read `(step, model_config)` from a checkpoint without restoring params or creating a model."""
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return int(ckpt["step"]), ckpt["model_config"]


def load_for_inference(
    path: str,
) -> tuple[Model, CharTokenizer, ModelConfig, int]:
    """Restore `(model, tokenizer, model_cfg, step)` from a checkpoint. Used to seed inference mode."""
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    model_cfg: ModelConfig = ckpt["model_config"]
    tokenizer: CharTokenizer = ckpt["tokenizer"]
    step = int(ckpt["step"])

    model = create_model(model_cfg, use_tp=False)
    _deserialize_params(model, ckpt["model_state"])
    return model, tokenizer, model_cfg, step
