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
from ttml import Sharding
from ttml.common.data import CharTokenizer
from ttml.trainers import SFTTrainer

from model_builders import Model, ModelConfig, create_model


def _tensor_meta(tensor: ttml.autograd.Tensor) -> dict:
    """Per-tensor header metadata (layout + dtype); the gathered data is streamed as a separate record."""
    val = tensor.get_value()
    if val.dtype == ttnn.DataType.FLOAT32:
        dtype = "FLOAT32"
    elif val.dtype == ttnn.DataType.BFLOAT16:
        dtype = "BFLOAT16"
    else:
        raise ValueError(f"checkpointing: unsupported tensor dtype {val.dtype} (only FLOAT32 and BFLOAT16)")
    return {"layout": "ROW_MAJOR" if val.get_layout() == ttnn.Layout.ROW_MAJOR else "TILE", "dtype": dtype}


def _tensor_from_record(meta: dict, data, mapper) -> ttml.autograd.Tensor:
    """Rebuild a Tensor from header `meta` + its streamed `data`, distributed onto the mesh via `mapper`
    (None for a fully-replicated tensor)."""
    layout = ttnn.Layout.ROW_MAJOR if meta["layout"] == "ROW_MAJOR" else ttnn.Layout.TILE
    if meta["dtype"] == "FLOAT32":
        arr, new_type = data, ttnn.DataType.FLOAT32
    else:
        arr, new_type = data.astype(ml_dtypes.bfloat16), ttnn.DataType.BFLOAT16
    return ttml.autograd.Tensor.from_numpy(arr, layout=layout, new_type=new_type, mapper=mapper)


def _walk_params(model: Model) -> tuple:
    """Model params -> (header skeleton `{name: meta}`, ordered tensor list) for streaming."""
    skeleton = {}
    tensors = []
    for name, param in model.parameters().items():
        skeleton[name] = _tensor_meta(param.tensor)
        tensors.append(param.tensor)
    return skeleton, tensors


def _walk_state(value) -> tuple:
    """Optimizer state_dict value -> (header skeleton, ordered tensor list).

    The skeleton mirrors the structure with `_tensor_meta` placeholders at each NamedParameters leaf;
    tensors are collected in the same DFS order the loader reads them back.
    """
    if isinstance(value, ttml.NamedParameters):
        skeleton = {"named_parameters": {}}
        tensors = []
        for name, tensor in value.items():
            skeleton["named_parameters"][name] = _tensor_meta(tensor)
            tensors.append(tensor)
        return skeleton, tensors
    if isinstance(value, dict):  # composite optimizers (e.g. MuonWithAdamW) nest sub-state dicts
        skeleton = {}
        tensors = []
        for key, sub in value.items():
            skeleton[key], sub_tensors = _walk_state(sub)
            tensors.extend(sub_tensors)
        return skeleton, tensors
    if isinstance(value, (bool, int, float)):
        return value, []
    raise ValueError(f"checkpointing: unsupported optimizer state of type {type(value).__name__}")


def _load_params(model: Model, skeleton: dict, f) -> None:
    """Stream model params back from `f` — one record per `skeleton` entry, in order."""
    params = model.parameters()
    restored = set()
    for name, meta in skeleton.items():
        data = pickle.load(f)  # read every record to keep the stream aligned, even when skipping
        if name not in params:
            continue
        mapper = Sharding.from_tensor(params[name]).derive_mapper()
        params[name].assign(_tensor_from_record(meta, data, mapper))
        restored.add(name)

    missing = set(params) - restored  # in model, not restored → left at init (dangerous)
    unexpected = set(skeleton) - set(params)  # in checkpoint, not in model → ignored
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint restore mismatch: "
            f"{len(missing)} model param(s) left at init "
            f"({sorted(missing)[:3]}{'...' if len(missing) > 3 else ''}), "
            f"{len(unexpected)} checkpoint param(s) ignored"
        )


def _load_optimizer(optimizer: ttml.optimizers.OptimizerBase, skeleton: dict, f) -> None:
    """Stream optimizer state back from `f`, resharding each moment per the live state dict, then set it."""
    live = optimizer.get_state_dict()

    def rebuild(node, live_node):
        if not isinstance(node, dict):
            return node  # scalar
        if "named_parameters" in node:
            named = ttml.NamedParameters()
            for name, meta in node["named_parameters"].items():
                data = pickle.load(f)
                mapper = Sharding.from_tensor(live_node[name]).derive_mapper()
                named[name] = _tensor_from_record(meta, data, mapper)
            return named
        return {key: rebuild(v, live_node[key]) for key, v in node.items()}  # nested sub-state

    optimizer.set_state_dict({key: rebuild(v, live[key]) for key, v in skeleton.items()})


def build_checkpoint_io(
    tokenizer: CharTokenizer | None,
    model_cfg: ModelConfig,
) -> tuple[Callable[[SFTTrainer, str], None], Callable[[SFTTrainer, str], int]]:
    """Return `(saver, loader)` closures. The checkpoint is a header record (step + tokenizer +
    model_config + a per-tensor layout/dtype skeleton) followed by one gathered-array record per tensor,
    so peak host memory stays ~one tensor instead of the whole model + optimizer."""

    def saver(trainer: SFTTrainer, path: str) -> None:
        if not path.endswith(".pkl"):
            path = path + ".pkl"
        model_skeleton, model_tensors = _walk_params(trainer.model)
        optimizer_skeleton, optimizer_tensors = _walk_state(trainer.optimizer.get_state_dict())
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(
                {
                    "step": trainer.step,
                    "tokenizer": tokenizer,
                    "model_config": model_cfg,
                    "model_state": model_skeleton,
                    "optimizer_state": optimizer_skeleton,
                },
                f,
            )
            for tensor in model_tensors + optimizer_tensors:  # gather one tensor at a time; freed after dump
                pickle.dump(Sharding.from_tensor(tensor).gather(tensor), f)
        os.replace(tmp_path, path)  # atomic rename: a crash mid-write leaves the previous checkpoint intact
        print(f"  Saved checkpoint to {path}")

    def loader(trainer: SFTTrainer, path: str) -> int:
        with open(path, "rb") as f:
            header = pickle.load(f)
            _load_params(trainer.model, header["model_state"], f)
            _load_optimizer(trainer.optimizer, header["optimizer_state"], f)
        return int(header["step"])

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
    """Read `(step, model_config)` from the checkpoint header alone — no tensor data is read."""
    with open(path, "rb") as f:
        header = pickle.load(f)
    return int(header["step"]), header["model_config"]


def load_for_inference(
    path: str,
) -> tuple[Model, CharTokenizer, ModelConfig, int]:
    """Restore `(model, tokenizer, model_cfg, step)` from a checkpoint. Used to seed inference mode."""
    with open(path, "rb") as f:
        header = pickle.load(f)
        model_cfg: ModelConfig = header["model_config"]
        tokenizer: CharTokenizer = header["tokenizer"]
        step = int(header["step"])
        model = create_model(model_cfg, use_tp=False)
        _load_params(model, header["model_state"], f)  # streams model params; optimizer records left unread
    return model, tokenizer, model_cfg, step
