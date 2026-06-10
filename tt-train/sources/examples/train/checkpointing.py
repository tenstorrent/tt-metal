# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint save/load + discovery helpers for the training entry point."""

from __future__ import annotations

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


def _serialize_params(model: Model) -> dict:
    """Serialize parameters into a pickle-friendly dict: float32 arrays + layout metadata per param."""
    state = {}
    for name, param in model.parameters().items():
        layout = param.tensor.get_value().get_layout()
        arr = param.tensor.to_numpy(ttnn.DataType.FLOAT32)
        state[name] = {
            "data": arr,
            "layout": layout.value if hasattr(layout, "value") else str(layout),
            "shape": arr.shape,
        }
    return state


def _restore_params(params: dict, model_state: dict) -> None:
    """Restore params from a `_serialize_params` dict into a parameter map (cast back to bf16)."""
    restored = set()
    for name, item in model_state.items():
        if name not in params:
            continue
        arr, layout_str = (item["data"], item.get("layout", "TILE")) if isinstance(item, dict) else (item, "TILE")
        layout = ttnn.Layout.ROW_MAJOR if "ROW_MAJOR" in str(layout_str) else ttnn.Layout.TILE
        tensor = ttml.autograd.Tensor.from_numpy(
            arr.astype(ml_dtypes.bfloat16), layout=layout, new_type=ttnn.DataType.BFLOAT16
        )
        params[name].assign(tensor)
        restored.add(name)

    missing = set(params) - restored  # in model, not restored → left at init (dangerous)
    unexpected = set(model_state) - set(params)  # in checkpoint, not in model → ignored
    if missing or unexpected:
        print(
            f"  [warn] checkpoint restore mismatch: "
            f"{len(missing)} model param(s) left at init "
            f"({sorted(missing)[:3]}{'…' if len(missing) > 3 else ''}), "
            f"{len(unexpected)} checkpoint param(s) ignored"
        )


def build_checkpoint_io(
    tokenizer: CharTokenizer | None,
    model_cfg: ModelConfig,
) -> tuple[Callable[[SFTTrainer, str], None], Callable[[SFTTrainer, str], int]]:
    """Return `(saver, loader)` closures. Saver writes step+params+tokenizer+model_config; loader restores params and returns step."""

    def saver(trainer: SFTTrainer, path: str) -> None:
        if not path.endswith(".pkl"):
            path = path + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "step": trainer.step,
                    "model_state": _serialize_params(trainer.model),
                    "tokenizer": tokenizer,
                    "model_config": model_cfg,
                },
                f,
            )
        print(f"  Saved checkpoint to {path}")

    def loader(trainer: SFTTrainer, path: str) -> int:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        _restore_params(trainer.model.parameters(), ckpt["model_state"])
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
    _restore_params(model.parameters(), ckpt["model_state"])
    return model, tokenizer, model_cfg, step
