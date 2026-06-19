# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint save/load + discovery for the training entry point."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from ttml.checkpointing import load_checkpoint, read_header, save_checkpoint
from ttml.common.data import CharTokenizer
from ttml.trainers import SFTTrainer

from model_builders import Model, ModelConfig, create_model


def build_checkpoint_io(
    tokenizer: CharTokenizer | None,
    model_cfg: ModelConfig,
) -> tuple[Callable[[SFTTrainer, str], None], Callable[[SFTTrainer, str], int]]:
    """Return `(saver, loader)` closures. The header carries step + tokenizer + model_config; the model and
    optimizer ride along as named tensor groups that `ttml.checkpointing` streams."""

    def saver(trainer: SFTTrainer, path: str) -> None:
        if not path.endswith(".pkl"):
            path = path + ".pkl"
        save_checkpoint(
            path,
            header={"step": trainer.step, "tokenizer": tokenizer, "model_config": model_cfg},
            model_params=trainer.model.parameters(),
            optimizer=trainer.optimizer,
        )
        print(f"  Saved checkpoint to {path}")

    def loader(trainer: SFTTrainer, path: str) -> int:
        header = load_checkpoint(path, model_params=trainer.model.parameters(), optimizer=trainer.optimizer)
        return int(header["step"])

    return saver, loader


def peek_checkpoint(path: str) -> tuple[int, ModelConfig]:
    """Read `(step, model_config)` from the checkpoint header alone — no tensor data is read."""
    header = read_header(path)
    return int(header["step"]), header["model_config"]


def load_for_inference(
    path: str,
) -> tuple[Model, CharTokenizer, ModelConfig, int]:
    """Restore `(model, tokenizer, model_cfg, step)` from a checkpoint. Used to seed inference mode."""
    header = read_header(path)
    model_cfg: ModelConfig = header["model_config"]
    model = create_model(model_cfg, use_tp=False)
    load_checkpoint(path, model_params=model.parameters())  # "optimizer" group present in the file is skipped
    return model, header["tokenizer"], model_cfg, int(header["step"])


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
