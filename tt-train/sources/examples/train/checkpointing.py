# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint save/load + discovery for the training entry point."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Callable

import ttml
from ttml.checkpointing import load_checkpoint, read_header, save_checkpoint
from ttml.common.data import CharTokenizer
from ttml.trainers import SFTTrainer

from model_builders import Model, ModelConfig, create_model


def _capture_rng_state() -> dict:
    return {"cpp": ttml.autograd.AutoContext.get_instance().get_generator_state()}


def _restore_rng_state(state: dict) -> None:
    ttml.autograd.AutoContext.get_instance().set_generator_state(state["cpp"])


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if round(size, 1) < 1024.0:  # compare the rounded value so we never display "1024.0 KB"
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _human_time(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def _show_progress() -> bool:
    """Show the tqdm bar only when stderr (where tqdm draws) is a TTY. Under a non-interactive stderr —
    CI, nohup, a log aggregator — its redraws land as one line per update, so the bar is disabled there."""
    return sys.stderr.isatty()


def build_checkpoint_io(
    tokenizer: CharTokenizer | None,
    model_cfg: ModelConfig,
) -> tuple[Callable[[SFTTrainer, str], None], Callable[[SFTTrainer, str], int]]:
    """Return `(saver, loader)` closures. The header carries step + tokenizer + model_config; the model and
    optimizer ride along as named tensor groups that `ttml.checkpointing` streams."""

    def saver(trainer: SFTTrainer, path: str) -> None:
        if not path.endswith(".pkl"):
            path = path + ".pkl"
        print(f"  Saving checkpoint to {path} (gathering to host, may take a while)...", flush=True)
        start = time.perf_counter()
        save_checkpoint(
            path,
            header={
                "step": trainer.step,
                "tokenizer": tokenizer,
                "model_config": model_cfg,
                "rng": _capture_rng_state(),
                "dataloader": trainer.train_dataloader.get_state_dict(),
            },
            model_params=trainer.model.parameters(),
            optimizer=trainer.optimizer,
            display_progress=_show_progress(),
        )
        elapsed = time.perf_counter() - start
        print(f"  Saved checkpoint to {path} ({_human_size(Path(path).stat().st_size)} in {_human_time(elapsed)})")

    def loader(trainer: SFTTrainer, path: str) -> int:
        print(f"  Loading checkpoint from {path} ...", flush=True)
        start = time.perf_counter()
        header = load_checkpoint(
            path,
            model_params=trainer.model.parameters(),
            optimizer=trainer.optimizer,
            display_progress=_show_progress(),
        )
        _restore_rng_state(header["rng"])
        trainer.train_dataloader.set_state_dict(header["dataloader"])
        elapsed = time.perf_counter() - start
        print(f"  Loaded checkpoint from {path} (in {_human_time(elapsed)})")
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
    model = create_model(model_cfg)  # inference: no tensor parallelism (TPStrategy.NONE)
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
