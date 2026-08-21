# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in GRPO logging callback.

:class:`GRPOMonitor` is the framework's default step logger for
:class:`~ttml.trainers.GRPOTrainer`. It writes a ``grpo_metrics.csv`` under the
config's ``output_dir``, prints a per-step log line, and (optionally) forwards
scalar metrics and a small completions table to Weights & Biases.

The trainer auto-appends a :class:`GRPOMonitor` to its callback list unless the
config sets ``disable_default_monitor=True``.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from typing import Any, Iterable

from .callback import TrainerCallback
from .grpo_trainer import GRPOConfig


try:
    import wandb as _wandb  # type: ignore
except ImportError:  # pragma: no cover - wandb is optional
    _wandb = None


# Fixed base column order in the CSV. Callback-timing columns are appended after
# these at ``on_train_begin`` (see :meth:`GRPOMonitor.on_train_begin`).
_BASE_COLUMNS = (
    "step",
    "reward_mean",
    "reward_std",
    "mean_completion_len",
    "min_completion_len",
    "max_completion_len",
    "lr",
    "step_time_s",
    "generation_time_s",
)


class GRPOMonitor(TrainerCallback):
    """CSV + console + optional wandb logger for GRPO training.

    Args:
        config: The :class:`GRPOConfig` in use. ``output_dir`` (where the CSV
            lands), ``report_to`` (``"none"`` or ``"wandb"``),
            ``log_completions`` and ``num_completions_to_print`` are read from
            it. The trainer wires this up automatically.
    """

    def __init__(self, config: GRPOConfig) -> None:
        self._output_dir = config.output_dir
        self._report_to = config.report_to
        self._log_completions = config.log_completions
        self._num_completions_to_print = config.num_completions_to_print
        # Trainer fires on_step_end every optimizer step (WeightSyncCallback
        # must not be logging-gated). The monitor self-gates so logging_steps
        # still controls CSV / console / wandb cadence.
        self._logging_steps = config.logging_steps

        self._csv_path = os.path.join(self._output_dir, "grpo_metrics.csv") if self._output_dir else None
        # Frozen list of CSV columns; populated in ``on_train_begin`` so we can
        # snapshot the callback class names alongside the base columns and keep
        # the header stable for the rest of the run.
        self._columns: list[str] = []
        # One-time-warning flags to avoid spamming logs on every step.
        self._warned_missing_wandb = False
        self._warned_unknown_columns: set[str] = set()

        # Resolve the wandb sink lazily. If the user asked for wandb but the
        # package is missing or ``wandb.init`` was never called, degrade to a
        # console-only sink and warn once.
        self._wandb_active = False

    # -- lifecycle -----------------------------------------------------------

    def on_train_begin(self, trainer: Any) -> None:
        # Snapshot the callback class names present at train-begin so the CSV
        # header lists a ``<ClassName>_time_s`` column per callback. Callbacks
        # added or removed mid-training won't grow the header.
        callback_time_columns = [f"{type(cb).__name__}_time_s" for cb in trainer.callbacks]
        self._columns = list(_BASE_COLUMNS) + callback_time_columns

        if self._csv_path is not None:
            os.makedirs(self._output_dir, exist_ok=True)
            with open(self._csv_path, mode="w", newline="") as f:
                csv.writer(f).writerow(self._columns)

        if self._report_to == "wandb":
            if _wandb is None:
                logging.warning(
                    "GRPOMonitor: report_to='wandb' but the 'wandb' package is not installed; "
                    "falling back to console + CSV logging only."
                )
                self._warned_missing_wandb = True
            elif getattr(_wandb, "run", None) is None:
                logging.warning(
                    "GRPOMonitor: report_to='wandb' but wandb.init() has not been called; "
                    "call wandb.init(...) before constructing the trainer to enable W&B logging."
                )
            else:
                self._wandb_active = True

    def on_step_end(self, trainer: Any, step: int, *args: Any, **kwargs: Any) -> None:
        if self._logging_steps <= 0 or step % self._logging_steps != 0:
            return
        # Read the mutable metrics dict the trainer exposes rather than kwargs
        # so callbacks earlier in the list can inject additional columns (e.g.
        # an EvalCallback writing ``trainer.metrics["eval_similarity"]``). The
        # trainer builds ``self.metrics`` immediately before firing callbacks
        # and populates the base entries from ``kwargs``; the two are the same
        # dict, but reading from ``trainer.metrics`` documents the contract.
        metrics: dict[str, Any] = getattr(trainer, "metrics", None) or dict(kwargs)

        self._log_console(step, metrics)
        self._write_csv_row(step, metrics)
        self._maybe_log_completions(step, metrics)
        self._maybe_log_wandb(step, metrics)

    def on_train_end(self, trainer: Any) -> None:
        logging.info("Training complete.")
        if self._wandb_active:
            _wandb.finish()

    # -- helpers -------------------------------------------------------------

    def _log_console(self, step: int, metrics: dict[str, Any]) -> None:
        nan = float("nan")
        # ``logging.basicConfig`` in the example scripts already prepends a
        # timestamp; keep the payload identical to the pre-refactor format so
        # existing log-parsing tooling doesn't break.
        logging.info(
            "Step %d | Reward: %.4f | Len: %.2f (min %d, max %d) tokens | Step: %.2fs | Gen: %.2fs",
            step,
            _as_float(metrics.get("reward_mean", nan)),
            _as_float(metrics.get("mean_completion_len", nan)),
            _as_int(metrics.get("min_completion_len", 0)),
            _as_int(metrics.get("max_completion_len", 0)),
            _as_float(metrics.get("step_time_s", nan)),
            _as_float(metrics.get("generation_time_s", nan)),
        )

    def _write_csv_row(self, step: int, metrics: dict[str, Any]) -> None:
        if self._csv_path is None:
            return

        row: list[Any] = []
        for column in self._columns:
            if column == "step":
                row.append(step)
                continue
            row.append(_format_cell(metrics.get(column, float("nan"))))

        # Warn once for any keys that would be dropped (columns that didn't
        # exist at ``on_train_begin``) so a callback adding a new metric
        # mid-run has an audit trail without churning the CSV schema.
        for key in metrics:
            if key in _NON_CSV_KEYS or key in self._columns or key in self._warned_unknown_columns:
                continue
            self._warned_unknown_columns.add(key)
            logging.warning(
                "GRPOMonitor: metric %r was not present at on_train_begin; "
                "it will not be written to %s (header is frozen).",
                key,
                self._csv_path,
            )

        with open(self._csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow(row)

    def _maybe_log_completions(self, step: int, metrics: dict[str, Any]) -> None:
        if not self._log_completions or self._num_completions_to_print <= 0:
            return
        completions: Iterable[str] = metrics.get("completions", []) or []
        prompts: Iterable[str] = metrics.get("prompts", []) or []
        rewards: Iterable[float] = metrics.get("rewards", []) or []
        # Zip stops at the shortest, so a short ``rewards`` list quietly limits
        # the number of rows we print — that's the correct behaviour when a
        # caller only forwarded a truncated sample.
        for i, (prompt, completion, reward) in enumerate(zip(prompts, completions, rewards)):
            if i >= self._num_completions_to_print:
                break
            logging.info(
                "[completion %d @ step %d] reward=%.4f\n  prompt=%r\n  completion=%r",
                i,
                step,
                _as_float(reward),
                _preview(prompt),
                _preview(completion),
            )

    def _maybe_log_wandb(self, step: int, metrics: dict[str, Any]) -> None:
        if not self._wandb_active:
            return
        # Only forward numeric scalars to wandb — string / list metrics (like
        # ``completions``) are handled separately as a ``wandb.Table`` below.
        scalar_payload: dict[str, Any] = {}
        for key, value in metrics.items():
            if key in _NON_CSV_KEYS:
                continue
            if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                scalar_payload[f"grpo/{key}"] = value
        if scalar_payload:
            _wandb.log(scalar_payload, step=step)

        if self._log_completions and self._num_completions_to_print > 0:
            prompts = list(metrics.get("prompts", []) or [])[: self._num_completions_to_print]
            completions = list(metrics.get("completions", []) or [])[: self._num_completions_to_print]
            rewards = list(metrics.get("rewards", []) or [])[: self._num_completions_to_print]
            if prompts and completions:
                table = _wandb.Table(columns=["step", "prompt", "completion", "reward"])
                for prompt, completion, reward in zip(prompts, completions, rewards):
                    table.add_data(step, _preview(prompt), _preview(completion), _as_float(reward))
                _wandb.log({"grpo/completions": table}, step=step)


# Metric keys that carry per-step sample payloads (not scalars) and should
# never be forwarded to the CSV or to wandb as scalars. Kept as a module-level
# constant so both the CSV writer and the wandb sink agree.
_NON_CSV_KEYS = frozenset({"completions", "prompts", "rewards"})


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_cell(value: Any) -> Any:
    # Preserve numeric types where possible; fall back to str() for anything
    # exotic so the CSV row never blows up on ``csv.writer``.
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def _preview(text: Any, max_len: int = 300) -> str:
    s = str(text).replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 1] + "\u2026"
