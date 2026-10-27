# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict
import csv
import gc
import inspect
import json
import logging
import math
import random
import sys
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, List, Optional, Tuple

import os
import numpy as np
import torch
import ttml
import ttnn
from ttml.common.schedulers import CosineAnnealingScheduler, LambdaScheduler, LinearScheduler, SequentialScheduler
from ttml import checkpointing
from ttml.common.utils import (
    create_optimizer,
    deallocate_tensors,
    no_grad,
    resolve_pad_token_id,
    round_up_to_tile,
    validate_clip_grad_norm,
)
from ttml.modules import RunMode

from ..callback import TrainerCallback
from .distributed import BatchLayout, exit_on_error, init_distributed_training, is_rollout_rank, resolve_batch_layout
from .grpo_ttml_model import CHECKPOINT_FILE, setup_ttml_model

if TYPE_CHECKING:
    from .rollout_batch_source import RolloutBatchSource, ScoredRolloutBatch

try:
    import wandb as _wandb  # type: ignore
except ImportError:  # pragma: no cover - wandb is optional
    _wandb = None


@dataclass
class RolloutBatch:
    """Describes multiple rollout samples (not a specific count).

    Fields:
        weight_version: Which theta version produced this batch. The trainer
            can use it to detect / down-weight stale samples.
        prompts: B ragged prompt token IDs.
        completions: B ragged completion token IDs (one per prompt in
            single-generation mode; per prompt-completion pair otherwise).
        logprobs: [B, max_completion_length] float32. Per-generated-token
            log pi_old(a_t | s_t): the full-vocab log_softmax of the raw logits,
            without temperature/top_k/top_p. Padding positions are don't-care; the
            trainer masks them out.
    """

    weight_version: int
    prompts: List[List[int]]
    completions: List[List[int]]
    logprobs: np.ndarray


class RolloutSampler(ABC):
    """Generates completions and their log-probs with updatable weights.

    weight_version starts at 0, or at -1 for a sampler booted without real weights; generate() then raises
    until the first update_weights. update_weights(weights, version) requires version > weight_version.
    """

    @abstractmethod
    def generate(self, prompts: List[List[int]]) -> RolloutBatch:
        """Generate completions for a batch of tokenised prompts and return them
        packaged (with per-token log pi_old and producer metadata) as a
        :class:`RolloutBatch`.
        """

    @property
    @abstractmethod
    def weight_version(self) -> int:
        ...

    @abstractmethod
    def update_weights(self, weights: Any, version: int) -> None:
        ...

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class RolloutBackend:
    sampler: str
    in_process: bool
    sync: bool
    weight_push_before_training: bool


ROLLOUT_BACKENDS = {
    "ttml_in_process_sync": RolloutBackend(
        sampler="ttml", in_process=True, sync=True, weight_push_before_training=False
    ),
    "ttt_remote_sync": RolloutBackend(sampler="ttt", in_process=False, sync=True, weight_push_before_training=True),
}


def build_rollout_sampler(
    source: str,
    *,
    max_completion_length: int,
    temperature: float,
    completions_per_prompt: int,
    model: Any = None,
    tokenizer: Any = None,
    batch_layout: Optional[BatchLayout] = None,
    model_source: Optional[str] = None,
    mesh_device: Any = None,
    remote_rollout: Optional[RemoteRolloutConfig] = None,
) -> RolloutSampler:
    """Build the :class:`RolloutSampler` named by ``source``.

    ``"ttml"`` builds a :class:`TTMLRolloutSampler` that generates with
    ``model`` and ``tokenizer`` (the trainer's own) on the already-open device.
    ``"ttt"`` builds a :class:`TTTRolloutSampler` for the Llama ``model_source``
    on the already-open ``mesh_device``, configured by ``remote_rollout``.
    """
    if source == "ttml":
        from .ttml_rollout_sampler import TTMLRolloutSampler

        return TTMLRolloutSampler(
            model, tokenizer, max_completion_length, temperature, completions_per_prompt, batch_layout
        )
    if source == "ttt":
        from .remote_rollout.llama_ttt_presets import bf16_attn_bfp8_mlp_optimizations, llama_stop_and_pad
        from .remote_rollout.ttt_rollout_sampler import TTTRolloutSampler

        if model_source is None or mesh_device is None or remote_rollout is None:
            raise ValueError("rollout sampler 'ttt' needs model_source, mesh_device and remote_rollout")
        stop_token_ids, pad_token_id = llama_stop_and_pad(model_source)
        return TTTRolloutSampler(
            mesh_device=mesh_device,
            model_source=model_source,
            max_batch_size=remote_rollout.max_batch_size,
            max_seq_len=remote_rollout.max_seq_len,
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_token_ids,
            pad_token_id=pad_token_id,
            completions_per_prompt=completions_per_prompt,
            max_completion_length=max_completion_length,
            temperature=temperature,
            top_k=remote_rollout.top_k,
            top_p=remote_rollout.top_p,
            dummy_weights=True,
        )
    raise ValueError(f"unknown rollout sampler {source!r}")


@dataclass
class RemoteRolloutConfig:
    """Rollout rank settings for rollout_backend="ttt_remote_sync"."""

    mesh_shape: List[int]
    max_batch_size: int
    max_seq_len: int
    top_k: int = 32
    top_p: float = 1.0


@dataclass
class GRPOConfig:
    epsilon: float
    # Number of completions resident on a single device within one micro-batch.
    # The across-mesh micro-batch size is per_device_train_batch_size *
    # num_devices, and the per micro-batch prompt count is derived from it (see
    # GRPOTrainer.train).
    per_device_train_batch_size: int
    num_iterations: int
    # Number of micro-batches per generation (effective) batch and per optimizer
    # step. The generation batch generates gradient_accumulation_steps *
    # per_device_train_batch_size * num_devices completions, then the trainer
    # accumulates gradients over micro-batches of size per_device_train_batch_size * num_devices
    # before each optimizer step. Larger values mean a larger effective batch per step.
    gradient_accumulation_steps: int
    # Metrics are accumulated every step and emitted every ``logging_steps``
    # steps as an interval-mean (except ``min_completion_len``,
    # ``max_completion_len`` and ``logprob_mismatch_max``, ``lr`` where minimum,
    # maximum, last value is emitted respectively).
    logging_steps: int
    output_dir: str
    checkpointing: bool
    checkpoint_interval: int
    prompts_to_train: int
    temperature: float
    max_completion_length: int
    num_generations: int
    warmup_steps: int
    # Which engine generates rollouts and how they reach the trainer (see ROLLOUT_BACKENDS above).
    # "ttml_in_process_sync": TTMLRolloutSampler with the trainer's policy model, same process.
    # "ttt_remote_sync": tt-transformers on rank 1 of a 2-rank tt-run job, on request.
    rollout_backend: str
    # LR schedule; values follow TRL's ``lr_scheduler_type``:
    #   "constant"             -> flat at base_lr; ``warmup_steps`` is ignored.
    #   "constant_with_warmup" -> linear warmup, then flat at base_lr.
    #   "cosine"               -> linear warmup, then half-cosine anneal to 0.
    #   "cosine_with_min_lr"   -> linear warmup, then half-cosine anneal to base_lr * ``min_lr_rate``.
    lr_scheduler_type: str = "constant_with_warmup"
    min_lr_rate: Optional[float] = None
    # Gradient clipping. Off by default (matches today).
    use_clip_grad_norm: bool = False
    clip_grad_norm_max_norm: float = 1.0
    log_completions: bool = False
    num_completions_to_print: int = 0
    report_to: str = "none"
    # Optional wandb run name, mirrored from TRL's ``TrainingArguments.run_name``.
    # ``None`` lets wandb auto-generate a name. Project / entity / mode
    # come from the ``WANDB_PROJECT`` / ``WANDB_ENTITY`` / ``WANDB_MODE`` env
    # vars, matching TRL + transformers conventions.
    run_name: Optional[str] = None
    # Escape hatch: when True the trainer does NOT auto-append a GRPOMonitor
    # callback (users can still add their own). Kept off by default so the
    # default experience prints step metrics + writes ``grpo_metrics.csv``.
    disable_default_monitor: bool = False
    # Deprecated/unused: the number of prompts per generation batch is now
    # derived at runtime from per_device_train_batch_size, num_devices, and
    # num_generations. Kept only so older configs that still set it construct
    # without error; the trainer ignores any value provided here.
    batch_size: Optional[int] = None
    # ttt_remote_sync: push the policy weights to the rollout rank at a generation-batch boundary
    # once N optimizer steps have passed since the last push.
    weight_sync_every: int = 1
    # ttt_remote_sync: rollout rank settings, from the top-level remote_rollout_config yaml section.
    remote_rollout: Optional[RemoteRolloutConfig] = None

    def __post_init__(self) -> None:
        # num_generations is the GRPO group size: each prompt must produce at least
        # one completion to form a group (advantages are computed within a group,
        # and the loss normalizes by the completion count). A value <= 0 yields
        # empty batches and divide-by-zero downstream, so fail fast here.
        if self.num_generations <= 0:
            raise ValueError(
                f"grpo_config: 'num_generations' must be > 0 (got {self.num_generations}); "
                "GRPO needs at least one completion per prompt."
            )

        # Other count fields that must be strictly positive: a value <= 0 produces
        # empty batches / divide-by-zero in batch sizing, loss normalization, or
        # the dataset loop. Fail fast at config-construction time.
        for _name, _val in (
            ("per_device_train_batch_size", self.per_device_train_batch_size),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("prompts_to_train", self.prompts_to_train),
        ):
            if _val <= 0:
                raise ValueError(f"grpo_config: '{_name}' must be > 0 (got {_val}).")

        # checkpoint_interval is only consulted when checkpointing is enabled, where
        # it drives ``num_steps % checkpoint_interval`` -- a value <= 0 would be a
        # modulo-by-zero. (When checkpointing is off the value is unused, so don't
        # constrain it.)
        if self.checkpointing and self.checkpoint_interval <= 0:
            raise ValueError(
                f"grpo_config: 'checkpoint_interval' must be > 0 when checkpointing is enabled "
                f"(got {self.checkpoint_interval})."
            )

        if not isinstance(self.warmup_steps, int) or isinstance(self.warmup_steps, bool) or self.warmup_steps < 0:
            raise ValueError(f"grpo_config: 'warmup_steps' must be a non-negative int (got {self.warmup_steps!r}).")

        # LR scheduler name must be one of the TRL ``lr_scheduler_type`` values
        # GRPOTrainer supports.
        _allowed_lr_scheduler = {"constant", "constant_with_warmup", "cosine", "cosine_with_min_lr"}
        if self.lr_scheduler_type not in _allowed_lr_scheduler:
            raise ValueError(
                f"grpo_config: 'lr_scheduler_type' must be one of {sorted(_allowed_lr_scheduler)} "
                f"(got {self.lr_scheduler_type!r})."
            )
        # ``min_lr_rate`` is a fraction of base_lr (TRL semantics). Only
        # consulted by "cosine_with_min_lr"; still validate the range so a
        # misconfigured yaml fails at construction rather than silently.
        if self.lr_scheduler_type == "cosine_with_min_lr" and self.min_lr_rate is None:
            raise ValueError("grpo_config: 'min_lr_rate' is required when 'lr_scheduler_type' is 'cosine_with_min_lr'.")
        if self.min_lr_rate is not None and not (0.0 <= self.min_lr_rate <= 1.0):
            raise ValueError(f"grpo_config: 'min_lr_rate' must be in [0.0, 1.0] (got {self.min_lr_rate}).")
        # Grad clip norm is only consulted when clipping is enabled; guard the
        # positive-only invariant so a misconfigured yaml fails loudly.
        if self.use_clip_grad_norm and self.clip_grad_norm_max_norm <= 0.0:
            raise ValueError(
                f"grpo_config: 'clip_grad_norm_max_norm' must be > 0 when 'use_clip_grad_norm' is True "
                f"(got {self.clip_grad_norm_max_norm})."
            )

        if not isinstance(self.rollout_backend, str):
            raise TypeError(
                f"grpo_config: 'rollout_backend' must be a str, got {type(self.rollout_backend).__name__}. "
                f"Supported values: {list(ROLLOUT_BACKENDS)}."
            )
        if self.rollout_backend not in ROLLOUT_BACKENDS:
            raise ValueError(
                f"grpo_config: 'rollout_backend' must be one of {list(ROLLOUT_BACKENDS)} "
                f"(got {self.rollout_backend!r})."
            )
        if self.weight_sync_every < 1:
            raise ValueError(f"grpo_config: 'weight_sync_every' must be >= 1 (got {self.weight_sync_every}).")
        if self.backend.in_process and self.weight_sync_every != 1:
            raise ValueError(
                f"grpo_config: 'weight_sync_every' does not apply to rollout_backend={self.rollout_backend!r}, "
                f"which shares the trainer's model (got {self.weight_sync_every})."
            )
        if not self.backend.in_process and self.remote_rollout is None:
            raise ValueError(
                f"grpo_config: rollout_backend={self.rollout_backend!r} needs a top-level "
                "'remote_rollout_config' section."
            )

        # ``report_to`` is intentionally a plain string in this framework
        # (unlike TRL, which accepts a list). Reject lists/tuples early so a
        # copy-pasted TRL config surfaces the difference immediately.
        if not isinstance(self.report_to, str):
            raise TypeError(
                f"grpo_config: 'report_to' must be a str, got {type(self.report_to).__name__}. "
                "Supported values: 'none', 'wandb'."
            )
        _allowed_report_to = {"none", "wandb"}
        if self.report_to not in _allowed_report_to:
            raise ValueError(
                f"grpo_config: 'report_to' must be one of {sorted(_allowed_report_to)} " f"(got {self.report_to!r})."
            )

        if not isinstance(self.num_completions_to_print, int) or isinstance(self.num_completions_to_print, bool):
            raise TypeError(
                "grpo_config: 'num_completions_to_print' must be an int, "
                f"got {type(self.num_completions_to_print).__name__}."
            )
        if self.num_completions_to_print < 0:
            raise ValueError(
                f"grpo_config: 'num_completions_to_print' must be >= 0 (got {self.num_completions_to_print})."
            )
        if self.log_completions and self.num_completions_to_print == 0:
            logging.warning(
                "grpo_config: 'log_completions' is True but 'num_completions_to_print' is 0; "
                "no completions will be logged. Set 'num_completions_to_print' > 0 to enable."
            )

        # Warn (once per construction) when a deprecated field is explicitly set.
        # TODO: remove this field and warning once all configs have migrated.
        if self.batch_size is not None:
            logging.warning(
                "grpo_config: 'batch_size' is deprecated and ignored; the generation batch "
                "size is now derived from per_device_train_batch_size, num_devices, "
                "num_generations, and gradient_accumulation_steps. Remove it from your config."
            )

    @property
    def backend(self) -> RolloutBackend:
        return ROLLOUT_BACKENDS[self.rollout_backend]


def get_grpo_config(yaml_config: dict, output_dir: str = "") -> GRPOConfig:
    """Build a :class:`GRPOConfig` from a top-level YAML config dict.

    Looks for ``training_config.grpo_config`` in ``yaml_config`` and constructs
    a :class:`GRPOConfig` from it. ``output_dir`` defaults to an empty string so
    callers can fill it in once they have picked a run directory.
    """
    tc = yaml_config.get("training_config", {})
    grpo_section = tc.get("grpo_config")
    if grpo_section is None:
        raise ValueError("training_config must contain a 'grpo_config' section")
    fields = dict(grpo_section)
    fields.setdefault("output_dir", output_dir)

    # Backwards-compatibility shim for the transition period.
    # ``micro_batch_size`` was renamed to ``per_device_train_batch_size``. Accept
    # the old name so existing configs keep working, mapping its value onto the
    # new field. TODO: deprecated — remove this shim (and the warning) once all
    # configs have migrated to ``per_device_train_batch_size``.
    if "micro_batch_size" in fields:
        old_value = fields.pop("micro_batch_size")
        if "per_device_train_batch_size" in fields and fields["per_device_train_batch_size"] != old_value:
            raise ValueError(
                "grpo_config: both 'micro_batch_size' (deprecated) and 'per_device_train_batch_size' are set with different values; "
                "remove 'micro_batch_size' and keep only 'per_device_train_batch_size'."
            )
        logging.warning(
            "grpo_config: 'micro_batch_size' is deprecated and will be removed; "
            "use 'per_device_train_batch_size' instead."
        )
        fields.setdefault("per_device_train_batch_size", old_value)

    remote_section = yaml_config.get("remote_rollout_config")
    if remote_section is not None:
        fields.setdefault("remote_rollout", RemoteRolloutConfig(**remote_section))

    return GRPOConfig(**fields)


# Fixed base column order in the CSV. Callback-timing columns are appended after
# these; any extra scalar keys populated on ``trainer.metrics`` follow. The
# header line is materialised together with the first row on the first logging
# step (see :meth:`GRPOMonitor._write_csv_header`).
_CSV_BASE_COLUMNS = (
    "step",
    "reward_mean",
    "reward_std",
    "mean_completion_len",
    "min_completion_len",
    "max_completion_len",
    "lr",
    "step_time_s",
    "generation_time_s",
    "reward_time_s",
    "rollout_wait_s",
    "weight_sync_time_s",
    "rollout_weight_version",
    "logprob_mismatch_mean",
    "logprob_mismatch_max",
    "clip_frac",
)


# Metric keys that carry per-step sample payloads (not scalars) and should
# never be forwarded to the CSV or to wandb as scalars. Kept as a module-level
# constant so both the CSV writer and the wandb sink agree.
_NON_CSV_KEYS = frozenset({"completions", "prompts", "rewards"})


class _MetricStats:
    """Constant-memory running stats for a single metric key."""

    __slots__ = ("sum", "count", "min", "max", "last")

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.min = math.inf
        self.max = -math.inf
        self.last = float("nan")

    def push(self, value: float) -> None:
        self.sum += value
        self.count += 1
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value
        self.last = value


# Per-key aggregation policy for interval emission. Anything not listed defaults
# to ``"mean"`` (matching TRL, which averages every metric it logs).
_AGG_POLICY: dict[str, str] = {
    "min_completion_len": "min",
    "max_completion_len": "max",
    "lr": "last",
    "logprob_mismatch_max": "max",
}


def _aggregate(key: str, r: _MetricStats) -> float:
    policy = _AGG_POLICY.get(key, "mean")
    if policy == "min":
        return r.min
    if policy == "max":
        return r.max
    if policy == "last":
        return r.last
    return r.sum / max(r.count, 1)


class GRPOMonitor(TrainerCallback):
    """CSV + console + optional wandb logger for GRPO training.

    The trainer auto-appends a :class:`GRPOMonitor` to its callback list unless
    the config sets ``disable_default_monitor=True``. It writes a
    ``grpo_metrics.csv`` under the config's ``output_dir``, prints a per-step
    log line, and — when ``report_to == "wandb"`` — calls ``wandb.init(...)``
    itself (matching the TRL / ``transformers`` convention) and forwards scalar
    metrics + a small completions table.

    Args:
        config: The :class:`GRPOConfig` in use. ``output_dir`` (where the CSV
            lands), ``report_to`` (``"none"`` or ``"wandb"``), ``run_name``
            (wandb run name), ``log_completions`` and
            ``num_completions_to_print`` are read from it. The trainer wires
            this up automatically.
    """

    def __init__(self, config: GRPOConfig) -> None:
        self._config = config
        self._output_dir = config.output_dir
        self._report_to = config.report_to
        self._run_name = config.run_name
        self._log_completions = config.log_completions
        self._num_completions_to_print = config.num_completions_to_print

        self._csv_path = os.path.join(self._output_dir, "grpo_metrics.csv") if self._output_dir else None
        # Frozen list of CSV columns; populated on the first ``_write_csv_row``
        # call so that the header reflects the actual contents of ``trainer.metrics``
        # at the first logging step (including keys populated by user callbacks
        # in their ``on_step_end``). Empty means "header not written yet".
        self._columns: list[str] = []
        # Snapshot of the callback classes present at ``on_train_begin`` — used
        # to seed the ``<Callback>_time_s`` columns in the header. Captured then
        # rather than at write time so mutations to ``trainer.callbacks`` after
        # training starts do not silently churn the CSV schema.
        self._callback_time_columns: list[str] = []
        # One-time-warning flags to avoid spamming logs on every step.
        self._warned_missing_wandb = False
        self._warned_unknown_columns: set[str] = set()

        self._wandb_active = False
        # Whether this callback owns the wandb run (i.e. it called
        # ``wandb.init`` itself). If a caller already started a run before the
        # trainer was constructed, we log into it but don't finish it in
        # ``on_train_end`` — that stays the caller's responsibility.
        self._wandb_owned = False

        # Per-key running stats accumulated every step; flushed to CSV / wandb
        # (as interval-mean, or the policy in ``_AGG_POLICY``) every
        # ``logging_steps`` steps.
        self._running: dict[str, _MetricStats] = defaultdict(_MetricStats)

    # -- lifecycle -----------------------------------------------------------

    def on_train_begin(self, trainer: Any) -> None:
        # Skip GRPOMonitor: its cost is deliberately outside step_time_s and it
        # never writes a ``{Callback}_time_s`` entry, so a column would always be empty.
        self._callback_time_columns = [
            f"{type(cb).__name__}_time_s" for cb in trainer.callbacks if not isinstance(cb, GRPOMonitor)
        ]

        if self._csv_path is not None:
            os.makedirs(self._output_dir, exist_ok=True)

        if self._report_to == "wandb":
            self._start_wandb(trainer)

    def on_step_end(self, trainer: Any, step: int, *args: Any, **kwargs: Any) -> None:
        metrics: dict[str, Any] = getattr(trainer, "metrics", None) or dict(kwargs)
        self._accumulate(metrics)

        if not self._is_logging_step(step):
            return

        row: dict[str, Any] = {k: _aggregate(k, r) for k, r in self._running.items()}
        for key in _NON_CSV_KEYS:
            if key in metrics:
                row[key] = metrics[key]

        self._log_console(step, row)
        self._write_csv_row(step, row)
        self._maybe_log_completions(step, row)
        self._maybe_log_wandb(step, row)
        self._running.clear()

    def _accumulate(self, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            if key in _NON_CSV_KEYS:
                continue
            if isinstance(value, (bool, int, float)) and not (isinstance(value, float) and math.isnan(value)):
                self._running[key].push(float(value))

    def _is_logging_step(self, step: int) -> bool:
        ls = self._config.logging_steps
        return ls > 0 and step % ls == 0

    def on_train_end(self, trainer: Any) -> None:
        logging.info("Training complete.")
        if self._wandb_active and self._wandb_owned:
            _wandb.finish()

    # -- helpers -------------------------------------------------------------

    def _start_wandb(self, trainer: Any) -> None:
        if _wandb is None:
            logging.warning(
                "GRPOMonitor: report_to='wandb' but the 'wandb' package is not installed; "
                "falling back to console + CSV logging only."
            )
            self._warned_missing_wandb = True
            return

        if getattr(_wandb, "run", None) is not None:
            # A run was already opened by the caller — log into it but don't
            # own its lifecycle. This preserves the escape hatch for users who
            # want full control over ``wandb.init`` (custom tags, groups, etc.).
            self._wandb_active = True
            self._wandb_owned = False
            logging.info("GRPOMonitor: reusing existing wandb run (%s).", _wandb.run.name)
            return

        # Project / entity / mode follow the TRL + transformers convention of
        # coming from ``WANDB_*`` env vars; only ``run_name`` is a config field.
        _wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            name=self._run_name,
            entity=os.environ.get("WANDB_ENTITY"),
            mode=os.environ.get("WANDB_MODE"),
            config=_build_wandb_config(trainer),
        )
        self._wandb_active = True
        self._wandb_owned = True
        logging.info(
            "GRPOMonitor: wandb.init() called (project=%s, run=%s).",
            os.environ.get("WANDB_PROJECT") or "<default>",
            self._run_name or "<auto>",
        )

    def _log_console(self, step: int, metrics: dict[str, Any]) -> None:
        logs = {"step": step, **{k: v for k, v in metrics.items() if k not in _NON_CSV_KEYS}}
        logging.info(logs)

    def _write_csv_row(self, step: int, metrics: dict[str, Any]) -> None:
        if self._csv_path is None:
            return

        if not self._columns:
            self._write_csv_header(metrics)

        row_values = [step if col == "step" else _format_cell(metrics.get(col, float("nan"))) for col in self._columns]
        with open(self._csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow(row_values)

        self._warn_unknown_keys(metrics)

    def _write_csv_header(self, metrics: dict[str, Any]) -> None:
        """Derive the CSV column list from the first row's metrics and write the header.

        Called once, on the first logging step. Columns are the base scalars,
        then the ``<Callback>_time_s`` columns captured in ``on_train_begin``,
        then any other numeric keys already populated on ``trainer.metrics``.
        """
        extras = [
            k
            for k in metrics
            if k not in _NON_CSV_KEYS and k not in _CSV_BASE_COLUMNS and k not in self._callback_time_columns
        ]
        self._columns = list(_CSV_BASE_COLUMNS) + self._callback_time_columns + extras
        with open(self._csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(self._columns)

    def _warn_unknown_keys(self, metrics: dict[str, Any]) -> None:
        """Log a one-time warning for any metric key that appeared after the header froze.

        On the first row this is a no-op — ``_write_csv_header`` derives the
        columns from ``metrics`` so every key present is already in
        ``self._columns``.
        """
        for key in metrics:
            if key in _NON_CSV_KEYS or key in self._columns or key in self._warned_unknown_columns:
                continue
            self._warned_unknown_columns.add(key)
            logging.warning(
                "GRPOMonitor: metric %r first appeared after the CSV header was frozen; "
                "it will not be written to %s.",
                key,
                self._csv_path,
            )

    def _maybe_log_completions(self, step: int, metrics: dict[str, Any]) -> None:
        if not self._log_completions or self._num_completions_to_print <= 0:
            return
        completions: Iterable[str] = metrics.get("completions", []) or []
        prompts: Iterable[str] = metrics.get("prompts", []) or []
        rewards: Iterable[float] = metrics.get("rewards", []) or []
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
            if isinstance(value, (bool, int, float)) and not (isinstance(value, float) and math.isnan(value)):
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


def _build_wandb_config(trainer: Any) -> dict:
    """Assemble the ``config`` payload logged to wandb at run start.

    Includes the full :class:`GRPOConfig` (via ``asdict``), the model source,
    and the raw optimizer dict — enough to reproduce a run from the wandb
    settings tab.
    """
    payload: dict = {}
    cfg = getattr(trainer, "config", None)
    if cfg is not None:
        try:
            payload.update(asdict(cfg))
        except TypeError:
            pass
    model_source = getattr(trainer, "model_source", None)
    if model_source is not None:
        payload["model_source"] = model_source
    optimizer_dict = getattr(trainer, "optimizer_dict", None)
    if optimizer_dict:
        payload["optimizer"] = dict(optimizer_dict)
    return payload


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_cell(value: Any) -> Any:
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def _preview(text: Any, max_len: int = 300) -> str:
    s = str(text).replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 1] + "\u2026"


def dispatch_reward(
    reward_func: Callable[..., List[float]],
    completions: List[str],
    prompts: List[str],
    batch_columns: dict,
) -> List[float]:
    sig = inspect.signature(reward_func)
    params = sig.parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    data_pool = {"completions": completions, "prompts": prompts, **batch_columns}

    if has_kwargs:
        return reward_func(**data_pool)

    call_kwargs = {name: data_pool[name] for name in params if name in data_pool}
    return reward_func(**call_kwargs)


def _derive_reward_names(funcs: List[Callable]) -> List[str]:
    """Derive a unique, human-readable name per reward function.

    Uses ``fn.__name__`` when available; falls back to ``reward_{i}`` for
    lambdas and other unnamed callables. Duplicate names are disambiguated
    with a ``_2`` / ``_3`` suffix so per-component metric keys stay unique.
    """
    names: List[str] = []
    seen: dict[str, int] = {}
    for i, fn in enumerate(funcs):
        raw = getattr(fn, "__name__", None) or f"reward_{i}"
        if raw == "<lambda>":
            raw = f"reward_{i}"
        if raw in seen:
            seen[raw] += 1
            names.append(f"{raw}_{seen[raw]}")
        else:
            seen[raw] = 1
            names.append(raw)
    return names


def compute_advantages_host(rewards_np: np.ndarray, group_size: int) -> np.ndarray:
    """Compute group-relative advantages on the host, kept in host order.

    ``rewards_np`` has shape ``[B]`` with ``B = num_groups * group_size`` and
    contiguous groups of length ``group_size`` (all completions of prompt 0,
    then prompt 1, ...). Returns an array of the same shape and order where
    each element has had its group (per-prompt) mean subtracted.

    Doing the group reduction on the host means the advantages never need to be
    co-located by group on a device, so groups are free to straddle devices.
    The advantages are deliberately returned in host order (NOT regrouped per
    device) so that each micro-batch slice can later be sharded along axis 0 in
    the exact same group-agnostic, host-order way as the :func:`layout_microbatch`
    token tensors that feed the trainer's policy forward. That alignment is what
    keeps every completion paired with its own advantage on every device; see
    :func:`upload_micro_advantages`.
    """
    B = rewards_np.shape[0]
    assert B % group_size == 0, "rewards length must be divisible by group_size"
    grouped = rewards_np.reshape(-1, group_size).astype(np.float32)
    advantages = grouped - grouped.mean(axis=1, keepdims=True)
    return advantages.reshape(B)


def upload_micro_advantages(adv_np: np.ndarray, mapper: Any, num_devices: int) -> Any:
    """Upload one micro-batch's advantages, sharded to match the policy log-probs.

    ``adv_np`` is the host-order advantage slice for a single micro-batch (shape
    ``[mb]``, where ``mb`` is the micro-batch size). It is sharded along axis 0
    across the mesh, so device ``d`` receives host rows
    ``[d * mb_local : (d + 1) * mb_local]`` — the SAME contiguous,
    group-agnostic split the trainer applies to the :func:`layout_microbatch`
    ``[mb, Tp]`` tensors for the very same micro-batch. Because both
    tensors are sharded the same way over the same host-order list, device-local
    row ``r`` of the advantages corresponds to device-local row ``r`` of the
    log-probs, i.e. the same completion.

    Returns a ``ttnn.Tensor`` of global shape ``[mb, 1]`` (per device
    ``[mb_local, 1]``), ready to broadcast-multiply the per-completion loss.
    """
    mb = adv_np.shape[0]
    assert mb % num_devices == 0, f"micro-batch size ({mb}) must be divisible by num_devices ({num_devices})"
    mb_local = mb // num_devices
    adv_4d = adv_np.reshape(mb, 1, 1, 1).astype(np.float32)
    adv_ttml = ttml.autograd.Tensor.from_numpy(adv_4d, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    adv_rm = ttnn.to_layout(adv_ttml.get_value(), ttnn.Layout.ROW_MAJOR)
    return ttnn.reshape(adv_rm, [mb_local, 1])


def iter_micro_batch(
    prompts: List[List[int]],
    completions: List[List[int]],
    micro_batch_size: int = 16,
) -> Iterator[Tuple[List[List[int]], List[List[int]]]]:
    for start in range(0, len(completions), micro_batch_size):
        end = min(start + micro_batch_size, len(completions))

        yield prompts[start:end], completions[start:end]


def _build_lr_scheduler(optimizer: Any, cfg: GRPOConfig, total_steps: int) -> Optional[Any]:
    """LR scheduler for ``cfg.lr_scheduler_type``, or ``None`` for a constant LR.

    Warmup trains 0-indexed step ``i`` at ``base_lr * (i + 1) / warmup_steps``; the cosine types then
    decay to ``base_lr * min_lr_rate`` (0 for ``"cosine"``) at ``total_steps``.
    """
    warmup_steps = cfg.warmup_steps
    # Even a no-op scheduler raises for optimizers with supports_lr_scheduling=False.
    if cfg.lr_scheduler_type == "constant" or (cfg.lr_scheduler_type == "constant_with_warmup" and warmup_steps <= 1):
        return None
    if not getattr(optimizer, "supports_lr_scheduling", True):
        raise ValueError(
            f"{type(optimizer).__name__} does not support LR schedules; use lr_scheduler_type='constant', "
            f"or 'constant_with_warmup' with warmup_steps <= 1 (got lr_scheduler_type="
            f"{cfg.lr_scheduler_type!r}, warmup_steps={warmup_steps})."
        )
    warmup = LinearScheduler(optimizer, 1.0 / warmup_steps, 1.0, max(1, warmup_steps - 1)) if warmup_steps > 0 else None
    decay_steps = total_steps - warmup_steps
    if cfg.lr_scheduler_type == "constant_with_warmup" or decay_steps <= 0:
        return warmup
    if cfg.lr_scheduler_type == "cosine_with_min_lr":
        min_lr_rate = float(cfg.min_lr_rate)

        # Keep the floor relative to each group's own base LR (HF semantics).
        def _cosine_with_floor(step: int) -> float:
            progress = min(step / decay_steps, 1.0)
            return min_lr_rate + (1.0 - min_lr_rate) * 0.5 * (1.0 + math.cos(math.pi * progress))

        decay = LambdaScheduler(optimizer, _cosine_with_floor)
    else:
        decay = CosineAnnealingScheduler(optimizer, decay_steps, 0.0)
    if warmup is None:
        return decay
    return SequentialScheduler(optimizer, [warmup, decay], [warmup_steps, decay_steps])


def layout_microbatch(
    prompts: List[List[int]],
    completions: List[List[int]],
    pad_token: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Right-padded next-token layout for one micro-batch.

    Row ``i`` holds ``seq = prompts[i] + completions[i]`` shifted for next-token
    prediction: ``inputs[i, :L] = seq[:-1]`` and ``targets[i, :L] = seq[1:]``
    with ``L = len(seq) - 1``; pad tokens fill ``[L, Tp)``. Completion token
    ``j`` is therefore predicted at column ``len(prompts[i]) - 1 + j``, which is
    where ``mask`` is 1.0. ``Tp`` is the tile-rounded max ``L`` over the
    micro-batch. Padding only on the right means a plain causal mask keeps real
    tokens from attending to pads.

    Returns ``(inputs [B, Tp] uint32, targets [B, Tp] uint32, mask [B, Tp] float32, Tp)``.
    """
    assert len(prompts) == len(completions)
    B = len(prompts)
    Tp = round_up_to_tile(max(len(p) + len(c) - 1 for p, c in zip(prompts, completions)))
    inputs = np.full((B, Tp), pad_token, dtype=np.uint32)
    targets = np.full((B, Tp), pad_token, dtype=np.uint32)
    mask = np.zeros((B, Tp), dtype=np.float32)
    for i, (p, c) in enumerate(zip(prompts, completions)):
        if len(p) < 2:
            raise ValueError("Prompt is too short")
        seq = list(p) + list(c)
        if len(seq) < 2:
            raise ValueError("Sequence is too short")
        L = len(seq) - 1
        inputs[i, :L] = np.asarray(seq[:-1], dtype=np.uint32)
        targets[i, :L] = np.asarray(seq[1:], dtype=np.uint32)
        mask[i, len(p) - 1 : len(p) - 1 + len(c)] = 1.0
    return inputs, targets, mask, Tp


def place_old_nlog_probs(
    prompts: List[List[int]],
    completions: List[List[int]],
    logprobs_rows: np.ndarray,
    Tp: int,
) -> np.ndarray:
    """Place the sampler's ``log pi_old`` rows into the :func:`layout_microbatch` layout.

    ``logprobs_rows`` is the ``[B, max_completion_length]`` slice of
    ``RolloutBatch.logprobs`` for this micro-batch, holding ``+log pi_old`` of
    completion token ``j`` at column ``j``. Returns ``[B, Tp]`` float32
    NEGATIVE log-probs (the sign the GRPO loss uses) with completion token ``j``
    of row ``r`` at column ``len(prompts[r]) - 1 + j``; every other column is 0.
    """
    old = np.zeros((len(prompts), Tp), dtype=np.float32)
    for r, (p, c) in enumerate(zip(prompts, completions)):
        s = len(p) - 1
        old[r, s : s + len(c)] = -logprobs_rows[r, : len(c)]
    return old


def clipping_stats(
    old_np: np.ndarray, new_np: np.ndarray, mask_np: np.ndarray, adv_np: np.ndarray, eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Per completion token: ``|log pi - log pi_old|``, and whether the clipped term is the minimum (zero gradient).

    ``old_np`` / ``new_np`` are ``[B, Tp]`` negative log-probs, ``mask_np`` the completion mask, ``adv_np`` ``[B]``.
    """
    log_ratio = old_np - new_np
    ratio = np.exp(log_ratio)
    adv = adv_np[:, None]
    clipped = ((ratio > 1.0 + eps) & (adv > 0)) | ((ratio < 1.0 - eps) & (adv < 0))
    tokens = mask_np > 0
    return np.abs(log_ratio[tokens]), clipped[tokens]


def save_checkpoint(
    model: Any,
    step: int,
    output_dir: str,
    tokenizer: Any = None,
    grpo_config: Optional[GRPOConfig] = None,
    optimizer: Any = None,
    model_source: Optional[str] = None,
    scheduler: Any = None,
    learning_rate: Optional[float] = None,
) -> None:
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"grpo_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    header = {
        "global_step": step,
        "learning_rate": learning_rate,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.random.get_rng_state()},
        "grpo_config": asdict(grpo_config) if grpo_config is not None else None,
    }
    checkpointing.save_checkpoint(
        os.path.join(ckpt_dir, CHECKPOINT_FILE), header=header, model_params=model.parameters(), optimizer=optimizer
    )

    if model_source:
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(model_source)
            hf_config.save_pretrained(ckpt_dir)
        except Exception as exc:
            logging.warning("Could not save HF config for %s: %s", model_source, exc)

    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)

    if grpo_config is not None:
        gen_config = {
            "temperature": grpo_config.temperature,
            "max_new_tokens": grpo_config.max_completion_length,
        }
        if tokenizer is not None:
            gen_config["eos_token_id"] = tokenizer.eos_token_id
            gen_config["pad_token_id"] = tokenizer.pad_token_id
        with open(os.path.join(ckpt_dir, "generation_config.json"), "w") as f:
            json.dump(gen_config, f, indent=2)

    with open(os.path.join(ckpt_dir, "timestamp.txt"), "w") as f:
        f.write(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC\n"))


def _validate_remote_inputs(config: GRPOConfig, transformer_config: Any, device_config: Any, model_source: str) -> None:
    if config.backend.sampler == "ttt" and transformer_config.model_type != "llama":
        raise ValueError(
            f"GRPOTrainer: rollout_backend={config.rollout_backend!r} supports Llama models only, "
            f"got model_type={transformer_config.model_type!r}."
        )
    if config.backend.in_process:
        return
    if getattr(device_config, "enable_fsdp", False):
        raise ValueError(
            f"GRPOTrainer: rollout_backend={config.rollout_backend!r} does not support FSDP: "
            "the weight push needs replicated parameters."
        )
    if os.path.exists(model_source):
        raise ValueError(
            f"GRPOTrainer: rollout_backend={config.rollout_backend!r} needs a HuggingFace Hub model id, "
            f"got local path {model_source!r}."
        )


class GRPOTrainer:
    def __init__(
        self,
        transformer_config: Any,
        device_config: Any,
        model_source: str,
        dataset_func: Callable[[], Any],
        config: GRPOConfig,
        reward_func: Optional[Callable[..., List[float]]] = None,
        optimizer_dict: Optional[dict] = None,
        callbacks: Optional[List[Any]] = None,
        reward_funcs: Optional[List[Callable[..., List[float]]]] = None,
    ) -> None:
        """
        Args:
            transformer_config: Model config; ``model_type`` selects the model
                family.
            device_config: Device mesh config the trainer opens.
            model_source: HuggingFace model ID or local checkpoint directory
                the trainer loads the model from, and whose HF config is saved
                with checkpoints.
            dataset_func: Zero-argument callable that returns the training
                dataset. The trainer calls it once, in the constructor.

        The trainer opens the device and builds the policy model and tokenizer
        (``self.model`` / ``self.tokenizer``) with
        :func:`~ttml.trainers.grpo_trainer.grpo_ttml_model.setup_ttml_model`.
        It also builds the optimizer, LR scheduler and the
        :class:`RolloutBatchSource` over the tokenized prompts
        (``self.rollout_batch_source``), so ``train()`` only runs the loop. With
        ``rollout_backend="ttml_in_process_sync"`` the source generates with ``self.rollout_sampler``,
        built from ``config.rollout_backend`` with the trainer's model and
        tokenizer. The trainer runs its own forward pass with gradients on the
        model and never changes its run mode.

        With ``rollout_backend="ttt_remote_sync"`` the same script runs on both tt-run ranks. Rank
        1 only opens the rollout mesh and builds its :class:`RolloutSampler`
        and RPC server; it never calls ``dataset_func``, and ``train()``
        serves rollouts there.
        """
        if optimizer_dict is None:
            raise ValueError("GRPOTrainer: 'optimizer_dict' is required.")
        if not callable(dataset_func):
            raise TypeError("GRPOTrainer: 'dataset_func' must be a zero-argument callable returning the dataset.")
        _validate_remote_inputs(config, transformer_config, device_config, model_source)

        self._init_rewards(reward_func, reward_funcs)

        # Constructor inputs (immutable during ``train``).
        self.config = config
        self.optimizer_dict = optimizer_dict
        self.callbacks: List[Any] = list(callbacks or [])
        self.model_source = model_source

        self.model: Any = None
        self.tokenizer: Any = None
        self.rollout_sampler: Optional[RolloutSampler] = None

        # Per-step accumulator, reset at the start of each rollout and of each
        # later mini-epoch. Callbacks can inject additional keys here
        # (e.g. an eval callback writing ``trainer.metrics["eval_similarity"]``)
        # and later callbacks (notably ``GRPOMonitor``) read the merged view.
        self.metrics: dict = {}
        self.global_step: int = 0

        # Transient timing state used to bracket ``step_time_s`` — written by
        # ``train()``, read by ``_publish_step_metrics``.
        self._step_start_time: float = 0.0
        self._weight_sync_time_s: float = 0.0

        # Resolved-at-setup state. Pre-declared with ``None`` sentinels so the
        # full trainer lifecycle is visible in one place; populated by
        # ``_setup()`` at the end of the constructor.
        self._train_called: bool = False
        self._pad_token: int = 0
        self._optimizer: Any = None
        self._lr_scheduler: Any = None
        self._monitor_cb: Optional[GRPOMonitor] = None
        self._batch_layout: Optional[BatchLayout] = None
        self._completions_per_microbatch: int = 0
        self._num_generation_batches: int = 0
        self.rollout_batch_source: Optional[RolloutBatchSource] = None
        self._rollout_client: Any = None
        self._rollout_mesh: Any = None
        self._rollout_server: Any = None

        init_distributed_training(config)
        if is_rollout_rank(config):
            # Rank 0 is blocked waiting for the handshake, terminating processes on a failure before handshake.
            with exit_on_error():
                self._init_remote_rollout()
        else:
            self._build_policy(transformer_config, device_config, dataset_func)

    def _build_policy(self, transformer_config: Any, device_config: Any, dataset_func: Callable[[], Any]) -> None:
        cfg = self.config

        # Auto-append the framework's default GRPOMonitor unless the config
        # opts out. Placed last so any user-supplied callbacks (e.g. eval)
        # get a chance to populate ``self.metrics`` before the monitor writes
        # a row.
        if not cfg.disable_default_monitor:
            # Detect the framework GRPOMonitor (or a subclass) by isinstance, and
            # a legacy local class also named ``GRPOMonitor`` by class name — pre-
            # refactor forks used to define their own monitor with that name.
            # Either way we skip auto-appending to avoid duplicate CSV writes.
            if not any(isinstance(cb, GRPOMonitor) or type(cb).__name__ == "GRPOMonitor" for cb in self.callbacks):
                self.callbacks.append(GRPOMonitor(cfg))

        if cfg.backend.in_process:
            self.model, self.tokenizer = setup_ttml_model(transformer_config, device_config, self.model_source)
            self._batch_layout = resolve_batch_layout()
            self.rollout_sampler = build_rollout_sampler(
                cfg.backend.sampler,
                model=self.model,
                tokenizer=self.tokenizer,
                batch_layout=self._batch_layout,
                max_completion_length=cfg.max_completion_length,
                temperature=cfg.temperature,
                completions_per_prompt=cfg.num_generations,
            )
            self._setup(dataset_func())
        else:
            from .remote_rollout.mpi_rollout import MPIRolloutClient
            from .remote_rollout.weight_bridge import TTT_RANK, HostWeightBridge

            # Rank 1 is blocked waiting for the handshake, terminating processes on a failure before handshake.
            with exit_on_error():
                autograd_ctx = ttml.autograd.AutoContext.get_instance()
                autograd_ctx.initialize_distributed_context(*sys.argv)
                self.model, self.tokenizer = setup_ttml_model(transformer_config, device_config, self.model_source)
                self._batch_layout = resolve_batch_layout()
                bridge = HostWeightBridge.init_sender(mesh=autograd_ctx.get_device(), peer_rank=TTT_RANK)
                self._rollout_client = MPIRolloutClient(peer_rank=TTT_RANK, bridge=bridge)
            try:
                self._setup(dataset_func())
            except BaseException:
                self._rollout_client.shutdown()
                raise

    def _init_remote_rollout(self) -> None:
        from .remote_rollout.mpi_rollout import MPIRolloutServer
        from .remote_rollout.weight_bridge import TTML_RANK, HostWeightBridge

        cfg = self.config
        rr = cfg.remote_rollout
        self._rollout_mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*rr.mesh_shape), offset=ttnn.MeshCoordinate(0, 0)
        )
        self.rollout_sampler = build_rollout_sampler(
            cfg.backend.sampler,
            model_source=self.model_source,
            mesh_device=self._rollout_mesh,
            remote_rollout=rr,
            max_completion_length=cfg.max_completion_length,
            temperature=cfg.temperature,
            completions_per_prompt=cfg.num_generations,
        )
        bridge = HostWeightBridge.init_receiver(
            mesh=self._rollout_mesh, peer_rank=TTML_RANK, submeshes=self.rollout_sampler.submeshes
        )
        self._rollout_server = MPIRolloutServer(peer_rank=TTML_RANK, bridge=bridge, sampler=self.rollout_sampler)

    def _init_rewards(
        self,
        reward_func: Optional[Callable[..., List[float]]],
        reward_funcs: Optional[List[Callable[..., List[float]]]],
    ) -> None:
        """Validate reward inputs (exactly one of ``reward_func`` /
        ``reward_funcs`` must be provided) and normalize onto a list.

        When ``reward_funcs`` is given, per-completion rewards from each
        function are summed element-wise at train time; per-component means
        are logged separately under ``{fn.__name__}_mean`` keys.
        """
        if (reward_func is None) == (reward_funcs is None):
            raise ValueError(
                "GRPOTrainer: pass exactly one of 'reward_func' (single callable) "
                "or 'reward_funcs' (non-empty list of callables)."
            )
        if reward_funcs is not None and not reward_funcs:
            raise ValueError("GRPOTrainer: 'reward_funcs' must be a non-empty list.")

        self.reward_funcs = list(reward_funcs) if reward_funcs is not None else [reward_func]
        self.reward_func = reward_func
        self._reward_func_names = _derive_reward_names(self.reward_funcs)

    def _time_callback(self, cb: Any, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Fire ``cb.<method_name>(*args, **kwargs)`` and accumulate its wall-clock
        time into ``self.metrics[<ClassName>_time_s]`` for the current step.

        ``GRPOMonitor`` is intentionally excluded from the accumulator: it runs
        last, after ``step_time_s`` has been sealed, and its own cost sits
        outside the step wall time by design.
        """
        cb_t0 = time.perf_counter()
        getattr(cb, method_name)(*args, **kwargs)
        if not isinstance(cb, GRPOMonitor):
            key = f"{type(cb).__name__}_time_s"
            self.metrics[key] = self.metrics.get(key, 0.0) + (time.perf_counter() - cb_t0)

    def _compute_grpo_loss(
        self,
        nlog_probs_old: ttml.autograd.Tensor,
        nlog_probs_new: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        adv_ttml: ttml.autograd.Tensor,
        completions_batch_len: int,
        eps: float,
        ddp_world_size: int = 1,
    ) -> ttml.autograd.Tensor:
        """Compute the clipped GRPO surrogate loss.

        ``completions_batch_len`` is the *global* number of completions in the
        optimizer step. Under DDP, ``ttml.core.distributed.synchronize_gradients``
        *averages* (not sums) the per-device gradients, so normalising the loss
        by the global count would leave an extra ``1 / ddp_world_size`` factor
        after that averaging. To keep gradients invariant to the device count we
        normalise by the *per-device* completion count instead
        (``completions_batch_len / ddp_world_size``); the gradient averaging then
        restores the intended global-mean gradient. ``ddp_world_size`` is 1 when
        DDP is disabled, leaving the single-device path unchanged.
        """
        B_local, Tp = nlog_probs_old.shape()
        ratio = ttml.ops.unary.exp(nlog_probs_old - nlog_probs_new)
        clipped_ratio = ttml.ops.unary.clip(ratio, 1.0 - eps, 1.0 + eps)

        surr1 = ratio * adv_ttml
        surr2 = clipped_ratio * adv_ttml
        surr = ttml.ops.binary.min(surr1, surr2)

        # Per-completion normalised weight: w[i,t] = mask[i,t] / max(sum_t(mask[i,t]), 1)
        mask_val = mask.get_value()
        tokens_per_completion = ttnn.maximum(ttnn.sum(mask_val, dim=1, keepdim=True), 1.0)
        weight_tt = ttml.autograd.create_tensor(ttnn.div(mask_val, tokens_per_completion), requires_grad=False)

        weighted_surr = surr * weight_tt
        weighted_surr_4d = ttml.ops.reshape.reshape(weighted_surr, [1, 1, B_local, Tp])
        per_device_batch_len = completions_batch_len / ddp_world_size
        return ttml.ops.unary.mean(weighted_surr_4d) * (-float(B_local) * float(Tp) / per_device_batch_len)

    def _setup(self, dataset: Any) -> None:
        """One-shot training setup: validate config, build optimizer, resolve
        the device-parallelism topology, and tokenize ``dataset``.

        Populates the ``self._foo`` attributes pre-declared in ``__init__`` (all
        the parallelism-topology and batching state that every phase helper
        below reads).
        """
        grpo_cfg = self.config
        tt_model = self.model
        tokenizer = self.tokenizer
        pad_token = resolve_pad_token_id(tokenizer)

        optimizer = create_optimizer(tt_model, self.optimizer_dict)
        if grpo_cfg.use_clip_grad_norm:
            validate_clip_grad_norm(tt_model)

        num_devices = self._batch_layout.num_shards

        # Derive the across-mesh micro-batch size (in completions), the per
        # micro-batch prompt count, and the generation (effective) batch size up
        # front, and validate the divisibility relationships so misconfigurations
        # fail with a clear message instead of a cryptic shard assert when the
        # :func:`layout_microbatch` tensors are uploaded for the trainer's
        # policy forward (or a silently ragged final micro-batch).
        #
        # ``per_device_train_batch_size`` is the number of completions resident
        # on a single device within one micro-batch, so the whole mesh handles
        # ``completions_per_microbatch = per_device_train_batch_size *
        # num_devices`` completions per micro-batch. The per micro-batch prompt
        # count is ``completions_per_microbatch // num_generations``. By
        # construction ``completions_per_microbatch`` is divisible by
        # ``num_devices``, so each micro-batch always shards evenly along axis 0.
        #
        # The generation (effective) batch spans ``gradient_accumulation_steps``
        # micro-batches: each batch generates ``grad_accum`` times the per
        # micro-batch prompt count, then the trainer runs one forward/backward
        # pass per micro-batch accumulating gradients before a single optimizer
        # step. Increasing ``gradient_accumulation_steps`` therefore generates
        # proportionally more completions per batch and trains over that many
        # micro-batches between optimizer steps.
        if grpo_cfg.per_device_train_batch_size <= 0:
            raise ValueError(
                f"per_device_train_batch_size must be positive, got {grpo_cfg.per_device_train_batch_size}"
            )
        grad_accum = grpo_cfg.gradient_accumulation_steps
        if grad_accum <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {grad_accum}")
        if grpo_cfg.num_generations <= 0:
            raise ValueError(f"num_generations must be positive, got {grpo_cfg.num_generations}")
        completions_per_microbatch = grpo_cfg.per_device_train_batch_size * num_devices
        if completions_per_microbatch % grpo_cfg.num_generations != 0:
            raise ValueError(
                f"per_device_train_batch_size * num_devices ({grpo_cfg.per_device_train_batch_size} * "
                f"{num_devices} = {completions_per_microbatch}) must be divisible by "
                f"num_generations ({grpo_cfg.num_generations}) so the per micro-batch prompt count is an integer"
            )
        prompts_per_microbatch = completions_per_microbatch // grpo_cfg.num_generations
        generation_batch_prompts = prompts_per_microbatch * grad_accum

        total_prompts = min(int(grpo_cfg.prompts_to_train), len(dataset))
        if total_prompts % generation_batch_prompts != 0:
            raise ValueError(
                f"prompts_to_train ({total_prompts}) must be divisible by the generation batch size "
                f"(prompts_per_microbatch * gradient_accumulation_steps = {prompts_per_microbatch} * "
                f"{grad_accum} = {generation_batch_prompts}) to avoid a ragged final batch that can break "
                "micro-batch sharding"
            )
        from .rollout_batch_source import build_rollout_batch_source, tokenize_prompts

        prompts, extra_columns = tokenize_prompts(dataset, tokenizer, total_prompts)

        # Publish outputs onto self for the per-batch helpers.
        self._pad_token = int(pad_token)
        self._optimizer = optimizer
        self._completions_per_microbatch = completions_per_microbatch
        self._monitor_cb = next((cb for cb in self.callbacks if isinstance(cb, GRPOMonitor)), None)
        self.rollout_batch_source = build_rollout_batch_source(
            grpo_cfg,
            sampler=self.rollout_sampler,
            client=self._rollout_client,
            prompts=prompts,
            extra_columns=extra_columns,
            batch_prompts=generation_batch_prompts,
            tokenizer=tokenizer,
            reward_funcs=self.reward_funcs,
        )
        self._num_generation_batches = total_prompts // generation_batch_prompts
        total_optimizer_steps = self._num_generation_batches * grpo_cfg.num_iterations
        self._lr_scheduler = _build_lr_scheduler(optimizer, grpo_cfg, total_optimizer_steps)

    # -- phase helpers -------------------------------------------------------
    #
    # Each helper maps to one phase of GRPO training. They take a few raw args,
    # return one primitive or nothing, and write any metrics they produce
    # directly into ``self.metrics``. ``train()`` composes them.

    def _iter_scored_batches(self) -> Iterator[ScoredRolloutBatch]:
        """Yield the rollout batch source's batches, starting a fresh
        ``self.metrics`` (and the ``step_time_s`` timer) for each one and
        recording its metrics. ``self.metrics`` is not reset once the source is
        exhausted, so ``on_train_end`` sees the final step's metrics.
        """
        batches = iter(self.rollout_batch_source)
        while True:
            start = time.perf_counter()
            scored = next(batches, None)
            if scored is None:
                return
            self.metrics = {"weight_sync_time_s": self._weight_sync_time_s}
            self._step_start_time = start - self._weight_sync_time_s
            self._weight_sync_time_s = 0.0
            self._record_batch_metrics(scored, wait_s=time.perf_counter() - start)
            yield scored

    def _record_batch_metrics(self, scored: ScoredRolloutBatch, wait_s: float) -> None:
        rollout, rewards = scored.rollout, scored.rewards
        self.metrics["generation_time_s"] = scored.generation_time_s
        self.metrics["reward_time_s"] = scored.reward_time_s
        self.metrics["rollout_wait_s"] = wait_s
        if len(self.reward_funcs) > 1:
            for name, arr in scored.reward_components.items():
                self.metrics[f"{name}_mean"] = float(arr.mean()) if arr.size else 0.0

        self.metrics["reward_mean"] = float(rewards.mean())
        self.metrics["reward_std"] = float(rewards.std())

        lens = [len(c) for c in rollout.completions]
        self.metrics["mean_completion_len"] = (sum(lens) / len(lens)) if lens else 0.0
        self.metrics["min_completion_len"] = min(lens) if lens else 0
        self.metrics["max_completion_len"] = max(lens) if lens else 0
        self.metrics["rollout_weight_version"] = rollout.weight_version

        cfg = self.config
        if cfg.log_completions and cfg.num_completions_to_print > 0:
            k = cfg.num_completions_to_print
            self.metrics["prompts"] = [self.tokenizer.decode(p) for p in rollout.prompts[:k]]
            self.metrics["completions"] = [
                self.tokenizer.decode(c, skip_special_tokens=True) for c in rollout.completions[:k]
            ]
            self.metrics["rewards"] = rewards[:k].tolist()

    def _record_clipping_stats(
        self, micro_batch_logprobs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> None:
        """Write ``logprob_mismatch_{mean,max}`` and ``clip_frac`` over every completion token of the step.

        Each entry is one micro-batch's ``(old_np, new_np, mask_np, adv_np)``, as :func:`clipping_stats` takes them.
        """
        stats = [clipping_stats(*mb, self.config.epsilon) for mb in micro_batch_logprobs]
        mismatch = np.concatenate([m for m, _ in stats])
        clipped = np.concatenate([c for _, c in stats])
        self.metrics["logprob_mismatch_mean"] = float(mismatch.mean()) if mismatch.size else 0.0
        self.metrics["logprob_mismatch_max"] = float(mismatch.max()) if mismatch.size else 0.0
        self.metrics["clip_frac"] = float(clipped.mean()) if clipped.size else 0.0

    def _compute_advantages(self, rewards_np: np.ndarray) -> np.ndarray:
        """Group-relative advantages on host (per-prompt mean subtracted)."""
        return compute_advantages_host(rewards_np, self.config.num_generations)

    def _optimize(self, batch: RolloutBatch, advantages_np: np.ndarray) -> None:
        """Per-micro-batch forward-with-grad + GRPO loss + backward.

        ``pi_old`` comes from ``batch.logprobs`` (the sampler's per-token
        ``log pi_old``) and stays on the host until its micro-batch runs, so it
        is fixed across ``num_iterations`` passes. Accumulates gradients across
        the whole generation batch; does NOT call ``optimizer.step()`` — that is
        ``_apply_gradients``.
        """
        assert (
            self.model.get_run_mode() == RunMode.TRAIN
        ), "GRPOTrainer: model left in eval mode (a sampler or callback didn't restore it)"
        self._optimizer.zero_grad()
        mb = self._completions_per_microbatch
        global_len = len(batch.completions)
        micro_batch_logprobs = []
        for i, (p, c) in enumerate(iter_micro_batch(batch.prompts, batch.completions, mb)):
            rows = slice(i * mb, i * mb + len(c))
            inputs, targets, mask_np, Tp = layout_microbatch(p, c, self._pad_token)
            old_np = place_old_nlog_probs(p, c, batch.logprobs[rows], Tp)
            nlog_new = self._compute_policy_nlog_probs(inputs, targets)
            self._compute_loss_and_backward(nlog_new, mask_np, old_np, advantages_np[rows], global_len)
            new_np = nlog_new.to_numpy(ttnn.DataType.FLOAT32, composer=self._batch_layout.composer)
            deallocate_tensors([nlog_new])
            micro_batch_logprobs.append((old_np, new_np, mask_np, advantages_np[rows]))
        self._record_clipping_stats(micro_batch_logprobs)

    def _compute_policy_nlog_probs(self, inputs: np.ndarray, targets: np.ndarray) -> ttml.autograd.Tensor:
        """Teacher-forced forward with grad over the :func:`layout_microbatch`
        tensors. Returns ``-log pi(target)`` of shape ``[B_local, Tp]``.
        """
        B, Tp = inputs.shape
        x = ttml.autograd.Tensor.from_numpy(
            inputs.reshape(B, 1, 1, Tp), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, self._batch_layout.mapper
        )
        # Per-device row count after sharding: correct for DDP, FSDP and single device alike.
        B_local = int(x.shape()[0])
        logits = self.model(x, None)
        tgt = ttml.autograd.Tensor.from_numpy(
            targets, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, self._batch_layout.mapper
        )
        nlog = ttml.ops.loss.cross_entropy_loss(logits, tgt, ttml.ops.ReduceType.NONE)
        return ttml.ops.reshape.reshape(nlog, [B_local, Tp])

    def _compute_loss_and_backward(
        self,
        nlog_new: ttml.autograd.Tensor,
        mask_np: np.ndarray,
        old_np: np.ndarray,
        adv_slice: np.ndarray,
        global_len: int,
    ) -> None:
        """One autograd transaction: upload advantages, ``pi_old`` and the
        completion mask, build the clipped GRPO surrogate on ``nlog_new``,
        backward, and free the intermediates.

        ``global_len`` is the completion count across the whole generation
        batch (all ``grad_accum`` micro-batches); the loss normalization uses
        it to yield a mean-over-effective-batch gradient after all micro-batch
        contributions accumulate.
        """
        # Advantages, pi_old, the mask and the policy log-probs are all sharded
        # along axis 0 over the identical host-order rows, so device-local row r
        # of every tensor refers to the same completion.
        adv_slice_val = upload_micro_advantages(adv_slice, self._batch_layout.mapper, self._batch_layout.num_shards)
        adv_ttml = ttml.autograd.create_tensor(adv_slice_val, requires_grad=False)
        mask = ttml.autograd.Tensor.from_numpy(
            mask_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, self._batch_layout.mapper
        )
        mask.set_requires_grad(False)

        new_val = nlog_new.get_value()
        nlog_old = ttml.autograd.Tensor.from_numpy(old_np, new_val.layout, new_val.dtype, self._batch_layout.mapper)
        nlog_old.set_requires_grad(False)
        assert list(nlog_old.shape()) == list(
            nlog_new.shape()
        ), f"pi_old shape {nlog_old.shape()} != pi shape {nlog_new.shape()}"

        loss = self._compute_grpo_loss(
            nlog_old,
            nlog_new,
            mask,
            adv_ttml,
            global_len,
            self.config.epsilon,
            ddp_world_size=self._batch_layout.grad_sync_world_size,
        )
        loss.backward(retain_graph=False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        deallocate_tensors([nlog_old, mask, adv_ttml, loss])

    def _apply_gradients(self) -> None:
        """Grad sync + ``on_before_optimizer_step`` + optional grad clip +
        ``optimizer.step()`` + ``zero_grad`` + LR scheduler step.
        Writes ``lr`` and per-callback timings to ``self.metrics``.

        Ordering:
          1. Grad sync (FSDP / parallelism-context DDP / named-mesh DDP).
          2. Callback ``on_before_optimizer_step`` fires (sees unclipped grads).
          3. ``ttml.core.clip_grad_norm`` AFTER grad sync so every rank clips
             the identical reduced grads. Guarded by ``config.use_clip_grad_norm``.
          4. ``optimizer.step()`` + ``zero_grad``.
          5. ``self._lr_scheduler.step()`` sets the next step's LR.
        """
        if self._batch_layout.fsdp_enabled:
            ttml.sync_gradients(self.model.parameters(), axis_names=self._batch_layout.fsdp_sync_axes)
        elif self._batch_layout.ddp_context_enabled:
            # Parallelism-context DDP (Llama): a parallelism context
            # is initialized, so use its gradient sync. ``sync_gradients`` would
            # be a silent no-op here — it reduces over named mesh axes, and this
            # path opens no named mesh (``maybe_mesh()`` is None), so it would
            # leave gradients un-averaged.
            ttml.core.distributed.synchronize_gradients(self.model.parameters())
        elif self._batch_layout.ddp_enabled:
            # Named-mesh DDP (Qwen3): no parallelism context exists,
            # so all-reduce + average grads over the "dp" mesh axis — the same
            # primitive FSDP uses. The loss normalization divides by
            # ``grad_sync_world_size`` (= the "dp" axis size here), matched to
            # this reduction.
            ttml.sync_gradients(self.model.parameters(), axis_names=("dp",))

        for cb in self.callbacks:
            self._time_callback(cb, "on_before_optimizer_step", self)

        if self.config.use_clip_grad_norm:
            ttml.core.clip_grad_norm(
                self.model.parameters(),
                self.config.clip_grad_norm_max_norm,
            )

        self.metrics["lr"] = self._optimizer.get_lr()
        self._optimizer.step()
        self._optimizer.zero_grad()
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    def _update_rollout_weights(self) -> None:
        start = time.perf_counter()
        self.rollout_batch_source.update_weights(self.model, self.global_step)
        self._weight_sync_time_s = time.perf_counter() - start

    def _publish_step_metrics(self) -> None:
        """Fire ``on_step_end`` for every non-monitor callback (their timings
        fold into ``self.metrics`` via ``_time_callback``), save a checkpoint
        if due, seal ``step_time_s``, and finally fire the monitor callback's
        ``on_step_end`` OUTSIDE the ``step_time_s`` window.
        """
        step = self.global_step

        for cb in self.callbacks:
            if cb is self._monitor_cb:
                continue
            self._time_callback(cb, "on_step_end", self, step, **self.metrics)

        self._maybe_checkpoint()

        # Seal step_time_s after all non-monitor work is done, so it covers
        # the full per-step wall time (rollout, host post-gen, training loop,
        # non-monitor callbacks, checkpoint). GRPOMonitor's own cost is
        # deliberately outside this window.
        self.metrics["step_time_s"] = time.perf_counter() - self._step_start_time

        if self._monitor_cb is not None:
            self._time_callback(self._monitor_cb, "on_step_end", self, step, **self.metrics)

    def _maybe_checkpoint(self) -> None:
        """Save a checkpoint on interval + fire ``on_save`` for non-monitor
        callbacks (their timings fold into ``self.metrics``).
        """
        step = self.global_step
        cfg = self.config
        if not cfg.checkpointing or step % cfg.checkpoint_interval != 0:
            return
        save_checkpoint(
            self.model,
            step,
            cfg.output_dir,
            tokenizer=self.tokenizer,
            grpo_config=cfg,
            optimizer=self._optimizer,
            model_source=self.model_source,
            scheduler=self._lr_scheduler,
            learning_rate=self.metrics["lr"],
        )
        ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", f"grpo_step_{step}")
        for cb in self.callbacks:
            if cb is self._monitor_cb:
                continue
            self._time_callback(cb, "on_save", self, step, ckpt_dir)

    # -- training loop -------------------------------------------------------

    def train(self) -> None:
        if self._train_called:
            raise RuntimeError("GRPOTrainer.train() can only be called once. Build a new GRPOTrainer to train again.")
        self._train_called = True
        if is_rollout_rank(self.config):
            self._serve_rollouts()
        else:
            self._train_policy()

    def _serve_rollouts(self) -> None:
        try:
            self._rollout_server.serve_forever()
        finally:
            self._rollout_server = None
            self.rollout_sampler = None
            gc.collect()
            ttnn.close_mesh_device(self._rollout_mesh)
            self._rollout_mesh = None

    def _train_policy(self) -> None:
        try:
            for cb in self.callbacks:
                cb.on_train_begin(self)
            if self.config.backend.weight_push_before_training:
                self._update_rollout_weights()
            for batch_idx, scored in enumerate(self._iter_scored_batches()):
                advantages_np = self._compute_advantages(scored.rewards)
                batch_metrics = dict(self.metrics)
                for mini_epoch in range(self.config.num_iterations):
                    if mini_epoch > 0:
                        # Metrics are averaged over steps; later mini-epochs don't generate, so they report 0.
                        self.metrics = {
                            **batch_metrics,
                            "generation_time_s": 0.0,
                            "reward_time_s": 0.0,
                            "rollout_wait_s": 0.0,
                            "weight_sync_time_s": 0.0,
                        }
                        self._step_start_time = time.perf_counter()
                    self._optimize(scored.rollout, advantages_np)
                    self._apply_gradients()
                    self.global_step += 1
                    self._publish_step_metrics()
                if batch_idx < self._num_generation_batches - 1:
                    self._update_rollout_weights()
        finally:
            self.rollout_batch_source.close()
        for cb in self.callbacks:
            cb.on_train_end(self)
