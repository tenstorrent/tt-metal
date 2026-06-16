# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""SFTTrainer callbacks used by the training entry point."""

from __future__ import annotations

import time
from typing import Any

import ttml
from ttml.datasets import Batch
from ttml.trainers import SFTTrainer, TrainerCallback

import moe_activation_logger

MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker


class ThroughputCallback(TrainerCallback):
    """Print wall-clock TPS / TFLOPS / MFU every `log_interval` steps."""

    def __init__(self, flops_per_token: float, peak_tflops: float, log_interval: int = 1) -> None:
        self._flops_per_token = flops_per_token
        self._peak_tflops = peak_tflops
        self._log_interval = max(1, int(log_interval))
        self._step_start: float | None = None
        self._tokens_in_step: int = 0
        self._dp_size: int = 1

    def on_train_begin(self, trainer: SFTTrainer) -> None:
        mesh = ttml.mesh()
        self._dp_size = mesh.axis_size("dp") if mesh.has_axis("dp") else 1
        self._step_start = time.time()
        self._tokens_in_step = 0

    def on_after_forward(self, trainer: SFTTrainer, batch: Batch, loss: float) -> None:
        shape = batch.input_ids.shape()
        # Per-rank micro-shard tokens × dp_size = global tokens processed this step.
        self._tokens_in_step += int(shape[0]) * int(shape[-1]) * self._dp_size

    def on_step_end(self, trainer: SFTTrainer, step: int, step_loss: float = 0.0, *args: Any, **kwargs: Any) -> None:
        if step % self._log_interval != 0 or self._step_start is None:
            self._step_start = time.time()
            self._tokens_in_step = 0
            return
        now = time.time()
        elapsed_ms = (now - self._step_start) * 1000.0
        tps = self._tokens_in_step / max(1e-6, elapsed_ms / 1000.0)
        line = f"Step: {step}, Loss: {step_loss:.6f}, Time: {elapsed_ms:.2f} ms, TPS: {tps:.0f}"
        if self._flops_per_token > 0 and elapsed_ms > 0:
            achieved = tps * self._flops_per_token / 1e12
            line += f", TFLOPS: {achieved:.3g}"
            if self._peak_tflops > 0:
                mfu = achieved / self._peak_tflops * 100.0
                line += f", MFU: {mfu:.3g}%"
        print(line)
        self._step_start = now
        self._tokens_in_step = 0


class MemoryTrackerCallback(TrainerCallback):
    """In-loop FORWARD_PASS / BACKWARD_PASS / FIRST_ITERATION_COMPLETE snapshots over step 1, then deregister.

    run_training opens the capture session (so its ENTRY/MODEL_CREATION/OPTIMIZER_CREATION setup
    snapshots are included); this callback only adds the per-step snapshots and closes the session.
    """

    def on_after_forward(self, trainer: SFTTrainer, batch: Batch, loss: float) -> None:
        MemoryUsageTracker.snapshot("FORWARD_PASS")

    def on_after_backward(self, trainer: SFTTrainer, batch: Batch) -> None:
        MemoryUsageTracker.snapshot("BACKWARD_PASS")

    def on_step_end(self, trainer: SFTTrainer, step: int, *args: Any, **kwargs: Any) -> None:
        MemoryUsageTracker.end_capture("FIRST_ITERATION_COMPLETE")
        MemoryUsageTracker.print_memory_usage()
        MemoryUsageTracker.clear()
        trainer.remove_callback(self)


class MoECallback(TrainerCallback):
    """DeepSeek-only: update expert routing bias each step; optionally log per-expert activation probs to CSV."""

    def __init__(self, log_path: str | None = None) -> None:
        self._log_path = log_path

    def on_step_end(self, trainer: SFTTrainer, step: int, *args: Any, **kwargs: Any) -> None:
        if not hasattr(trainer.model, "get_moe_layers"):
            return
        moe_layers = trainer.model.get_moe_layers()
        # Log BEFORE update_expert_bias: update resets the underlying _token_counts buffer,
        # which the logger reads to compute activation probabilities.
        if self._log_path and moe_activation_logger.should_log_step(step):
            moe_activation_logger.log_step_expert_balance(self._log_path, step, moe_layers)
        for layer in moe_layers:
            layer.update_expert_bias()


class AverageLossCallback(TrainerCallback):
    """Running mean of `step_loss` for the final-summary `Average loss` line."""

    def __init__(self) -> None:
        self.sum: float = 0.0
        self.count: int = 0

    def on_step_end(self, trainer: SFTTrainer, step: int, step_loss: float = 0.0, *args: Any, **kwargs: Any) -> None:
        self.sum += step_loss
        self.count += 1

    @property
    def average(self) -> float:
        return self.sum / max(1, self.count)
