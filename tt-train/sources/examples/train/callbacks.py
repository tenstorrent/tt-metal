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
DramFootprintTracker = ttml.core.utils.DramFootprintTracker


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
    """In-loop per-micro-step snapshots over step 1, then deregister.

    run_training opens the capture session (so its ENTRY/MODEL_CREATION/OPTIMIZER_CREATION setup
    snapshots are included); this callback only adds the per-step snapshots and closes the session.

    Each grad-accumulation micro-step gets a *uniquely named* snapshot (FORWARD_PASS_i / BACKWARD_PASS_i).
    Reusing one name per pass overwrote the stored trace, collapsing every micro-step to the last one --
    which hides the gradient buffers allocated on micro-step 1 and retained across the rest, so the
    stitched cumulative peak under-reports by that amount. Unique names keep each micro-step's real net.
    """

    def __init__(self) -> None:
        self._micro_step = 0

    def on_after_forward(self, trainer: SFTTrainer, batch: Batch, loss: float) -> None:
        self._micro_step += 1
        MemoryUsageTracker.snapshot(f"FORWARD_PASS_{self._micro_step}")

    def on_after_backward(self, trainer: SFTTrainer, batch: Batch) -> None:
        MemoryUsageTracker.snapshot(f"BACKWARD_PASS_{self._micro_step}")

    def on_step_end(self, trainer: SFTTrainer, step: int, *args: Any, **kwargs: Any) -> None:
        MemoryUsageTracker.end_capture("FIRST_ITERATION_COMPLETE")
        MemoryUsageTracker.print_memory_usage()
        MemoryUsageTracker.clear()
        trainer.remove_callback(self)


class DramFootprintCallback(TrainerCallback):
    """Track the peak DRAM footprint over the step, then print it and stop.

    The peak lands in the first step's backward (all activations + gradients + loss transients
    co-resident) -- no need to track the whole run. Near-zero
    overhead (a running max/min sampled on the allocation path), so it is always on.
    Once the window closes (during training) it prints the footprint and deregisters.
    """

    # Peak usage lands in step 1. Tunable.
    _PEAK_WINDOW_STEPS = 1

    def __init__(self) -> None:
        self.footprint: Any = None  # DramFootprint (per device) captured over the window
        self._active = False

    def on_train_begin(self, trainer: SFTTrainer) -> None:
        DramFootprintTracker.begin()
        self._active = True

    def on_step_end(self, trainer: SFTTrainer, step: int, *args: Any, **kwargs: Any) -> None:
        if self._active and step >= self._PEAK_WINDOW_STEPS:
            self._close_and_report(step)
            trainer.remove_callback(self)

    def on_train_end(self, trainer: SFTTrainer) -> None:
        # Run ended before the window closed (fewer steps than the window): capture and report.
        if self._active:
            self._close_and_report(trainer.step)

    def _close_and_report(self, steps: int) -> None:
        self.footprint = DramFootprintTracker.end()
        self._active = False
        mb = 1024 * 1024
        arena = ttml.core.utils.dram_arena_bytes()
        reserved = ttml.core.utils.dram_reserved_bytes()
        peak = self.footprint.peak_allocated_bytes
        pct = 100 * peak / arena if arena else 0.0
        print(f"=== DRAM footprint ===")
        print(f"  peak usage: {peak / mb:,.0f} MB/device ({pct:.1f}% of {arena / mb:,.0f} MB arena)")
        print(f"  reserved (outside arena): {reserved / mb:,.0f} MB/device")
        print(
            f"  largest allocatable block at the tightest point: {self.footprint.min_largest_free_bytes / mb:,.0f} MB/device"
        )


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
