# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Performance and device-capacity helpers shared by the training examples.

Collects the pieces used to report throughput and MFU (model FLOPS utilization)
so every example measures them the same way:

  - :func:`get_device_peak_tflops_bf16` – theoretical per-device BF16 peak, the
    denominator for MFU.
  - :func:`get_available_device_memory_in_bytes` – total device DRAM.
  - :class:`PerformanceMeter` – rolling samples/sec and tokens/sec.
"""

from __future__ import annotations

from time import time

import ttnn
from ttnn.device import is_blackhole, is_wormhole_b0
import ttml


def get_device_peak_tflops_bf16() -> float:
    """Per-device theoretical BF16 TFLOPS. Whole-mesh peak = this × num_devices."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    # Per-core BF16 TFLOPS for each supported TT architecture.
    if is_wormhole_b0(device):
        per_core = 1.0
    elif is_blackhole(device):
        per_core = 1.35
    else:
        raise ValueError(f"Unknown device: {device.arch()}")
    return num_cores * per_core


def get_available_device_memory_in_bytes() -> int:
    """Get the total amount of device DRAM available on the system."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    dram_view = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
    total_dram = dram_view.total_bytes_per_bank * dram_view.num_banks * ttnn.get_num_devices()
    return total_dram


class PerformanceMeter:
    def __init__(self, cfg, window_size=10):
        self.cfg = cfg
        self.steps = []
        self.window_size = window_size

    def step(self):
        self.steps.append(time())
        if len(self.steps) > self.window_size:
            self.steps.pop(0)

    def get_metrics(self):
        time_window = self.steps[-1] - self.steps[0]
        if time_window == 0:
            return 0, 0

        samples = len(self.steps) * self.cfg.batch_size * self.cfg.gradient_accumulation_steps
        samples_per_second = samples / time_window
        tokens_per_second = samples * self.cfg.seq_len / time_window
        return samples_per_second, tokens_per_second
