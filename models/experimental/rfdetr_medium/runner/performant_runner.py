# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
RF-DETR Medium performant runner with 2 command queue overlapped execution.

Pipeline structure:
  CQ1 (write): host-to-device input transfer
  CQ0 (compute): model forward pass

RF-DETR has a host barrier (two-stage top-K selection) between
backbone+projector and decoder+heads, so full trace capture across the
entire forward is not possible. Instead, we use 2CQ to overlap H2D
input transfer with computation.
"""

import torch
import ttnn
from loguru import logger

from models.experimental.rfdetr_medium.runner.performant_runner_infra import RFDETRPerformanceRunnerInfra


class RFDETRPerformantRunner:
    def __init__(self, device, batch_size=1):
        self.device = device
        self.batch_size = batch_size

        self.runner_infra = RFDETRPerformanceRunnerInfra(device, batch_size)

        self.tt_inputs_host, self.tt_input_dram = self.runner_infra.setup_dram_input()

    def _warmup_2cqs(self, num_warmup=2):
        """
        Warm up with 2CQ overlapped I/O to compile all kernels.

        Phase 1: JIT compilation (first run compiles kernels)
        Phase 2+: Optimized runs (cached programs)
        """
        self.op_event = ttnn.record_event(self.device, 0)

        for i in range(num_warmup):
            logger.info(f"Warmup iteration {i+1}/{num_warmup}...")

            ttnn.wait_for_event(1, self.op_event)
            ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_input_dram, 1)
            self.write_event = ttnn.record_event(self.device, 1)
            ttnn.wait_for_event(0, self.write_event)

            self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_input_dram, ttnn.L1_MEMORY_CONFIG)
            self.op_event = ttnn.record_event(self.device, 0)

            self.runner_infra.run_full()
            ttnn.synchronize_device(self.device)

        logger.info("Warmup complete.")

    def _execute_2cqs_inference(self, tt_inputs_host=None):
        """Execute one inference with 2CQ overlapped H2D."""
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host

        # CQ1: H2D transfer (overlapped with previous iteration's post-processing)
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_input_dram, 1)
        self.write_event = ttnn.record_event(self.device, 1)

        # CQ0: Wait for transfer, then compute
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_input_dram, ttnn.L1_MEMORY_CONFIG)
        self.op_event = ttnn.record_event(self.device, 0)

        self.runner_infra.run_full()

        return self.runner_infra.cls_output, self.runner_infra.bbox_output

    def run(self, torch_input_tensor=None):
        """Run one inference with 2CQ. Returns (cls_output, bbox_output) on device."""
        if torch_input_tensor is not None:
            tt_inputs_host = self.runner_infra.setup_l1_input(torch_input_tensor)
        else:
            tt_inputs_host = self.tt_inputs_host
        return self._execute_2cqs_inference(tt_inputs_host)

    def run_no_trace(self, torch_input_tensor=None):
        """Run without 2CQ (plain forward, for comparison)."""
        inp = torch_input_tensor if torch_input_tensor is not None else self.runner_infra.torch_input
        img = inp.permute(0, 2, 3, 1)
        img = torch.nn.functional.pad(img, (0, 1, 0, 0, 0, 0, 0, 0))
        self.runner_infra.input_tensor = ttnn.from_torch(img, dtype=ttnn.bfloat16, device=self.device)

        self.runner_infra.run_full()
        return self.runner_infra.cls_output, self.runner_infra.bbox_output

    def release(self):
        """Cleanup (no traces to release in 2CQ-only mode)."""
