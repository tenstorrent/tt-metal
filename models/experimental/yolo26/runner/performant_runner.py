# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performant runner for YOLO26 with trace and 2CQ support.

Implements optimized inference pipelines using:
- Trace capture and replay for reduced host overhead
- 2 Command Queues (2CQ) for overlapped data transfer and compute
"""

import ttnn
from loguru import logger

from models.experimental.yolo26.runner.yolo26_test_infra import create_test_infra


class YOLO26PerformantRunner:
    """
    High-performance YOLO26 runner with trace 2CQ support.

    This runner captures the model execution as a trace and uses
    2 command queues to overlap data transfer with computation.
    """

    def __init__(
        self,
        device,
        batch_size: int = 1,
        input_size: int = 640,
        variant: str = "yolo26n",
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        torch_input_tensor=None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.input_size = input_size
        self.variant = variant

        # Create test infrastructure
        self.runner_infra = create_test_infra(
            device,
            batch_size,
            input_size,
            variant,
            act_dtype,
            weight_dtype,
            torch_input_tensor,
        )

        # Setup DRAM sharded input
        (
            self.tt_inputs_host,
            sharded_mem_config_DRAM,
            self.input_mem_config,
        ) = self.runner_infra.setup_dram_sharded_input(device)
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)

        # Trace state
        self.tid = None
        self.op_event = None
        self.write_event = None
        self.input_tensor = None

    def capture_trace_2cq(self):
        """
        Capture YOLO26 execution trace with 2CQ setup.

        This performs:
        1. JIT compilation run
        2. Optimized warmup run
        3. Trace capture
        """
        logger.info("Capturing YOLO26 trace with 2CQ...")

        # Initialize the op event so we can write
        self.op_event = ttnn.record_event(self.device, 0)

        # === First run: JIT compilation ===
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        spec = self.runner_infra.input_tensor.spec
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.dealloc_output()

        # === Second run: Optimized execution ===
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.dealloc_output()

        # === Capture trace ===
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)

        trace_input_addr = self.runner_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)

        assert (
            trace_input_addr == self.input_tensor.buffer_address()
        ), "Trace input address mismatch - memory allocation changed during trace"

        logger.info("Trace capture complete")

    def execute_trace_2cq(self, tt_inputs_host=None):
        """
        Execute the captured trace with 2CQ overlapping.

        Uses event synchronization to overlap:
        - CQ1: Host-to-device data transfer
        - CQ0: Model execution (trace replay)

        Args:
            tt_inputs_host: Optional input tensor (defaults to stored input)

        Returns:
            Output tensor from model
        """
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host

        # Wait for previous op to complete before writing new input
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)

        # Wait for write to complete before executing
        ttnn.wait_for_event(0, self.write_event)
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)

        # Execute trace
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)

        return self.runner_infra.output_tensor

    def run(self, torch_input_tensor=None):
        """
        Run YOLO26 inference with trace 2CQ.

        Args:
            torch_input_tensor: Optional PyTorch input tensor

        Returns:
            Output tensor
        """
        if torch_input_tensor is not None:
            tt_inputs_host, _ = self.runner_infra.setup_l1_sharded_input(self.device, torch_input_tensor)
        else:
            tt_inputs_host = self.tt_inputs_host

        return self.execute_trace_2cq(tt_inputs_host)

    def release(self):
        """Release the captured trace."""
        if self.tid is not None:
            ttnn.release_trace(self.device, self.tid)
            self.tid = None


class YOLO26Trace2CQPipeline:
    """
    Full pipeline for YOLO26 trace 2CQ benchmarking.

    Provides warmup, measurement, and cleanup phases.
    """

    def __init__(
        self,
        device,
        batch_size: int = 1,
        input_size: int = 640,
        variant: str = "yolo26n",
    ):
        self.device = device
        self.batch_size = batch_size
        self.input_size = input_size
        self.runner = YOLO26PerformantRunner(
            device,
            batch_size,
            input_size,
            variant,
        )

    def setup(self):
        """Setup and capture trace."""
        self.runner.capture_trace_2cq()
        return self

    def warmup(self, num_iterations: int = 10):
        """Run warmup iterations."""
        logger.info(f"Running {num_iterations} warmup iterations...")
        for _ in range(num_iterations):
            self.runner.run()
        ttnn.synchronize_device(self.device)
        return self

    def benchmark(self, num_iterations: int = 100):
        """
        Run benchmark iterations.

        Returns:
            List of output tensors
        """
        outputs = []
        for _ in range(num_iterations):
            output = self.runner.run()
            outputs.append(output)
        ttnn.synchronize_device(self.device)
        return outputs

    def cleanup(self):
        """Release resources."""
        self.runner.release()
