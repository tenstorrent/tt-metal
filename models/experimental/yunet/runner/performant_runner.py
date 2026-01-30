# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from models.experimental.yunet.runner.performant_runner_infra import YunetPerformanceRunnerInfra


class YunetPerformantRunner:
    """
    Performant runner for YUNet using Trace + 2 Command Queues.

    Uses double-buffering to overlap host-to-device transfer with device computation.
    - Buffer A: Used by current trace execution
    - Buffer B: Receiving next input via PCIe transfer
    - Alternate between buffers each iteration
    """

    def __init__(
        self,
        device,
        input_height=640,
        input_width=640,
        act_dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.input_height = input_height
        self.input_width = input_width

        # Create runner infrastructure
        self.runner_infra = YunetPerformanceRunnerInfra(
            device,
            batch_size=1,
            input_height=input_height,
            input_width=input_width,
            act_dtype=act_dtype,
        )

        # Setup DRAM input
        self.tt_inputs_host, self.dram_mem_config = self.runner_infra.setup_dram_input(device)

        # Allocate TWO persistent DRAM buffers for double-buffering
        self.tt_image_res_0 = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, input_height, input_width, 3]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            self.dram_mem_config,
        )
        self.tt_image_res_1 = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, input_height, input_width, 3]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            self.dram_mem_config,
        )

        self.current_buffer = 0  # Track which buffer is active

        # Capture TWO traces, one for each buffer
        self._capture_double_buffer_traces()

    def _capture_double_buffer_traces(self):
        """Capture two traces for double-buffering."""
        # Initialize events
        self.op_event = ttnn.record_event(self.device, 0)
        self.write_event = ttnn.record_event(self.device, 1)

        # === Warmup and JIT compile using buffer 0 ===
        logger.info("Running JIT config iteration...")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res_0, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)

        self.runner_infra.input_tensor = self.tt_image_res_0
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.dealloc_output()

        # === Optimized run ===
        logger.info("Running optimized iteration...")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res_0, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)

        self.runner_infra.input_tensor = self.tt_image_res_0
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.dealloc_output()

        # === Capture trace 0 (using buffer 0) ===
        logger.info("Capturing trace 0 (buffer 0)...")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res_0, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)

        self.runner_infra.input_tensor = self.tt_image_res_0
        self.runner_infra.dealloc_output()

        self.tid_0 = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        ttnn.end_trace_capture(self.device, self.tid_0, cq_id=0)

        # === Capture trace 1 (using buffer 1) ===
        logger.info("Capturing trace 1 (buffer 1)...")
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res_1, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)

        self.runner_infra.input_tensor = self.tt_image_res_1
        self.runner_infra.dealloc_output()

        self.tid_1 = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        ttnn.end_trace_capture(self.device, self.tid_1, cq_id=0)

        self.op_event = ttnn.record_event(self.device, 0)

        logger.info("Double-buffer traces captured successfully!")

    def _execute_double_buffer_inference(self, tt_inputs_host):
        """
        Execute inference with double-buffering.

        While trace N executes on buffer A, copy next input to buffer B.
        Then swap buffers for next iteration.
        """
        if self.current_buffer == 0:
            # Execute trace 0 (reads from buffer 0)
            # Meanwhile, copy next input to buffer 1
            ttnn.wait_for_event(1, self.op_event)
            ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res_1, 1)
            self.write_event = ttnn.record_event(self.device, 1)

            # Execute trace (non-blocking)
            ttnn.execute_trace(self.device, self.tid_0, cq_id=0, blocking=False)
            self.op_event = ttnn.record_event(self.device, 0)

            # Next iteration will use buffer 1
            self.current_buffer = 1
        else:
            # Execute trace 1 (reads from buffer 1)
            # Meanwhile, copy next input to buffer 0
            ttnn.wait_for_event(1, self.op_event)
            ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res_0, 1)
            self.write_event = ttnn.record_event(self.device, 1)

            # Execute trace (non-blocking)
            ttnn.execute_trace(self.device, self.tid_1, cq_id=0, blocking=False)
            self.op_event = ttnn.record_event(self.device, 0)

            # Next iteration will use buffer 0
            self.current_buffer = 0

        return self.runner_infra.output_tensors

    def run(self, torch_input_tensor=None):
        """
        Run inference on a torch input tensor.

        Args:
            torch_input_tensor: Input tensor in NHWC format (batch, height, width, channels).
                               If None, uses the default input from setup.

        Returns:
            Tuple of (cls_outputs, box_outputs, obj_outputs, kpt_outputs)
        """
        if torch_input_tensor is not None:
            tt_inputs_host = ttnn.from_torch(
                torch_input_tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        else:
            tt_inputs_host = self.tt_inputs_host

        output = self._execute_double_buffer_inference(tt_inputs_host)
        return output

    def release(self):
        """Release traces and cleanup resources."""
        ttnn.release_trace(self.device, self.tid_0)
        ttnn.release_trace(self.device, self.tid_1)
        logger.info("Traces released.")
