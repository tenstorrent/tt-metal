# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from models.experimental.yunet.runner.performant_runner_infra import YunetPerformanceRunnerInfra


class YunetPerformantRunner:
    """
    Performant runner for YUNet using Trace + 2 Command Queues.

    Uses 2CQ to overlap host-to-device input copy (CQ1) with trace execution (CQ0).
    Note: For interleaved DRAM inputs, true overlap requires sharded memory configs.
    With interleaved memory, E2E is limited by max(copy_time, trace_time) + overhead.
    """

    def __init__(
        self,
        device,
        input_height=320,
        input_width=320,
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

        # Allocate persistent DRAM buffer for input
        self.tt_image_res = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, input_height, input_width, 3]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            self.dram_mem_config,
        )

        # Capture trace
        self._capture_trace()

    def _capture_trace(self):
        """Capture trace for the model."""
        # Initialize op event
        self.op_event = ttnn.record_event(self.device, 0)

        # First run configures convs JIT
        logger.info("Running first iteration (JIT config)...")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.tt_image_res
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.dealloc_output()

        # Optimized run
        logger.info("Running optimized iteration...")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.tt_image_res
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.dealloc_output()

        # Capture trace
        logger.info("Capturing trace...")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.tt_image_res
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()

        # Begin trace capture
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)

        logger.info("Trace captured successfully!")

    def execute_inference(self, tt_inputs_host=None):
        """Execute one inference iteration using trace + 2CQ.

        Uses 2 command queues to overlap host-to-device transfer (CQ1) with
        trace execution (CQ0). The PCIe transfer is the bottleneck (~315 MB/s),
        so E2E FPS is limited by transfer time for large inputs.
        """
        if tt_inputs_host is None:
            tt_inputs_host = self.tt_inputs_host

        # Wait for previous trace to finish using input buffer
        ttnn.wait_for_event(1, self.op_event)

        # Copy new input to device (CQ1)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)

        # Wait for copy to finish before starting trace
        ttnn.wait_for_event(0, self.write_event)

        # Record that we're about to use the input buffer
        self.op_event = ttnn.record_event(self.device, 0)

        # Execute trace (non-blocking)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)

        return self.runner_infra.output_tensors

    def execute_inference_device_only(self):
        """Execute trace only, no host-to-device copy. For benchmarking device speed."""
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
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
            # Create host tensor from torch input
            tt_inputs_host = ttnn.from_torch(
                torch_input_tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        else:
            tt_inputs_host = self.tt_inputs_host

        output = self.execute_inference(tt_inputs_host)
        return output

    def release(self):
        """Release trace and cleanup resources."""
        ttnn.release_trace(self.device, self.tid)
        logger.info("Trace released.")
