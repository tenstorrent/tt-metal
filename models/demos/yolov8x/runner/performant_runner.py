# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov8x.runner.performant_runner_infra import YOLOv8xPerformanceRunnerInfra


class YOLOv8xPerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size,
    ):
        self.runner_infra = YOLOv8xPerformanceRunnerInfra(
            device,
            device_batch_size,
        )
        self.device = device
        (
            self.tt_inputs_host,
            sharded_mem_config_DRAM,
            self.input_mem_config,
        ) = self.runner_infra.setup_dram_sharded_input(device)
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)
        self.capture_yolov8x_trace_2cqs()

    def _prepare_input_for_compute(self):
        """
        Transfers input from host to device using CQ1,
        synchronizes with CQ0, and sets up input tensor.
        """
        # CQ1: transfer input tensor
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)

        # CQ0: wait for input transfer to finish
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)

    def capture_yolov8x_trace_2cqs(self):
        # Initial op event
        self.op_event = ttnn.record_event(self.device, 0)

        # === First Run: Setup and JIT ===
        self._prepare_input_for_compute()
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()

        # === Optimized Run ===
        self._prepare_input_for_compute()
        self.runner_infra.run()
        self.runner_infra.validate()

        # === Capture Trace ===
        self._prepare_input_for_compute()
        self.runner_infra.dealloc_output()
        trace_input_addr = self.runner_infra.input_tensor.buffer_address()

        # Begin trace capture on compute queue
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(self.runner_infra.input_tensor.spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)

        assert trace_input_addr == self.input_tensor.buffer_address()

    def execute_yolov8x_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host

        # Data transfer and re-sharding
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)

        ttnn.wait_for_event(0, self.write_event)

        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)

        return self.runner_infra.output_tensor

    def run(self, torch_input_tensor):
        tt_inputs_host, _ = self.runner_infra.setup_l1_sharded_input(self.device, torch_input_tensor)
        return self.execute_yolov8x_trace_2cqs_inference(tt_inputs_host)

    def release(self):
        ttnn.release_trace(self.device, self.tid)
