# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.retinanet.runner.performant_runner_infra import RetinaNetPerformanceRunnerInfra
from loguru import logger


class RetinaNetPerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size=1,
        model_location_generator=None,
        resolution=(512, 512),
        torch_input_tensor=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
        input_path=None,
        model_config=None,
    ):
        self.device = device
        self.resolution = resolution
        self.torch_input_tensor = torch_input_tensor
        self.num_devices = self.device.get_num_devices()
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer

        self.runner_infra = RetinaNetPerformanceRunnerInfra(
            device,
            device_batch_size,
            model_config,
            model_location_generator,
            resolution=resolution,
            torch_input_tensor=self.torch_input_tensor,
            inputs_mesh_mapper=self.inputs_mesh_mapper,
            weights_mesh_mapper=self.weights_mesh_mapper,
            outputs_mesh_composer=self.outputs_mesh_composer,
            input_path=input_path,
        )

        # Setup DRAM interleaved input
        self.tt_inputs_host, self.input_mem_config = self.runner_infra.setup_dram_interleaved_input()

        # Allocate persistent device tensor (NOT ttnn.to_device)
        self.tt_image_res = ttnn.allocate_tensor_on_device(
            self.tt_inputs_host.shape,
            self.tt_inputs_host.dtype,
            self.tt_inputs_host.layout,
            device,
            ttnn.DRAM_MEMORY_CONFIG,  # Interleaved DRAM
        )

        self._capture_retinanet_trace_2cqs()

    def _capture_retinanet_trace_2cqs(self):
        # Initialize the op event so we can write
        self.op_event = ttnn.record_event(self.device, 0)

        # Allocate persistent DRAM tensor
        self.input_dram_tensor = ttnn.allocate_tensor_on_device(
            self.tt_inputs_host.shape,
            self.tt_inputs_host.dtype,
            self.tt_inputs_host.layout,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # First run (JIT compilation) - weights will be automatically prepared and cached on device
        logger.info("[JIT] Starting JIT compilation run")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.input_dram_tensor, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.input_dram_tensor
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()
        logger.info("[JIT] Compilation complete")

        # Optimized run - reuse cached prepared weights
        logger.info("[OPT] Starting optimized run")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.input_dram_tensor, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.input_dram_tensor
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()
        logger.info("[OPT] Optimized run complete")

        # Capture trace - weights should already be on device from previous runs
        logger.info("[TRACE] Starting trace capture")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.input_dram_tensor, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.input_dram_tensor  # Use DRAM tensor directly
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()

        trace_input_addr = self.runner_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        # Allocate persistent input tensor
        self.input_tensor = ttnn.allocate_tensor_on_device(self.runner_infra.input_tensor.spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        assert trace_input_addr == self.input_tensor.buffer_address()
        logger.info("[TRACE] Trace capture complete")

    def _execute_retinanet_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return {
            "regression": self.runner_infra.regression_output,
            "classification": self.runner_infra.classification_output,
        }

    def _validate(self):
        self.runner_infra.validate()

    def run(self, torch_input_tensor, check_pcc=False):
        n, c, h, w = torch_input_tensor.shape
        tt_inputs_host, _ = self.runner_infra._setup_l1_sharded_input(self.device, torch_input_tensor)
        output = self._execute_retinanet_trace_2cqs_inference(tt_inputs_host)
        if check_pcc:
            self._validate()
        return output

    def release(self):
        ttnn.release_trace(self.device, self.tid)
