# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch.nn.functional as F
import torch

import ttnn
from models.experimental.yolov8s_world.runner.performant_runner_infra import YOLOv8sWorldPerformanceRunnerInfra

from tests.ttnn.utils_for_testing import assert_with_pcc


def create_yolov8s_world_input_tensors(
    device, batch_size=1, input_channels=3, input_height=640, input_width=640, model=False
):
    torch_input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    memory_config = None
    if model:
        ttnn_input_tensor = F.pad(ttnn_input_tensor, (0, 29))
        memory_config = ttnn.create_sharded_memory_config(
            [6400, 32],
            core_grid=device.core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    memory_config = memory_config if memory_config else ttnn.L1_MEMORY_CONFIG
    ttnn_input_tensor = ttnn.from_torch(
        ttnn_input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config
    )
    return torch_input_tensor, ttnn_input_tensor


class YOLOv8sWorldPerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size=1,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        model_location_generator=None,
        resolution=(640, 640),
        torch_input_tensor=None,
    ):
        self.device = device
        self.resolution = resolution
        self.torch_input_tensor = torch_input_tensor
        self.runner_infra = YOLOv8sWorldPerformanceRunnerInfra(
            device,
            device_batch_size,
            act_dtype,
            weight_dtype,
            model_location_generator,
            resolution=resolution,
        )

        (
            self.tt_inputs_host,
            sharded_mem_config_DRAM,
            self.input_mem_config,
        ) = self.runner_infra.setup_dram_sharded_input(device)
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)

    def _capture_yolov8s_world_trace_2cqs(self):
        # Initialize the op event so we can write
        self.op_event = ttnn.record_event(self.device, 0)

        # First run configures convs JIT
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        spec = self.runner_infra.input_tensor.spec
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()

        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()

        # Capture
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()
        trace_input_addr = self.runner_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        assert trace_input_addr == self.input_tensor.buffer_address()

    def _execute_yolov8s_world_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        # TODO: Add in place support to ttnn to_memory_config
        if self.input_tensor.is_sharded():
            self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)
        return self.runner_infra.output_tensor

    def _validate(self, input_tensor, result_output_tensor):
        torch_output_tensor = self.runner_infra.torch_output_tensor
        assert_with_pcc(torch_output_tensor, result_output_tensor, 0.99)

    def run(self, torch_input_tensor, check_pcc=False):
        n, h, w, c = torch_input_tensor.shape
        torch_input_tensor = F.pad(torch_input_tensor, (0, 29), mode="constant", value=0)
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        output = self._execute_yolov8s_world_trace_2cqs_inference(tt_inputs_host)
        if check_pcc:
            torch_input_tensor = torch_input_tensor.reshape(n, h, w, c)
            torch_input_tensor = torch_input_tensor.permute(0, 3, 1, 2)
            self._validate(torch_input_tensor, output)

        return output

    def release(self):
        ttnn.release_trace(self.device, self.tid)
