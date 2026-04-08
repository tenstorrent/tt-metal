# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov8l.runner.performant_runner_infra import YOLOv8lPerformanceRunnerInfra

try:
    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class YOLOv8lPerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size,
        inp_h=None,
        inp_w=None,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        mesh_mapper=None,
        mesh_composer=None,
        weights_mesh_mapper=None,
        model_location_generator=None,
    ):
        self.device = device
        self.mesh_mapper = mesh_mapper
        self.mesh_composer = mesh_composer
        self.weights_mesh_mapper = weights_mesh_mapper
        self.runner_infra = YOLOv8lPerformanceRunnerInfra(
            device,
            device_batch_size,
            inp_h=inp_h,
            inp_w=inp_w,
            mesh_mapper=self.mesh_mapper,
            mesh_composer=self.mesh_composer,
            weights_mesh_mapper=self.weights_mesh_mapper,
            model_location_generator=model_location_generator,
        )
        (
            self.tt_inputs_host,
            sharded_mem_config_DRAM,
            self.input_mem_config,
        ) = self.runner_infra._setup_dram_sharded_input(device)
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)
        self._capture_yolov8l_trace_2cqs()

    # def _convert_tensor_to_input_config(self, tensor):
    #     """Convert tensor to the appropriate memory configuration for input."""
    #     if hasattr(self.input_mem_config, "buffer_type") and self.input_mem_config.buffer_type == ttnn.BufferType.L1:
    #         if tensor.is_sharded():
    #             # Create an L1 sharded memory config matching the tensor shape
    #             l1_shard_spec = tensor.memory_config().shard_spec
    #             l1_sharded_config = ttnn.MemoryConfig(
    #                 memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    #                 buffer_type=ttnn.BufferType.L1,
    #                 shard_spec=l1_shard_spec
    #             )
    #             l1_tensor = ttnn.to_memory_config(tensor, l1_sharded_config)
    #             # Then convert to L1 interleaved
    #             return ttnn.to_memory_config(l1_tensor, ttnn.L1_MEMORY_CONFIG)
    #         else:
    #             return ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
    #     else:
    #         return ttnn.to_memory_config(tensor, self.input_mem_config)

    # def _convert_tensor_to_input_config(self, tensor):
    #     """Convert tensor to the appropriate memory configuration for input."""
    #     # Always use DRAM during trace capture to avoid L1 OOM
    #     return ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)

    def _convert_tensor_to_input_config(self, tensor):
        """Convert tensor to the appropriate memory configuration for input."""
        # Keep original DRAM-sharded format during trace capture
        return tensor

    # def _convert_tensor_to_input_config(self, tensor):
    #     """Convert tensor to the appropriate memory configuration for input."""
    #     # Use L1 for trace capture to ensure address consistency
    #     if tensor.is_sharded():
    #         # Create L1 sharded config matching the tensor shape
    #         l1_shard_spec = tensor.memory_config().shard_spec
    #         l1_sharded_config = ttnn.MemoryConfig(
    #             memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    #             buffer_type=ttnn.BufferType.L1,
    #             shard_spec=l1_shard_spec
    #         )
    #         l1_tensor = ttnn.to_memory_config(tensor, l1_sharded_config)
    #         return ttnn.to_memory_config(l1_tensor, ttnn.L1_MEMORY_CONFIG)
    #     else:
    #         return ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
    # def _convert_tensor_to_input_config(self, tensor):
    #     """Convert tensor to the appropriate memory configuration for input."""
    #     # Always use DRAM during trace capture to avoid L1 OOM
    #     return ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)

    # def _convert_tensor_to_input_config(self, tensor):
    #     """Convert tensor to the appropriate memory configuration for input."""
    #     if hasattr(self.input_mem_config, "buffer_type") and self.input_mem_config.buffer_type == ttnn.BufferType.L1:
    #         if tensor.is_sharded():
    #             try:
    #                 # Try L1 sharded first
    #                 l1_shard_spec = tensor.memory_config().shard_spec
    #                 l1_sharded_config = ttnn.MemoryConfig(
    #                     memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    #                     buffer_type=ttnn.BufferType.L1,
    #                     shard_spec=l1_shard_spec
    #                 )
    #                 l1_tensor = ttnn.to_memory_config(tensor, l1_sharded_config)
    #                 return ttnn.to_memory_config(l1_tensor, ttnn.L1_MEMORY_CONFIG)
    #             except RuntimeError as e:
    #                 if "Out of Memory" in str(e):
    #                     # Fall back to DRAM if L1 allocation fails
    #                     return ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
    #                 else:
    #                     raise
    #         else:
    #             return ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
    #     else:
    #         return ttnn.to_memory_config(tensor, self.input_mem_config)

    # def _convert_tensor_to_input_config(self, tensor):
    #     """Convert tensor to the appropriate memory configuration for input."""
    #     if hasattr(self.input_mem_config, "buffer_type") and self.input_mem_config.buffer_type == ttnn.BufferType.L1:
    #         if tensor.is_sharded():
    #             try:
    #                 # Try L1 sharded first
    #                 l1_shard_spec = tensor.memory_config().shard_spec
    #                 l1_sharded_config = ttnn.MemoryConfig(
    #                     memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    #                     buffer_type=ttnn.BufferType.L1,
    #                     shard_spec=l1_shard_spec
    #                 )
    #                 l1_tensor = ttnn.to_memory_config(tensor, l1_sharded_config)
    #                 return ttnn.to_memory_config(l1_tensor, ttnn.L1_MEMORY_CONFIG)
    #             except RuntimeError as e:
    #                 if "Out of Memory" in str(e):
    #                     # Convert DRAM-sharded to L1-sharded first, then to DRAM-interleaved
    #                     l1_shard_spec = tensor.memory_config().shard_spec
    #                     l1_sharded_config = ttnn.MemoryConfig(
    #                         memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    #                         buffer_type=ttnn.BufferType.L1,
    #                         shard_spec=l1_shard_spec
    #                     )
    #                     l1_tensor = ttnn.to_memory_config(tensor, l1_sharded_config)
    #                     return ttnn.to_memory_config(l1_tensor, ttnn.DRAM_MEMORY_CONFIG)
    #                 else:
    #                     raise
    #         else:
    #             return ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
    #     else:
    #         return ttnn.to_memory_config(tensor, self.input_mem_config)

    def _capture_yolov8l_trace_2cqs(self):
        # Initialize the op event so we can write
        self.op_event = ttnn.record_event(self.device, 0)

        # First run configures convs JIT
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        l1_spec = self.runner_infra.input_tensor.spec  # For first run
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()

        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self._convert_tensor_to_input_config(self.tt_image_res)
        dram_spec = self.runner_infra.input_tensor.spec  # For trace
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()

        # Capture
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self._convert_tensor_to_input_config(self.tt_image_res)
        self.op_event = ttnn.record_event(self.device, 0)
        # Deallocate output to ensure input gets same address after trace
        self.runner_infra.dealloc_output()
        trace_input_addr = self.runner_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(dram_spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        # assert trace_input_addr == self.input_tensor.buffer_address()

    def _execute_yolov8l_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        # TODO: Add in place support to ttnn to_memory_config
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)

        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)

        return self.runner_infra.output_tensor

    def run(self, torch_input_tensor):
        tt_inputs_host, _ = self.runner_infra._setup_l1_sharded_input(self.device, torch_input_tensor)
        return self._execute_yolov8l_trace_2cqs_inference(tt_inputs_host)

    def release(self):
        ttnn.release_trace(self.device, self.tid)
