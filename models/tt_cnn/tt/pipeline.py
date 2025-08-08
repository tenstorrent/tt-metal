# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable, Optional

import ttnn

from .executor import (
    Executor,
    MultiCQTracedModelExecutor,
    MultiCQTracedModelWithSeparateIOExecutor,
    TracedModelExecutor,
)


@dataclass
class PipelineConfig:
    use_trace: bool = True
    num_command_queues: int = 1
    separate_io_queue: bool = False


class Pipeline:
    def __init__(self, executor: Executor):
        self.executor = executor
        self.output_tensors = []
        self.preallocated_output_tensors = False

    def compile(self, host_input):
        self.executor.compile(host_input)
        return self

    def enqueue(self, host_inputs: list):
        if not all([input.storage_type() == ttnn.StorageType.HOST for input in host_inputs]):
            raise ValueError("All input tensors must be on host")
        if self.preallocated_output_tensors:
            if len(host_inputs) > len(self.output_tensors):
                raise ValueError(
                    f"Number of inputs ({len(host_inputs)}) exceeds the number of pre-allocated output tensors ({len(self.output_tensors)})."
                )
            for i, output_tensor in enumerate(self.executor.execute(host_inputs)):
                ttnn.copy_device_to_host_tensor(
                    output_tensor, self.output_tensors[i], blocking=False, cq_id=self.executor.get_read_cq()
                )
        else:
            self.output_tensors = []
            for t in self.executor.execute(host_inputs):
                host_tensor = t.cpu(blocking=False, cq_id=self.executor.get_read_cq())
                self.output_tensors.append(host_tensor)
        return self

    def preallocate_output_tensors_on_host(
        self, number_of_tensors_to_allocate, output_shape, output_dtype, output_layout
    ):
        self.preallocated_output_tensors = True
        if not isinstance(output_shape, ttnn.Shape):
            output_shape = ttnn.Shape(output_shape)
        self.output_tensors = [
            ttnn.allocate_tensor_on_host(output_shape, output_dtype, output_layout, self.executor.device)
            for _ in range(number_of_tensors_to_allocate)
        ]
        return self

    def pop_all(self):
        ttnn.synchronize_device(self.executor.device)
        return self.output_tensors

    def cleanup(self):
        self.executor.cleanup()


def pipeline(executor: Executor):
    return Pipeline(executor)


def create_pipeline_from_config(
    config: PipelineConfig,
    model: Callable,
    device,
    dram_input_memory_config: ttnn.MemoryConfig,
    l1_input_memory_config: ttnn.MemoryConfig,
    dram_output_memory_config: Optional[ttnn.MemoryConfig] = None,
    output_shape: Optional[tuple] = None,
    output_dtype: Optional[ttnn.DataType] = None,
):
    if not config.use_trace:
        raise NotImplementedError(
            "Non-traced executor is not currently implemented. All executors require trace=True in the current version."
        )

    executor = None
    if config.num_command_queues == 1:
        executor = TracedModelExecutor(
            model,
            device,
            dram_input_memory_config=dram_input_memory_config,
            l1_input_memory_config=l1_input_memory_config,
        )
    elif config.num_command_queues == 2:
        if config.separate_io_queue:
            if not all([dram_output_memory_config, output_shape, output_dtype]):
                raise ValueError(
                    "dram_output_memory_config, output_shape, and output_dtype must be provided for separate IO queue configuration."
                )
            executor = MultiCQTracedModelWithSeparateIOExecutor(
                model,
                device,
                dram_input_memory_config=dram_input_memory_config,
                l1_input_memory_config=l1_input_memory_config,
                dram_output_memory_config=dram_output_memory_config,
                output_shape=output_shape,
                output_dtype=output_dtype,
            )
        else:
            executor = MultiCQTracedModelExecutor(
                model,
                device,
                dram_input_memory_config=dram_input_memory_config,
                l1_input_memory_config=l1_input_memory_config,
            )
    else:
        raise ValueError(f"Unsupported pipeline configuration: {config}")

    if executor is None:
        raise ValueError(f"Could not create executor for configuration: {config}")

    return pipeline(executor)
