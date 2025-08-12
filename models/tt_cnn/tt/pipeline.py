# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable, Optional

from loguru import logger

import ttnn

from .executor import (
    Executor,
    ModelExecutor,
    MultiCQTracedModelOverlappedInputExecutor,
    MultiCQTracedModelPipelinedIOExecutor,
    TracedModelExecutor,
)


def _determine_num_cores_for_even_sharding(shard_dim: int, max_cores: int):
    number_of_cores = max_cores
    while shard_dim % number_of_cores != 0:
        assert number_of_cores > 0, "Unable to find core grid"
        number_of_cores = number_of_cores - 1
    return number_of_cores


def get_memory_config_for_persistent_dram_tensor(shape, shard_strategy, dram_grid_size):
    if len(shape) < 2:
        raise ValueError("Shape must be 2D or higher (was {shape})")
    if dram_grid_size.y != 1:
        raise ValueError(f"Only 1D DRAM grid is supported (was {dram_grid_size})")

    # Force even shards because uneven width-sharding is not supported properly transferring from host (#22396)
    total_number_of_dram_cores = dram_grid_size.x
    shard_dim = shape[-1] if shard_strategy == ttnn.TensorMemoryLayout.WIDTH_SHARDED else shape[-2]
    dram_cores_for_even_sharding = _determine_num_cores_for_even_sharding(shard_dim, total_number_of_dram_cores)

    if shard_dim % dram_cores_for_even_sharding != 0:
        raise ValueError(
            f"Number of DRAM cores must evenly divide sharded tensor (was {shard_dim} and {dram_cores_for_even_sharding})"
        )

    if shard_strategy == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        output_dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores_for_even_sharding - 1, 0))}
            ),
            [shape[-2], shape[-1] // dram_cores_for_even_sharding],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, output_dram_shard_spec)
    elif shard_strategy == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        output_dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores_for_even_sharding - 1, 0))}
            ),
            [shape[-2] // dram_cores_for_even_sharding, shape[-1]],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, output_dram_shard_spec)
    else:
        raise ValueError(f"Unsupported shard strategy: {shard_strategy}")


@dataclass
class PipelineConfig:
    use_trace: bool = True
    num_command_queues: int = 1
    all_transfers_on_separate_command_queue: bool = False

    def __str__(self):
        return (
            f"PipelineConfig(\n"
            f"    use_trace: {self.use_trace}\n"
            f"    num_command_queues: {self.num_command_queues}\n"
            f"    all_transfers_on_separate_command_queue: {self.all_transfers_on_separate_command_queue}\n"
            f")"
        )


class Pipeline:
    def __init__(self, executor: Executor):
        self.executor = executor
        self.output_tensors = []
        self.preallocated_output_tensors = False

    def compile(self, host_input):
        logger.debug(f"Compiling pipeline model with input_shape={list(host_input.shape)}")
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
        logger.debug(f"Preallocating memory for {number_of_tensors_to_allocate} output tensors on host")
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
    l1_input_memory_config: ttnn.MemoryConfig,
    dram_input_memory_config: ttnn.MemoryConfig = None,
    dram_output_memory_config: Optional[ttnn.MemoryConfig] = None,
):
    logger.debug(f"Creating pipeline from config:\n{config}")

    executor = None
    if config.num_command_queues == 1:
        if config.use_trace:
            if not dram_input_memory_config:
                raise ValueError(
                    "dram_input_memory_config must be provided when using trace all_transfers_on_separate_command_queue=True."
                )
            logger.debug("Creating TracedModelExecutor with single command queue")
            executor = TracedModelExecutor(
                model,
                device,
                dram_input_memory_config=dram_input_memory_config,
                l1_input_memory_config=l1_input_memory_config,
            )
        else:
            logger.debug("Creating ModelExecutor with single command queue")
            executor = ModelExecutor(
                model,
                device,
                l1_input_memory_config=l1_input_memory_config,
            )
    elif config.num_command_queues == 2:
        if not config.use_trace:
            raise ValueError("Non-traced runs not supported when using multiple command queues")
        if config.all_transfers_on_separate_command_queue:
            if not dram_output_memory_config:
                raise ValueError(
                    "dram_output_memory_config must be provided when all_transfers_on_separate_command_queue=True."
                )
            executor = MultiCQTracedModelPipelinedIOExecutor(
                model,
                device,
                dram_input_memory_config=dram_input_memory_config,
                l1_input_memory_config=l1_input_memory_config,
                dram_output_memory_config=dram_output_memory_config,
            )
        else:
            executor = MultiCQTracedModelOverlappedInputExecutor(
                model,
                device,
                dram_input_memory_config=dram_input_memory_config,
                l1_input_memory_config=l1_input_memory_config,
            )
    else:
        raise ValueError(f"Unsupported pipeline configuration: {config}")

    if executor is None:
        raise ValueError(f"Could not create executor for configuration: {config}")

    logger.debug(f"Created executor with type={type(executor).__name__}")
    return pipeline(executor)
