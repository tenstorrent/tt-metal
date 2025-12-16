# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Iterable, Optional

from loguru import logger

import ttnn

from models.tt_cnn.tt.executor import (
    Executor,
    ModelExecutor,
    MultiCQModelOverlappedInputExecutor,
    MultiCQTracedModelOverlappedInputExecutor,
    MultiCQTracedModelPipelinedIOExecutor,
)
from models.tt_cnn.tt.pipeline import PipelineConfig, pipeline


class CustomTracedModelExecutor(Executor):
    """
    Custom TracedModelExecutor that transfers host input directly to L1 memory
    instead of going through DRAM. This is a custom implementation with modified
    behavior compared to the standard TracedModelExecutor.

    Runs a model using tracing on a single command-queue.

    This executor compiles the model once to generate optimized kernels, then captures
    a trace of the execution for efficient repeated runs.

    Best suited for scenarios where input/output transfers are not the bottleneck
    and simplicity is preferred over maximum throughput.
    """

    def __init__(self, model: Callable, device, l1_input_memory_config, cq_id=0):
        self.model = model
        self.device = device
        self.cq_id = cq_id

        self.l1_input_memory_config = l1_input_memory_config

        self.l1_input_tensor = None
        self.output_tensor = None
        self._compilation_output_tensor = None
        self.output_schema = None

        self.trace_id = None
        self.input_trace_addr = None

    def get_output_schema(self):
        if self.output_schema is None:
            raise RuntimeError("Executor must be compiled before getting the output schema.")
        return self.output_schema

    def _create_output_schema(self, output):
        if isinstance(output, ttnn.Tensor):
            return (output.shape, output.dtype, output.layout)
        elif isinstance(output, (list, tuple)):
            return [self._create_output_schema(t) for t in output]
        else:
            raise TypeError(f"Unsupported type in model output: {type(output)}")

    def _deallocate_structured_tensor(self, tensor_struct, force=False):
        if isinstance(tensor_struct, ttnn.Tensor):
            if tensor_struct.is_allocated():
                ttnn.deallocate(tensor_struct, force=force)
        elif isinstance(tensor_struct, (list, tuple)):
            for t in tensor_struct:
                self._deallocate_structured_tensor(t, force=force)

    def get_read_cq(self):
        return self.cq_id

    def compile(self, host_input):
        """
        Compiles the model by running it once and then captures a trace.
        """
        self._validate_input(host_input)
        self._allocate_persistent_tensors(host_input)
        self._run_model_for_compilation(host_input)
        self.output_schema = self._create_output_schema(self._compilation_output_tensor)
        self._capture_execution_trace(host_input)
        ttnn.synchronize_device(self.device)

    def _validate_input(self, host_input):
        if host_input.storage_type() != ttnn.StorageType.HOST:
            raise ValueError("Input tensor must be on host")

    def _allocate_persistent_tensors(self, host_input):
        # No persistent tensors needed at this stage - L1 tensor allocated after trace capture
        pass

    def _run_model_for_compilation(self, host_input):
        l1_input_for_compile = self._prepare_l1_input(host_input)
        self._compilation_output_tensor = self.model(l1_input_for_compile)
        ttnn.deallocate(l1_input_for_compile)

    def _capture_execution_trace(self, host_input):
        l1_input_for_trace, spec = self._prepare_input_for_trace_capture(host_input)

        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=self.cq_id)

        self.output_tensor = self.model(l1_input_for_trace)
        ttnn.deallocate(l1_input_for_trace)

        self._validate_trace_address_consistency(spec)
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=self.cq_id)

    def _prepare_input_for_trace_capture(self, host_input):
        l1_input_for_trace = self._prepare_l1_input(host_input)
        self.input_trace_addr = l1_input_for_trace.buffer_address()
        spec = l1_input_for_trace.spec

        self._deallocate_structured_tensor(self._compilation_output_tensor, force=True)

        return l1_input_for_trace, spec

    def _validate_trace_address_consistency(self, spec):
        """
        Validates that allocating a persistent L1 tensor will use the expected address
        that was captured during trace recording.
        """
        self.l1_input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        actual_addr = self.l1_input_tensor.buffer_address()

        if self.input_trace_addr != actual_addr:
            raise RuntimeError(
                f"L1 input tensor address mismatch: trace captured {self.input_trace_addr}, "
                f"but persistent tensor allocated at {actual_addr}"
            )

    def _prepare_l1_input(self, host_input):
        """
        Transfers host input directly to L1 memory.
        """
        return ttnn.to_device(host_input, device=self.device, memory_config=self.l1_input_memory_config)

    def _execute_single(self, input_tensor):
        """
        Executes the traced model for a single input tensor.
        """
        # Transfer host input directly to L1
        ttnn.copy_host_to_device_tensor(input_tensor, self.l1_input_tensor, cq_id=self.cq_id)

        # Validate address consistency with captured trace
        actual_addr = self.l1_input_tensor.buffer_address()
        if actual_addr != self.input_trace_addr:
            raise RuntimeError(
                f"L1 input tensor address mismatch during execution: "
                f"expected {self.input_trace_addr}, got {actual_addr}"
            )

        ttnn.execute_trace(self.device, self.trace_id, cq_id=self.cq_id, blocking=False)
        return self.output_tensor

    def execute(self, host_inputs: list) -> Iterable[ttnn.Tensor]:
        """
        Executes the traced model for a batch of input tensors.
        """
        for host_input in host_inputs:
            yield self._execute_single(host_input)

    def cleanup(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)


def create_pipeline_from_config(
    config: PipelineConfig,
    model: Callable,
    device,
    l1_input_memory_config: ttnn.MemoryConfig,
    dram_input_memory_config: ttnn.MemoryConfig = None,
    dram_output_memory_config: Optional[ttnn.MemoryConfig] = None,
):
    """
    Custom create_pipeline_from_config that uses CustomTracedModelExecutor
    for traced single command queue configurations.

    For other configurations, this falls back to the standard pipeline creation.
    """
    logger.debug(f"Creating pipeline from config:\n{config}")

    executor = None
    if config.num_command_queues == 1:
        if config.use_trace:
            logger.debug("Creating CustomTracedModelExecutor with single command queue")
            executor = CustomTracedModelExecutor(
                model,
                device,
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
            # Non-traced multi-CQ executor that overlaps input transfers
            if config.all_transfers_on_separate_command_queue:
                raise ValueError("Pipelined I/O (all_transfers_on_separate_command_queue) requires tracing")
            if not dram_input_memory_config:
                raise ValueError("dram_input_memory_config must be provided when using multi-CQ executor")
            logger.debug("Creating MultiCQModelOverlappedInputExecutor with multiple command queues (non-traced)")
            executor = MultiCQModelOverlappedInputExecutor(
                model,
                device,
                dram_input_memory_config=dram_input_memory_config,
                l1_input_memory_config=l1_input_memory_config,
            )
        elif config.all_transfers_on_separate_command_queue:
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
