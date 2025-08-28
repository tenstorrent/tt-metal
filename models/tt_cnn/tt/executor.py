# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Union

import ttnn


class Executor(ABC):
    @abstractmethod
    def compile(self, host_input):
        pass

    @abstractmethod
    def execute(self, host_inputs: list) -> Iterable[Union[ttnn.Tensor, list]]:
        pass

    @abstractmethod
    def cleanup(self):
        pass

    @abstractmethod
    def get_read_cq(self) -> int:
        pass

    @abstractmethod
    def get_output_schema(self):
        """
        Returns the schema of the model's output. The schema is a nested structure
        of lists/tuples mirroring the model's output, with tensors replaced by a
        (shape, dtype, layout) tuple.
        """


class ModelExecutor(Executor):
    def __init__(self, model: Callable, device, l1_input_memory_config, cq_id=0):
        """
        Executor that runs a model on a single command-queue.
        """
        self.model = model
        self.device = device
        self.cq_id = cq_id
        self.l1_input_memory_config = l1_input_memory_config
        self.output_schema = None

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
        Compiles the model by running it once.
        """
        self._validate_input(host_input)
        output_tensor = self._execute_single(host_input)
        self.output_schema = self._create_output_schema(output_tensor)
        self._deallocate_structured_tensor(output_tensor, force=True)
        ttnn.synchronize_device(self.device)

    def _validate_input(self, host_input):
        if host_input.storage_type() != ttnn.StorageType.HOST:
            raise ValueError("Input tensor must be on host")

    def _execute_single(self, input_tensor):
        l1_input_tensor = ttnn.to_device(input_tensor, device=self.device, memory_config=self.l1_input_memory_config)
        output_tensor = self.model(l1_input_tensor)
        if l1_input_tensor.is_allocated():
            ttnn.deallocate(l1_input_tensor, force=True)
        return output_tensor

    def execute(self, host_inputs: list) -> Iterable[ttnn.Tensor]:
        for host_input in host_inputs:
            yield self._execute_single(host_input)

    def cleanup(self):
        pass


class TracedModelExecutor(Executor):
    """
    Runs a model using tracing on a single command-queue.

    This executor compiles the model once to generate optimized kernels, then captures
    a trace of the execution for efficient repeated runs.

    Best suited for scenarios where input/output transfers are not the bottleneck
    and simplicity is preferred over maximum throughput.
    """

    def __init__(self, model: Callable, device, dram_input_memory_config, l1_input_memory_config, cq_id=0):
        self.model = model
        self.device = device
        self.cq_id = cq_id

        self.dram_input_memory_config = dram_input_memory_config
        self.l1_input_memory_config = l1_input_memory_config

        self.dram_input_tensor = None
        self.l1_input_tensor = None
        self.output_tensor = None
        self._compilation_output_tensor = None

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
        self.dram_input_tensor = ttnn.allocate_tensor_on_device(
            host_input.shape, host_input.dtype, host_input.layout, self.device, self.dram_input_memory_config
        )

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
        Transfers host input to device DRAM and reshards to L1 memory.
        """
        self._validate_dram_input_tensor()
        ttnn.copy_host_to_device_tensor(host_input, self.dram_input_tensor, cq_id=self.cq_id)
        return ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)

    def _validate_dram_input_tensor(self):
        if self.dram_input_tensor is None:
            raise RuntimeError("DRAM input tensor is not allocated")
        if self.dram_input_tensor.storage_type() != ttnn.StorageType.DEVICE:
            raise RuntimeError("DRAM input tensor must be on device")
        if self.dram_input_tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
            raise RuntimeError("DRAM input tensor must be in DRAM memory")

    def _execute_single(self, input_tensor):
        """
        Executes the traced model for a single input tensor.
        """
        # Transfer input to device and reshard to L1
        ttnn.copy_host_to_device_tensor(input_tensor, self.dram_input_tensor, cq_id=self.cq_id)

        # Reuse persistent L1 tensor by resharding in-place
        self.l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config, self.l1_input_tensor)

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


class MultiCQModelOverlappedInputExecutor(Executor):
    """
    Multi-command queue model executor that overlaps input transfers with model execution
    for improved throughput without using tracing.

    Uses two command queues:
    - CQ_OPS_AND_OUTPUT_READ (0): Model operations and output reading
    - CQ_INPUT_WRITE (1): Input tensor writing to device

    This executor can transfer the next input while the model is processing the current input,
    reducing the impact of input transfer latency on overall throughput.
    """

    CQ_OPS_AND_OUTPUT_READ = 0
    CQ_INPUT_WRITE = 1

    def __init__(
        self,
        model: Callable,
        device,
        dram_input_memory_config,
        l1_input_memory_config,
    ):
        self.model = model
        self.device = device
        self.dram_input_memory_config = dram_input_memory_config
        self.l1_input_memory_config = l1_input_memory_config
        self.output_schema = None

        self.dram_input_tensor = None
        self.op_event = None

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
        return self.CQ_OPS_AND_OUTPUT_READ

    def compile(self, host_input):
        """
        Compiles the model by running it once to generate kernels using multi-command queue synchronization.
        """
        if host_input.storage_type() != ttnn.StorageType.HOST:
            raise ValueError("Input tensor must be on host")

        self.dram_input_tensor = ttnn.allocate_tensor_on_device(
            host_input.shape, host_input.dtype, host_input.layout, self.device, self.dram_input_memory_config
        )

        self.op_event = ttnn.record_event(self.device, self.CQ_OPS_AND_OUTPUT_READ)

        compilation_output = self._compile_model(host_input)
        self.output_schema = self._create_output_schema(compilation_output)
        self._deallocate_structured_tensor(compilation_output, force=True)
        ttnn.synchronize_device(self.device)

    def _compile_model(self, host_input):
        """
        Runs the model once to compile kernels.
        """
        # Transfer input to device DRAM
        ttnn.wait_for_event(self.CQ_INPUT_WRITE, self.op_event)
        ttnn.copy_host_to_device_tensor(host_input, self.dram_input_tensor, cq_id=self.CQ_INPUT_WRITE)
        write_event = ttnn.record_event(self.device, self.CQ_INPUT_WRITE)

        # Reshard to L1 and run model
        ttnn.wait_for_event(self.CQ_OPS_AND_OUTPUT_READ, write_event)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)
        output_tensor = self.model(l1_input_tensor)
        self.op_event = ttnn.record_event(self.device, self.CQ_OPS_AND_OUTPUT_READ)
        if l1_input_tensor.is_allocated():
            ttnn.deallocate(l1_input_tensor, force=True)
        return output_tensor

    def _execute_single(self, input_tensor):
        """
        Executes the model for a single input tensor using multi-CQ synchronization.
        Returns the output tensor (result available after synchronization).
        """
        # Transfer input to device DRAM
        ttnn.wait_for_event(self.CQ_INPUT_WRITE, self.op_event)
        ttnn.copy_host_to_device_tensor(input_tensor, self.dram_input_tensor, cq_id=self.CQ_INPUT_WRITE)
        write_event = ttnn.record_event(self.device, self.CQ_INPUT_WRITE)

        # Execute model
        ttnn.wait_for_event(self.CQ_OPS_AND_OUTPUT_READ, write_event)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)
        output_tensor = self.model(l1_input_tensor)
        self.op_event = ttnn.record_event(self.device, self.CQ_OPS_AND_OUTPUT_READ)
        if l1_input_tensor.is_allocated():
            ttnn.deallocate(l1_input_tensor, force=True)

        return output_tensor

    def execute(self, host_inputs: list) -> Iterable[ttnn.Tensor]:
        """Executes the model for a batch of input tensors."""
        for host_input in host_inputs:
            yield self._execute_single(host_input)

    def cleanup(self):
        """No trace to release for non-traced executor."""


class MultiCQTracedModelOverlappedInputExecutor(Executor):
    """
    Multi-command queue traced model executor that overlaps input transfers with model execution
    for improved throughput.

    Uses two command queues:
    - CQ_OPS_AND_OUTPUT_READ (0): Model operations and output reading
    - CQ_INPUT_WRITE (1): Input tensor writing to device

    This executor can transfer the next input while the model is processing the current input,
    reducing the impact of input transfer latency on overall throughput.
    """

    CQ_OPS_AND_OUTPUT_READ = 0
    CQ_INPUT_WRITE = 1

    def __init__(
        self,
        model: Callable,
        device,
        dram_input_memory_config,
        l1_input_memory_config,
    ):
        self.model = model
        self.device = device
        self.dram_input_memory_config = dram_input_memory_config
        self.l1_input_memory_config = l1_input_memory_config
        self.output_schema = None

        self.dram_input_tensor = None
        self.l1_input_tensor = None
        self.output_tensor = None
        self._compilation_output_tensor = None

        self.trace_id = None
        self.op_event = None

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
        return self.CQ_OPS_AND_OUTPUT_READ

    def compile(self, host_input):
        """
        Compiles the model by running it once to generate kernels, then capturing a trace
        for efficient repeated execution using multi-command queue synchronization.
        """
        if host_input.storage_type() != ttnn.StorageType.HOST:
            raise ValueError("Input tensor must be on host")

        self.dram_input_tensor = ttnn.allocate_tensor_on_device(
            host_input.shape, host_input.dtype, host_input.layout, self.device, self.dram_input_memory_config
        )

        self.op_event = ttnn.record_event(self.device, self.CQ_OPS_AND_OUTPUT_READ)

        self._compile_model(host_input)
        self.output_schema = self._create_output_schema(self._compilation_output_tensor)
        self._capture_trace(host_input)
        ttnn.synchronize_device(self.device)

    def _compile_model(self, host_input):
        """
        Runs the model once to compile kernels.
        """
        # Transfer input to device DRAM
        ttnn.wait_for_event(self.CQ_INPUT_WRITE, self.op_event)
        ttnn.copy_host_to_device_tensor(host_input, self.dram_input_tensor, cq_id=self.CQ_INPUT_WRITE)
        write_event = ttnn.record_event(self.device, self.CQ_INPUT_WRITE)

        # Reshard to L1 and run model
        ttnn.wait_for_event(self.CQ_OPS_AND_OUTPUT_READ, write_event)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)
        self.op_event = ttnn.record_event(self.device, self.CQ_OPS_AND_OUTPUT_READ)

        self._compilation_output_tensor = self.model(l1_input_tensor)

        # Cleanup L1 input tensor only
        ttnn.deallocate(l1_input_tensor)

    def _capture_trace(self, host_input):
        """
        Captures a trace of the model execution for efficient repeated runs.
        Sets up persistent L1 input tensor and validates memory addresses.
        """
        # Transfer input to device DRAM
        ttnn.wait_for_event(self.CQ_INPUT_WRITE, self.op_event)
        ttnn.copy_host_to_device_tensor(host_input, self.dram_input_tensor, cq_id=self.CQ_INPUT_WRITE)
        write_event = ttnn.record_event(self.device, self.CQ_INPUT_WRITE)

        # Prepare L1 input for trace
        ttnn.wait_for_event(self.CQ_OPS_AND_OUTPUT_READ, write_event)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)
        ttnn.record_event(self.device, self.CQ_OPS_AND_OUTPUT_READ)

        # Record tensor details for validation
        input_trace_addr = l1_input_tensor.buffer_address()
        spec = l1_input_tensor.spec

        # Force cleanup of the compilation output tensor to ensure that the subsequent allocation
        # of the persistent L1 input tensor occurs at the expected device memory address. This is
        # necessary because trace capture relies on the L1 input tensor being allocated at the same
        # address as during the initial trace.
        self._deallocate_structured_tensor(self._compilation_output_tensor, force=True)

        # Capture trace
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=self.CQ_OPS_AND_OUTPUT_READ)
        self.output_tensor = self.model(l1_input_tensor)
        if l1_input_tensor.is_allocated():
            ttnn.deallocate(l1_input_tensor, force=True)

        # Allocate persistent L1 input tensor and validate address
        self.l1_input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        actual_addr = self.l1_input_tensor.buffer_address()
        if input_trace_addr != actual_addr:
            raise RuntimeError(f"L1 input tensor address mismatch: expected {input_trace_addr}, got {actual_addr}")

        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=self.CQ_OPS_AND_OUTPUT_READ)
        self.op_event = ttnn.record_event(self.device, self.CQ_OPS_AND_OUTPUT_READ)

    def _execute_single(self, input_tensor):
        """
        Executes the traced model for a single input tensor using multi-CQ synchronization.
        Returns the output tensor (result available after synchronization).
        """
        # Transfer input to device DRAM
        ttnn.wait_for_event(self.CQ_INPUT_WRITE, self.op_event)
        ttnn.copy_host_to_device_tensor(input_tensor, self.dram_input_tensor, cq_id=self.CQ_INPUT_WRITE)
        write_event = ttnn.record_event(self.device, self.CQ_INPUT_WRITE)

        # Execute traced model
        ttnn.wait_for_event(self.CQ_OPS_AND_OUTPUT_READ, write_event)
        self.l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config, self.l1_input_tensor)
        self.op_event = ttnn.record_event(self.device, self.CQ_OPS_AND_OUTPUT_READ)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=self.CQ_OPS_AND_OUTPUT_READ, blocking=False)

        return self.output_tensor

    def execute(self, host_inputs: list) -> Iterable[ttnn.Tensor]:
        """Executes the traced model for a batch of input tensors."""
        for host_input in host_inputs:
            yield self._execute_single(host_input)

    def cleanup(self):
        """Releases the captured trace to free device resources."""
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)


class MultiCQTracedModelPipelinedIOExecutor(Executor):
    """
    Multi-command queue traced model executor that uses separate queues for I/O and operations.

    This executor uses two command queues to enable parallel execution:
    - CQ_OPS (0): Model operations and compute
    - CQ_IO (1): Input/output tensor transfers

    This design allows for simultaneous execution of input/output tensor transfers and model
    execution, which can improve throughput when the input/output tensor transfers are large.
    Requires at least 2 inputs for proper pipelining.
    """

    CQ_OPS = 0
    CQ_IO = 1

    def __init__(
        self,
        model: Callable,
        device,
        dram_input_memory_config,
        l1_input_memory_config,
        dram_output_memory_config,
    ):
        self.model = model
        self.device = device

        self.dram_input_memory_config = dram_input_memory_config
        self.l1_input_memory_config = l1_input_memory_config
        self.dram_output_memory_config = dram_output_memory_config
        self.output_schema = None

        self.dram_input_tensor = None
        self.dram_output_tensor = None
        self.l1_input_tensor = None
        self.output_tensor = None
        self._compilation_output_tensor = None

        self.trace_id = None
        self.first_op_event = None
        self.read_event = None
        self.write_event = None
        self.last_op_event = None

    def get_output_schema(self):
        if self.output_schema is None:
            raise RuntimeError("Executor must be compiled before getting the output schema.")
        return self.output_schema

    def _create_output_schema(self, output):
        if isinstance(output, ttnn.Tensor):
            return (output.shape, output.dtype, output.layout)
        elif isinstance(output, (list, tuple)):
            raise TypeError("MultiCQTracedModelPipelinedIOExecutor does not support multiple outputs.")
        else:
            raise TypeError(f"Unsupported type in model output: {type(output)}")

    def _deallocate_structured_tensor(self, tensor_struct, force=False):
        if isinstance(tensor_struct, ttnn.Tensor):
            if tensor_struct.is_allocated():
                ttnn.deallocate(tensor_struct, force=force)
        elif isinstance(tensor_struct, (list, tuple)):
            raise TypeError("MultiCQTracedModelPipelinedIOExecutor does not support multiple outputs.")

    def get_read_cq(self):
        return self.CQ_IO

    def compile(self, host_input):
        """
        Compiles the model with pipelined I/O by running it once to generate kernels,
        then capturing a trace for efficient repeated execution using multi-command queue
        synchronization with pipelined I/O operations.
        """
        if host_input.storage_type() != ttnn.StorageType.HOST:
            raise ValueError("Input tensor must be on host")

        self.dram_input_tensor = ttnn.allocate_tensor_on_device(
            host_input.shape, host_input.dtype, host_input.layout, self.device, self.dram_input_memory_config
        )

        self.first_op_event = ttnn.record_event(self.device, self.CQ_OPS)
        self.read_event = ttnn.record_event(self.device, self.CQ_IO)

        self._compile_model(host_input)
        self.output_schema = self._create_output_schema(self._compilation_output_tensor)

        output_shape, output_dtype, _ = self.output_schema

        self.dram_output_tensor = ttnn.allocate_tensor_on_device(
            output_shape, output_dtype, ttnn.ROW_MAJOR_LAYOUT, self.device, self.dram_output_memory_config
        )

        self._capture_trace(host_input)
        ttnn.synchronize_device(self.device)

    def _compile_model(self, host_input):
        """
        Runs the model once to compile kernels and determine output shape/dtype.
        Returns: (output_shape, output_dtype)
        """
        # Transfer input to DRAM
        ttnn.wait_for_event(self.CQ_IO, self.first_op_event)
        ttnn.copy_host_to_device_tensor(host_input, self.dram_input_tensor, cq_id=self.CQ_IO)
        self.write_event = ttnn.record_event(self.device, self.CQ_IO)

        # Execute model and get output shape/dtype
        ttnn.wait_for_event(self.CQ_OPS, self.write_event)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)
        self.first_op_event = ttnn.record_event(self.device, self.CQ_OPS)
        self._compilation_output_tensor = self.model(l1_input_tensor)

        # Transfer output to DRAM
        ttnn.wait_for_event(self.CQ_OPS, self.read_event)

        ttnn.reshard(
            self._compilation_output_tensor, self.dram_output_memory_config, output_tensor=self.dram_output_tensor
        )
        self.last_op_event = ttnn.record_event(self.device, self.CQ_OPS)

        # Cleanup compilation tensors
        ttnn.deallocate(l1_input_tensor)

    def _capture_trace(self, host_input):
        """
        Captures a trace of the model execution for efficient repeated runs.
        Sets up persistent L1 input tensor and validates memory addresses.
        """
        # Transfer input to DRAM and then reshard to L1
        ttnn.wait_for_event(self.CQ_IO, self.first_op_event)
        ttnn.copy_host_to_device_tensor(host_input, self.dram_input_tensor, cq_id=self.CQ_IO)
        self.write_event = ttnn.record_event(self.device, self.CQ_IO)
        ttnn.wait_for_event(self.CQ_OPS, self.write_event)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)

        input_trace_addr = l1_input_tensor.buffer_address()
        spec = l1_input_tensor.spec

        # Force cleanup of the compilation output tensor to ensure that the subsequent allocation
        # of the persistent L1 input tensor occurs at the expected device memory address. This is
        # necessary because trace capture relies on the L1 input tensor being allocated at the same
        # address as during the initial trace.
        self._deallocate_structured_tensor(self._compilation_output_tensor, force=True)

        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=self.CQ_OPS)
        self.output_tensor = self.model(l1_input_tensor)
        if l1_input_tensor.is_allocated():
            ttnn.deallocate(l1_input_tensor, force=True)

        # Allocate persistent L1 input tensor and validate address
        self.l1_input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        actual_addr = self.l1_input_tensor.buffer_address()
        if input_trace_addr != actual_addr:
            raise RuntimeError(f"L1 input tensor address mismatch: expected {input_trace_addr}, got {actual_addr}")

        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=self.CQ_OPS)

    def _execute_model_and_transfer_output(self):
        """
        Executes the traced model on current input and transfers output to DRAM.
        This is the core execution step that gets pipelined with I/O operations.
        """
        ttnn.wait_for_event(self.CQ_OPS, self.write_event)
        self.l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config, self.l1_input_tensor)
        self.first_op_event = ttnn.record_event(self.device, self.CQ_OPS)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=self.CQ_OPS, blocking=False)

        ttnn.wait_for_event(self.CQ_OPS, self.read_event)
        ttnn.reshard(self.output_tensor, self.dram_output_memory_config, output_tensor=self.dram_output_tensor)
        self.last_op_event = ttnn.record_event(self.device, self.CQ_OPS)

    def _transfer_input_to_device(self, input_tensor):
        """Transfers input tensor to device DRAM using I/O queue."""
        ttnn.wait_for_event(self.CQ_IO, self.first_op_event)
        ttnn.copy_host_to_device_tensor(input_tensor, self.dram_input_tensor, cq_id=self.CQ_IO)
        self.write_event = ttnn.record_event(self.device, self.CQ_IO)

    def _read_output_from_device(self):
        """Reads output tensor from device DRAM using I/O queue."""
        ttnn.wait_for_event(self.CQ_IO, self.last_op_event)
        output = self.dram_output_tensor
        self.read_event = ttnn.record_event(self.device, self.CQ_IO)
        return output

    def execute(self, host_inputs: list) -> Iterable[ttnn.Tensor]:
        """
        Executes the traced model for a batch of input tensors using pipelined I/O.

        This implementation requires at least 2 inputs to enable proper pipelining:
        - First input is transferred to device
        - For inputs 1 to N-1: process current input while transferring next input
        - For input N: process final input and read its output
        """
        num_inputs = len(host_inputs)
        if num_inputs < 2:
            raise ValueError(f"Pipelined I/O executor requires at least 2 inputs for pipelining (got {num_inputs})")

        self._transfer_input_to_device(host_inputs[0])

        # Process inputs 1 to N-1 with pipelining
        for i in range(1, num_inputs):
            self._execute_model_and_transfer_output()
            self._transfer_input_to_device(host_inputs[i])
            yield self._read_output_from_device()

        self._execute_model_and_transfer_output()
        yield self._read_output_from_device()

    def cleanup(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
