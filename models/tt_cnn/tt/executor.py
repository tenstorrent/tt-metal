# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Callable, Iterable

import ttnn


class Executor(ABC):
    @abstractmethod
    def compile(self, host_input):
        pass

    @abstractmethod
    def execute(self, host_inputs: list) -> Iterable[ttnn.Tensor]:
        pass

    @abstractmethod
    def cleanup(self):
        pass

    @abstractmethod
    def get_read_cq(self) -> int:
        pass


class ModelExecutor(Executor):
    def __init__(self, model: Callable, device, l1_input_memory_config, cq_id=0):
        """
        Executor that runs a model on a single command-queue.
        """
        self.model = model
        self.device = device
        self.cq_id = cq_id
        self.l1_input_memory_config = l1_input_memory_config

    def get_read_cq(self):
        return self.cq_id

    def compile(self, host_input):
        """
        Compiles the model by running it once.
        """
        self._validate_input(host_input)
        self._execute_single(host_input)
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

    def get_read_cq(self):
        return self.cq_id

    def compile(self, host_input):
        """
        Compiles the model by running it once and then captures a trace.
        """
        self._validate_input(host_input)
        self._allocate_persistent_tensors(host_input)
        self._run_model_for_compilation(host_input)
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

        self._compilation_output_tensor.deallocate(force=True)

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

        self.dram_input_tensor = None
        self.l1_input_tensor = None
        self.output_tensor = None
        self._compilation_output_tensor = None

        self.trace_id = None
        self.op_event = None

    def get_read_cq(self):
        return self.CQ_OPS_AND_OUTPUT_READ

    def compile(self, host_input):
        """
        Compiles the model by running it once to generate kernels, then capturing a trace
        for efficient repeated execution using multi-command queue synchronization.
        """
        self._validate_input(host_input)
        self._allocate_persistent_tensors(host_input)

        op_event = ttnn.record_event(self.device, self.CQ_OPS_AND_OUTPUT_READ)
        op_event = self._run_model_for_compilation(host_input, op_event)
        self._capture_execution_trace(host_input, op_event)

        ttnn.synchronize_device(self.device)

    def _validate_input(self, host_input):
        """Validates that the input tensor is on host memory."""
        if host_input.storage_type() != ttnn.StorageType.HOST:
            raise ValueError("Input tensor must be on host")

    def _allocate_persistent_tensors(self, host_input):
        """Allocates persistent DRAM input tensor that will be reused across executions."""
        self.dram_input_tensor = ttnn.allocate_tensor_on_device(
            host_input.shape, host_input.dtype, host_input.layout, self.device, self.dram_input_memory_config
        )

    def _run_model_for_compilation(self, host_input, op_event):
        """
        Runs the model once to compile it, using multi-CQ synchronization pattern.
        Returns the final op_event for subsequent trace capture.
        """
        # Transfer input from host to device DRAM (CQ 1)
        write_event = self._transfer_input_to_device(host_input, op_event)

        # Reshard to L1 and run model (CQ 0)
        op_event = self._execute_model_operations(write_event)

        return op_event

    def _capture_execution_trace(self, host_input, op_event):
        """
        Captures a trace of the model execution for efficient repeated runs.
        Validates that persistent tensors are allocated at expected addresses.
        """
        # Prepare input for trace capture
        write_event = self._transfer_input_to_device(host_input, op_event)
        l1_input_tensor, input_trace_addr, spec = self._prepare_trace_input(write_event)

        # Deallocate the previous output tensor to ensure correct memory allocation order
        # This ensures our persistent input tensor gets allocated at the right address
        # Note: This refers to output_tensor from _run_model_for_compilation
        ttnn.deallocate(self._compilation_output_tensor, force=True)

        # Begin trace capture and run model
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=self.CQ_OPS_AND_OUTPUT_READ)
        self.output_tensor = self.model(l1_input_tensor)
        if l1_input_tensor.is_allocated():
            ttnn.deallocate(l1_input_tensor, force=True)  # Needed if model does not deallocate its input

        # Allocate persistent L1 input tensor and validate address consistency
        self._allocate_and_validate_l1_tensor(spec, input_trace_addr)

        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=self.CQ_OPS_AND_OUTPUT_READ)
        self.op_event = ttnn.record_event(self.device, self.CQ_OPS_AND_OUTPUT_READ)

    def _transfer_input_to_device(self, host_input, op_event):
        """Transfers input from host to device DRAM using input write queue."""
        ttnn.wait_for_event(self.CQ_INPUT_WRITE, op_event)
        ttnn.copy_host_to_device_tensor(host_input, self.dram_input_tensor, cq_id=self.CQ_INPUT_WRITE)
        return ttnn.record_event(self.device, self.CQ_INPUT_WRITE)

    def _execute_model_operations(self, write_event):
        """Executes model operations on the ops queue after input is ready."""
        ttnn.wait_for_event(self.CQ_OPS_AND_OUTPUT_READ, write_event)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)
        op_event = ttnn.record_event(self.device, self.CQ_OPS_AND_OUTPUT_READ)

        self._compilation_output_tensor = self.model(l1_input_tensor)
        ttnn.deallocate(l1_input_tensor)

        return op_event

    def _prepare_trace_input(self, write_event):
        """Prepares L1 input tensor for trace capture and records its address."""
        ttnn.wait_for_event(self.CQ_OPS_AND_OUTPUT_READ, write_event)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)
        ttnn.record_event(self.device, self.CQ_OPS_AND_OUTPUT_READ)

        # Record tensor details for validation after trace capture
        input_trace_addr = l1_input_tensor.buffer_address()
        spec = l1_input_tensor.spec

        return l1_input_tensor, input_trace_addr, spec

    def _allocate_and_validate_l1_tensor(self, spec, expected_addr):
        """Allocates persistent L1 input tensor and validates it uses the expected address."""
        self.l1_input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        actual_addr = self.l1_input_tensor.buffer_address()

        if expected_addr != actual_addr:
            raise RuntimeError(f"L1 input tensor address mismatch: expected {expected_addr}, got {actual_addr}")

    def _execute_single(self, input_tensor):
        """
        Executes the traced model for a single input tensor using multi-CQ synchronization.
        Returns the output tensor (result available after synchronization).
        """
        # Wait for previous operation to complete then write new input
        ttnn.wait_for_event(self.CQ_INPUT_WRITE, self.op_event)
        ttnn.copy_host_to_device_tensor(input_tensor, self.dram_input_tensor, cq_id=self.CQ_INPUT_WRITE)
        write_event = ttnn.record_event(self.device, self.CQ_INPUT_WRITE)

        # Wait for input write, then execute operations
        ttnn.wait_for_event(self.CQ_OPS_AND_OUTPUT_READ, write_event)

        # Reuse persistent L1 tensor by resharding in-place
        self.l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config, self.l1_input_tensor)

        # Signal completion to input writer and execute traced model
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

    def get_read_cq(self):
        return self.CQ_IO

    def compile(self, host_input):
        """
        Compiles the model with pipelined I/O by running it once to generate kernels,
        then capturing a trace for efficient repeated execution using multi-command queue
        synchronization with pipelined I/O operations.
        """
        self._validate_input(host_input)
        self._allocate_persistent_input_tensor(host_input)

        first_op_event, read_event = self._initialize_synchronization_events()

        (
            first_op_event,
            write_event,
            last_op_event,
            output_shape,
            output_dtype,
        ) = self._run_model_for_compilation_with_io(host_input, first_op_event, read_event)

        # Allocate output tensor based on inferred shape and dtype from compilation run
        self._allocate_persistent_output_tensor(output_shape, output_dtype)

        self._capture_execution_trace_with_io(host_input, first_op_event, write_event)

        self._store_event_state(first_op_event, read_event, write_event, last_op_event)
        ttnn.synchronize_device(self.device)

    def _validate_input(self, host_input):
        """Validates that the input tensor is on host memory."""
        if host_input.storage_type() != ttnn.StorageType.HOST:
            raise ValueError("Input tensor must be on host")

    def _allocate_persistent_input_tensor(self, host_input):
        """Allocates persistent DRAM input tensor for reuse across executions."""
        self.dram_input_tensor = ttnn.allocate_tensor_on_device(
            host_input.shape, host_input.dtype, host_input.layout, self.device, self.dram_input_memory_config
        )

    def _allocate_persistent_output_tensor(self, output_shape, output_dtype):
        """Allocates persistent DRAM output tensor based on inferred shape and dtype."""
        self.dram_output_tensor = ttnn.allocate_tensor_on_device(
            output_shape, output_dtype, ttnn.ROW_MAJOR_LAYOUT, self.device, self.dram_output_memory_config
        )

    def _initialize_synchronization_events(self):
        first_op_event = ttnn.record_event(self.device, self.CQ_OPS)
        read_event = ttnn.record_event(self.device, self.CQ_IO)
        return first_op_event, read_event

    def _run_model_for_compilation_with_io(self, host_input, first_op_event, read_event):
        """
        Runs the model once for compilation using the I/O pipeline pattern.
        Returns updated events and output tensor shape/dtype for subsequent trace capture.
        """
        # Transfer input using I/O queue
        write_event = self._transfer_input_for_compilation(host_input, first_op_event)

        # Execute model and transfer output using ops queue
        first_op_event, last_op_event, output_shape, output_dtype = self._execute_model_and_transfer_output(
            write_event, read_event
        )

        return first_op_event, write_event, last_op_event, output_shape, output_dtype

    def _transfer_input_for_compilation(self, host_input, first_op_event):
        """Transfers input from host to device DRAM using I/O queue."""
        ttnn.wait_for_event(self.CQ_IO, first_op_event)
        ttnn.copy_host_to_device_tensor(host_input, self.dram_input_tensor, cq_id=self.CQ_IO)
        return ttnn.record_event(self.device, self.CQ_IO)

    def _execute_model_and_transfer_output(self, write_event, read_event):
        """
        Executes model operations and transfers output to DRAM using ops queue.
        Returns updated synchronization events and output tensor shape/dtype.
        """
        # Execute model operations
        ttnn.wait_for_event(self.CQ_OPS, write_event)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)
        first_op_event = ttnn.record_event(self.device, self.CQ_OPS)
        self._compilation_output_tensor = self.model(l1_input_tensor)

        # Transfer output to DRAM
        ttnn.wait_for_event(self.CQ_OPS, read_event)
        ttnn.reshard(
            self._compilation_output_tensor, self.dram_output_memory_config, output_tensor=self.dram_output_tensor
        )
        last_op_event = ttnn.record_event(self.device, self.CQ_OPS)

        # Extract output info before cleanup
        output_shape = self._compilation_output_tensor.shape
        output_dtype = self._compilation_output_tensor.dtype

        # Cleanup compilation tensors
        ttnn.deallocate(l1_input_tensor)
        ttnn.deallocate(self._compilation_output_tensor)

        return first_op_event, last_op_event, output_shape, output_dtype

    def _capture_execution_trace_with_io(self, host_input, first_op_event, write_event):
        """
        Captures a trace of the model execution for efficient repeated runs with I/O pipeline.
        Validates that persistent tensors are allocated at expected addresses.
        """
        # Prepare input for trace capture
        write_event = self._prepare_input_for_trace_capture(host_input, first_op_event)
        l1_input_tensor, input_trace_addr, spec = self._prepare_trace_input_with_io(write_event)

        # Begin trace capture and run model
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=self.CQ_OPS)
        self.output_tensor = self.model(l1_input_tensor)
        if l1_input_tensor.is_allocated():
            ttnn.deallocate(l1_input_tensor, force=True)  # Needed if model does not deallocate its input

        # Allocate persistent L1 input tensor and validate address consistency
        self._allocate_and_validate_l1_tensor(spec, input_trace_addr)

        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=self.CQ_OPS)

    def _prepare_input_for_trace_capture(self, host_input, first_op_event):
        """Transfers input for trace capture using I/O queue."""
        ttnn.wait_for_event(self.CQ_IO, first_op_event)
        ttnn.copy_host_to_device_tensor(host_input, self.dram_input_tensor, cq_id=self.CQ_IO)
        return ttnn.record_event(self.device, self.CQ_IO)

    def _prepare_trace_input_with_io(self, write_event):
        """Prepares L1 input tensor for trace capture and records its address."""
        ttnn.wait_for_event(self.CQ_OPS, write_event)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)

        # Record tensor details for validation after trace capture
        input_trace_addr = l1_input_tensor.buffer_address()
        spec = l1_input_tensor.spec

        # Force cleanup of the compilation output tensor to ensure that the subsequent allocation
        # of the persistent L1 input tensor occurs at the expected device memory address. This is
        # necessary because trace capture relies on the L1 input tensor being allocated at the same
        # address as during the initial trace.
        self._compilation_output_tensor.deallocate(force=True)

        return l1_input_tensor, input_trace_addr, spec

    def _allocate_and_validate_l1_tensor(self, spec, expected_addr):
        """Allocates persistent L1 input tensor and validates it uses the expected address."""
        self.l1_input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        actual_addr = self.l1_input_tensor.buffer_address()

        if expected_addr != actual_addr:
            raise RuntimeError(f"L1 input tensor address mismatch: expected {expected_addr}, got {actual_addr}")

    def _store_event_state(self, first_op_event, read_event, write_event, last_op_event):
        """Stores synchronization events for use during execution."""
        self.first_op_event = first_op_event
        self.read_event = read_event
        self.write_event = write_event
        self.last_op_event = last_op_event

    def _execute_traced_model_and_transfer_output(self):
        """
        Executes the traced model and transfers output to DRAM.
        Updates synchronization events and returns them.
        """
        # Execute traced model on current input
        ttnn.wait_for_event(self.CQ_OPS, self.write_event)
        self.l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config, self.l1_input_tensor)
        self.first_op_event = ttnn.record_event(self.device, self.CQ_OPS)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=self.CQ_OPS, blocking=False)

        # Transfer output to DRAM for reading
        ttnn.wait_for_event(self.CQ_OPS, self.read_event)
        ttnn.reshard(self.output_tensor, self.dram_output_memory_config, output_tensor=self.dram_output_tensor)
        self.last_op_event = ttnn.record_event(self.device, self.CQ_OPS)

    def _transfer_next_input(self, input_tensor):
        """Transfers the next input tensor to device DRAM using I/O queue."""
        ttnn.wait_for_event(self.CQ_IO, self.first_op_event)
        ttnn.copy_host_to_device_tensor(input_tensor, self.dram_input_tensor, cq_id=self.CQ_IO)
        self.write_event = ttnn.record_event(self.device, self.CQ_IO)

    def _read_output_from_device(self):
        """Reads output tensor from device DRAM using I/O queue."""
        ttnn.wait_for_event(self.CQ_IO, self.last_op_event)
        output = self.dram_output_tensor
        self.read_event = ttnn.record_event(self.device, self.CQ_IO)
        return output

    def _issue_input_and_get_prev_output(self, input_tensor):
        """
        Processes the current input while issuing the next input in parallel.
        """
        # Execute model on current input and transfer output to DRAM
        self._execute_traced_model_and_transfer_output()

        # Transfer next input (parallel with output reading)
        self._transfer_next_input(input_tensor)

        # Read output from previous computation
        return self._read_output_from_device()

    def _get_last_output(self):
        """
        Processes the final input and returns its output.

        This handles the last input in the pipeline where there's no next input
        to transfer, so we only need to execute the model and read the output.
        """
        # Execute model on final input and transfer output to DRAM
        self._execute_traced_model_and_transfer_output()

        # Read final output (no next input to transfer)
        return self._read_output_from_device()

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

        self._transfer_first_input(host_inputs[0])
        for i in range(1, num_inputs):
            yield self._issue_input_and_get_prev_output(host_inputs[i])
        yield self._get_last_output()

    def _transfer_first_input(self, first_input):
        ttnn.wait_for_event(self.CQ_IO, self.first_op_event)
        ttnn.copy_host_to_device_tensor(first_input, self.dram_input_tensor, cq_id=self.CQ_IO)
        self.write_event = ttnn.record_event(self.device, self.CQ_IO)

    def cleanup(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
