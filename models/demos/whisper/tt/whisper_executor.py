# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Traced executor for Whisper decoder.

This module provides a TracedWhisperDecoderExecutor class that captures and replays
trace execution for the Whisper decoder, eliminating host dispatch overhead during
autoregressive decoding.
"""

from typing import Callable

from loguru import logger

import ttnn


class TracedWhisperDecoderExecutor:
    """
    Traced executor for Whisper decoder that captures decoder execution and replays it efficiently.

    Note: This executor expects DEVICE tensors as input (already on device in DRAM).
    It skips DRAM staging and copies directly from input to persistent L1 tensor.

    Usage:
        # First, run decoder once to compile kernels and populate cross-attention cache
        decoder_output = decoder(decoder_hidden_states, ...)

        # Then create executor and capture trace
        executor = TracedWhisperDecoderExecutor(decoder_fn, device, l1_mem_config)
        executor.compile(sample_device_input)

        # Use trace for subsequent iterations
        for step in decode_steps:
            output = executor.execute(device_tensor)
            ttnn.plus_one(current_decode_pos)  # Outside trace
        executor.cleanup()
    """

    def __init__(
        self,
        model_fn: Callable,
        device,
        l1_input_memory_config: ttnn.MemoryConfig,
        cq_id: int = 0,
    ):
        """
        Initialize the traced decoder executor.

        Args:
            model_fn: The decoder function to trace. Should accept L1 input tensor and return output.
            device: The TTNN device.
            l1_input_memory_config: Memory config for L1 input tensor.
            cq_id: Command queue ID (default 0 for single CQ).
        """
        self.model_fn = model_fn
        self.device = device
        self.cq_id = cq_id

        self.l1_input_memory_config = l1_input_memory_config

        # Persistent tensors (no DRAM staging needed)
        self.l1_input_tensor = None
        self.output_tensor = None
        self._compilation_output_tensor = None

        # Trace state
        self.trace_id = None
        self.input_trace_addr = None

    def compile(self, device_input: ttnn.Tensor):
        """
        Compile the model by running it once to set up memory state, then capture a trace.

        Args:
            device_input: Sample input tensor on DEVICE with correct shape/dtype.
        """
        logger.debug("Compiling traced Whisper decoder executor")

        self._validate_input(device_input)
        self._run_model_for_compilation(device_input)
        self._capture_execution_trace(device_input)
        ttnn.synchronize_device(self.device)

        logger.debug("Trace capture complete")

    def _validate_input(self, device_input: ttnn.Tensor):
        """Validate that input tensor is on device."""
        if device_input.storage_type() != ttnn.StorageType.DEVICE:
            raise ValueError("Input tensor must be on DEVICE for trace compilation")

    def _run_model_for_compilation(self, device_input: ttnn.Tensor):
        """Run the model once to set up memory state for trace capture."""
        # Move device input directly to L1 and run model
        l1_input_for_compile = ttnn.to_memory_config(device_input, self.l1_input_memory_config)
        self._compilation_output_tensor = self.model_fn(l1_input_for_compile)

        # Cleanup L1 input tensor
        if l1_input_for_compile.is_allocated():
            ttnn.deallocate(l1_input_for_compile)

    def _capture_execution_trace(self, device_input: ttnn.Tensor):
        """Capture execution trace for efficient replay."""
        # Move device input directly to L1
        l1_input_for_trace = ttnn.to_memory_config(device_input, self.l1_input_memory_config)

        # Record tensor address and spec for validation
        self.input_trace_addr = l1_input_for_trace.buffer_address()
        spec = l1_input_for_trace.spec

        # Force cleanup of compilation output to ensure address consistency
        # This is necessary because trace capture relies on the L1 input tensor
        # being allocated at the same address as during the initial trace
        if self._compilation_output_tensor is not None:
            self._deallocate_tensor(self._compilation_output_tensor, force=True)

        # Begin trace capture
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=self.cq_id)

        # Run model under trace
        self.output_tensor = self.model_fn(l1_input_for_trace)

        # Deallocate L1 input inside trace
        if l1_input_for_trace.is_allocated():
            ttnn.deallocate(l1_input_for_trace, force=True)

        # Allocate persistent L1 input tensor and validate address
        self.l1_input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        actual_addr = self.l1_input_tensor.buffer_address()

        if self.input_trace_addr != actual_addr:
            raise RuntimeError(
                f"L1 input tensor address mismatch: trace captured {self.input_trace_addr}, "
                f"but persistent tensor allocated at {actual_addr}"
            )

        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=self.cq_id)

    def _deallocate_tensor(self, tensor, force: bool = False):
        """Deallocate a tensor, handling nested structures."""
        if isinstance(tensor, ttnn.Tensor):
            if tensor.is_allocated():
                ttnn.deallocate(tensor, force=force)
        elif isinstance(tensor, (list, tuple)):
            for t in tensor:
                self._deallocate_tensor(t, force=force)

    def execute(self, device_input: ttnn.Tensor) -> ttnn.Tensor:
        """
        Execute the traced decoder for a single input tensor.

        Args:
            device_input: Input tensor on DEVICE (decoder hidden states in DRAM).

        Returns:
            Output tensor from decoder (on device).
        """
        if self.trace_id is None:
            raise RuntimeError("Executor must be compiled before execution")

        # Copy device input directly to persistent L1 tensor (skip DRAM staging)
        self.l1_input_tensor = ttnn.to_memory_config(
            device_input, self.l1_input_memory_config, output_tensor=self.l1_input_tensor
        )

        # Validate address consistency
        actual_addr = self.l1_input_tensor.buffer_address()
        if actual_addr != self.input_trace_addr:
            raise RuntimeError(
                f"L1 input tensor address mismatch during execution: "
                f"expected {self.input_trace_addr}, got {actual_addr}"
            )

        # Execute trace
        ttnn.execute_trace(self.device, self.trace_id, cq_id=self.cq_id, blocking=False)

        return self.output_tensor

    def cleanup(self):
        """Release the captured trace and free device resources."""
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None
            logger.debug("Released Whisper decoder trace")


def get_decoder_trace_memory_configs(device, input_shape, dtype=ttnn.bfloat16):
    """
    Get memory configurations for traced decoder execution.

    Args:
        device: The TTNN device.
        input_shape: Shape of decoder hidden states input.
        dtype: Data type for tensors.

    Returns:
        L1 memory config for decoder input.
    """
    # L1 config - interleaved for decoder input
    l1_memory_config = ttnn.L1_MEMORY_CONFIG

    return l1_memory_config
