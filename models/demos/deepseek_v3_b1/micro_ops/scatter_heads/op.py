# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
ScatterHeads micro-op implementation.

Scatters data from few input cores to many output cores.
Input: 8 cores, each with shard shape (8, 512) = 8 rows × 512 elements
Output: 64 cores, each with shard shape (1, 512) = 1 row × 512 elements

Each input core's 8 rows are scattered to 8 different output cores.
"""

from typing import Optional

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class ScatterHeads:
    """
    ScatterHeads operation using ttnn.generic_op.

    Scatters data from input cores to output cores where each output core
    reads one row from a specific input core.
    """

    @staticmethod
    def golden(input_tensor):
        """
        PyTorch reference implementation of scatter_heads for validation.

        Args:
            input_tensor: Input tensor of shape (num_input_cores * rows_per_core, width)
                          e.g., (8*8, 512) = (64, 512)

        Returns:
            Output tensor with same shape but re-ordered for scatter pattern
        """
        # For scatter_heads, the data is simply re-distributed
        # Input: 8 shards of (8, 512) stacked = (64, 512)
        # Output: 64 shards of (1, 512) stacked = (64, 512)
        # The logical data is the same, just sharded differently
        return input_tensor.clone()

    @staticmethod
    def op(
        input_tensor,
        output_tensor,
        core_mapping: Optional[list] = None,
        rows_per_input_core: Optional[int] = 8,
    ):
        """
        Execute scatter_heads operation using generic_op.

        Args:
            input_tensor: Input tensor sharded on input_cores, each shard is (rows_per_input_core, width)
            output_tensor: Pre-allocated output tensor sharded on output_cores, each shard is (1, width)
            core_mapping: Optional list of (output_core, input_core, row_offset) tuples.
                          If None, uses default mapping where output core i reads from
                          input core (i // rows_per_input_core), row (i % rows_per_input_core).
                          Each tuple specifies:
                          - output_core: CoreCoord of the output core
                          - input_core: CoreCoord of the input core to read from
                          - row_offset: Which row (0-indexed) to read from the input core's shard

        Returns:
            Output tensor with scattered data
        """
        device = input_tensor.device()

        # Get memory configs
        input_memory_config = input_tensor.memory_config()
        output_memory_config = output_tensor.memory_config()

        # Get shard shapes
        input_shard_shape = input_memory_config.shard_spec.shape
        output_shard_shape = output_memory_config.shard_spec.shape

        # Validate shapes
        input_height, input_width = input_shard_shape
        output_height, output_width = output_shard_shape

        assert output_height == 1, f"Expected output height 1, got {output_height}"
        assert input_width == output_width, f"Width mismatch: input {input_width} vs output {output_width}"

        element_size = 2

        # Size of one row (what each output core reads)
        row_size_bytes = output_width * element_size

        # Get input tensor buffer address (runtime arg)
        input_buffer_addr = input_tensor.buffer_address()

        # Build the all_cores set (input + output)
        output_core_range_set = output_memory_config.shard_spec.grid
        all_cores = output_core_range_set
        output_num_pages = 1

        # CB indices
        dst_cb = 0

        # ========================================================================
        # Extract cores from shard specs
        # ========================================================================

        # Get input cores from input tensor's shard spec
        input_core_range_set = input_memory_config.shard_spec.grid
        input_core_list = ttnn.corerange_to_cores(input_core_range_set, row_wise=True)

        # Get output cores from output tensor's shard spec
        output_core_list = ttnn.corerange_to_cores(output_core_range_set, row_wise=True)

        num_input_cores = len(input_core_list)
        num_output_cores = len(output_core_list)

        # Build per-core compile-time args
        src_noc_x_per_core = []
        src_noc_y_per_core = []
        src_row_offset_per_core = []

        if core_mapping is not None:
            # Use user-provided mapping
            assert (
                len(core_mapping) == num_output_cores
            ), f"core_mapping must have {num_output_cores} entries, got {len(core_mapping)}"
            for output_core, input_core, row_offset in core_mapping:
                noc_core = device.worker_core_from_logical_core(input_core)
                src_noc_x_per_core.append((output_core, noc_core.x))
                src_noc_y_per_core.append((output_core, noc_core.y))
                src_row_offset_per_core.append((output_core, row_offset))
        else:
            # Use default mapping: output core i reads from input core (i // rows_per_input_core),
            # row (i % rows_per_input_core)
            assert (
                num_output_cores == num_input_cores * rows_per_input_core
            ), f"Expected {num_input_cores * rows_per_input_core} output cores, got {num_output_cores}"
            for i, output_core in enumerate(output_core_list):
                input_core_idx = i // rows_per_input_core
                row_offset = i % rows_per_input_core

                input_core = input_core_list[input_core_idx]
                noc_core = device.worker_core_from_logical_core(input_core)

                src_noc_x_per_core.append((output_core, noc_core.x))
                src_noc_y_per_core.append((output_core, noc_core.y))
                src_row_offset_per_core.append((output_core, row_offset))

        # ========================================================================
        # Named compile-time args for each RISC processor
        # ========================================================================

        # NCRISC (Reader) named compile-time args
        ncrisc_named_compile_time_args = [
            ("data_size_bytes", row_size_bytes),
            ("dst_cb", dst_cb),
            ("dst_num_pages", output_num_pages),
            ("src_addr", input_buffer_addr),
        ]

        brisc_named_compile_time_args = []
        trisc_named_compile_time_args = []

        # ========================================================================
        # Circular buffer descriptors
        # ========================================================================

        # CB 0: Destination (output tensor, on output cores)
        dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(dst_cb, output_tensor)

        # ========================================================================
        # Unified kernel descriptor
        # ========================================================================
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/scatter_heads/kernels/scatter_heads_kernel.cpp",
            core_ranges=all_cores,
            # NCRISC named compile-time args
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            # BRISC named compile-time args
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            # TRISC named compile-time args
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            # Per-core compile-time role differentiation
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_output_core",
                    core_range=output_core_range_set,
                    value=1,
                    other_value=0,
                )
            ],
            # Per-core compile-time args for source mapping
            per_core_compile_time_descriptors=[
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="src_noc_x",
                    core_values=src_noc_x_per_core,
                    other_value=0,
                ),
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="src_noc_y",
                    core_values=src_noc_y_per_core,
                    other_value=0,
                ),
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="src_row_offset",
                    core_values=src_row_offset_per_core,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[dst_cb_descriptor],
            semaphores=[],
        )

        # Execute generic op
        io_tensors = [input_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
