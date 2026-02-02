# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Single-core SDPA S*K operation.

Computes the attention output: S @ V
where V is a subset of K, with v_width tiles read from K.
"""

import torch

import ttnn


class SdpaSingleCore:
    """
    Single-core SDPA S*K computation using ttnn.generic_op.

    Computes attention output by multiplying S with K in chunks.
    V is a subset of K, with v_width tiles read from K.
    """

    @staticmethod
    def golden(s, k, v_width):
        """
        PyTorch reference implementation for validation.

        Args:
            s: Attention scores tensor [height, seq_len]
            k: Key/Value tensor [seq_len, head_dim] (will be transposed if transpose_k=True)
            transpose_k: Whether to transpose K before matmul

        Returns:
            Attention output tensor [height, head_dim]
        """
        return torch.matmul(s, k[:, :v_width])

    @staticmethod
    def op(
        s_tensor,
        k_tensor,
        out_tensor,
        chunk_size=1,
        num_tiles_k=1,
        num_tiles_v=1,
    ):
        """
        Execute SDPA S*K using generic_op.

        Args:
            s_tensor: Attention scores input tensor (must be sharded)
            k_tensor: Key/Value input tensor (must be sharded)
            out_tensor: Pre-allocated output tensor (must be sharded)
            chunk_size: Number of tiles per K chunk
            num_tiles_k: Number of tiles in the K dimension
            num_tiles_v: Number of tiles in the V dimension

        Returns:
            Output tensor with attention output
        """
        all_cores = s_tensor.memory_config().shard_spec.grid

        # CB indices
        cb_s = 0
        cb_k = 1
        cb_out = 2

        # Create CB descriptors for sharded tensors (uses each tensor's own tile)
        s_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_s, s_tensor)
        k_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_k, k_tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out_tensor)

        # Reader kernel
        reader_compile_time_args = [
            cb_s,
            cb_k,
            chunk_size,
            num_tiles_k,
        ]

        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/sdpa/kernels/sdpa_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=reader_compile_time_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        # Writer kernel
        out_num_tiles = num_tiles_v
        writer_compile_time_args = [
            cb_out,
            out_num_tiles,
        ]

        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/sdpa/kernels/sdpa_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=writer_compile_time_args,
            config=ttnn.WriterConfigDescriptor(),
        )

        # Compute kernel
        compute_compile_time_args = [
            cb_s,
            cb_k,
            cb_out,
            chunk_size,
            num_tiles_k,
            num_tiles_v,
        ]

        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/sdpa/kernels/sdpa_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=compute_compile_time_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )

        print(reader_compile_time_args)
        print(writer_compile_time_args)
        print(compute_compile_time_args)

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
            cbs=[
                s_cb_descriptor,
                k_cb_descriptor,
                out_cb_descriptor,
            ],
        )

        # Execute generic op
        io_tensors = [
            s_tensor,
            k_tensor,
            out_tensor,
        ]
        ttnn.generic_op(io_tensors, program_descriptor)

        return out_tensor
