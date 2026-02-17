# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Single-core SDPA Q*K^T operation.

Computes the attention scores: QK^T = Q @ K^T
where Q and K are chunked and processed iteratively.
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.utils import float_to_uint32


class SdpaSingleCore:
    """
    Single-core SDPA Q*K^T computation using ttnn.generic_op.

    Computes attention scores by multiplying Q with transposed K in chunks.
    """

    @staticmethod
    def golden(q, k, v_size, scale=1.0):
        """
        PyTorch reference implementation for validation.

        Args:
            q: Query tensor [height, k_dim]
            k: Key tensor [seq_len, k_dim]
            v_size: Number of elements in the v dimension
        Returns:
            Attention scores tensor [height, v_size]
        """
        mm1 = torch.matmul(q, k.T)
        max_values = mm1.max(dim=-1, keepdim=True).values
        sub_max = mm1 - max_values
        exp_sub_max = torch.exp(sub_max * scale)
        sum_values = exp_sub_max.sum(dim=-1, keepdim=True)
        mm2 = torch.matmul(exp_sub_max, k[:, :v_size])
        return mm2, max_values, sum_values

    @staticmethod
    def op(
        q_tensor,
        k_tensor,
        out_tensor,
        stats_tensor,
        chunk_size=1,
        num_chunks=1,
        num_tiles_k=1,
        num_tiles_v=1,
        scale=1.0,
    ):
        """
        Execute SDPA Q*K^T using generic_op.

        Args:
            q_tensor: Query input tensor (must be sharded)
            k_tensor: Key input tensor (must be sharded)
            out_tensor: Pre-allocated output tensor (must be sharded)
            stats_tensor: Pre-allocated stats tensor (must be sharded)
            chunk_size: Number of tiles per K chunk
            num_chunks: Number of K chunks to process
            num_tiles_k: Number of tiles in the K dimension
            scale: Scale factor for exponent
        Returns:
            Output tensor with attention scores
        """
        all_cores = q_tensor.memory_config().shard_spec.grid

        # TODO: Add missing validation
        assert num_tiles_v <= num_tiles_k, f"num_tiles_v must be less than or equal to num_tiles_k"

        # CB indices
        cb_q = 0
        cb_k = 1
        cb_out = 2
        cb_stats = 3

        # Create CB descriptors for sharded tensors (uses each tensor's own tile)
        q_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_q, q_tensor)
        k_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_k, k_tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out_tensor)
        stats_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_stats, stats_tensor)

        num_tiles_stats = 1

        # Reader kernel
        reader_compile_time_args = [
            cb_q,
            cb_k,
            chunk_size,
            num_chunks,
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
        writer_compile_time_args = [
            cb_out,
            cb_stats,
            num_tiles_v,
            num_tiles_stats,
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
            cb_q,
            cb_k,
            cb_out,
            cb_stats,
            chunk_size,
            num_chunks,
            num_tiles_k,
            num_tiles_v,
            num_tiles_stats,
            float_to_uint32(scale),
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

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
            cbs=[
                q_cb_descriptor,
                k_cb_descriptor,
                out_cb_descriptor,
                stats_cb_descriptor,
            ],
        )

        # Execute generic op
        io_tensors = [
            q_tensor,
            k_tensor,
            out_tensor,
            stats_tensor,
        ]
        ttnn.generic_op(io_tensors, program_descriptor)

        return out_tensor, stats_tensor
