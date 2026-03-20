# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Eltwise Add with Compressed Tensor.

Computes: out = A + decompress(B_compressed)
"""


import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import CompressedTensor
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreCompileTimeDescriptor, UnifiedKernelDescriptor


class EltwiseAddCompressed:
    @staticmethod
    def op(
        a_tensor: ttnn.Tensor,
        ct: CompressedTensor,
        output_tensor: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        A (bf16) + B (bfp8) = C (bf16) using standard add_tiles.
        """
        # Use only cores that have compressed data — empty cores (from uneven shards)
        # would hang because CB1 has no backing tensor on those cores.
        core_grid = ct.get_data_core_range_set()
        all_cores = ttnn.corerange_to_cores(core_grid)

        # CB indices
        cb_in0 = 0
        cb_in1 = 1
        cb_out = 2

        # Per-shard tile count (for multi-core, each core processes its shard)
        a_shard_shape = a_tensor.memory_config().shard_spec.shape
        num_tiles = (a_shard_shape[0] // 32) * (a_shard_shape[1] // 32)

        # CB0: A tensor — standard
        cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in0, a_tensor)

        # CB1: compressed data — per-core or lockstep depending on ct mode
        cb1_descs = ct.cb_descriptor_from_compressed_tensor(cb_in1)

        # CB2: output tensor — standard
        cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_out, output_tensor)

        # Number of pages for sharded buffer setup
        cb_in0_num_pages = num_tiles
        # Compressed data: push 1 page covering the whole shard.
        # The kernel manually advances addr_b per tile.
        cb_in1_num_pages = 1

        compile_time_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles", num_tiles),
            ("cb_in0_num_pages", cb_in0_num_pages),
            ("cb_in1_num_pages", cb_in1_num_pages),
        ]

        # assign_l1_addr: per-core in per_core_allocation mode, uniform otherwise
        per_core_descriptors = []
        if ct._per_core_allocation:
            per_core_descriptors.append(
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="assign_l1_addr",
                    core_values=[(core, ct.get_assignment_l1_address_per_core(core)) for core in all_cores],
                    other_value=0,
                )
            )
        else:
            compile_time_args.append(("assign_l1_addr", ct.get_assignment_l1_address()))

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/eltwise_add_compressed/kernel.cpp",
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=compile_time_args,
            brisc_named_compile_time_args=compile_time_args,
            trisc_named_compile_time_args=compile_time_args,
            per_core_compile_time_descriptors=per_core_descriptors,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[cb0_desc, *cb1_descs, cb2_desc],
            semaphores=[],
        )

        io_tensors = [a_tensor, *ct.get_data_tensors(), *ct.get_assignment_tensors(), output_tensor]
        ttnn.generic_op(io_tensors, program_descriptor)
        return output_tensor
