# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
DRAM Streaming Matmul with Compressed Weights — Multi-Expert.

Computes: for each selected expert e:
  output[e * per_core_N : (e+1) * per_core_N] += in0[1, K] @ decompress(in1_e[K, per_core_N])

Each expert's weight shard is packed at its natural size (no DRAM padding; F1a).
Expert addressing uses a per-expert byte offset table CB instead of fixed stride.

Constraints:
  - cores_per_bank = 1 (one compute core per DRAM bank)
  - selected_experts_k <= 16
  - Output tensor has selected_experts_k * N columns (stacked per-expert results, not summed)

The offset table CB is replicated to all compute cores. It holds one uint32 offset
per selected expert: offsets[e] = cts[e].data.buffer_address() - cts[0].data.buffer_address().
The kernel starts each expert's DRAM read at in1_tensor_addr + offsets[e].
"""

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import (
    _TILE_SIZES,
    _compute_subblock_metadata,
)
from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import _CB_ADDR_SHIFT
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

KERNEL_PATH = "models/demos/deepseek_v3_b1/micro_ops/dram_streaming_experts_matmul_compressed/kernels/dram_streaming_experts_matmul_compressed_kernel.cpp"


class DRAMStreamingExpertsMatmulCompressed:
    @staticmethod
    def op(
        input_a: ttnn.Tensor,
        cts: list,
        output_tensor: ttnn.Tensor,
        subblock_k: int = None,
    ) -> ttnn.Tensor:
        """
        A [M, K] @ decompress(B_e [K, N]) for each selected expert e.

        B_e tensors are WIDTH_SHARDED in DRAM with variable packed sizes (F1a).
        Each expert's tiles are streamed from DRAM in variable-size subblocks.
        L1 holds stacked block-size metadata and tile-format info for all experts.

        Output has shape [M, selected_experts_k * N] (per-expert results stacked,
        not summed). Summation is performed by the caller (e.g. MoeOp).

        Args:
            input_a: Activation tensor, HEIGHT_SHARDED (replicated) on compute cores.
                     Shape [M, K], M=1 for decode.
            cts: List of CompressedTensors, one per selected expert. All must share
                 the same DRAM bank topology and K/N dimensions.
            output_tensor: Output tensor, WIDTH_SHARDED on compute cores.
                           Shape [M, selected_experts_k * N_total].
            subblock_k: K subblock size in tiles. None means full K.
        """
        selected_experts_k = len(cts)
        assert 1 <= selected_experts_k <= 16, f"selected_experts_k={selected_experts_k} must be in [1, 16]"

        device = input_a.device()

        # Compute cores — one per DRAM bank
        in1_noc = ttnn.NOC.NOC_0
        primary_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(in1_noc)
        num_banks = len(primary_worker_cores)

        compute_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in primary_worker_cores]
        )
        num_total_cores = num_banks

        # Shapes from shard specs
        a_shard_shape = input_a.memory_config().shard_spec.shape
        M = a_shard_shape[0]
        K = a_shard_shape[1]
        Kt = K // 32
        out_shard_shape = output_tensor.memory_config().shard_spec.shape
        per_core_N_total = out_shard_shape[1] // 32  # N tiles per core in the output
        per_core_N = per_core_N_total // selected_experts_k  # N tiles per core per expert

        assert per_core_N_total == per_core_N * selected_experts_k, (
            f"Output per_core_N_total={per_core_N_total} must equal "
            f"selected_experts_k={selected_experts_k} * per_core_N={per_core_N}"
        )

        if subblock_k is None:
            subblock_k = Kt
        assert Kt % subblock_k == 0, f"Kt ({Kt}) must be divisible by subblock_k ({subblock_k})"
        num_subblocks_k = Kt // subblock_k
        assert subblock_k % 2 == 0, f"subblock_k ({subblock_k}) must be even"

        logger.debug(
            f"DRAMStreamingExpertsMatmulCompressed: M={M}, K={K}, Kt={Kt}, "
            f"per_core_N={per_core_N}, selected_experts_k={selected_experts_k}, "
            f"subblock_k={subblock_k}, num_subblocks_k={num_subblocks_k}, "
            f"num_banks={num_banks}"
        )

        # CB indices
        cb_in0 = 0  # A tensor (HEIGHT_SHARDED, replicated)
        cb_in1 = 1  # Working buffer for streaming
        cb_out = 2  # Output tensor

        # CB0: A tensor — tensor-backed
        cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in0, input_a)

        # CB1: Working buffer — tensor-backed (3 × max_subblock_bytes per slot)
        max_tile_size = _TILE_SIZES[0]  # bfp8 = 1088
        num_in1_buffers = 3
        max_subblock_bytes = subblock_k * max_tile_size
        cb_in1_total_bytes = num_in1_buffers * max_subblock_bytes

        in1_tile = ttnn.Tile([32, 32])
        in1_cb_tiles = subblock_k * num_in1_buffers
        in1_backing_shard_shape = (32, in1_cb_tiles * 32)
        in1_backing_total_width = in1_cb_tiles * 32 * num_total_cores
        in1_backing_shard_spec = ttnn.ShardSpec(compute_cores, in1_backing_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        in1_backing_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, in1_backing_shard_spec
        )
        in1_backing_torch = torch.zeros([1, 1, 32, in1_backing_total_width]).bfloat16().float()
        in1_backing_tensor = ttnn.from_torch(
            in1_backing_torch,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=in1_backing_mem_config,
            tile=in1_tile,
        )
        cb_in1_base = in1_backing_tensor.buffer_address()
        cb_in1_base_shifted = (cb_in1_base >> _CB_ADDR_SHIFT) - 1
        max_subblock_bytes_shifted = max_subblock_bytes >> _CB_ADDR_SHIFT

        cb1_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in1, in1_backing_tensor)

        # CB2: Output tensor — tensor-backed
        cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_out, output_tensor)

        # ====================================================================
        # CB3: Expert byte offset table
        # Replicated across all cores. Each core gets selected_experts_k uint32 values.
        # offsets[e] = cts[e].data.buffer_address() - cts[0].data.buffer_address()
        # ====================================================================
        base_addr = cts[0].data.buffer_address()
        expert_offsets_u32 = np.array([int(ct.data.buffer_address()) - int(base_addr) for ct in cts], dtype=np.uint32)
        assert expert_offsets_u32[0] == 0, "First expert offset must be 0"

        # Replicate offset table to all cores
        offsets_bytes_per_core = expert_offsets_u32.view(np.uint8)  # shape (selected_experts_k * 4,)
        offsets_np = np.tile(offsets_bytes_per_core, (num_total_cores, 1))  # (num_cores, k*4)
        offsets_torch = torch.from_numpy(offsets_np.astype(np.int8)).view(torch.uint8)

        offsets_shard_spec = ttnn.ShardSpec(compute_cores, [1, selected_experts_k * 4], ttnn.ShardOrientation.ROW_MAJOR)
        offsets_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, offsets_shard_spec
        )
        offsets_tensor = ttnn.from_torch(
            offsets_torch,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=offsets_mem_config,
        )
        expert_offsets_l1_addr = offsets_tensor.buffer_address()

        cbs = [cb0_desc, cb1_desc, cb2_desc]

        # DRAM base address (from expert-0's data tensor)
        in1_buffer_addr = cts[0].data.buffer_address()

        # DRAM shard cores from expert-0's tensor
        dram_cores = ttnn.corerange_to_cores(cts[0].data.memory_config().shard_spec.grid)

        # ====================================================================
        # Build per-core stacked metadata (block sizes + tile format infos)
        # Stacked order: expert 0 all iters, expert 1 all iters, ...
        # Within each expert: (n=0, sb_k=0), (n=0, sb_k=1), ..., (n=1, sb_k=0), ...
        # ====================================================================
        meta_entries_per_core = selected_experts_k * per_core_N * num_subblocks_k
        fmt_entries_per_core = selected_experts_k * subblock_k * num_subblocks_k * per_core_N

        per_core_block_sizes = {}
        per_core_fmt_pairs = {}

        bank_id_core_values = []
        vc_core_values = []
        bank_ids = []

        for bank_idx, primary_core in enumerate(primary_worker_cores):
            bank_id = bank_idx % num_banks
            vc = bank_id & 0x3
            for j in range(bank_idx):
                prev_core = primary_worker_cores[j]
                if prev_core.y == primary_core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
                    vc = (vc + 1) & 0x3
                    break
            bank_ids.append(bank_id)

            stacked_block_sizes = []
            stacked_tile_infos = []

            total_iteration = 0
            for ct in cts:
                shard_assignment = ct.get_assignment_per_shard(dram_cores[bank_idx])
                block_sizes, tile_infos = _compute_subblock_metadata(
                    device,
                    shard_assignment,
                    subblock_k,
                    per_core_N,
                    num_subblocks_k,
                    cb_in1_base_shifted,
                    max_subblock_bytes_shifted,
                    num_in1_buffers,
                    start_iteration=total_iteration,
                )
                stacked_block_sizes.extend(block_sizes)
                stacked_tile_infos.extend(tile_infos)
                total_iteration += per_core_N * num_subblocks_k

            per_core_block_sizes[bank_idx] = stacked_block_sizes
            per_core_fmt_pairs[bank_idx] = stacked_tile_infos

            bank_id_core_values.append((primary_core, bank_id))
            vc_core_values.append((primary_core, vc))

        # ====================================================================
        # Upload stacked block-size metadata to L1
        # ====================================================================
        meta_flat = []
        for core_idx in range(num_total_cores):
            meta_flat.extend(per_core_block_sizes[core_idx])

        meta_torch = torch.tensor(meta_flat, dtype=torch.int32).reshape(num_total_cores, meta_entries_per_core)
        meta_shard_spec = ttnn.ShardSpec(compute_cores, [1, meta_entries_per_core * 4], ttnn.ShardOrientation.ROW_MAJOR)
        meta_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, meta_shard_spec)
        meta_tensor = ttnn.from_torch(
            meta_torch.view(torch.uint8).reshape(num_total_cores, meta_entries_per_core * 4),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=meta_mem_config,
        )
        meta_l1_addr = meta_tensor.buffer_address()

        # ====================================================================
        # Upload stacked tile format metadata to L1
        # ====================================================================
        fmt_flat = []
        for core_idx in range(num_total_cores):
            fmt_flat.extend(per_core_fmt_pairs[core_idx])

        fmt_np = np.array(fmt_flat, dtype=np.uint32).view(np.int32).reshape(num_total_cores, fmt_entries_per_core)
        fmt_torch = torch.from_numpy(fmt_np)
        fmt_shard_spec = ttnn.ShardSpec(compute_cores, [1, fmt_entries_per_core * 4], ttnn.ShardOrientation.ROW_MAJOR)
        fmt_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, fmt_shard_spec)
        fmt_tensor = ttnn.from_torch(
            fmt_torch.view(torch.uint8).reshape(num_total_cores, fmt_entries_per_core * 4),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=fmt_mem_config,
        )
        fmt_l1_addr = fmt_tensor.buffer_address()

        # NOC max page size (arch-dependent)
        arch = device.arch()
        if arch == ttnn.device.Arch.WORMHOLE_B0:
            noc_max_page_size = 8192
        elif arch == ttnn.device.Arch.BLACKHOLE:
            noc_max_page_size = 16384
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # ====================================================================
        # Compile-time args
        # ====================================================================
        ncrisc_named_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles_k", Kt),
            ("in1_tensor_addr", in1_buffer_addr),
            ("subblock_k", subblock_k),
            ("per_core_n", per_core_N),
            ("out_num_tiles", per_core_N_total),
            ("num_subblocks_k", num_subblocks_k),
            ("meta_l1_addr", meta_l1_addr),
            ("cb_in1_size_bytes", cb_in1_total_bytes),
            ("noc_max_page_size", noc_max_page_size),
            ("selected_experts_k", selected_experts_k),
            ("expert_offsets_l1_addr", expert_offsets_l1_addr),
        ]

        brisc_named_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles_k", Kt),
        ]

        trisc_named_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("subblock_k", subblock_k),
            ("per_core_n", per_core_N),
            ("num_subblocks_k", num_subblocks_k),
            ("selected_experts_k", selected_experts_k),
            ("fmt_l1_addr", fmt_l1_addr),
        ]

        per_core_descriptors = [
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="bank_id",
                core_values=bank_id_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="vc",
                core_values=vc_core_values,
                other_value=0,
            ),
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=KERNEL_PATH,
            core_ranges=compute_cores,
            ncrisc_named_compile_time_args=ncrisc_named_args,
            brisc_named_compile_time_args=brisc_named_args,
            trisc_named_compile_time_args=trisc_named_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_active_core",
                    core_range=compute_cores,
                    value=1,
                    other_value=0,
                ),
            ],
            per_core_compile_time_descriptors=per_core_descriptors,
        )

        kernel_descriptors = unified_kernel.get_kernel_descriptors()

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernel_descriptors.kernels,
            cbs=cbs,
            semaphores=[],
        )

        io_tensors = [
            input_a,
            cts[0].data,
            output_tensor,
            in1_backing_tensor,
            offsets_tensor,
            meta_tensor,
            fmt_tensor,
        ]
        ttnn.generic_op(io_tensors, program_descriptor)

        return output_tensor
