# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Simplified DRAM Streaming Matmul Operation.

This implements a multicore matmul with DRAM sharded weights where:
- Input A (in0): REPLICATED on all compute cores (each core has full tensor)
- Input B (in1): WIDTH_SHARDED in DRAM as [K, N], per-shard tiles reordered to column-major
- Output: Computed directly on compute cores, sharded across N dimension

The in1 tensor keeps shape [K, N] with WIDTH_SHARDED:
- Each bank gets [K, per_core_N]
- Tiles are pre-shuffled within each shard from row-major to column-major
- This makes K tiles contiguous for each N column in physical memory

Compute loop (per core):
  for n in per_core_N:
    wait for in1 Kx1 stick (K tiles contiguous in memory)
    for k in K: matmul_tiles(in0[k], in1[k], out[n])
    pack out[n]

No multicast needed - each core has its own copy of in0.
CBs are backed directly by tensors (no L1-to-L1 copies).
"""

import torch
from loguru import logger

import ttnn


def get_max_page_size_and_num_pages(device, num_tiles, tile_size):
    """
    Calculate optimal page size and number of pages for NOC transfers.

    The NOC has a maximum burst size that varies by architecture:
    - Wormhole: 8192 bytes
    - Blackhole: 16384 bytes

    Returns (page_size, num_pages) where:
    - page_size is the largest multiple of tile_size that fits in NOC max
    - num_pages is total_size / page_size
    """
    total_size = num_tiles * tile_size

    # Get NOC max page size based on architecture
    arch = device.arch()
    if arch == ttnn.device.Arch.WORMHOLE_B0:
        noc_max_page_size = 8192
    elif arch == ttnn.device.Arch.BLACKHOLE:
        noc_max_page_size = 16384
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Calculate page size as largest multiple of tile_size that fits
    page_size = (noc_max_page_size // tile_size) * tile_size

    # Ensure total_size is divisible by page_size
    while total_size % page_size != 0 and page_size >= tile_size:
        page_size -= tile_size

    num_pages = total_size // page_size
    return page_size, num_pages


class DRAMStreamingMatmul:
    """
    Simplified DRAM streaming matmul using ttnn.generic_op.

    - Input A: REPLICATED on compute cores (each core has full [M, K])
    - Input B: WIDTH_SHARDED in DRAM, pre-transposed so K×1 column sticks are contiguous
    - Output: WIDTH_SHARDED on compute cores [M, N_per_core]
    """

    @staticmethod
    def golden(
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        fused_activation: str = None,
    ) -> torch.Tensor:
        """PyTorch reference implementation with optional fused activation."""
        result = input_a @ input_b
        if fused_activation is not None:
            activation_fn = {
                "silu": torch.nn.functional.silu,
            }.get(fused_activation.lower())
            if activation_fn is None:
                raise ValueError(f"Unknown activation for golden: {fused_activation}")
            result = activation_fn(result)
        return result

    @staticmethod
    def op(
        input_a: ttnn.Tensor,
        input_b: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        fp32_dest_acc_en: bool = False,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        math_approx_mode: bool = False,
        subblock_k: int = None,  # K subblock size in tiles, None means full K
        fused_activation: str = None,  # "silu" to fuse SiLU activation
    ) -> ttnn.Tensor:
        """
        Execute simplified DRAM streaming matmul.

        Key simplifications:
        - in0 is REPLICATED on compute cores (no multicast)
        - in1 is WIDTH_SHARDED [K, N] with per-shard tiles shuffled to column-major order
          (K tiles contiguous per N column in physical memory)
        - CBs are backed directly by tensors (no L1-to-L1 copies)
        - Simple loop: for each N output tile, accumulate across K
        """
        device = input_a.device()

        # Get tiles
        in0_tile = input_a.get_tile()
        in1_tile = input_b.get_tile()

        in0_tile_shape = in0_tile.tile_shape
        in1_tile_shape = in1_tile.tile_shape

        # Get dimensions from shard specs (since tensors are sharded)
        # in0 is HEIGHT_SHARDED (replicated), shard shape is [M, K]
        in0_shard_shape = input_a.memory_config().shard_spec.shape
        M = in0_shard_shape[0]
        K = in0_shard_shape[1]

        # in1 is WIDTH_SHARDED on [K, N], shard shape is [K, per_core_N]
        in1_shard_shape = input_b.memory_config().shard_spec.shape
        K_from_in1 = in1_shard_shape[0]

        # Calculate dimensions in tiles
        Mt = M // in0_tile_shape[0]
        Kt = K // in0_tile_shape[1]
        per_core_N = in1_shard_shape[1] // in1_tile_shape[1]

        # Validate
        assert Mt == 1, f"Mt must be 1 for simplified matmul, got {Mt}"
        assert K == K_from_in1, f"K dimension mismatch: {K} vs {K_from_in1}"

        # Determine subblock_k (K subblock size in tiles)
        # Default to full K if not specified
        if subblock_k is None:
            subblock_k = Kt
        assert Kt % subblock_k == 0, f"Kt ({Kt}) must be divisible by subblock_k ({subblock_k})"
        num_subblocks_k = Kt // subblock_k

        logger.debug(f"Kt={Kt}, subblock_k={subblock_k}, num_subblocks_k={num_subblocks_k}")

        # Determine subblock_w based on fp32_dest_acc_en and per_core_N
        # FP32 dest: 8 dest regs (full sync) or 4 (half sync)
        # BF16/FP16 dest: 16 dest regs (full sync) or 8 (half sync)
        if fp32_dest_acc_en:
            if per_core_N <= 8:
                max_subblock_w = 8
            else:
                max_subblock_w = 4
        else:
            if per_core_N <= 16:
                max_subblock_w = 16
            else:
                max_subblock_w = 8

        # Find largest subblock_w that evenly divides per_core_N
        subblock_w = max_subblock_w
        while subblock_w > 1 and per_core_N % subblock_w != 0:
            subblock_w -= 1

        logger.debug(f"subblock_w={subblock_w}, max_subblock_w={max_subblock_w}")

        # Data formats
        in1_dtype = input_b.dtype

        # Get compute cores
        in1_noc = ttnn.NOC.NOC_0
        all_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(in1_noc)
        num_cores = len(all_worker_cores)

        logger.debug(f"num_cores={num_cores}, Mt={Mt}, Kt={Kt}, per_core_N={per_core_N}")

        # Tile size for in1 (used for NOC transfers and CB sizing)
        in1_tile_size = in1_tile.get_tile_size(in1_dtype)

        # Calculate page size for NOC transfers (respects max NOC burst size)
        # Each block is subblock_k tiles (one K subblock)
        in1_page_size, in1_num_pages = get_max_page_size_and_num_pages(device, subblock_k, in1_tile_size)
        in1_block_size_bytes = subblock_k * in1_tile_size

        logger.debug(
            f"in1_page_size={in1_page_size}, in1_num_pages={in1_num_pages}, in1_block_size={in1_block_size_bytes}"
        )

        # CB sizes
        # in0: K tiles (full tensor, tensor-backed - size determined by tensor)
        # in1: 3 * num_subblocks_k buffers for DRAM read pipelining
        # Transaction IDs must stay within NOC_MAX_TRANSACTION_ID (0xF = 15)
        num_in1_buffers = 3 * num_subblocks_k
        assert num_in1_buffers <= 15, (
            f"num_in1_buffers ({num_in1_buffers}) exceeds NOC_MAX_TRANSACTION_ID (15). "
            f"Consider reducing subblock_k to satisfy: 3 * (Kt / subblock_k) <= 15 "
            f"(current: 3 * ({Kt} / {subblock_k}) = {num_in1_buffers})"
        )
        in1_CB_tiles = subblock_k * num_in1_buffers
        in1_CB_size = in1_CB_tiles * in1_tile_size

        # Output: per_core_N tiles (tensor-backed - size determined by tensor)
        out_num_tiles = per_core_N

        # Core ranges - use specific compute cores
        compute_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in all_worker_cores]
        )

        # Tile descriptor for in1 (used in CB1 format descriptor)
        in1_tile_desc = ttnn.TileDescriptor(in1_tile)

        # ========== CIRCULAR BUFFERS ==========

        # CB 0: in0 - BACKED BY INPUT TENSOR (replicated)
        cb0_descriptor = ttnn.cb_descriptor_from_sharded_tensor(0, input_a)

        # CB 1: in1 - working buffer for DRAM reads (K tiles double buffered)
        cb1_format = ttnn.CBFormatDescriptor(
            buffer_index=1,
            data_format=in1_dtype,
            page_size=in1_tile_size,
            tile=in1_tile_desc,
        )
        cb1_descriptor = ttnn.CBDescriptor(
            total_size=in1_CB_size,
            core_ranges=compute_cores,
            format_descriptors=[cb1_format],
        )

        cb_descriptors = [cb0_descriptor, cb1_descriptor]

        # CB 4: output - BACKED BY OUTPUT TENSOR
        cb4_descriptor = ttnn.cb_descriptor_from_sharded_tensor(4, output_tensor)
        cb_descriptors.append(cb4_descriptor)

        # ========== KERNEL ARGS ==========

        # CB IDs
        cb_id_in0 = 0
        cb_id_in1 = 1
        cb_id_out = 4

        # Kernel defines
        mm_kernel_defines = []
        if fp32_dest_acc_en:
            mm_kernel_defines.append(("FP32_DEST_ACC_EN", "1"))

        # Fused SiLU activation (only silu supported for now)
        if fused_activation is not None:
            if fused_activation.lower() != "silu":
                raise ValueError(f"Only 'silu' activation is supported, got: {fused_activation}")
            mm_kernel_defines.append(("FUSE_SILU", "1"))

        # Get buffer addresses
        in1_buffer_addr = input_b.buffer_address()

        # in0 reader compile args - just push all K tiles once
        in0_reader_compile_args = [
            cb_id_in0,
            Kt,  # num_tiles_k
        ]

        # in1 reader compile args
        in1_reader_compile_args = [
            cb_id_in1,
            cb_id_out,
            in1_buffer_addr,
            in1_page_size,
            in1_num_pages,
            subblock_k,  # tiles per K subblock
            per_core_N,
            in1_block_size_bytes,
            out_num_tiles,
            num_subblocks_k,
        ]

        # Compute compile args
        compute_compile_args = [
            cb_id_in0,
            cb_id_in1,
            cb_id_out,
            subblock_k,  # tiles per K subblock
            per_core_N,
            subblock_w,
            num_subblocks_k,
            in0_tile_shape[0],  # tile_r_dim (m) for SFPU SILU iterations
        ]

        # Runtime args (per-core: bank_id and vc)
        in1_reader_rt_args = []
        num_dram_banks = len(all_worker_cores)
        bank_ids = []
        for i, worker_core in enumerate(all_worker_cores):
            core = ttnn.CoreCoord(worker_core.x, worker_core.y)
            bank_id = i % num_dram_banks

            vc = bank_id & 0x3
            for j in range(i):
                prev_core = all_worker_cores[j]
                if prev_core.y == worker_core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
                    vc = (vc + 1) & 0x3
                    break
            bank_ids.append(bank_id)

            rt_args = [bank_id, vc]
            in1_reader_rt_args.append((core, rt_args))

        # ========== KERNELS ==========
        KERNEL_DIR = "models/demos/deepseek_v3_b1/micro_ops/dram_streaming_matmul/kernels"

        # in0 reader - just push all K tiles once (tensor-backed CB)
        in0_reader_kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/reader_in0.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=compute_cores,
            compile_time_args=in0_reader_compile_args,
            runtime_args=[],
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1,
                noc=ttnn.NOC.NOC_0,
            ),
        )

        in1_reader_kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/reader_in1.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=compute_cores,
            compile_time_args=in1_reader_compile_args,
            runtime_args=in1_reader_rt_args,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=in1_noc,
            ),
        )

        compute_kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=compute_cores,
            compile_time_args=compute_compile_args,
            defines=mm_kernel_defines,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=math_fidelity,
                fp32_dest_acc_en=fp32_dest_acc_en,
                math_approx_mode=math_approx_mode,
            ),
        )

        # Create program
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[in0_reader_kernel, in1_reader_kernel, compute_kernel],
            semaphores=[],
            cbs=cb_descriptors,
        )

        # Execute
        io_tensors = [input_a, input_b, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
