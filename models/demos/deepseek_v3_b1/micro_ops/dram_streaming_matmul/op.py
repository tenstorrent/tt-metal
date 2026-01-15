# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Simplified DRAM Streaming Matmul Operation.

This implements a multicore matmul with DRAM sharded weights where:
- Input A (in0): REPLICATED on all compute cores (each core has full tensor)
- Input B (in1): Stored in DRAM, streamed by each compute core
- Output: Computed directly on compute cores, sharded across N dimension

No multicast needed - each core has its own copy of in0.
CBs are backed directly by tensors (no L1-to-L1 copies).
"""

from typing import Optional, Tuple

import torch
from loguru import logger

import ttnn


def get_batch_size(shape):
    """Calculate batch size from tensor shape (all dims except last 2)."""
    if len(shape) <= 2:
        return 1
    batch = 1
    for i in range(len(shape) - 2):
        batch *= shape[i]
    return batch


def get_max_page_size_and_num_pages(device, num_tiles: int, tile_size: int) -> Tuple[int, int]:
    """Calculate optimal page size and number of pages for DRAM reads."""
    total_size = num_tiles * tile_size

    arch = device.arch()
    if arch == ttnn.device.Arch.WORMHOLE_B0:
        noc_max_page_size = 8192
    elif arch == ttnn.device.Arch.BLACKHOLE:
        noc_max_page_size = 16384
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    page_size = (noc_max_page_size // tile_size) * tile_size
    while total_size % page_size != 0 and page_size >= tile_size:
        page_size -= tile_size

    num_pages = total_size // page_size
    return page_size, num_pages


def get_matmul_subblock_params(per_core_M: int, per_core_N: int, fp32_dest_acc_en: bool = False) -> Tuple[int, int]:
    """Get optimal subblock dimensions for matmul."""
    max_subblock_tiles = 4 if fp32_dest_acc_en else 8

    out_subblock_h = 1
    out_subblock_w = min(per_core_N, max_subblock_tiles)

    for h in range(min(per_core_M, max_subblock_tiles), 0, -1):
        if per_core_M % h == 0:
            for w in range(min(per_core_N, max_subblock_tiles // h), 0, -1):
                if per_core_N % w == 0 and h * w <= max_subblock_tiles:
                    out_subblock_h = h
                    out_subblock_w = w
                    return out_subblock_h, out_subblock_w

    return out_subblock_h, out_subblock_w


class DRAMStreamingMatmul:
    """
    Simplified DRAM streaming matmul using ttnn.generic_op.

    - Input A: REPLICATED on compute cores (each core has full [M, K])
    - Input B: WIDTH_SHARDED in DRAM, streamed per core
    - Output: WIDTH_SHARDED on compute cores [M, N_per_core]
    """

    @staticmethod
    def golden(
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        activation: Optional[str] = None,
    ) -> torch.Tensor:
        """PyTorch reference implementation."""
        output = input_a @ input_b

        if bias is not None:
            output = output + bias

        if activation is not None:
            if activation.lower() == "relu":
                output = torch.relu(output)
            elif activation.lower() == "gelu":
                output = torch.nn.functional.gelu(output)
            elif activation.lower() == "silu":
                output = torch.nn.functional.silu(output)

        return output

    @staticmethod
    def op(
        input_a: ttnn.Tensor,
        input_b: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        bias: Optional[ttnn.Tensor] = None,
        in0_block_w: int = 1,
        fused_activation: Optional[str] = None,
        fp32_dest_acc_en: bool = False,
        packer_l1_acc: bool = False,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        math_approx_mode: bool = False,
        untilize_out: bool = False,
    ) -> ttnn.Tensor:
        """
        Execute simplified DRAM streaming matmul.

        Key simplifications:
        - in0 is REPLICATED on compute cores (no multicast)
        - CBs are backed directly by tensors (no L1-to-L1 copies)
        - Each core streams its portion of in1 from DRAM
        """
        device = input_a.device()

        # Get tensor shapes
        a_shape = input_a.shape
        b_shape = input_b.shape

        # Get tiles
        in0_tile = input_a.get_tile()
        in1_tile = input_b.get_tile()
        out_tile = output_tensor.get_tile()

        in0_tile_shape = in0_tile.tile_shape
        in1_tile_shape = in1_tile.tile_shape

        # Calculate dimensions in tiles
        B = 1
        Mt = (get_batch_size(a_shape) * a_shape[-2]) // in0_tile_shape[0]
        Kt = a_shape[-1] // in0_tile_shape[1]
        Nt = b_shape[-1] // in1_tile_shape[1]

        # Validate
        assert a_shape[-1] == b_shape[-2], f"K dimension mismatch: {a_shape[-1]} vs {b_shape[-2]}"
        assert Kt % in0_block_w == 0, f"Kt ({Kt}) must be divisible by in0_block_w ({in0_block_w})"

        # Data formats
        in0_dtype = input_a.dtype
        in1_dtype = input_b.dtype
        out_dtype = output_tensor.dtype
        bias_dtype = bias.dtype if bias is not None else ttnn.bfloat16

        # Get compute cores
        in1_noc = ttnn.NOC.NOC_0
        all_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(in1_noc)
        num_cores = len(all_worker_cores)

        logger.debug(f"num_cores={num_cores}, Mt={Mt}, Kt={Kt}, Nt={Nt}")

        # Per-core dimensions
        # in0: REPLICATED - each core has full [M, K]
        # out: WIDTH_SHARDED - each core has [M, N/num_cores]
        per_core_M = 1  # Always 1 tile in M dimension
        per_core_N_compute = (Nt + num_cores - 1) // num_cores
        per_core_N_in1_sender = per_core_N_compute

        # Subblock optimization
        out_subblock_h, out_subblock_w = get_matmul_subblock_params(per_core_M, per_core_N_compute, fp32_dest_acc_en)

        max_subblock_w = 4 if fp32_dest_acc_en else 8
        if out_subblock_h == 1 and out_subblock_w < max_subblock_w:
            num_subblock_w = per_core_N_compute // out_subblock_w
            for new_w in range(out_subblock_w + 1, max_subblock_w + 1):
                new_num_subblock_w = (per_core_N_compute + new_w - 1) // new_w
                if new_num_subblock_w < num_subblock_w:
                    num_subblock_w = new_num_subblock_w
                    out_subblock_w = new_w
            per_core_N_compute = out_subblock_w * num_subblock_w

        logger.debug(f"per_core_M={per_core_M}, per_core_N_compute={per_core_N_compute}")
        logger.debug(f"out_subblock_h={out_subblock_h}, out_subblock_w={out_subblock_w}")

        num_blocks = Kt // in0_block_w

        # Tile sizes
        in0_tile_size = in0_tile.get_tile_size(in0_dtype)
        in1_tile_size = in1_tile.get_tile_size(in1_dtype)
        out_tile_size = out_tile.get_tile_size(out_dtype)
        bias_tile_size = in1_tile.get_tile_size(bias_dtype) if bias else out_tile_size

        # Intermediate format
        packer_l1_acc_en = packer_l1_acc and num_blocks > 1
        interm_dtype = ttnn.bfloat16
        interm_tile_size = out_tile.get_tile_size(interm_dtype)

        # CB sizes
        # in0: full tensor per core (replicated), double buffered for streaming
        in0_block_tiles = per_core_M * in0_block_w
        in0_CB_tiles = in0_block_tiles * 2 if B * num_blocks > 1 else in0_block_tiles

        # in1: per-core portion from DRAM
        in1_block_tiles = per_core_N_in1_sender * in0_block_w
        in1_CB_tiles = in1_block_tiles * 3 if B * num_blocks > 1 else in1_block_tiles
        in1_CB_size = in1_CB_tiles * in1_tile_size

        # Output
        out_block_tiles = per_core_M * per_core_N_compute
        interm_CB_size = out_block_tiles * interm_tile_size

        # Bias
        bias_block_tiles = per_core_N_in1_sender if bias else 0
        bias_CB_size = bias_block_tiles * bias_tile_size

        # DRAM read page sizes
        in1_page_size, in1_num_pages = get_max_page_size_and_num_pages(device, in1_block_tiles, in1_tile_size)
        bias_page_size, bias_num_pages = (
            get_max_page_size_and_num_pages(device, bias_block_tiles, bias_tile_size) if bias else (0, 0)
        )

        # Core ranges - use specific compute cores, not bounding box
        compute_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in all_worker_cores]
        )

        # Tile descriptors
        in0_tile_desc = ttnn.TileDescriptor(in0_tile)
        in1_tile_desc = ttnn.TileDescriptor(in1_tile)
        out_tile_desc = ttnn.TileDescriptor(out_tile)

        # ========== CIRCULAR BUFFERS ==========
        # Key: CB0 and CB4 are backed directly by tensors (no copies needed)

        # CB 0: in0 - BACKED BY INPUT TENSOR (replicated)
        # Each core reads directly from its local copy of in0
        cb0_descriptor = ttnn.cb_descriptor_from_sharded_tensor(0, input_a)

        # CB 1: in1 - working buffer for DRAM reads
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

        # CB 3: bias (if present)
        if bias is not None:
            cb3_format = ttnn.CBFormatDescriptor(
                buffer_index=3,
                data_format=bias_dtype,
                page_size=bias_tile_size,
                tile=in1_tile_desc,
            )
            cb3_descriptor = ttnn.CBDescriptor(
                total_size=bias_CB_size,
                core_ranges=compute_cores,
                format_descriptors=[cb3_format],
            )
            cb_descriptors.append(cb3_descriptor)

        # CB 4: output - BACKED BY OUTPUT TENSOR
        # Compute writes directly to output tensor memory
        cb4_descriptor = ttnn.cb_descriptor_from_sharded_tensor(4, output_tensor)
        cb_descriptors.append(cb4_descriptor)

        # CB 5: intermediate (for packer L1 acc)
        cb5_format = ttnn.CBFormatDescriptor(
            buffer_index=5,
            data_format=interm_dtype,
            page_size=interm_tile_size,
            tile=out_tile_desc,
        )
        cb5_descriptor = ttnn.CBDescriptor(
            total_size=interm_CB_size,
            core_ranges=compute_cores,
            format_descriptors=[cb5_format],
        )
        cb_descriptors.append(cb5_descriptor)

        # ========== SEMAPHORES ==========
        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(
                id=0,
                core_type=ttnn.CoreType.WORKER,
                core_ranges=compute_cores,
                initial_value=0,
            ),
        ]

        # ========== KERNEL ARGS ==========

        # Kernel defines
        mm_kernel_defines = []
        if bias is not None:
            mm_kernel_defines.append(("FUSE_BIAS", "1"))
        if fused_activation and fused_activation.lower() == "relu":
            mm_kernel_defines.append(("PACK_RELU", "1"))
        if packer_l1_acc_en:
            mm_kernel_defines.append(("PACKER_L1_ACC", "1"))
        if fp32_dest_acc_en:
            mm_kernel_defines.append(("FP32_DEST_ACC_EN", "1"))

        # CB IDs
        cb_id_in0 = 0
        cb_id_in1 = 1
        cb_id_in3 = 3  # bias
        cb_id_out = 4

        # in0 reader compile args - simple, just signals tiles are ready
        in0_block_num_tiles = out_subblock_h * in0_block_w * (per_core_M // out_subblock_h)
        in0_reader_compile_args = [
            cb_id_in0,
            in0_block_num_tiles,
            num_blocks,
        ]

        # Get buffer addresses (same for all cores, so can be compile args)
        in1_buffer_addr = input_b.buffer_address()
        bias_buffer_addr = bias.buffer_address() if bias is not None else 0

        # in1 reader compile args
        in1_reader_compile_args = [
            cb_id_in1,
            cb_id_in3,
            cb_id_out,
            in1_buffer_addr,
            bias_buffer_addr,
            in1_page_size,
            in1_num_pages,
            per_core_N_in1_sender,
            per_core_N_in1_sender * in0_block_w,
            num_blocks,
            out_block_tiles,
        ]
        if bias is not None:
            in1_reader_compile_args.extend([bias_page_size, bias_num_pages, 1])

        # Compute kernel compile args
        in0_num_subblocks = per_core_M // out_subblock_h
        in1_num_subblocks = per_core_N_compute // out_subblock_w
        in0_subblock_num_tiles = out_subblock_h * in0_block_w
        out_subblock_num_tiles = out_subblock_h * out_subblock_w

        compute_compile_args = [
            in0_block_w,
            in0_num_subblocks,
            in0_block_num_tiles,
            in0_subblock_num_tiles,
            in1_num_subblocks,
            in1_block_tiles,
            per_core_N_in1_sender,
            num_blocks,
            1,  # num_blocks_w_dim
            1,  # num_blocks_h_dim
            out_subblock_h,
            out_subblock_w,
            out_subblock_num_tiles,
            B,
            out_block_tiles,
            1 if untilize_out else 0,
            0,  # get_batch_from_reader
            0,  # in0_transpose_tile
        ]

        # in1 reader runtime args (per-core: bank_id and vc)
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

        # in0 reader - just signals tiles are ready (no copy, CB backed by tensor)
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

        in1_reader_defines = []
        if bias is not None:
            in1_reader_defines.append(("FUSE_BIAS", "1"))

        in1_reader_kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/reader_in1.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=compute_cores,
            compile_time_args=in1_reader_compile_args,
            defines=in1_reader_defines,
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
            semaphores=semaphore_descriptors,
            cbs=cb_descriptors,
        )

        # Execute
        io_tensors = [input_a, input_b, output_tensor]
        if bias is not None:
            io_tensors.insert(2, bias)

        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
