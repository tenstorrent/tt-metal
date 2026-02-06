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

Uses unified kernel style with UnifiedKernelDescriptor.
Core logic is in unified_kernels/dram_streaming_matmul.hpp.
bank_id and vc are computed in-kernel from grid position using linear_id_in_grid.
"""

import math

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


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
        mul_tensor: torch.Tensor = None,
        scalar_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        """PyTorch reference implementation with optional fused activation, mul, and scalar.

        If mul_tensor is provided, computes:
            activation(input_a @ input_b) * mul_tensor
        If scalar_tensor is also provided (1x16 tensor, uses index 0 as scalar):
            activation(input_a @ input_b) * mul_tensor * scalar_tensor[0]
        Otherwise:
            input_a @ input_b with optional fused_activation
        """
        result = input_a @ input_b
        if fused_activation is not None:
            activation_fn = {
                "silu": torch.nn.functional.silu,
            }.get(fused_activation.lower())
            if activation_fn is None:
                raise ValueError(f"Unknown activation for golden: {fused_activation}")
            result = activation_fn(result)
        if mul_tensor is not None:
            result = result * mul_tensor
        if scalar_tensor is not None:
            # Use index 0 as the scalar value (scalar_tensor is [1, 16])
            scalar_value = scalar_tensor.flatten()[0]
            result = result * scalar_value
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
        index_tensor: ttnn.Tensor = None,  # Expert index tensor for indexed access
        mul_tensor: ttnn.Tensor = None,  # Optional tensor to multiply with matmul output (16x16 tiles)
        mm_out_tensor: ttnn.Tensor = None,  # Optional intermediate tensor for matmul output (1x32 tiles)
        scalar_tensor: ttnn.Tensor = None,  # Optional scalar tensor (16x16 tile) to multiply after mul
    ) -> ttnn.Tensor:
        """
        Execute simplified DRAM streaming matmul.

        Key simplifications:
        - in0 is REPLICATED on compute cores (no multicast)
        - in1 is WIDTH_SHARDED [K, N] with per-shard tiles shuffled to column-major order
          (K tiles contiguous per N column in physical memory)
        - CBs are backed directly by tensors (no L1-to-L1 copies)
        - Simple loop: for each N output tile, accumulate across K

        Expert indexing (optional):
        - index_tensor: HEIGHT_SHARDED tensor with expert indices [1, 16] per core
          The first element is the expert index to use
        - When enabled, reads K tiles from in1 starting at expert_idx * K_tiles offset
          (K_tiles derived from input_a shard shape, no separate expert_k parameter needed)
        """
        enable_indexing = index_tensor is not None
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
        # When indexing is enabled, in1 has [K * num_experts, N] stacked experts
        in1_shard_shape = input_b.memory_config().shard_spec.shape
        K_from_in1 = in1_shard_shape[0]

        # Calculate dimensions in tiles
        Mt = M // in0_tile_shape[0]
        Kt = K // in0_tile_shape[1]
        per_core_N = in1_shard_shape[1] // in1_tile_shape[1]

        # Validate
        assert Mt == 1, f"Mt must be 1 for simplified matmul, got {Mt}"
        # With indexing, in1 is the first expert tensor (K rows); other experts are contiguous in DRAM
        # Kernel uses expert_idx * expert_size offset to access different experts
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

        # CB 5: index tensor (optional, for expert indexing)
        cb_id_index = 5
        if enable_indexing:
            cb5_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_id_index, index_tensor)
            cb_descriptors.append(cb5_descriptor)

        # Matmul output CB setup:
        # - When mul disabled: cb_id_mm_out = 4, backed by output_tensor (final output)
        # - When mul enabled: cb_id_mm_out = 8, backed by mm_out_tensor (intermediate)
        #   Plus CB 4,6,7 with 16x16 format for mul operation
        enable_mul = mul_tensor is not None
        cb_id_mul_in1 = 6  # mul_tensor viewed as 16x16
        cb_id_mul_in0 = 7  # mm_out viewed as 16x16

        if enable_mul:
            assert mm_out_tensor is not None, "mm_out_tensor required when mul_tensor is provided"
            cb_id_mm_out = 8  # matmul writes to intermediate CB

            # Get 16x16 tile format for the mul CBs
            mul_tile_16x16 = ttnn.Tile([16, 16])
            mul_tile_desc = ttnn.TileDescriptor(mul_tile_16x16)
            mul_dtype = mul_tensor.dtype
            mul_tile_size = mul_tile_16x16.get_tile_size(mul_dtype)

            # CB 4: output with 16x16 tile format, backed by output_tensor's memory
            cb4_format = ttnn.CBFormatDescriptor(
                buffer_index=4,
                data_format=mul_dtype,
                page_size=mul_tile_size,
                tile=mul_tile_desc,
            )
            cb4_descriptor = ttnn.CBDescriptor(
                total_size=mul_tile_size,  # 1 tile of 16x16
                core_ranges=compute_cores,
                format_descriptors=[cb4_format],
            )
            cb4_descriptor.set_buffer_from_tensor(output_tensor)
            cb_descriptors.append(cb4_descriptor)

            # CB 6: mul_in1 with 16x16 tile format, backed by mul_tensor's memory
            cb6_format = ttnn.CBFormatDescriptor(
                buffer_index=cb_id_mul_in1,
                data_format=mul_dtype,
                page_size=mul_tile_size,
                tile=mul_tile_desc,
            )
            cb6_descriptor = ttnn.CBDescriptor(
                total_size=mul_tile_size,  # 1 tile of 16x16
                core_ranges=compute_cores,
                format_descriptors=[cb6_format],
            )
            cb6_descriptor.set_buffer_from_tensor(mul_tensor)
            cb_descriptors.append(cb6_descriptor)

            # CB 7: mul_in0 with 16x16 tile format, backed by mm_out_tensor's memory
            cb7_format = ttnn.CBFormatDescriptor(
                buffer_index=cb_id_mul_in0,
                data_format=mul_dtype,
                page_size=mul_tile_size,
                tile=mul_tile_desc,
            )
            cb7_descriptor = ttnn.CBDescriptor(
                total_size=mul_tile_size,  # 1 tile of 16x16
                core_ranges=compute_cores,
                format_descriptors=[cb7_format],
            )
            cb7_descriptor.set_buffer_from_tensor(mm_out_tensor)
            cb_descriptors.append(cb7_descriptor)

            # CB 8: mm_out - matmul writes here (1x32 tiles, same as tensor)
            cb_mm_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_id_mm_out, mm_out_tensor)
            cb_descriptors.append(cb_mm_out_descriptor)
        else:
            cb_id_mm_out = 4  # matmul writes directly to output CB

            # CB 4: output - matmul writes here
            cb_mm_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_id_mm_out, output_tensor)
            cb_descriptors.append(cb_mm_out_descriptor)

        # Optional scalar tensor CB (for scalar multiply after mul)
        enable_scalar_mul = scalar_tensor is not None
        cb_id_scalar = 9  # 16x16 CB for mul operation (BRISC fills this)
        cb_id_scalar_src = 10  # CB backed by scalar tensor (NCRISC sets up, BRISC reads)

        if enable_scalar_mul:
            assert enable_mul, "scalar_tensor requires mul_tensor to be provided"
            scalar_dtype = scalar_tensor.dtype

            # CB 9: 16x16 working CB for scalar mul (NOT backed by tensor)
            # BRISC will read scalar from CB 10 and write one value here
            scalar_tile_16x16 = ttnn.Tile([16, 16])
            scalar_tile_desc_16x16 = ttnn.TileDescriptor(scalar_tile_16x16)
            scalar_tile_size_16x16 = scalar_tile_16x16.get_tile_size(scalar_dtype)

            cb9_format = ttnn.CBFormatDescriptor(
                buffer_index=cb_id_scalar,
                data_format=scalar_dtype,
                page_size=scalar_tile_size_16x16,
                tile=scalar_tile_desc_16x16,
            )
            cb9_descriptor = ttnn.CBDescriptor(
                total_size=scalar_tile_size_16x16,  # 1 tile of 16x16
                core_ranges=compute_cores,
                format_descriptors=[cb9_format],
            )
            cb_descriptors.append(cb9_descriptor)

            # CB 10: scalar source CB backed by scalar tensor (1x16 tile)
            cb10_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_id_scalar_src, scalar_tensor)
            cb_descriptors.append(cb10_descriptor)

        # ========== KERNEL ARGS ==========

        # CB IDs
        cb_id_in0 = 0
        cb_id_in1 = 1
        cb_id_out = 4  # final output CB

        # Fused SiLU activation (only silu supported for now)
        fuse_silu = 0
        if fused_activation is not None:
            if fused_activation.lower() != "silu":
                raise ValueError(f"Only 'silu' activation is supported, got: {fused_activation}")
            fuse_silu = 1

        # Get buffer addresses
        in1_buffer_addr = input_b.buffer_address()

        # Create per-core compile-time args for bank_id and vc
        # Each core gets a unique bank_id based on its position in all_worker_cores
        # VC conflict resolution: avoid NOC contention for cores on same row
        num_dram_banks = len(all_worker_cores)
        bank_id_core_values = []
        vc_core_values = []
        bank_ids = []
        for idx, core in enumerate(all_worker_cores):
            bank_id = idx % num_dram_banks
            # VC conflict resolution - avoid NOC contention for cores on same row
            vc = bank_id & 0x3
            for j in range(idx):
                prev_core = all_worker_cores[j]
                if prev_core.y == core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
                    vc = (vc + 1) & 0x3
                    break
            bank_ids.append(bank_id)
            bank_id_core_values.append((core, bank_id))
            vc_core_values.append((core, vc))

        # ========== UNIFIED KERNEL DESCRIPTOR ==========
        # Core logic is in unified_kernels/dram_streaming_matmul.hpp
        KERNEL_PATH = (
            "models/demos/deepseek_v3_b1/micro_ops/dram_streaming_matmul/kernels/dram_streaming_matmul_kernel.cpp"
        )

        # Number of output tiles for mul (using 16x16 tiles)
        # Total elements per core = M * per_core_N * tile_width
        # 16x16 tile = 256 elements
        if enable_mul:
            total_elements = M * per_core_N * in1_tile_shape[1]  # M * N_tiles * tile_width
            mul_num_tiles = math.ceil(total_elements / 256)
        else:
            mul_num_tiles = 0

        # Named compile-time args for NCRISC (DRAM streaming - uses NOC_0)
        ncrisc_named_compile_time_args = [
            # Sharded buffer setup args
            ("dram_mm_cb_in0", cb_id_in0),
            ("dram_mm_num_tiles_k", Kt),
            # DRAM streaming args
            ("dram_mm_cb_in1", cb_id_in1),
            ("dram_mm_cb_out", cb_id_mm_out),  # matmul output CB (4 or 8)
            ("dram_mm_in1_tensor_addr", in1_buffer_addr),
            ("dram_mm_in1_page_size", in1_page_size),
            ("dram_mm_in1_num_pages", in1_num_pages),
            ("dram_mm_subblock_k", subblock_k),
            ("dram_mm_per_core_n", per_core_N),
            ("dram_mm_in1_block_size_bytes", in1_block_size_bytes),
            ("dram_mm_out_num_tiles", out_num_tiles),
            ("dram_mm_num_subblocks_k", num_subblocks_k),
            # Expert indexing parameters
            ("dram_mm_enable_indexing", 1 if enable_indexing else 0),
            ("dram_mm_cb_index", cb_id_index if enable_indexing else 0),
            ("dram_mm_index_offset", 0),  # TODO: make configurable, offset into index tensor
            # Mul parameters
            ("dram_mm_enable_mul", 1 if enable_mul else 0),
            ("dram_mm_cb_mul_in1", cb_id_mul_in1),
            ("dram_mm_mul_num_tiles", mul_num_tiles),
            # Scalar mul parameters (NCRISC sets up scalar source CB)
            ("dram_mm_enable_scalar_mul", 1 if enable_scalar_mul else 0),
            ("dram_mm_cb_scalar_src", cb_id_scalar_src if enable_scalar_mul else 0),
        ]

        # Named compile-time args for BRISC (no-op for DRAM streaming, handles mul)
        brisc_named_compile_time_args = [
            # Mul parameters
            ("dram_mm_enable_mul", 1 if enable_mul else 0),
            ("dram_mm_cb_mul_in0", cb_id_mul_in0),
            ("dram_mm_cb_mul_in1", cb_id_mul_in1),
            ("dram_mm_cb_final_out", cb_id_out),
            ("dram_mm_mul_num_tiles", mul_num_tiles),
            # Scalar mul parameters
            ("dram_mm_enable_scalar_mul", 1 if enable_scalar_mul else 0),
            ("dram_mm_cb_scalar", cb_id_scalar if enable_scalar_mul else 0),
            ("dram_mm_cb_scalar_src", cb_id_scalar_src if enable_scalar_mul else 0),
        ]

        # Named compile-time args for TRISC (compute)
        trisc_named_compile_time_args = [
            ("dram_mm_cb_in0", cb_id_in0),
            ("dram_mm_cb_in1", cb_id_in1),
            ("dram_mm_cb_out", cb_id_mm_out),  # matmul output CB (4 or 8)
            ("dram_mm_subblock_k", subblock_k),
            ("dram_mm_per_core_n", per_core_N),
            ("dram_mm_subblock_w", subblock_w),
            ("dram_mm_num_subblocks_k", num_subblocks_k),
            ("dram_mm_tile_r_dim", in0_tile_shape[0]),
            ("dram_mm_fuse_silu", fuse_silu),
            # Mul parameters (16x16 tiles)
            ("dram_mm_enable_mul", 1 if enable_mul else 0),
            ("dram_mm_cb_mul_in0", cb_id_mul_in0),
            ("dram_mm_cb_mul_in1", cb_id_mul_in1),
            ("dram_mm_cb_mul_out", cb_id_out),
            ("dram_mm_mul_num_tiles", mul_num_tiles),
            # Scalar mul parameters
            ("dram_mm_enable_scalar_mul", 1 if enable_scalar_mul else 0),
            ("dram_mm_cb_scalar", cb_id_scalar if enable_scalar_mul else 0),
        ]

        # Unified kernel descriptor (same pattern as matmul)
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=KERNEL_PATH,
            core_ranges=compute_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=math_fidelity,
                fp32_dest_acc_en=fp32_dest_acc_en,
                math_approx_mode=math_approx_mode,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="dram_mm_is_active_core",
                    core_range=compute_cores,
                    value=1,
                    other_value=0,
                ),
            ],
            per_core_compile_time_descriptors=[
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="dram_mm_bank_id",
                    core_values=bank_id_core_values,
                    other_value=0,
                ),
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="dram_mm_vc",
                    core_values=vc_core_values,
                    other_value=0,
                ),
            ],
        )

        kernel_descriptors = unified_kernel.get_kernel_descriptors()

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernel_descriptors.kernels,
            semaphores=[],
            cbs=cb_descriptors,
        )

        # Execute
        io_tensors = [input_a, input_b, output_tensor]
        if enable_indexing:
            io_tensors.append(index_tensor)
        if enable_mul:
            io_tensors.append(mul_tensor)
            io_tensors.append(mm_out_tensor)
        if enable_scalar_mul:
            io_tensors.append(scalar_tensor)
        ttnn.generic_op(io_tensors, program_descriptor)

        # Output is written in-place to output_tensor
        return output_tensor
