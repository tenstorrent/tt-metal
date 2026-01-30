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

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
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


class DRAMStreamingMatmulProgramInfo:
    """
    Contains program descriptor and metadata for kernel fusion.

    This class holds all the information needed to fuse this op into a global program:
    - program_descriptor: The ProgramDescriptor for this op
    - io_tensors: Input/output tensors
    - num_cbs: Number of CBs used by this op
    - num_semaphores: Number of semaphores used by this op
    - num_runtime_args_per_core: Number of runtime args per core
    - core_ranges: The core ranges this op runs on
    """

    def __init__(
        self,
        program_descriptor,
        io_tensors,
        num_cbs,
        num_semaphores,
        num_runtime_args_per_core,
        core_ranges,
    ):
        self.program_descriptor = program_descriptor
        self.io_tensors = io_tensors
        self.num_cbs = num_cbs
        self.num_semaphores = num_semaphores
        self.num_runtime_args_per_core = num_runtime_args_per_core
        self.core_ranges = core_ranges


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
        # Create program info with dsm_ prefix for standalone execution
        program_info = DRAMStreamingMatmul.create_program_info(
            input_a=input_a,
            input_b=input_b,
            output_tensor=output_tensor,
            fp32_dest_acc_en=fp32_dest_acc_en,
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            subblock_k=subblock_k,
            fused_activation=fused_activation,
            cb_offset=0,
            sem_offset=0,
            prefix="dsm_",
        )

        # Execute
        output = ttnn.generic_op(program_info.io_tensors, program_info.program_descriptor)
        return output

    @staticmethod
    def create_program_info(
        input_a: ttnn.Tensor,
        input_b: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        fp32_dest_acc_en: bool = False,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        math_approx_mode: bool = False,
        subblock_k: int = None,
        fused_activation: str = None,
        cb_offset: int = 0,
        sem_offset: int = 0,
        prefix: str = "",
        kernel_dir: str = "models/demos/deepseek_v3_b1/micro_ops/dram_streaming_matmul/kernels",
    ) -> DRAMStreamingMatmulProgramInfo:
        """
        Create program descriptor without executing.

        This method is used for kernel fusion - it returns all the information
        needed to fuse this op into a global program.

        Args:
            prefix: Prefix for named compile-time args (default "" for fusion, "dsm_" for standalone)
            cb_offset: Starting CB ID for this op (default 0). Op uses 3 CBs.
            sem_offset: Starting semaphore ID for this op (default 0). Op uses 0 semaphores.
            kernel_dir: Directory containing kernel files

        Returns:
            DRAMStreamingMatmulProgramInfo containing program descriptor and metadata.
        """
        # CB IDs are offset from cb_offset
        # This op uses 3 CBs: in0, in1, out
        cb_id_in0 = cb_offset + 0
        cb_id_in1 = cb_offset + 1
        cb_id_out = cb_offset + 2
        device = input_a.device()

        # Get tiles
        in0_tile = input_a.get_tile()
        in1_tile = input_b.get_tile()

        in0_tile_shape = in0_tile.tile_shape
        in1_tile_shape = in1_tile.tile_shape

        # Get dimensions from shard specs
        in0_shard_shape = input_a.memory_config().shard_spec.shape
        M = in0_shard_shape[0]
        K = in0_shard_shape[1]

        in1_shard_shape = input_b.memory_config().shard_spec.shape
        K_from_in1 = in1_shard_shape[0]

        # Calculate dimensions in tiles
        Mt = M // in0_tile_shape[0]
        Kt = K // in0_tile_shape[1]
        per_core_N = in1_shard_shape[1] // in1_tile_shape[1]

        # Validate
        assert Mt == 1, f"Mt must be 1 for simplified matmul, got {Mt}"
        assert K == K_from_in1, f"K dimension mismatch: {K} vs {K_from_in1}"

        # Determine subblock_k
        if subblock_k is None:
            subblock_k = Kt
        assert Kt % subblock_k == 0, f"Kt ({Kt}) must be divisible by subblock_k ({subblock_k})"
        num_subblocks_k = Kt // subblock_k

        # Determine subblock_w
        if fp32_dest_acc_en:
            max_subblock_w = 8 if per_core_N <= 8 else 4
        else:
            max_subblock_w = 16 if per_core_N <= 16 else 8

        subblock_w = max_subblock_w
        while subblock_w > 1 and per_core_N % subblock_w != 0:
            subblock_w -= 1

        # Data formats
        in1_dtype = input_b.dtype

        # Get compute cores
        in1_noc = ttnn.NOC.NOC_0
        all_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(in1_noc)

        # Tile size for in1
        in1_tile_size = in1_tile.get_tile_size(in1_dtype)

        # Calculate page size for NOC transfers
        in1_page_size, in1_num_pages = get_max_page_size_and_num_pages(device, subblock_k, in1_tile_size)
        in1_block_size_bytes = subblock_k * in1_tile_size

        # CB sizes
        num_in1_buffers = 3 * num_subblocks_k
        assert num_in1_buffers <= 15, f"num_in1_buffers ({num_in1_buffers}) exceeds NOC_MAX_TRANSACTION_ID (15)"
        in1_CB_tiles = subblock_k * num_in1_buffers
        in1_CB_size = in1_CB_tiles * in1_tile_size

        out_num_tiles = per_core_N

        # Core ranges
        compute_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in all_worker_cores]
        )

        # Tile descriptor for in1
        in1_tile_desc = ttnn.TileDescriptor(in1_tile)

        # ========== CIRCULAR BUFFERS ==========

        # CB in0 - BACKED BY INPUT TENSOR (replicated)
        cb0_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_id_in0, input_a)

        # CB in1 - working buffer for DRAM reads
        cb1_format = ttnn.CBFormatDescriptor(
            buffer_index=cb_id_in1,
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

        # CB out - BACKED BY OUTPUT TENSOR
        cb4_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_id_out, output_tensor)
        cb_descriptors.append(cb4_descriptor)

        # ========== KERNEL ARGS ==========

        # Get buffer addresses
        in1_buffer_addr = input_b.buffer_address()

        # Runtime args (per-core: bank_id and vc) for BRISC
        brisc_runtime_args = []
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
            brisc_runtime_args.append((core, rt_args))

        # ========== UNIFIED KERNEL ==========

        # Named compile-time args for NCRISC (reader_in0)
        ncrisc_named_compile_time_args = [
            (f"{prefix}in0", cb_id_in0),
            (f"{prefix}num_tiles_k", Kt),
        ]

        # Named compile-time args for BRISC (reader_in1)
        brisc_named_compile_time_args = [
            (f"{prefix}in1", cb_id_in1),
            (f"{prefix}out", cb_id_out),
            (f"{prefix}in1_tensor_addr", in1_buffer_addr),
            (f"{prefix}in1_page_size", in1_page_size),
            (f"{prefix}in1_num_pages", in1_num_pages),
            (f"{prefix}subblock_k", subblock_k),
            (f"{prefix}per_core_N", per_core_N),
            (f"{prefix}in1_block_size_bytes", in1_block_size_bytes),
            (f"{prefix}out_num_tiles", out_num_tiles),
            (f"{prefix}num_subblocks_k", num_subblocks_k),
        ]

        # Named compile-time args for TRISC (compute)
        trisc_named_compile_time_args = [
            (f"{prefix}in0", cb_id_in0),
            (f"{prefix}in1", cb_id_in1),
            (f"{prefix}out", cb_id_out),
            (f"{prefix}subblock_k", subblock_k),
            (f"{prefix}per_core_N", per_core_N),
            (f"{prefix}subblock_w", subblock_w),
            (f"{prefix}num_subblocks_k", num_subblocks_k),
            (f"{prefix}tile_r_dim", in0_tile_shape[0]),
        ]

        # Compute config defines
        trisc_defines = []
        if fused_activation is not None:
            if fused_activation.lower() != "silu":
                raise ValueError(f"Only 'silu' activation is supported, got: {fused_activation}")
            trisc_defines.append(("FUSE_SILU", "1"))

        # Create unified kernel descriptor
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=f"{kernel_dir}/dram_streaming_matmul_kernel.cpp",
            core_ranges=compute_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            brisc_runtime_args=brisc_runtime_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=math_fidelity,
                fp32_dest_acc_en=fp32_dest_acc_en,
                math_approx_mode=math_approx_mode,
                defines=trisc_defines,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_active_core",
                    core_range=compute_cores,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            semaphores=[],
            cbs=cb_descriptors,
        )

        io_tensors = [input_a, input_b, output_tensor]

        return DRAMStreamingMatmulProgramInfo(
            program_descriptor=program_descriptor,
            io_tensors=io_tensors,
            num_cbs=3,  # in0, in1, out
            num_semaphores=0,  # This op doesn't use semaphores
            num_runtime_args_per_core=2,  # bank_id, vc
            core_ranges=compute_cores,
        )
