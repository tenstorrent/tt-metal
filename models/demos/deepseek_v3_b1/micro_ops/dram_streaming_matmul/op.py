# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
DRAM Streaming Matmul Operation.

This implements a multicore matmul with DRAM sharded weights where:
- Input A (in0): Sharded in L1 across storage cores, multicast to compute cores
- Input B (in1): Stored in DRAM, streamed by each compute core
- Output: Computed results resharded back to storage cores

The operation uses multicast to distribute input A from storage cores to all
compute cores, while each compute core independently reads its portion of
weights from DRAM.
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
    """
    Calculate optimal page size and number of pages for DRAM reads.

    Args:
        device: The device to query architecture from
        num_tiles: Number of tiles to read
        tile_size: Size of each tile in bytes

    Returns:
        Tuple of (page_size, num_pages)
    """
    total_size = num_tiles * tile_size

    # NOC max burst size varies by architecture
    arch = device.arch()
    if arch == ttnn.device.Arch.WORMHOLE_B0:
        noc_max_page_size = 8192
    elif arch == ttnn.device.Arch.BLACKHOLE:
        noc_max_page_size = 16384
    else:
        raise RuntimeError(f"Unsupported architecture for DRAM sharded matmul: {arch}")

    page_size = (noc_max_page_size // tile_size) * tile_size
    while total_size % page_size != 0 and page_size >= tile_size:
        page_size -= tile_size

    num_pages = total_size // page_size
    return page_size, num_pages


def get_matmul_subblock_params(per_core_M: int, per_core_N: int, fp32_dest_acc_en: bool = False) -> Tuple[int, int]:
    """
    Get optimal subblock dimensions for matmul.

    Args:
        per_core_M: M dimension tiles per core
        per_core_N: N dimension tiles per core
        fp32_dest_acc_en: Whether FP32 dest accumulation is enabled

    Returns:
        Tuple of (out_subblock_h, out_subblock_w)
    """
    # Simple heuristic - can be refined based on actual hardware constraints
    max_subblock_tiles = 4 if fp32_dest_acc_en else 8

    out_subblock_h = 1
    out_subblock_w = min(per_core_N, max_subblock_tiles)

    # Try to find factors that work
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
    DRAM streaming matmul implementation using ttnn.generic_op.

    This operation performs matmul where:
    - Input A is sharded in L1 and multicast to all compute cores
    - Input B (weights) is stored in DRAM and streamed per core
    - Output is resharded back to storage cores

    Supports optional bias fusion and activation functions.
    """

    @staticmethod
    def golden(
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        activation: Optional[str] = None,
    ) -> torch.Tensor:
        """
        PyTorch reference implementation for validation.

        Args:
            input_a: Input tensor A [M, K]
            input_b: Input tensor B [K, N]
            bias: Optional bias tensor [1, N]
            activation: Optional activation function name ("relu", etc.)

        Returns:
            Output tensor [M, N]
        """
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
        per_core_M: int = 1,
        per_core_N: int = 1,
        fused_activation: Optional[str] = None,
        fp32_dest_acc_en: bool = False,
        packer_l1_acc: bool = False,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        math_approx_mode: bool = False,
        untilize_out: bool = False,
    ) -> ttnn.Tensor:
        """
        Execute DRAM streaming matmul operation using generic_op.

        Args:
            input_a: Input tensor A, sharded in L1 [M, K]
            input_b: Input tensor B (weights) in DRAM [K, N]
            output_tensor: Pre-allocated output tensor, sharded [M, N]
            bias: Optional bias tensor in DRAM [1, N]
            in0_block_w: Block width for K dimension (in tiles)
            per_core_M: M tiles per compute core
            per_core_N: N tiles per storage core (for output sharding)
            fused_activation: Optional activation function ("relu", etc.)
            fp32_dest_acc_en: Enable FP32 accumulation in dest registers
            packer_l1_acc: Enable L1 accumulation in packer
            math_fidelity: Math fidelity level
            math_approx_mode: Enable math approximation mode
            untilize_out: Whether to untilize the output

        Returns:
            Output tensor with matmul result
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
        B = 1  # Batch size (simplified)
        Mt = (get_batch_size(a_shape) * a_shape[-2]) // in0_tile_shape[0]
        Kt = a_shape[-1] // in0_tile_shape[1]
        Nt = b_shape[-1] // in1_tile_shape[1]

        # Validate dimensions
        assert a_shape[-1] == b_shape[-2], f"K dimension mismatch: {a_shape[-1]} vs {b_shape[-2]}"
        assert Kt % in0_block_w == 0, f"Kt ({Kt}) must be divisible by in0_block_w ({in0_block_w})"

        # Get data formats
        in0_dtype = input_a.dtype
        in1_dtype = input_b.dtype
        out_dtype = output_tensor.dtype
        bias_dtype = bias.dtype if bias is not None else ttnn.bfloat16

        # Get shard specs
        assert input_a.is_sharded(), "Input A must be sharded"
        assert output_tensor.is_sharded(), "Output must be sharded"

        input_shard_spec = input_a.memory_config().shard_spec
        output_shard_spec = output_tensor.memory_config().shard_spec

        input_all_storage_cores = input_shard_spec.grid
        output_all_storage_cores = output_shard_spec.grid

        # Get compute grid
        compute_grid = device.compute_with_storage_grid_size()
        num_mcast_cores = compute_grid.x * compute_grid.y

        # Get optimal worker core assignment for DRAM reads
        # in1 is the reader of weights/output writer, uses optimal read NOC
        # in0 is for multicast, uses optimal write NOC
        # On WH/BH: NOC_0 is preferred for reads, NOC_1 for writes
        in0_noc = ttnn.NOC.NOC_1  # preferred for dram write (mcast)
        in1_noc = ttnn.NOC.NOC_0  # preferred for dram read (weights)

        all_worker_cores_ordered = device.get_optimal_dram_bank_to_logical_worker_assignment(in1_noc)
        num_dram_banks = len(all_worker_cores_ordered)
        num_worker_cores = num_dram_banks

        # Calculate per-core dimensions
        per_core_N_compute = (Nt + num_worker_cores - 1) // num_worker_cores
        per_core_N_in1_sender = per_core_N_compute  # Keep original value for in1 kernel
        per_core_N_storage = per_core_N

        # Get subblock parameters
        out_subblock_h, out_subblock_w = get_matmul_subblock_params(per_core_M, per_core_N_compute, fp32_dest_acc_en)

        # Optimize subblock width if needed (pads per_core_N_compute for compute kernel)
        max_subblock_w = 4 if fp32_dest_acc_en else 8
        if out_subblock_h == 1 and out_subblock_w < max_subblock_w:
            num_subblock_w = per_core_N_compute // out_subblock_w
            for new_w in range(out_subblock_w + 1, max_subblock_w + 1):
                new_num_subblock_w = (per_core_N_compute + new_w - 1) // new_w
                if new_num_subblock_w < num_subblock_w:
                    num_subblock_w = new_num_subblock_w
                    out_subblock_w = new_w
            per_core_N_compute = out_subblock_w * num_subblock_w

        logger.debug(
            f"per_core_M={per_core_M}, per_core_N_compute={per_core_N_compute}, per_core_N_in1_sender={per_core_N_in1_sender}"
        )

        num_blocks = Kt // in0_block_w

        # Calculate tile sizes
        in0_tile_size = in0_tile.get_tile_size(in0_dtype)
        in1_tile_size = in1_tile.get_tile_size(in1_dtype)
        out_tile_size = out_tile.get_tile_size(out_dtype)
        bias_tile_size = in1_tile.get_tile_size(bias_dtype) if bias else out_tile_size

        # Intermediate format
        packer_l1_acc_en = packer_l1_acc and num_blocks > 1
        if packer_l1_acc_en:
            interm_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
        else:
            interm_dtype = ttnn.float32 if fp32_dest_acc_en else out_dtype
        interm_dtype = ttnn.bfloat16  # Match C++ behavior
        interm_tile_size = out_tile.get_tile_size(interm_dtype)

        # CB sizes
        in0_block_tiles = per_core_M * in0_block_w
        in0_CB_tiles = in0_block_tiles * 2 if B * num_blocks > 1 else in0_block_tiles
        in0_CB_size = in0_CB_tiles * in0_tile_size

        # in1 uses original per_core_N (unpadded) for DRAM reads
        in1_block_tiles = per_core_N_in1_sender * in0_block_w
        in1_CB_tiles = in1_block_tiles * 3 if B * num_blocks > 1 else in1_block_tiles
        in1_CB_size = in1_CB_tiles * in1_tile_size

        out_block_tiles = per_core_M * per_core_N_compute
        out_CB_size = out_block_tiles * out_tile_size
        interm_CB_size = out_block_tiles * interm_tile_size

        out_reshard_tiles = per_core_M * per_core_N_storage
        out_reshard_CB_size = out_reshard_tiles * out_tile_size

        in0_shard_width_tiles = input_shard_spec.shape[1] // in0_tile_shape[1]
        in2_block_tiles = per_core_M * in0_shard_width_tiles
        in2_CB_size = in2_block_tiles * in0_tile_size

        # bias uses original per_core_N (unpadded)
        bias_block_tiles = per_core_N_in1_sender if bias else 0
        bias_CB_size = bias_block_tiles * bias_tile_size

        # Get page sizes for DRAM reads
        in1_page_size, in1_num_pages = get_max_page_size_and_num_pages(device, in1_block_tiles, in1_tile_size)
        bias_page_size, bias_num_pages = (
            get_max_page_size_and_num_pages(device, bias_block_tiles, bias_tile_size) if bias else (0, 0)
        )

        # Create core ranges
        input_storage_cores_vec = list(input_all_storage_cores.ranges())
        num_storage_cores = input_all_storage_cores.num_cores()
        num_blocks_per_shard = num_blocks // num_storage_cores

        # Build bounding box for all cores
        all_cores_list = []
        for cr in input_all_storage_cores.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    all_cores_list.append(ttnn.CoreCoord(x, y))

        for core in all_worker_cores_ordered:
            if core not in all_cores_list:
                all_cores_list.append(core)

        # Find bounding box
        min_x = min(c.x for c in all_cores_list)
        max_x = max(c.x for c in all_cores_list)
        min_y = min(c.y for c in all_cores_list)
        max_y = max(c.y for c in all_cores_list)

        bounding_box = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(min_x, min_y), ttnn.CoreCoord(max_x, max_y))])

        # Physical core coordinates for multicast
        top_left_physical = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        bottom_right_physical = device.worker_core_from_logical_core(
            ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
        )

        # Swap start/end based on NOC (in0_noc is used for mcast)
        if in0_noc == ttnn.NOC.NOC_1:
            start_core_noc = bottom_right_physical
            end_core_noc = top_left_physical
        else:
            start_core_noc = top_left_physical
            end_core_noc = bottom_right_physical

        # Tile descriptors
        in0_tile_desc = ttnn.TileDescriptor(in0_tile)
        in1_tile_desc = ttnn.TileDescriptor(in1_tile)
        out_tile_desc = ttnn.TileDescriptor(out_tile)

        # ========== CIRCULAR BUFFER DESCRIPTORS ==========

        # CB 0: in0 - Input A block buffer
        cb0_format = ttnn.CBFormatDescriptor(
            buffer_index=0,
            data_format=in0_dtype,
            page_size=in0_tile_size,
            tile=in0_tile_desc,
        )
        cb0_descriptor = ttnn.CBDescriptor(
            total_size=in0_CB_size,
            core_ranges=bounding_box,
            format_descriptors=[cb0_format],
        )

        # CB 1: in1 - Weights buffer
        cb1_format = ttnn.CBFormatDescriptor(
            buffer_index=1,
            data_format=in1_dtype,
            page_size=in1_tile_size,
            tile=in1_tile_desc,
        )
        cb1_descriptor = ttnn.CBDescriptor(
            total_size=in1_CB_size,
            core_ranges=bounding_box,
            format_descriptors=[cb1_format],
        )

        # CB 2: in2 - Sharded input A (backed by tensor)
        cb2_descriptor = ttnn.cb_descriptor_from_sharded_tensor(2, input_a)

        # CB 3: bias (if present)
        cb_descriptors = [cb0_descriptor, cb1_descriptor, cb2_descriptor]

        if bias is not None:
            cb3_format = ttnn.CBFormatDescriptor(
                buffer_index=3,
                data_format=bias_dtype,
                page_size=bias_tile_size,
                tile=in1_tile_desc,
            )
            cb3_descriptor = ttnn.CBDescriptor(
                total_size=bias_CB_size,
                core_ranges=bounding_box,
                format_descriptors=[cb3_format],
            )
            cb_descriptors.append(cb3_descriptor)

        # CB 4: output
        cb4_format = ttnn.CBFormatDescriptor(
            buffer_index=4,
            data_format=out_dtype,
            page_size=out_tile_size,
            tile=out_tile_desc,
        )
        cb4_descriptor = ttnn.CBDescriptor(
            total_size=out_CB_size,
            core_ranges=bounding_box,
            format_descriptors=[cb4_format],
        )
        cb_descriptors.append(cb4_descriptor)

        # CB 5: intermediate
        cb5_format = ttnn.CBFormatDescriptor(
            buffer_index=5,
            data_format=interm_dtype,
            page_size=interm_tile_size,
            tile=out_tile_desc,
        )
        cb5_descriptor = ttnn.CBDescriptor(
            total_size=interm_CB_size,
            core_ranges=bounding_box,
            format_descriptors=[cb5_format],
        )
        cb_descriptors.append(cb5_descriptor)

        # CB 6: output reshard (backed by tensor)
        cb6_descriptor = ttnn.cb_descriptor_from_sharded_tensor(6, output_tensor)
        cb_descriptors.append(cb6_descriptor)

        # ========== SEMAPHORE DESCRIPTORS ==========

        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(
                id=0,
                core_type=ttnn.CoreType.WORKER,
                core_ranges=bounding_box,
                initial_value=0,  # INVALID
            ),
            ttnn.SemaphoreDescriptor(
                id=1,
                core_type=ttnn.CoreType.WORKER,
                core_ranges=bounding_box,
                initial_value=0,  # INVALID
            ),
            ttnn.SemaphoreDescriptor(
                id=2,
                core_type=ttnn.CoreType.WORKER,
                core_ranges=bounding_box,
                initial_value=1,  # VALID
            ),
        ]

        # ========== KERNEL DESCRIPTORS ==========

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
        mm_kernel_defines.append(("MATMUL_DRAM_SHARDED", "1"))

        in1_transpose_tile = in1_tile.transpose_within_face and in1_tile.transpose_of_faces
        if in1_transpose_tile:
            mm_kernel_defines.append(("IN1_TRANSPOSE_TILE", "1"))

        # in0 sender compile time args
        in0_block_num_tiles = out_subblock_h * in0_block_w * (per_core_M // out_subblock_h)
        in0_sender_compile_args = [
            in0_block_num_tiles,
            in0_block_num_tiles * in0_tile_size,
            0,  # in0_last_ktile_w (assuming no padding needed)
            0,  # semaphore 0 id
            1,  # semaphore 1 id
            num_worker_cores,
            num_mcast_cores,
            num_blocks,
            start_core_noc.x,
            start_core_noc.y,
            end_core_noc.x,
            end_core_noc.y,
            2,  # semaphore 2 id
            num_blocks_per_shard,
        ]

        # in1 sender/writer compile time args
        # Uses per_core_N_in1_sender (original) for in1 block args
        # Uses per_core_N_compute (padded) for output stride
        in1_sender_compile_args = [
            in1_page_size,
            in1_num_pages,
            per_core_N_in1_sender,  # in1_block_w (original)
            per_core_N_in1_sender * in0_block_w,  # in1_block_num_tiles (original)
            num_blocks,
            out_block_tiles,
            per_core_N_compute * out_tile_size,  # out_tensor_stride_w_bytes (padded)
            per_core_N_storage * out_tile_size,  # out_reshard_tensor_stride_w_bytes
            per_core_M,
        ]
        if bias is not None:
            in1_sender_compile_args.extend([bias_page_size, bias_num_pages, 1])

        # Compute kernel compile args
        in0_num_subblocks = per_core_M // out_subblock_h
        in1_num_subblocks = per_core_N_compute // out_subblock_w  # Uses padded value
        in0_subblock_num_tiles = out_subblock_h * in0_block_w
        out_subblock_num_tiles = out_subblock_h * out_subblock_w

        compute_compile_args = [
            in0_block_w,
            in0_num_subblocks,
            in0_block_num_tiles,
            in0_subblock_num_tiles,
            in1_num_subblocks,
            in1_block_tiles,  # Uses per_core_N_in1_sender (original)
            per_core_N_in1_sender,  # in1_per_core_w - uses original value
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

        # Build lists of storage cores and worker cores
        input_storage_cores_list = []
        for cr in input_all_storage_cores.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    input_storage_cores_list.append((x, y))

        worker_cores_list = [(c.x, c.y) for c in all_worker_cores_ordered]

        # Find common cores (both storage and worker)
        storage_worker_common = []
        for coord in worker_cores_list[:]:
            if coord in input_storage_cores_list:
                storage_worker_common.append(coord)

        # Remove common cores from worker_cores_list (they become mcast senders)
        mcast_receiver_coords = [c for c in worker_cores_list if c not in storage_worker_common]

        # mcast_senders = storage cores (sorted by y then x)
        mcast_sender_coords = sorted(input_storage_cores_list, key=lambda c: (c[1], c[0]))

        # Get physical coordinates of mcast senders for runtime args
        storage_cores_physical_x = []
        storage_cores_physical_y = []
        for x, y in mcast_sender_coords:
            phys = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            storage_cores_physical_x.append(phys.x)
            storage_cores_physical_y.append(phys.y)

        # Get output storage core physical coordinates
        output_coords = []
        for cr in output_all_storage_cores.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    output_coords.append((x, y))
        # Sort output coords for consistent ordering
        output_coords = sorted(output_coords, key=lambda c: (c[1], c[0]))

        output_cores_physical_x = []
        output_cores_physical_y = []
        for x, y in output_coords:
            phys = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            output_cores_physical_x.append(phys.x)
            output_cores_physical_y.append(phys.y)

        # Build sets for quick lookup
        storage_cores_set = set(input_storage_cores_list)
        worker_cores_set = set(worker_cores_list)
        common_cores_set = set(storage_worker_common)

        # in0 sender runtime args - iterate in specific order matching C++
        in0_sender_rt_args = []

        # First, set args for mcast senders (storage cores)
        sender_id = 0
        for x, y in mcast_sender_coords:
            core = ttnn.CoreCoord(x, y)
            # mcast sender - 1, mcast sender + compute - 2
            if (x, y) in common_cores_set:
                worker_core_type = 2  # sender + compute
            else:
                worker_core_type = 1  # sender only

            rt_args = [worker_core_type, sender_id, 0]  # type, sender_id, is_last_ktile_padded
            rt_args.extend(storage_cores_physical_x)
            rt_args.extend(storage_cores_physical_y)
            in0_sender_rt_args.append((core, rt_args))
            sender_id += 1

        # Then, set args for mcast receivers (worker cores not in storage)
        for x, y in mcast_receiver_coords:
            core = ttnn.CoreCoord(x, y)
            worker_core_type = 3  # receiver + compute
            rt_args = [worker_core_type, 0, 0]
            rt_args.extend(storage_cores_physical_x)
            rt_args.extend(storage_cores_physical_y)
            in0_sender_rt_args.append((core, rt_args))

        # Finally, set args for idle cores in bounding box
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if (x, y) not in storage_cores_set and (x, y) not in worker_cores_set:
                    core = ttnn.CoreCoord(x, y)
                    rt_args = [0]  # idle
                    in0_sender_rt_args.append((core, rt_args))

        # in1 sender/writer runtime args
        # First, set non-worker cores in bounding box
        in1_sender_rt_args = []
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if (x, y) not in worker_cores_set:
                    core = ttnn.CoreCoord(x, y)
                    rt_args = [0]  # is_worker_core = false
                    in1_sender_rt_args.append((core, rt_args))

        # Get buffer addresses for DRAM tensors
        in1_buffer_addr = input_b.buffer_address()
        bias_buffer_addr = bias.buffer_address() if bias is not None else 0

        # Then, set worker cores in order
        bank_ids = []
        for i, worker_core in enumerate(all_worker_cores_ordered):
            core = ttnn.CoreCoord(worker_core.x, worker_core.y)
            bank_id = i % num_dram_banks

            # Calculate VC avoiding conflicts with cores on same row
            vc = bank_id & 0x3
            for j in range(i):
                prev_core = all_worker_cores_ordered[j]
                if prev_core.y == worker_core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
                    vc = (vc + 1) & 0x3
                    break
            bank_ids.append(bank_id)

            # Simplified resharding - each worker writes to one storage core
            storage_core_idx = i % len(output_coords) if output_coords else 0

            rt_args = [
                1,  # is_worker_core = true
                in1_buffer_addr,  # in1_tensor_addr
                bias_buffer_addr,  # bias_tensor_addr
                bank_id,
                vc,
                1,  # num_cores_write_back
                0,  # reshard_tensor_start_offset
                per_core_N_storage * out_tile_size,  # per_core_N_reshard_bytes
            ]
            if output_cores_physical_x:
                rt_args.append(output_cores_physical_x[storage_core_idx])
                rt_args.append(output_cores_physical_y[storage_core_idx])
            else:
                rt_args.extend([0, 0])

            in1_sender_rt_args.append((core, rt_args))

        # Compute runtime args - all cores in bounding box
        compute_rt_args = []
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                core = ttnn.CoreCoord(x, y)
                is_worker = (x, y) in worker_cores_set
                compute_rt_args.append((core, [1 if is_worker else 0]))

        # Create kernel descriptors
        # Local kernel paths relative to tt-metal root
        KERNEL_DIR = "models/demos/deepseek_v3_b1/micro_ops/dram_streaming_matmul/kernels"

        in0_sender_kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/reader_in0_sender.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=bounding_box,
            compile_time_args=in0_sender_compile_args,
            runtime_args=in0_sender_rt_args,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1,
                noc=in0_noc,
            ),
        )

        in1_writer_defines = [("OUT_SHARDED", "1"), ("SKIP_MCAST", "1")]
        if bias is not None:
            in1_writer_defines.append(("FUSE_BIAS", "1"))

        in1_sender_kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/reader_in1_sender_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=bounding_box,
            compile_time_args=in1_sender_compile_args,
            defines=in1_writer_defines,
            runtime_args=in1_sender_rt_args,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=in1_noc,
            ),
        )

        compute_kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/bmm_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=bounding_box,
            compile_time_args=compute_compile_args,
            defines=mm_kernel_defines,
            runtime_args=compute_rt_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=math_fidelity,
                fp32_dest_acc_en=fp32_dest_acc_en,
                math_approx_mode=math_approx_mode,
            ),
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[in0_sender_kernel, in1_sender_kernel, compute_kernel],
            semaphores=semaphore_descriptors,
            cbs=cb_descriptors,
        )

        # Execute generic op
        io_tensors = [input_a, input_b, output_tensor]
        if bias is not None:
            io_tensors.insert(2, bias)

        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
