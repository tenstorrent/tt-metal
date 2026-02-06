# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
MoE Routed Expert fused operation.

This implements the MoE routed expert computation:
1. Input: [1, K] tensor on sender core (outside compute grid)
2. Mcast input to N compute cores
3. Each core computes: routing matmul + sigmoid
4. Gather outputs back to sender core
5. Gate: top-K expert selection with normalized scores
6. Mcast expert indices to compute cores
7. DRAM streaming matmul + SiLU with indexed expert weights
8. Output: expert computation result [1, N] on compute cores
"""

import math

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
    """
    total_size = num_tiles * tile_size

    arch = device.arch()
    if arch == ttnn.device.Arch.WORMHOLE_B0:
        noc_max_page_size = 8192
    elif arch == ttnn.device.Arch.BLACKHOLE:
        noc_max_page_size = 16384
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    page_size = (noc_max_page_size // tile_size) * tile_size
    while total_size % page_size != 0 and page_size >= tile_size:
        page_size -= tile_size

    num_pages = total_size // page_size
    return page_size, num_pages


def setup_eltwise_mul(
    in0_tensor,
    in1_tensor,
    out_tensor,
    core_ranges,
    cb_in0_index,
    cb_in1_index,
    cb_out_index,
    per_core_n,
    cb_scalar_index,
    cb_scalar_src_index,
    scalar_src_tensor,
):
    """
    Set up parameters and CB descriptors for element-wise multiply with CB aliasing and scalar multiply.

    The mul operation uses 16x16 tiles while the input tensors use 1x32 tiles.
    This function creates aliased CB descriptors that view the same memory with 16x16 tile format.
    Always includes scalar multiply: out = in0 * in1 * scalar

    Args:
        in0_tensor: First input tensor (e.g., up_proj matmul output)
        in1_tensor: Second input tensor (e.g., gate_proj matmul output)
        out_tensor: Output tensor for fused result
        core_ranges: CoreRangeSet for compute cores
        cb_in0_index: CB index for first input (aliased)
        cb_in1_index: CB index for second input (aliased)
        cb_out_index: CB index for output
        per_core_n: Number of output tiles per core (in 1x32 format)
        cb_scalar_index: CB index for scalar working buffer (16x16 tile format)
        cb_scalar_src_index: CB index for scalar source (receives mcasted scalar)
        scalar_src_tensor: Tensor backing the scalar source CB (for CB descriptor)

    Returns:
        Dictionary with mul_num_tiles and CB descriptors
    """
    # Compute mul_num_tiles: total elements / 256 (16x16 tile)
    M = 1  # Output row dim
    tile_width = 32  # 1x32 tiles from matmul
    total_elements = M * per_core_n * tile_width
    mul_num_tiles = math.ceil(total_elements / 256)

    # 16x16 tile format for mul operation
    TILE_16x16 = ttnn.Tile((16, 16))
    tile_16x16_size = TILE_16x16.get_tile_size(ttnn.bfloat16)
    tile_16x16_desc = ttnn.TileDescriptor(TILE_16x16)

    # CB for in0: alias of in0_tensor with 16x16 tile format
    cb_in0_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_in0_index,
        data_format=ttnn.bfloat16,
        page_size=tile_16x16_size,
        tile=tile_16x16_desc,
    )
    cb_in0_descriptor = ttnn.CBDescriptor(
        total_size=mul_num_tiles * tile_16x16_size,
        core_ranges=core_ranges,
        format_descriptors=[cb_in0_format],
    )
    cb_in0_descriptor.set_buffer_from_tensor(in0_tensor)

    # CB for in1: alias of in1_tensor with 16x16 tile format
    cb_in1_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_in1_index,
        data_format=ttnn.bfloat16,
        page_size=tile_16x16_size,
        tile=tile_16x16_desc,
    )
    cb_in1_descriptor = ttnn.CBDescriptor(
        total_size=mul_num_tiles * tile_16x16_size,
        core_ranges=core_ranges,
        format_descriptors=[cb_in1_format],
    )
    cb_in1_descriptor.set_buffer_from_tensor(in1_tensor)

    # CB for out: output with 16x16 tile format (tensor-backed)
    cb_out_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_out_index,
        data_format=ttnn.bfloat16,
        page_size=tile_16x16_size,
        tile=tile_16x16_desc,
    )
    cb_out_descriptor = ttnn.CBDescriptor(
        total_size=mul_num_tiles * tile_16x16_size,
        core_ranges=core_ranges,
        format_descriptors=[cb_out_format],
    )
    cb_out_descriptor.set_buffer_from_tensor(out_tensor)

    # CB for scalar source: receives mcasted scalar (tensor-backed)
    cb_scalar_src_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_scalar_src_index, scalar_src_tensor)

    # CB for scalar working buffer: 16x16 tile format for compute
    # Only needs 1 tile (single scalar value at position [0,0])
    cb_scalar_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_scalar_index,
        data_format=ttnn.bfloat16,
        page_size=tile_16x16_size,
        tile=tile_16x16_desc,
    )
    cb_scalar_descriptor = ttnn.CBDescriptor(
        total_size=tile_16x16_size,
        core_ranges=core_ranges,
        format_descriptors=[cb_scalar_format],
    )

    return {
        "mul_num_tiles": mul_num_tiles,
        "cb_in0_descriptor": cb_in0_descriptor,
        "cb_in1_descriptor": cb_in1_descriptor,
        "cb_out_descriptor": cb_out_descriptor,
        "cb_scalar_index": cb_scalar_index,
        "cb_scalar_src_index": cb_scalar_src_index,
        "cb_scalar_descriptor": cb_scalar_descriptor,
        "cb_scalar_src_descriptor": cb_scalar_src_descriptor,
    }


def setup_dram_matmul(
    device,
    weights_tensor,
    output_tensor,
    core_ranges,
    cb_in1_index,
    cb_out_index,
    fp32_dest_acc_en,
    num_subblocks_k,
):
    """
    Set up parameters and CB descriptors for a DRAM streaming matmul operation.

    Args:
        device: TT device
        weights_tensor: Weight tensor (WIDTH_SHARDED in DRAM)
        output_tensor: Output tensor (WIDTH_SHARDED in L1)
        core_ranges: CoreRangeSet for compute cores
        cb_in1_index: CB index for weights working buffer
        cb_out_index: CB index for output
        fp32_dest_acc_en: Whether FP32 dest accumulation is enabled
        num_subblocks_k: Number of K subblocks

    Returns:
        Dictionary with all computed parameters and CB descriptors
    """
    # Get parameters from weights tensor
    weights_tile = weights_tensor.get_tile()
    weights_dtype = weights_tensor.dtype
    weights_tile_size = weights_tile.get_tile_size(weights_dtype)
    weights_shard_shape = weights_tensor.memory_config().shard_spec.shape
    K = weights_shard_shape[0]
    per_core_n = weights_shard_shape[1] // weights_tile.tile_shape[1]
    Kt = K // weights_tile.tile_shape[0]

    # Calculate subblock_k
    subblock_k = Kt // num_subblocks_k
    assert Kt % num_subblocks_k == 0, f"Kt ({Kt}) must be divisible by num_subblocks ({num_subblocks_k})"

    # Calculate page size for NOC transfers
    in1_page_size, in1_num_pages = get_max_page_size_and_num_pages(device, subblock_k, weights_tile_size)
    in1_block_size_bytes = subblock_k * weights_tile_size

    # CB in1: weights working buffer
    num_in1_buffers = 3 * num_subblocks_k
    assert num_in1_buffers <= 15, f"num_in1_buffers ({num_in1_buffers}) exceeds NOC_MAX_TRANSACTION_ID (15)"
    in1_CB_tiles = subblock_k * num_in1_buffers
    in1_CB_size = in1_CB_tiles * weights_tile_size

    weights_tile_desc = ttnn.TileDescriptor(weights_tile)
    cb_in1_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_in1_index,
        data_format=weights_dtype,
        page_size=weights_tile_size,
        tile=weights_tile_desc,
    )
    cb_in1_descriptor = ttnn.CBDescriptor(
        total_size=in1_CB_size,
        core_ranges=core_ranges,
        format_descriptors=[cb_in1_format],
    )

    # CB out: output (tensor-backed)
    cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out_index, output_tensor)

    # Calculate subblock_w based on fp32_dest_acc_en and per_core_n
    if fp32_dest_acc_en:
        max_subblock_w = 8 if per_core_n <= 8 else 4
    else:
        max_subblock_w = 16 if per_core_n <= 16 else 8

    subblock_w = max_subblock_w
    while subblock_w > 1 and per_core_n % subblock_w != 0:
        subblock_w -= 1

    # Tile dimensions
    tile_r_dim = weights_tile.tile_shape[0]

    return {
        # Dimension parameters
        "per_core_n": per_core_n,
        "Kt": Kt,
        "tile_r_dim": tile_r_dim,
        # Subblock parameters
        "num_subblocks_k": num_subblocks_k,
        "subblock_k": subblock_k,
        "subblock_w": subblock_w,
        # NOC transfer parameters
        "in1_page_size": in1_page_size,
        "in1_num_pages": in1_num_pages,
        "in1_block_size_bytes": in1_block_size_bytes,
        # Output parameters
        "out_num_tiles": per_core_n,
        # Buffer address
        "in1_tensor_addr": weights_tensor.buffer_address(),
        # CB descriptors
        "cb_in1_descriptor": cb_in1_descriptor,
        "cb_out_descriptor": cb_out_descriptor,
    }


def setup_gather(
    device,
    receiver_core,
    sender_core_ranges,
    num_senders,
    data_size_bytes_per_sender,
    src_cb,
    src_num_pages,
    dst_cb,
    dst_tensor,
    noc0_receiver_semaphore_id,
    noc1_receiver_semaphore_id,
    row_major=True,
    use_explicit_sender_index=False,
):
    """
    Set up parameters for a gather operation.

    Gather collects data from multiple sender cores to a single receiver core.

    Args:
        device: TT device
        receiver_core: Logical CoreCoord of the receiver (single core)
        sender_core_ranges: CoreRangeSet of sender cores
        num_senders: Total number of sender cores
        data_size_bytes_per_sender: Bytes each sender sends
        src_cb: Source CB index on sender cores
        src_num_pages: Number of pages to wait for in source CB
        dst_cb: Destination CB index on receiver core
        dst_tensor: Destination tensor on receiver core (for buffer address and CB descriptor)
        noc0_receiver_semaphore_id: Semaphore ID for NOC0 senders
        noc1_receiver_semaphore_id: Semaphore ID for NOC1 senders
        row_major: Grid traversal order (True=row-major, False=column-major)
        use_explicit_sender_index: If True, use explicit per-core sender index (for scattered cores)

    Returns:
        Dictionary with gather parameters for NCRISC and BRISC compile-time args
    """
    # Get receiver NOC coordinates
    receiver_core_noc = device.worker_core_from_logical_core(receiver_core)

    # Get sender grid bounding box
    sender_ranges = list(sender_core_ranges.ranges())
    sender_min_x = min(r.start.x for r in sender_ranges)
    sender_min_y = min(r.start.y for r in sender_ranges)
    sender_max_x = max(r.end.x for r in sender_ranges)
    sender_max_y = max(r.end.y for r in sender_ranges)

    # All senders use NOC0 for simplicity
    noc0_num_senders = num_senders
    noc1_num_senders = 0

    # Calculate dst_num_pages from destination tensor shard shape
    dst_shard_shape = dst_tensor.memory_config().shard_spec.shape
    dst_tile = dst_tensor.get_tile()
    dst_num_pages = (dst_shard_shape[0] * dst_shard_shape[1]) // (dst_tile.tile_shape[0] * dst_tile.tile_shape[1])

    # CB descriptor for destination
    dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(dst_cb, dst_tensor)

    return {
        # NCRISC (sender) args
        "dest_noc_x": receiver_core_noc.x,
        "dest_noc_y": receiver_core_noc.y,
        "data_size_bytes": data_size_bytes_per_sender,
        "receiver_semaphore_id": noc0_receiver_semaphore_id,
        "src_cb": src_cb,
        "src_num_pages": src_num_pages,
        "sender_grid_start_x": sender_min_x,
        "sender_grid_start_y": sender_min_y,
        "sender_grid_end_x": sender_max_x,
        "sender_grid_end_y": sender_max_y,
        "row_major": 1 if row_major else 0,
        "receiver_data_addr": dst_tensor.buffer_address(),
        # BRISC (receiver) args
        "noc0_num_senders": noc0_num_senders,
        "noc1_num_senders": noc1_num_senders,
        "noc0_receiver_semaphore_id": noc0_receiver_semaphore_id,
        "noc1_receiver_semaphore_id": noc1_receiver_semaphore_id,
        "dst_cb": dst_cb,
        "dst_num_pages": dst_num_pages,
        # CB descriptor
        "dst_cb_descriptor": dst_cb_descriptor,
        # Config
        "use_explicit_sender_index": use_explicit_sender_index,
    }


def setup_mcast(
    device,
    sender_core,
    mcast_grid,
    src_cb,
    src_tensor,
    dst_cb,
    dst_tensor,
    sender_semaphore_id,
    receiver_semaphore_id,
    data_size_bytes,
):
    """
    Set up parameters for a multicast operation.

    Mcast broadcasts data from a sender core to all cores in the mcast grid.

    Args:
        device: TT device
        sender_core: Logical CoreCoord of the sender (single core)
        mcast_grid: CoreRangeSet of destination cores (rectangular grid)
        src_cb: Source CB index on sender core
        src_tensor: Source tensor on sender core (for num_pages calculation)
        dst_cb: Destination CB index on receiver cores
        dst_tensor: Destination tensor on receiver cores (for CB descriptor)
        sender_semaphore_id: Semaphore ID for sender
        receiver_semaphore_id: Semaphore ID for receivers
        data_size_bytes: Total data size to mcast in bytes

    Returns:
        Dictionary with mcast parameters for compile-time args
    """
    # Get mcast grid bounds
    mcast_ranges = list(mcast_grid.ranges())
    mcast_grid_range = mcast_ranges[0]  # Single rectangular range
    mcast_grid_start = mcast_grid_range.start
    mcast_grid_end = mcast_grid_range.end

    # Get NOC coordinates for mcast destination
    dest_noc_start_core = device.worker_core_from_logical_core(mcast_grid_start)
    dest_noc_end_core = device.worker_core_from_logical_core(mcast_grid_end)

    # Calculate number of cores in mcast grid
    num_cores = (mcast_grid_end.x - mcast_grid_start.x + 1) * (mcast_grid_end.y - mcast_grid_start.y + 1)

    # Check if sender is part of receiver grid
    sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
    is_sender_part_of_receiver_grid = mcast_grid.contains(sender_core_grid)

    # Calculate num_pages from source tensor shard shape
    src_shard_shape = src_tensor.memory_config().shard_spec.shape
    src_tile = src_tensor.get_tile()
    src_num_pages = (src_shard_shape[0] * src_shard_shape[1]) // (src_tile.tile_shape[0] * src_tile.tile_shape[1])

    # CB descriptor for destination
    dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(dst_cb, dst_tensor)

    return {
        # Sender args (BRISC)
        "dest_noc_start_x": dest_noc_start_core.x,
        "dest_noc_start_y": dest_noc_start_core.y,
        "dest_noc_end_x": dest_noc_end_core.x,
        "dest_noc_end_y": dest_noc_end_core.y,
        "num_cores": num_cores,
        "sender_semaphore_id": sender_semaphore_id,
        "receiver_semaphore_id": receiver_semaphore_id,
        "data_size_bytes": data_size_bytes,
        "src_cb": src_cb,
        "src_num_pages": src_num_pages,
        "is_sender_part_of_receiver_grid": is_sender_part_of_receiver_grid,
        # Receiver args (NCRISC)
        "dst_cb": dst_cb,
        "dst_num_pages": src_num_pages,  # Same as src_num_pages
        # CB descriptor
        "dst_cb_descriptor": dst_cb_descriptor,
    }


def setup_sram_matmul(
    in0_cb,
    in1_cb,
    out_cb,
    weights_tensor,
    output_tensor,
    k_num_tiles,
    fused_activation=0,
):
    """
    Set up parameters for an SRAM matmul operation.

    SRAM matmul computes: output = input @ weights with optional fused activation.
    Weights and output are sharded in L1 (SRAM).

    Args:
        in0_cb: Input CB index (receives mcasted input)
        in1_cb: Weights CB index
        out_cb: Output CB index
        weights_tensor: Weight tensor (WIDTH_SHARDED in L1)
        output_tensor: Output tensor (WIDTH_SHARDED in L1)
        k_num_tiles: K dimension in tiles
        fused_activation: Activation to fuse (0=none, 1=sigmoid, 2=silu)

    Returns:
        Dictionary with matmul parameters and CB descriptors
    """
    # Get per-core output width in tiles from weights tensor
    weights_tile = weights_tensor.get_tile()
    weights_shard_shape = weights_tensor.memory_config().shard_spec.shape
    weights_shard_width = weights_shard_shape[1]
    out_w = weights_shard_width // weights_tile.tile_shape[1]

    # Get core grid from weights tensor
    core_grid = weights_tensor.memory_config().shard_spec.grid

    # CB descriptors
    weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in1_cb, weights_tensor)
    output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

    return {
        # CB indices
        "in0_cb": in0_cb,
        "in1_cb": in1_cb,
        "out_cb": out_cb,
        # Matmul parameters
        "k_num_tiles": k_num_tiles,
        "out_w": out_w,
        "fused_activation": fused_activation,
        # Core grid
        "core_grid": core_grid,
        "num_cores": core_grid.num_cores(),
        # CB descriptors
        "weights_cb_descriptor": weights_cb_descriptor,
        "output_cb_descriptor": output_cb_descriptor,
    }


def setup_eltwise_add(
    in0_tensor,
    in1_tensor,
    out_tensor,
    cb_in0_index,
    cb_in1_index,
    cb_out_index,
):
    """
    Set up parameters and CB descriptors for element-wise add with per-core indexing.

    Used after down_proj to add fused_add tensor. Each core uses sender_index to
    offset into the replicated in1 tensor.

    Args:
        in0_tensor: First input tensor (e.g., down_proj output), WIDTH_SHARDED
        in1_tensor: Second input tensor (e.g., fused_add), HEIGHT_SHARDED (replicated)
        out_tensor: Output tensor, WIDTH_SHARDED (padded to 32x32 tile)
        cb_in0_index: CB index for first input (aliased with 32x32 tile format)
        cb_in1_index: CB index for second input (replicated tensor)
        cb_out_index: CB index for output

    Returns:
        Dictionary with eltwise_add parameters and CB descriptors
    """
    # Get core ranges from in0_tensor (same as previous mm)
    core_ranges = in0_tensor.memory_config().shard_spec.grid
    compute_cores_list = ttnn.corerange_to_cores(core_ranges, row_wise=True)
    # Get tensor info
    in0_dtype = in0_tensor.dtype
    in0_shard_shape = in0_tensor.memory_config().shard_spec.shape
    in1_shard_shape = in1_tensor.memory_config().shard_spec.shape

    # Dimensions
    width_per_core = in0_shard_shape[1]  # per-core width (e.g., 896)
    total_width = in1_shard_shape[1]  # full width of replicated tensor (e.g., 7168)

    # Element size
    if in0_dtype == ttnn.bfloat16:
        element_size_bytes = 2
    else:
        raise ValueError(f"Unsupported dtype: {in0_dtype}")

    slice_size_bytes = width_per_core * element_size_bytes

    # Use 32x32 tile view for CB (not tensor's actual tile format)
    # This is CB aliasing - tensor uses 1x32 tiles but CB views as 32x32
    cb_tile_h = 32
    cb_tile_w = 32
    cb_tile_size_bytes = cb_tile_h * cb_tile_w * element_size_bytes
    cb_tile = ttnn.Tile([cb_tile_h, cb_tile_w])
    cb_tile_desc = ttnn.TileDescriptor(cb_tile)

    # CB sizes
    in0_size_bytes = slice_size_bytes
    in1_size_bytes = total_width * element_size_bytes

    # Number of pages for CB wait (total_size / page_size)
    # cb_in0: 1 page (in0_size_bytes / in0_size_bytes = 1)
    # cb_in1: multiple pages (in1_size_bytes / in0_size_bytes)
    in0_wait_tiles = in0_size_bytes // in0_size_bytes  # 1
    in1_wait_tiles = in1_size_bytes // in0_size_bytes

    # Number of output tiles (based on 32x32 CB view, not tensor tile format)
    out_shard_shape = out_tensor.memory_config().shard_spec.shape
    out_shard_elements = out_shard_shape[0] * out_shard_shape[1]
    cb_tile_elements = cb_tile_h * cb_tile_w
    num_tiles = out_shard_elements // cb_tile_elements
    assert num_tiles == 1, f"Expected 1 tile (32x32 view), got {num_tiles}"

    # CB for in0: down_proj output aliased with 32x32 tile format
    cb_in0_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_in0_index,
        data_format=in0_dtype,
        page_size=in0_size_bytes,
        tile=cb_tile_desc,
    )
    cb_in0_descriptor = ttnn.CBDescriptor(
        total_size=in0_size_bytes,
        core_ranges=core_ranges,
        format_descriptors=[cb_in0_format],
    )
    cb_in0_descriptor.set_buffer_from_tensor(in0_tensor)

    # CB for in1: replicated tensor (tensor-backed)
    cb_in1_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_in1_index,
        data_format=in0_dtype,
        page_size=in0_size_bytes,  # page_size = slice size for reading
        tile=cb_tile_desc,
    )
    cb_in1_descriptor = ttnn.CBDescriptor(
        total_size=in1_size_bytes,
        core_ranges=core_ranges,
        format_descriptors=[cb_in1_format],
    )
    cb_in1_descriptor.set_buffer_from_tensor(in1_tensor)

    # CB for out: output (tensor-backed, uses 32x32 CB view)
    cb_out_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_out_index,
        data_format=in0_dtype,
        page_size=cb_tile_size_bytes,
        tile=cb_tile_desc,
    )
    cb_out_descriptor = ttnn.CBDescriptor(
        total_size=num_tiles * cb_tile_size_bytes,
        core_ranges=core_ranges,
        format_descriptors=[cb_out_format],
    )
    cb_out_descriptor.set_buffer_from_tensor(out_tensor)

    # Per-core sender_index values
    sender_index_core_values = []
    for idx, core in enumerate(compute_cores_list):
        sender_index_core_values.append((core, idx))

    return {
        # CB indices
        "cb_in0": cb_in0_index,
        "cb_in1": cb_in1_index,
        "cb_out": cb_out_index,
        # Dimensions
        "num_tiles": num_tiles,
        "slice_size_bytes": slice_size_bytes,
        # Wait tiles
        "cb_in0_wait_tiles": in0_wait_tiles,
        "cb_in1_wait_tiles": in1_wait_tiles,
        # CB descriptors
        "cb_in0_descriptor": cb_in0_descriptor,
        "cb_in1_descriptor": cb_in1_descriptor,
        "cb_out_descriptor": cb_out_descriptor,
        # Per-core values
        "sender_index_core_values": sender_index_core_values,
    }


def setup_gate(
    input_cb,
    bias_cb,
    indices_cb,
    output_cb,
    output_indices_cb,
    input_tensor,
    bias_tensor,
    indices_tensor,
    output_scores_tensor,
    output_indices_tensor,
    eps=1e-20,
    scaling_factor=2.5,
    enable_sigmoid=False,
):
    """
    Set up parameters for the MoE gate operation.

    The gate computes top-K expert selection with normalized scores.

    Args:
        input_cb: Input CB index (receives gathered matmul output)
        bias_cb: Bias CB index
        indices_cb: Indices CB index
        output_cb: Output scores CB index
        output_indices_cb: Output indices CB index
        input_tensor: Input tensor (gathered matmul output)
        bias_tensor: Bias tensor
        indices_tensor: Indices tensor
        output_scores_tensor: Output scores tensor
        output_indices_tensor: Output indices tensor
        eps: Epsilon for numerical stability (default 1e-20)
        scaling_factor: Scaling factor for gate scores (default 2.5)
        enable_sigmoid: Whether to apply sigmoid (default False, already done in matmul)

    Returns:
        Dictionary with gate parameters and CB descriptors
    """
    import struct

    # Convert float parameters to uint32 bit patterns
    eps_uint32 = int.from_bytes(struct.pack("f", eps), byteorder="little")
    scaling_factor_uint32 = int.from_bytes(struct.pack("f", scaling_factor), byteorder="little")

    # CB descriptors
    input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)
    bias_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(bias_cb, bias_tensor)
    indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(indices_cb, indices_tensor)
    output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_scores_tensor)
    output_indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_indices_cb, output_indices_tensor)

    return {
        # CB indices
        "input_cb": input_cb,
        "bias_cb": bias_cb,
        "indices_cb": indices_cb,
        "output_cb": output_cb,
        "output_indices_cb": output_indices_cb,
        # Parameters (as uint32 bit patterns)
        "eps": eps_uint32,
        "scaling_factor": scaling_factor_uint32,
        "enable_sigmoid": 1 if enable_sigmoid else 0,
        # CB descriptors
        "input_cb_descriptor": input_cb_descriptor,
        "bias_cb_descriptor": bias_cb_descriptor,
        "indices_cb_descriptor": indices_cb_descriptor,
        "output_cb_descriptor": output_cb_descriptor,
        "output_indices_cb_descriptor": output_indices_cb_descriptor,
    }


class MoeRoutedExpert:
    """
    MoE Routed Expert fused operation implementation using ttnn.generic_op.
    """

    # Fused activation enum values (must match matmul.hpp FusedActivation enum)
    ACTIVATION_NONE = 0
    ACTIVATION_SIGMOID = 1
    ACTIVATION_SILU = 2

    @staticmethod
    def golden(
        input_tensor,
        routing_weights_tensor,
        bias_tensor,
        gate_proj_weights_dict=None,
        up_proj_weights_dict=None,
        down_proj_weights_dict=None,
        fused_add_tensor=None,
        eps=1e-20,
        scaling_factor=2.5,
        use_hardcoded_expert_index=False,
        hardcoded_expert_index=0,
        explicit_expert_scale=None,
    ):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) [1, K]
            routing_weights_tensor: Routing matmul weights (torch.Tensor) [K, N_routing]
            bias_tensor: Gate bias tensor (torch.Tensor) [1, 8, 32] or [16, 16]
            gate_proj_weights_dict: Dict mapping expert_idx -> gate_proj weights [1, 1, K, N_expert] (optional)
            up_proj_weights_dict: Dict mapping expert_idx -> up_proj weights [1, 1, K, N_expert] (optional)
            down_proj_weights_dict: Dict mapping expert_idx -> down_proj weights [1, 1, N_expert, K] (optional)
            fused_add_tensor: Tensor to add after down_proj [1, 1, 1, K] (optional)
            eps: Epsilon for numerical stability in gate
            scaling_factor: Scaling factor for gate scores

        Returns:
            Tuple of (top8_scores, top8_indices, final_output) tensors
            final_output is None if gate_proj_weights_dict is None
            final_output = down_proj_output + fused_add (if fused_add provided)
            down_proj_output = (silu(gate_proj) * up_proj) @ down_proj_weights
        """
        import torch

        from models.demos.deepseek_v3_b1.micro_ops.deepseek_moe_gate.op import DeepseekMoeGateSingleCore

        # 1. Routing matmul + sigmoid
        logits = input_tensor.float() @ routing_weights_tensor.float()
        scores = torch.sigmoid(logits)

        # 2. Gate: top-8 selection with normalized scores
        # Reshape to (1, 8, 32) for gate golden
        gate_input = scores.reshape(1, 8, 32)
        top8_scores, top8_indices = DeepseekMoeGateSingleCore.golden(
            gate_input, bias_tensor.float(), eps, scaling_factor, enable_sigmoid=False
        )

        # 3. Expert matmuls (if expert weights provided)
        if gate_proj_weights_dict is not None:
            # Get the selected expert index
            if use_hardcoded_expert_index:
                selected_expert_idx = hardcoded_expert_index
            else:
                selected_expert_idx = int(top8_indices[0, 0].item())

            # gate_proj: input @ weights + SiLU
            gate_proj_weights = gate_proj_weights_dict[selected_expert_idx]
            input_for_expert = input_tensor.reshape(1, 1, 1, -1).float()
            gate_proj_output = input_for_expert @ gate_proj_weights.float()
            gate_proj_output = torch.nn.functional.silu(gate_proj_output)

            # up_proj: input @ weights (no activation)
            up_proj_weights = up_proj_weights_dict[selected_expert_idx]
            up_proj_output = input_for_expert @ up_proj_weights.float()

            # Get expert scale (use explicit if provided, otherwise from gate output position 0)
            if explicit_expert_scale is not None:
                expert_scale = explicit_expert_scale
            else:
                expert_scale = top8_scores[0, 0].float()

            # Fused output: silu(gate_proj) * up_proj * expert_scale
            fused_output = gate_proj_output * up_proj_output * expert_scale

            # down_proj: fused_output @ weights (no activation)
            if down_proj_weights_dict is not None:
                down_proj_weights = down_proj_weights_dict[selected_expert_idx]
                down_proj_output = fused_output @ down_proj_weights.float()

                # Add fused_add if provided
                if fused_add_tensor is not None:
                    final_output = down_proj_output + fused_add_tensor.float()
                    return top8_scores, top8_indices, final_output
                return top8_scores, top8_indices, down_proj_output

            return top8_scores, top8_indices, fused_output

        return top8_scores, top8_indices, None

    @staticmethod
    def op(
        input_tensor,
        mcast_output_tensor,
        gate_mm_weights_tensor,
        gate_mm_output_tensor,
        gate_input_tensor,
        gate_bias_tensor,
        gate_indices_tensor,
        gate_output_scores_tensor,
        gate_output_indices_tensor,
        expert_index_tensor,
        expert_scale_tensor,
        gate_proj_weights_tensor,
        gate_proj_output_tensor,
        up_proj_weights_tensor,
        up_proj_mm_out_tensor,
        fused_output_tensor,
        down_proj_gather_output_tensor,
        down_proj_mcast_output_tensor,
        down_proj_weights_tensor,
        down_proj_output_tensor,
        fused_add_tensor,
        final_output_tensor,
        use_hardcoded_expert_index=False,  # For testing: always use expert index 0
    ):
        """
        Execute the full MoE routed expert fused operation:
        mcast + matmul + sigmoid + gather + gate + index_mcast + expert_scale_mcast + gate_proj + up_proj + mul +
        down_proj_gather + down_proj_mcast + down_proj

        Args:
            input_tensor: Input tensor [1, K] sharded on single sender core (outside matmul grid)
            mcast_output_tensor: Mcast output tensor [1, K] sharded on all cores that need input
                                 (routing matmul cores + DRAM matmul cores)
            gate_mm_weights_tensor: Gate matmul weights [K, N_routing] width-sharded on matmul cores
            gate_mm_output_tensor: Gate matmul output [1, N_routing] width-sharded (routing matmul output)
            gate_input_tensor: Gate input tensor [16, 16] on sender core (receives gathered output)
            gate_bias_tensor: Gate bias tensor [16, 16] on sender core
            gate_indices_tensor: Gate indices tensor [16, 16] on sender core
            gate_output_scores_tensor: Gate output scores [1, 16] on sender core (source for expert_scale mcast)
            gate_output_indices_tensor: Gate output indices [1, 16] on sender core (source for index mcast)
            expert_index_tensor: Expert index [1, 16] on mcast grid (receives mcasted indices)
            expert_scale_tensor: Expert scale [1, 16] on mcast grid (receives mcasted scale)
            gate_proj_weights_tensor: Expert gate_proj weights [K, N_expert] width-sharded in DRAM (first expert)
            gate_proj_output_tensor: Expert gate_proj matmul+SiLU output [1, N_expert] width-sharded on DRAM matmul cores
            up_proj_weights_tensor: Expert up_proj weights [K, N_expert] width-sharded in DRAM (first expert)
            up_proj_mm_out_tensor: Expert up_proj matmul output (intermediate) [1, N_expert] width-sharded
            fused_output_tensor: Fused output silu(gate_proj) * up_proj * expert_scale [1, N_expert] width-sharded
            down_proj_gather_output_tensor: Gathered fused output [1, N_expert] on sender core
            down_proj_mcast_output_tensor: Mcasted fused output [1, N_expert] on mcast grid
            down_proj_weights_tensor: down_proj weights [N_expert, K] width-sharded in DRAM (first expert)
            down_proj_output_tensor: down_proj output [1, K] width-sharded on DRAM matmul cores
            fused_add_tensor: fused_add tensor [1, K] height-sharded (replicated on all gate_proj cores)
            final_output_tensor: final output [1, K] width-sharded on gate_proj cores (padded to 32x32 tile)

        Returns:
            Tuple of (gate_output_scores_tensor, gate_output_indices_tensor, final_output_tensor)
        """
        # Get tensor properties
        input_shape = input_tensor.shape
        data_format = input_tensor.dtype

        # Tile definitions
        TILE_1x32 = ttnn.Tile((1, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)

        # Calculate K dimension in tiles
        K = input_shape[1]
        num_tiles_k = K // TILE_1x32.tile_shape[1]  # K / 32

        # Get input core grid (single core for input)
        input_memory_config = input_tensor.memory_config()
        input_core_grid = input_memory_config.shard_spec.grid
        input_core_ranges = list(input_core_grid.ranges())
        sender_core = input_core_ranges[0].start

        # Get device and compute grid
        mesh_device = input_tensor.device()
        device_grid_size = mesh_device.compute_with_storage_grid_size()

        # Get mesh shape (1x1 for single device)
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        # Get first device for setup calculations
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor)
        device = input_tensors_per_device[0].device()

        # Get DRAM matmul cores from output tensor's shard spec
        gate_proj_output_memory_config = gate_proj_output_tensor.memory_config()
        gate_proj_core_ranges = gate_proj_output_memory_config.shard_spec.grid

        # Mcast grid: use the mcast_output_tensor's rectangular grid directly
        mcast_grid = mcast_output_tensor.memory_config().shard_spec.grid

        # Semaphore IDs (used by both input mcast and down_proj_mcast)
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1

        # CB indices
        input_cb = 0  # Input tensor (sharded on sender core)
        gate_mm_input_cb = 1  # Mcast destination CB (receives input on all cores) - also used as expert matmul input
        gate_mm_weights_cb = 2  # Matmul weights (sharded on all cores)
        gate_mm_output_cb = 3  # Matmul output (intermediate on compute cores)
        gate_input_cb = 4  # Gate input (gathered output, tensor-backed on sender core)
        gate_bias_cb = 5  # Gate bias (tensor-backed on sender core)
        gate_indices_cb = 6  # Gate indices (tensor-backed on sender core)
        gate_output_cb = 7  # Gate output scores (tensor-backed on sender core)
        gate_output_indices_cb = 8  # Gate output indices (tensor-backed on sender core)
        gate_proj_cb_in1 = 9  # DRAM matmul weights working buffer
        gate_proj_cb_index = 10  # DRAM matmul index (receives mcasted indices)
        gate_proj_cb_out = 11  # DRAM matmul output (tensor-backed on compute cores)
        up_proj_cb_in1 = 12  # up_proj matmul weights working buffer
        up_proj_cb_mm_out = 13  # up_proj matmul output intermediate (before mul)
        # Mul CBs for fusing up_proj * gate_proj
        mul_cb_in0 = 14  # up_proj mm output aliased as 16x16 (same memory as up_proj_cb_mm_out)
        mul_cb_in1 = 15  # gate_proj output aliased as 16x16 (same memory as gate_proj_cb_out)
        mul_cb_out = 16  # fused output (tensor-backed)
        # down_proj CBs (gather reads directly from mul_cb_out, no separate CB needed)
        down_proj_gather_dst_cb = 17  # gathered fused output on sender core (tensor-backed)
        down_proj_mcast_dst_cb = 18  # mcasted fused output on compute cores (tensor-backed)
        down_proj_cb_in1 = 19  # down_proj weights working buffer
        down_proj_cb_out = 20  # down_proj output (tensor-backed)
        # Scalar multiply CBs for expert scale
        mul_cb_scalar_src = 21  # Receives mcasted expert scale (tensor-backed on gate_proj cores)
        mul_cb_scalar = 22  # Working buffer for scalar multiply (16x16 tile format)
        # Eltwise add CBs (after down_proj)
        # Note: add_cb_in0 must be separate CB with 32x32 tile format (not reusing down_proj_cb_out)
        # because down_proj_cb_out has 1x32 tile format but eltwise_add needs 32x32 view
        add_cb_in0 = 23  # down_proj output aliased with 32x32 tile format
        add_cb_in1 = 24  # fused_add (tensor-backed, replicated on all gate_proj cores)
        add_cb_out = 25  # final output (tensor-backed)

        # ========== Input Mcast Setup ==========
        input_mcast_data_size_bytes = num_tiles_k * tile_1x32_size

        input_mcast_params = setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=input_cb,
            src_tensor=input_tensor,
            dst_cb=gate_mm_input_cb,
            dst_tensor=mcast_output_tensor,
            sender_semaphore_id=mcast_data_sender_semaphore_id,
            receiver_semaphore_id=mcast_data_receiver_semaphore_id,
            data_size_bytes=input_mcast_data_size_bytes,
        )

        # ========== Gate MM (SRAM Matmul) Setup ==========
        gate_mm_params = setup_sram_matmul(
            in0_cb=gate_mm_input_cb,
            in1_cb=gate_mm_weights_cb,
            out_cb=gate_mm_output_cb,
            weights_tensor=gate_mm_weights_tensor,
            output_tensor=gate_mm_output_tensor,
            k_num_tiles=num_tiles_k,
            fused_activation=MoeRoutedExpert.ACTIVATION_SIGMOID,
        )

        # CB descriptors
        # CB 0: Input tensor (sharded on sender core)
        input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)

        # CB 1: Mcast destination (tensor-backed, sharded on all cores that need input)
        gate_mm_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_mm_input_cb, mcast_output_tensor)

        # Gather semaphore IDs (used by both gate_mm gather and down_proj_gather)
        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3

        # ========== Gate MM Gather Setup ==========
        # Gather data size: each compute core sends out_w tiles
        gate_mm_output_tile = gate_mm_output_tensor.get_tile()
        gate_mm_output_tile_size = gate_mm_output_tile.get_tile_size(data_format)
        gate_mm_gather_data_size_bytes = gate_mm_params["out_w"] * gate_mm_output_tile_size

        gate_mm_gather_params = setup_gather(
            device=device,
            receiver_core=sender_core,
            sender_core_ranges=gate_mm_params["core_grid"],
            num_senders=gate_mm_params["num_cores"],
            data_size_bytes_per_sender=gate_mm_gather_data_size_bytes,
            src_cb=gate_mm_params["out_cb"],
            src_num_pages=gate_mm_params["out_w"],
            dst_cb=gate_input_cb,
            dst_tensor=gate_input_tensor,
            noc0_receiver_semaphore_id=gather_noc0_receiver_semaphore_id,
            noc1_receiver_semaphore_id=gather_noc1_receiver_semaphore_id,
            row_major=False,  # Column-major grid
            use_explicit_sender_index=False,
        )

        # ========== Gate Setup ==========
        gate_params = setup_gate(
            input_cb=gate_input_cb,
            bias_cb=gate_bias_cb,
            indices_cb=gate_indices_cb,
            output_cb=gate_output_cb,
            output_indices_cb=gate_output_indices_cb,
            input_tensor=gate_input_tensor,
            bias_tensor=gate_bias_tensor,
            indices_tensor=gate_indices_tensor,
            output_scores_tensor=gate_output_scores_tensor,
            output_indices_tensor=gate_output_indices_tensor,
            eps=1e-20,
            scaling_factor=2.5,
            enable_sigmoid=False,  # Sigmoid already done in matmul
        )

        # ========== Index Mcast Setup ==========
        # CB 10: DRAM matmul index (receives mcasted indices from gate)
        gate_proj_cb_index_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_proj_cb_index, expert_index_tensor)

        # Get index tile size for compile-time args
        index_tile = expert_index_tensor.get_tile()
        index_dtype = expert_index_tensor.dtype
        index_tile_size = index_tile.get_tile_size(index_dtype)

        # Index mcast parameters
        index_mcast_sender_semaphore_id = mcast_data_sender_semaphore_id
        index_mcast_receiver_semaphore_id = mcast_data_receiver_semaphore_id
        index_mcast_num_pages = 1
        index_mcast_data_size_bytes = index_tile_size

        # ========== Expert Scale Mcast Setup ==========
        # Get expert scale tile size
        expert_scale_tile = expert_scale_tensor.get_tile()
        expert_scale_dtype = expert_scale_tensor.dtype
        expert_scale_tile_size = expert_scale_tile.get_tile_size(expert_scale_dtype)

        # Expert scale mcast parameters (different semaphores to avoid race condition with back-to-back mcasts)
        # Using UpdateSemaphoreAddr=true in kernel allows using different semaphore addresses
        expert_scale_mcast_sender_semaphore_id = 4  # Different from index_mcast
        expert_scale_mcast_receiver_semaphore_id = 5  # Different from index_mcast
        expert_scale_mcast_num_pages = 1
        expert_scale_mcast_data_size_bytes = expert_scale_tile_size

        # ========== DRAM Streaming Matmul Setup ==========
        gate_proj_params = setup_dram_matmul(
            device=device,
            weights_tensor=gate_proj_weights_tensor,
            output_tensor=gate_proj_output_tensor,
            core_ranges=gate_proj_core_ranges,
            cb_in1_index=gate_proj_cb_in1,
            cb_out_index=gate_proj_cb_out,
            fp32_dest_acc_en=True,
            num_subblocks_k=4,
        )
        gate_proj_cb_in1_descriptor = gate_proj_params["cb_in1_descriptor"]
        gate_proj_cb_out_descriptor = gate_proj_params["cb_out_descriptor"]

        # ========== up_proj Matmul Setup ==========
        up_proj_params = setup_dram_matmul(
            device=device,
            weights_tensor=up_proj_weights_tensor,
            output_tensor=up_proj_mm_out_tensor,  # Intermediate output (before mul)
            core_ranges=gate_proj_core_ranges,  # Same cores as gate_proj
            cb_in1_index=up_proj_cb_in1,
            cb_out_index=up_proj_cb_mm_out,  # Write to intermediate CB
            fp32_dest_acc_en=True,
            num_subblocks_k=4,
        )
        up_proj_cb_in1_descriptor = up_proj_params["cb_in1_descriptor"]
        up_proj_cb_mm_out_descriptor = up_proj_params["cb_out_descriptor"]

        # ========== Mul Setup (up_proj * gate_proj * expert_scale -> fused output) ==========
        mul_params = setup_eltwise_mul(
            in0_tensor=up_proj_mm_out_tensor,
            in1_tensor=gate_proj_output_tensor,
            out_tensor=fused_output_tensor,
            core_ranges=gate_proj_core_ranges,
            cb_in0_index=mul_cb_in0,
            cb_in1_index=mul_cb_in1,
            cb_out_index=mul_cb_out,
            per_core_n=gate_proj_params["per_core_n"],
            cb_scalar_index=mul_cb_scalar,
            cb_scalar_src_index=mul_cb_scalar_src,
            scalar_src_tensor=expert_scale_tensor,
        )
        mul_num_tiles = mul_params["mul_num_tiles"]
        mul_cb_in0_descriptor = mul_params["cb_in0_descriptor"]
        mul_cb_in1_descriptor = mul_params["cb_in1_descriptor"]
        mul_cb_out_descriptor = mul_params["cb_out_descriptor"]
        mul_cb_scalar_descriptor = mul_params["cb_scalar_descriptor"]
        mul_cb_scalar_src_descriptor = mul_params["cb_scalar_src_descriptor"]

        num_gate_proj_cores = gate_proj_core_ranges.num_cores()

        # ========== down_proj_gather Setup ==========
        # Data size per core: only valid data (per_core_n tiles of 1x32)
        # Note: mul outputs 16x16 tiles with padding, but we only send the valid portion
        down_proj_gather_data_size_bytes = gate_proj_params["per_core_n"] * tile_1x32_size

        down_proj_gather_params = setup_gather(
            device=device,
            receiver_core=sender_core,
            sender_core_ranges=gate_proj_core_ranges,
            num_senders=num_gate_proj_cores,
            data_size_bytes_per_sender=down_proj_gather_data_size_bytes,
            src_cb=mul_cb_out,
            src_num_pages=mul_num_tiles,
            dst_cb=down_proj_gather_dst_cb,
            dst_tensor=down_proj_gather_output_tensor,
            noc0_receiver_semaphore_id=gather_noc0_receiver_semaphore_id,
            noc1_receiver_semaphore_id=gather_noc1_receiver_semaphore_id,
            row_major=True,
            use_explicit_sender_index=True,  # Scattered optimal DRAM bank cores
        )
        down_proj_gather_dst_cb_descriptor = down_proj_gather_params["dst_cb_descriptor"]

        # ========== down_proj_mcast Setup ==========
        # Calculate mcast data size (fused output size)
        fused_output_shard = down_proj_gather_output_tensor.memory_config().shard_spec.shape
        down_proj_mcast_num_tiles = (fused_output_shard[0] * fused_output_shard[1]) // (
            TILE_1x32.tile_shape[0] * TILE_1x32.tile_shape[1]
        )
        down_proj_mcast_data_size_bytes = down_proj_mcast_num_tiles * tile_1x32_size

        down_proj_mcast_params = setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=down_proj_gather_dst_cb,
            src_tensor=down_proj_gather_output_tensor,
            dst_cb=down_proj_mcast_dst_cb,
            dst_tensor=down_proj_mcast_output_tensor,
            sender_semaphore_id=mcast_data_sender_semaphore_id,
            receiver_semaphore_id=mcast_data_receiver_semaphore_id,
            data_size_bytes=down_proj_mcast_data_size_bytes,
        )
        down_proj_mcast_dst_cb_descriptor = down_proj_mcast_params["dst_cb_descriptor"]

        # ========== down_proj DRAM Matmul Setup ==========
        down_proj_params = setup_dram_matmul(
            device=device,
            weights_tensor=down_proj_weights_tensor,
            output_tensor=down_proj_output_tensor,
            core_ranges=gate_proj_core_ranges,  # Same cores as gate_proj/up_proj
            cb_in1_index=down_proj_cb_in1,
            cb_out_index=down_proj_cb_out,
            fp32_dest_acc_en=True,  # Use FP32 accumulation for down_proj
            num_subblocks_k=2,
        )
        down_proj_cb_in1_descriptor = down_proj_params["cb_in1_descriptor"]
        down_proj_cb_out_descriptor = down_proj_params["cb_out_descriptor"]

        # ========== Eltwise Add Setup (down_proj + fused_add) ==========
        add_params = setup_eltwise_add(
            in0_tensor=down_proj_output_tensor,
            in1_tensor=fused_add_tensor,
            out_tensor=final_output_tensor,
            cb_in0_index=add_cb_in0,
            cb_in1_index=add_cb_in1,
            cb_out_index=add_cb_out,
        )
        add_cb_in0_descriptor = add_params["cb_in0_descriptor"]
        add_cb_in1_descriptor = add_params["cb_in1_descriptor"]
        add_cb_out_descriptor = add_params["cb_out_descriptor"]

        # Helper to create NCRISC compile-time args with chip-specific mesh_chip_id
        def create_ncrisc_compile_time_args(mesh_chip_id: int) -> list:
            return [
                # Mcast sender sharded buffer setup (for sender core)
                ("mcast_src_cb", input_mcast_params["src_cb"]),
                ("mcast_src_num_pages", input_mcast_params["src_num_pages"]),
                # Mcast receiver args
                ("mcast_data_receiver_semaphore", input_mcast_params["receiver_semaphore_id"]),
                ("mcast_dst_cb", input_mcast_params["dst_cb"]),
                ("mcast_dst_num_pages", input_mcast_params["dst_num_pages"]),
                # Matmul reader args (for sharded buffer setup)
                ("gate_mm_in0", gate_mm_params["in0_cb"]),
                ("gate_mm_in1", gate_mm_params["in1_cb"]),
                ("gate_mm_k_num_tiles", gate_mm_params["k_num_tiles"]),
                ("gate_mm_out_w", gate_mm_params["out_w"]),
                # Gather sender args (compute cores send to sender core)
                ("gather_dest_noc_x", gate_mm_gather_params["dest_noc_x"]),
                ("gather_dest_noc_y", gate_mm_gather_params["dest_noc_y"]),
                ("gather_data_size_bytes", gate_mm_gather_params["data_size_bytes"]),
                ("gather_receiver_semaphore_id", gate_mm_gather_params["receiver_semaphore_id"]),
                ("gather_src_cb", gate_mm_gather_params["src_cb"]),
                ("gather_src_num_pages", gate_mm_gather_params["src_num_pages"]),
                ("gather_sender_grid_start_x", gate_mm_gather_params["sender_grid_start_x"]),
                ("gather_sender_grid_start_y", gate_mm_gather_params["sender_grid_start_y"]),
                ("gather_sender_grid_end_x", gate_mm_gather_params["sender_grid_end_x"]),
                ("gather_sender_grid_end_y", gate_mm_gather_params["sender_grid_end_y"]),
                ("gather_row_major", gate_mm_gather_params["row_major"]),
                ("gather_receiver_data_addr", gate_mm_gather_params["receiver_data_addr"]),
                # Gate reader args (sender core)
                ("gate_input_cb", gate_params["input_cb"]),
                ("gate_bias_cb", gate_params["bias_cb"]),
                ("gate_input_indices_cb", gate_params["indices_cb"]),
                # Index mcast receiver args (compute cores)
                ("index_mcast_receiver_semaphore", index_mcast_receiver_semaphore_id),
                ("gate_proj_cb_index", gate_proj_cb_index),
                ("index_mcast_num_pages", index_mcast_num_pages),
                # Expert scale mcast receiver args (compute cores)
                ("expert_scale_mcast_receiver_semaphore", expert_scale_mcast_receiver_semaphore_id),
                ("mul_cb_scalar_src", mul_cb_scalar_src),
                ("expert_scale_mcast_num_pages", expert_scale_mcast_num_pages),
                # Mul reader args (setup mul_in1 buffer)
                ("mul_cb_in1", mul_cb_in1),
                ("mul_num_tiles", mul_num_tiles),
                # down_proj_gather sender args (gate_proj cores send fused output to sender core)
                ("down_proj_gather_dest_noc_x", down_proj_gather_params["dest_noc_x"]),
                ("down_proj_gather_dest_noc_y", down_proj_gather_params["dest_noc_y"]),
                ("down_proj_gather_data_size_bytes", down_proj_gather_params["data_size_bytes"]),
                ("down_proj_gather_receiver_semaphore_id", down_proj_gather_params["receiver_semaphore_id"]),
                ("down_proj_gather_src_cb", down_proj_gather_params["src_cb"]),
                ("down_proj_gather_src_num_pages", down_proj_gather_params["src_num_pages"]),
                ("down_proj_gather_sender_grid_start_x", down_proj_gather_params["sender_grid_start_x"]),
                ("down_proj_gather_sender_grid_start_y", down_proj_gather_params["sender_grid_start_y"]),
                ("down_proj_gather_sender_grid_end_x", down_proj_gather_params["sender_grid_end_x"]),
                ("down_proj_gather_sender_grid_end_y", down_proj_gather_params["sender_grid_end_y"]),
                ("down_proj_gather_row_major", down_proj_gather_params["row_major"]),
                ("down_proj_gather_receiver_data_addr", down_proj_gather_params["receiver_data_addr"]),
                # down_proj_mcast receiver args (compute cores)
                ("down_proj_mcast_receiver_semaphore", down_proj_mcast_params["receiver_semaphore_id"]),
                ("down_proj_mcast_dst_cb", down_proj_mcast_params["dst_cb"]),
                ("down_proj_mcast_dst_num_pages", down_proj_mcast_params["dst_num_pages"]),
                # Eltwise add args (CB indices and wait tiles for setup_sharded_buffer)
                ("add_cb_in0", add_cb_in0),
                ("add_cb_in1", add_cb_in1),
                ("add_cb_in0_wait_tiles", add_params["cb_in0_wait_tiles"]),
                ("add_cb_in1_wait_tiles", add_params["cb_in1_wait_tiles"]),
                # DRAM streaming matmul reader args (compute cores) - uses NOC_0
                ("gate_proj_cb_in1", gate_proj_cb_in1),
                ("gate_proj_cb_out", gate_proj_cb_out),
                ("gate_proj_in1_tensor_addr", gate_proj_params["in1_tensor_addr"]),
                ("gate_proj_in1_page_size", gate_proj_params["in1_page_size"]),
                ("gate_proj_in1_num_pages", gate_proj_params["in1_num_pages"]),
                ("gate_proj_subblock_k", gate_proj_params["subblock_k"]),
                ("gate_proj_per_core_n", gate_proj_params["per_core_n"]),
                ("gate_proj_in1_block_size_bytes", gate_proj_params["in1_block_size_bytes"]),
                ("gate_proj_out_num_tiles", gate_proj_params["out_num_tiles"]),
                ("gate_proj_num_subblocks_k", gate_proj_params["num_subblocks_k"]),
                ("gate_proj_index_offset", mesh_chip_id),  # Index offset (chip_id for mesh mode)
                # up_proj matmul reader args (compute cores) - writes to intermediate CB
                ("up_proj_cb_in1", up_proj_cb_in1),
                ("up_proj_in1_tensor_addr", up_proj_params["in1_tensor_addr"]),
                ("up_proj_in1_page_size", up_proj_params["in1_page_size"]),
                ("up_proj_in1_num_pages", up_proj_params["in1_num_pages"]),
                ("up_proj_subblock_k", up_proj_params["subblock_k"]),
                ("up_proj_per_core_n", up_proj_params["per_core_n"]),
                ("up_proj_in1_block_size_bytes", up_proj_params["in1_block_size_bytes"]),
                ("up_proj_out_num_tiles", up_proj_params["out_num_tiles"]),
                ("up_proj_num_subblocks_k", up_proj_params["num_subblocks_k"]),
                ("up_proj_cb_index", gate_proj_cb_index),  # Reuses same index CB as gate_proj
                ("up_proj_index_offset", mesh_chip_id),  # Index offset (chip_id for mesh mode)
                ("up_proj_cb_mm_out", up_proj_cb_mm_out),  # Intermediate output for up_proj (before mul)
                # down_proj DRAM matmul reader args (compute cores)
                ("down_proj_cb_in1", down_proj_cb_in1),
                ("down_proj_cb_out", down_proj_cb_out),
                ("down_proj_in1_tensor_addr", down_proj_params["in1_tensor_addr"]),
                ("down_proj_in1_page_size", down_proj_params["in1_page_size"]),
                ("down_proj_in1_num_pages", down_proj_params["in1_num_pages"]),
                ("down_proj_subblock_k", down_proj_params["subblock_k"]),
                ("down_proj_per_core_n", down_proj_params["per_core_n"]),
                ("down_proj_in1_block_size_bytes", down_proj_params["in1_block_size_bytes"]),
                ("down_proj_out_num_tiles", down_proj_params["out_num_tiles"]),
                ("down_proj_num_subblocks_k", down_proj_params["num_subblocks_k"]),
                ("down_proj_cb_index", gate_proj_cb_index),  # Reuses same index CB
                ("down_proj_index_offset", mesh_chip_id),  # Index offset (chip_id for mesh mode)
                # Testing flag: use hardcoded expert index instead of gate output
                ("use_hardcoded_expert_index", 1 if use_hardcoded_expert_index else 0),
            ]

        # Helper to create BRISC compile-time args with chip-specific mesh_chip_id
        def create_brisc_compile_time_args(mesh_chip_id: int) -> list:
            return [
                # Mcast sender args
                ("mcast_dest_noc_start_x", input_mcast_params["dest_noc_start_x"]),
                ("mcast_dest_noc_start_y", input_mcast_params["dest_noc_start_y"]),
                ("mcast_dest_noc_end_x", input_mcast_params["dest_noc_end_x"]),
                ("mcast_dest_noc_end_y", input_mcast_params["dest_noc_end_y"]),
                ("mcast_num_cores", input_mcast_params["num_cores"]),
                ("mcast_data_sender_semaphore", input_mcast_params["sender_semaphore_id"]),
                ("mcast_data_receiver_semaphore", input_mcast_params["receiver_semaphore_id"]),
                ("mcast_data_size_bytes", input_mcast_params["data_size_bytes"]),
                ("mcast_src_cb", input_mcast_params["src_cb"]),
                ("mcast_dst_cb", input_mcast_params["dst_cb"]),
                ("mcast_src_num_pages", input_mcast_params["src_num_pages"]),
                ("mcast_is_part_of_receiver_grid", input_mcast_params["is_sender_part_of_receiver_grid"]),
                # Gather receiver args (sender core receives from compute cores)
                ("gather_noc0_num_senders", gate_mm_gather_params["noc0_num_senders"]),
                ("gather_noc1_num_senders", gate_mm_gather_params["noc1_num_senders"]),
                ("gather_noc0_receiver_semaphore_id", gate_mm_gather_params["noc0_receiver_semaphore_id"]),
                ("gather_noc1_receiver_semaphore_id", gate_mm_gather_params["noc1_receiver_semaphore_id"]),
                ("gather_dst_cb", gate_mm_gather_params["dst_cb"]),
                ("gather_dst_num_pages", gate_mm_gather_params["dst_num_pages"]),
                # Gate writer args (sender core)
                ("gate_output_cb", gate_params["output_cb"]),
                ("gate_output_indices_cb", gate_params["output_indices_cb"]),
                # Index mcast sender args (sender core broadcasts indices to compute cores)
                ("index_mcast_sender_semaphore", index_mcast_sender_semaphore_id),
                ("index_mcast_receiver_semaphore", index_mcast_receiver_semaphore_id),
                ("index_mcast_data_size_bytes", index_mcast_data_size_bytes),
                ("index_mcast_num_pages", index_mcast_num_pages),
                ("gate_proj_cb_index", gate_proj_cb_index),
                # Expert scale mcast sender args (sender core broadcasts scale to compute cores)
                ("expert_scale_mcast_sender_semaphore", expert_scale_mcast_sender_semaphore_id),
                ("expert_scale_mcast_receiver_semaphore", expert_scale_mcast_receiver_semaphore_id),
                ("expert_scale_mcast_data_size_bytes", expert_scale_mcast_data_size_bytes),
                ("expert_scale_mcast_num_pages", expert_scale_mcast_num_pages),
                ("mul_cb_scalar_src", mul_cb_scalar_src),
                ("mul_cb_scalar", mul_cb_scalar),
                (
                    "mul_scalar_index_offset",
                    mesh_chip_id,
                ),  # Index into scalar source tensor (offset by chip_id for mesh mode)
                # Mul writer args (waits for final output)
                ("mul_cb_out", mul_cb_out),
                ("mul_num_tiles", mul_num_tiles),
                # down_proj_gather receiver args (sender core receives fused output from gate_proj cores)
                ("down_proj_gather_noc0_num_senders", down_proj_gather_params["noc0_num_senders"]),
                ("down_proj_gather_noc1_num_senders", down_proj_gather_params["noc1_num_senders"]),
                ("down_proj_gather_noc0_receiver_semaphore_id", down_proj_gather_params["noc0_receiver_semaphore_id"]),
                ("down_proj_gather_noc1_receiver_semaphore_id", down_proj_gather_params["noc1_receiver_semaphore_id"]),
                ("down_proj_gather_dst_cb", down_proj_gather_params["dst_cb"]),
                ("down_proj_gather_dst_num_pages", down_proj_gather_params["dst_num_pages"]),
                # down_proj_mcast sender args (sender core broadcasts fused output to compute cores)
                ("down_proj_mcast_sender_semaphore", down_proj_mcast_params["sender_semaphore_id"]),
                ("down_proj_mcast_receiver_semaphore", down_proj_mcast_params["receiver_semaphore_id"]),
                ("down_proj_mcast_data_size_bytes", down_proj_mcast_params["data_size_bytes"]),
                ("down_proj_mcast_src_cb", down_proj_mcast_params["src_cb"]),
                ("down_proj_mcast_dst_cb", down_proj_mcast_params["dst_cb"]),
                ("down_proj_mcast_src_num_pages", down_proj_mcast_params["src_num_pages"]),
            ]

        # Named compile-time args for TRISC (matmul + sigmoid compute + gate compute + dram matmul compute)
        trisc_named_compile_time_args = [
            ("gate_mm_in0", gate_mm_params["in0_cb"]),
            ("gate_mm_in1", gate_mm_params["in1_cb"]),
            ("gate_mm_out", gate_mm_params["out_cb"]),
            ("gate_mm_k_num_tiles", gate_mm_params["k_num_tiles"]),
            ("gate_mm_out_w", gate_mm_params["out_w"]),
            ("gate_mm_fused_activation", gate_mm_params["fused_activation"]),
            # Gate compute args (sender core)
            ("gate_input_cb", gate_params["input_cb"]),
            ("gate_bias_cb", gate_params["bias_cb"]),
            ("gate_input_indices_cb", gate_params["indices_cb"]),
            ("gate_output_cb", gate_params["output_cb"]),
            ("gate_output_indices_cb", gate_params["output_indices_cb"]),
            ("gate_eps", gate_params["eps"]),
            ("gate_scaling_factor", gate_params["scaling_factor"]),
            ("gate_enable_sigmoid", gate_params["enable_sigmoid"]),
            # DRAM streaming matmul compute args (compute cores)
            ("gate_proj_cb_in0", gate_mm_params["in0_cb"]),  # Reuses mcasted input
            ("gate_proj_cb_in1", gate_proj_cb_in1),
            ("gate_proj_cb_out", gate_proj_cb_out),
            ("gate_proj_subblock_k", gate_proj_params["subblock_k"]),
            ("gate_proj_per_core_n", gate_proj_params["per_core_n"]),
            ("gate_proj_subblock_w", gate_proj_params["subblock_w"]),
            ("gate_proj_num_subblocks_k", gate_proj_params["num_subblocks_k"]),
            ("gate_proj_tile_r_dim", gate_proj_params["tile_r_dim"]),
            ("gate_proj_fuse_silu", 1),  # Always use SiLU for expert computation
            ("gate_proj_fp32_dest_acc_en", 1),  # Use FP32 accumulation for gate_proj
            # up_proj matmul compute args (compute cores) - writes to intermediate CB
            ("up_proj_cb_in0", gate_mm_params["in0_cb"]),  # Reuses mcasted input (same as gate_proj)
            ("up_proj_cb_in1", up_proj_cb_in1),
            ("up_proj_subblock_k", up_proj_params["subblock_k"]),
            ("up_proj_per_core_n", up_proj_params["per_core_n"]),
            ("up_proj_subblock_w", up_proj_params["subblock_w"]),
            ("up_proj_num_subblocks_k", up_proj_params["num_subblocks_k"]),
            ("up_proj_tile_r_dim", up_proj_params["tile_r_dim"]),
            ("up_proj_fuse_silu", 0),  # No SiLU for up_proj
            ("up_proj_fp32_dest_acc_en", 1),  # Use FP32 accumulation for up_proj
            ("up_proj_cb_mm_out", up_proj_cb_mm_out),  # Intermediate output for up_proj (before mul)
            # Mul compute args (up_proj * gate_proj * expert_scale -> fused output)
            ("mul_cb_in0", mul_cb_in0),  # up_proj output aliased as 16x16
            ("mul_cb_in1", mul_cb_in1),  # gate_proj output aliased as 16x16
            ("mul_cb_out", mul_cb_out),  # final fused output
            ("mul_num_tiles", mul_num_tiles),
            ("mul_cb_scalar", mul_cb_scalar),  # scalar working buffer for expert scale
            ("up_proj_per_core_n", up_proj_params["per_core_n"]),  # tiles in mm_out format for cb_wait
            # down_proj matmul compute args (compute cores)
            ("down_proj_cb_in0", down_proj_mcast_dst_cb),  # Mcasted fused output
            ("down_proj_cb_in1", down_proj_cb_in1),
            ("down_proj_cb_out", down_proj_cb_out),
            ("down_proj_subblock_k", down_proj_params["subblock_k"]),
            ("down_proj_per_core_n", down_proj_params["per_core_n"]),
            ("down_proj_subblock_w", down_proj_params["subblock_w"]),
            ("down_proj_num_subblocks_k", down_proj_params["num_subblocks_k"]),
            ("down_proj_tile_r_dim", down_proj_params["tile_r_dim"]),
            ("down_proj_fuse_silu", 0),  # No SiLU for down_proj
            ("down_proj_fp32_dest_acc_en", 1),  # Use FP32 accumulation for down_proj
            # Testing flag: use hardcoded expert index 0 instead of gate output
            ("use_hardcoded_expert_index", 1 if use_hardcoded_expert_index else 0),
            # Eltwise add compute args (down_proj + fused_add)
            ("add_cb_in0", add_cb_in0),
            ("add_cb_in1", add_cb_in1),
            ("add_cb_out", add_cb_out),
            ("add_num_tiles", add_params["num_tiles"]),
            ("add_cb_in0_wait_tiles", add_params["cb_in0_wait_tiles"]),
            ("add_cb_in1_wait_tiles", add_params["cb_in1_wait_tiles"]),
            ("add_slice_size_bytes", add_params["slice_size_bytes"]),
        ]

        # Semaphore descriptors
        full_device_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )

        mcast_sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_data_sender_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        mcast_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_data_receiver_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        gather_noc0_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=gather_noc0_receiver_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        gather_noc1_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=gather_noc1_receiver_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        # Expert scale mcast uses separate semaphores (IDs 4 and 5) to avoid race condition
        # with back-to-back mcasts. Kernel uses UpdateSemaphoreAddr=true to update addresses.
        expert_scale_mcast_sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=expert_scale_mcast_sender_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=1,  # Sender semaphore starts as VALID (1)
        )

        expert_scale_mcast_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=expert_scale_mcast_receiver_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        # Per-core compile-time args for DRAM matmul bank_id and vc
        # Each DRAM matmul core gets a unique bank_id based on its optimal DRAM bank assignment
        # IMPORTANT: Use get_optimal_dram_bank_to_logical_worker_assignment() to get the correct
        # core-to-bank mapping. Do NOT use corerange_to_cores() iteration order as bank_id!
        gate_proj_optimal_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        # Build mapping from core to bank_id based on optimal assignment
        core_to_bank_id = {}
        for bank_id, core in enumerate(gate_proj_optimal_workers):
            core_to_bank_id[(core.x, core.y)] = bank_id
        bank_id_core_values = []
        vc_core_values = []
        sender_idx_core_values = []  # For down_proj_gather sender idx
        bank_ids = []
        for idx, core in enumerate(gate_proj_optimal_workers):
            bank_id = core_to_bank_id[(core.x, core.y)]
            # VC conflict resolution - avoid NOC contention for cores on same row
            vc = bank_id & 0x3
            for j in range(idx):
                prev_core = gate_proj_optimal_workers[j]
                if prev_core.y == core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
                    vc = (vc + 1) & 0x3
                    break
            bank_ids.append(bank_id)
            bank_id_core_values.append((core, bank_id))
            vc_core_values.append((core, vc))
            sender_idx_core_values.append((core, idx))  # Explicit sender idx

        # Common unified_compile_time_core_descriptors and per_core_compile_time_descriptors
        # (shared between single device and mesh modes)
        unified_compile_time_core_descriptors = [
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_sender_core",
                core_range=sender_core,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_mcast_grid_core",
                core_range=mcast_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_gate_mm_core",
                core_range=gate_mm_params["core_grid"],
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_gate_proj_core",
                core_range=gate_proj_core_ranges,
                value=1,
                other_value=0,
            ),
        ]

        per_core_compile_time_descriptors = [
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="gate_proj_bank_id",
                core_values=bank_id_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="gate_proj_vc",
                core_values=vc_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="up_proj_bank_id",
                core_values=bank_id_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="up_proj_vc",
                core_values=vc_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="down_proj_bank_id",
                core_values=bank_id_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="down_proj_vc",
                core_values=vc_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="down_proj_gather_sender_idx",
                core_values=sender_idx_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="add_sender_index",
                core_values=add_params["sender_index_core_values"],
                other_value=0,
            ),
        ]

        # CB descriptors list (shared across devices)
        cb_descriptors = [
            input_cb_descriptor,
            gate_mm_input_cb_descriptor,
            gate_mm_params["weights_cb_descriptor"],
            gate_mm_params["output_cb_descriptor"],
            gate_params["input_cb_descriptor"],
            gate_params["bias_cb_descriptor"],
            gate_params["indices_cb_descriptor"],
            gate_params["output_cb_descriptor"],
            gate_params["output_indices_cb_descriptor"],
            gate_proj_cb_in1_descriptor,
            gate_proj_cb_index_descriptor,
            gate_proj_cb_out_descriptor,
            up_proj_cb_in1_descriptor,
            up_proj_cb_mm_out_descriptor,
            mul_cb_in0_descriptor,
            mul_cb_in1_descriptor,
            mul_cb_out_descriptor,
            mul_cb_scalar_src_descriptor,
            mul_cb_scalar_descriptor,
            down_proj_gather_dst_cb_descriptor,
            down_proj_mcast_dst_cb_descriptor,
            down_proj_cb_in1_descriptor,
            down_proj_cb_out_descriptor,
            add_cb_in0_descriptor,
            add_cb_in1_descriptor,
            add_cb_out_descriptor,
        ]

        # Semaphore descriptors list (shared across devices)
        semaphore_descriptors = [
            mcast_sender_semaphore_descriptor,
            mcast_receiver_semaphore_descriptor,
            gather_noc0_semaphore_descriptor,
            gather_noc1_semaphore_descriptor,
            expert_scale_mcast_sender_semaphore_descriptor,
            expert_scale_mcast_receiver_semaphore_descriptor,
        ]

        # IO tensors list
        io_tensors = [
            input_tensor,
            mcast_output_tensor,
            gate_mm_weights_tensor,
            gate_mm_output_tensor,
            gate_input_tensor,
            gate_bias_tensor,
            gate_indices_tensor,
            gate_output_scores_tensor,
            gate_output_indices_tensor,
            expert_index_tensor,
            gate_proj_weights_tensor,
            gate_proj_output_tensor,
            up_proj_weights_tensor,
            up_proj_mm_out_tensor,
            fused_output_tensor,
            expert_scale_tensor,
            down_proj_gather_output_tensor,
            down_proj_mcast_output_tensor,
            down_proj_weights_tensor,
            down_proj_output_tensor,
            fused_add_tensor,
            final_output_tensor,
        ]

        # Create per-device programs (single device = 1x1 mesh with chip_id=0)
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                chip_id = row * mesh_cols + col

                # Create unified kernel with chip_id-specific mesh_chip_id
                chip_unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/moe_routed_expert/moe_routed_expert_kernel.cpp",
                    core_ranges=full_device_grid,
                    ncrisc_named_compile_time_args=create_ncrisc_compile_time_args(chip_id),
                    brisc_named_compile_time_args=create_brisc_compile_time_args(chip_id),
                    trisc_named_compile_time_args=trisc_named_compile_time_args,
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=False,
                        dst_full_sync_en=False,
                    ),
                    unified_compile_time_core_descriptors=unified_compile_time_core_descriptors,
                    per_core_compile_time_descriptors=per_core_compile_time_descriptors,
                )

                # Create program for this device
                program = ttnn.ProgramDescriptor(
                    kernels=chip_unified_kernel.get_kernel_descriptors().kernels,
                    cbs=cb_descriptors,
                    semaphores=semaphore_descriptors,
                )

                # Assign to mesh coordinate
                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute with mesh program descriptor
        ttnn.generic_op(io_tensors, mesh_program_descriptor)

        return gate_output_scores_tensor, gate_output_indices_tensor, final_output_tensor
