# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for KVPE cache initialization and management.
"""

import socket

import torch
from loguru import logger

import ttnn

# This is a predefined constant for the number of contiguous tokens in a DRAM bank
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32
BH_NUM_DRAM_BANKS = 8


def create_kv_chunk_address_table(config, mesh_device, mesh_shape, seq_len, sp_axis, tt_kvpe_cache, chunk_size_bytes):
    """
    Create and populate a KV chunk address table for disaggregation.

    Args:
        config: KvChunkAddressTableConfig
        mesh_device: Mesh device for TT
        mesh_shape: Shape of mesh device
        seq_len: Sequence length
        sp_axis: Sequence parallel axis
        tt_kvpe_cache: Initialized KVPE cache on device
        chunk_size_bytes: Size of each chunk in bytes

    Returns:
        lookup_table: Populated KvChunkAddressTable
    """
    lookup_table = ttnn.experimental.disaggregation.KvChunkAddressTable(config)

    host_name = socket.gethostname()
    logger.info(f"Host name: {host_name}")

    # Create device groups that contain replicated data
    # Data is replicated on each column of the mesh
    device_group_idx_per_row = []

    rank = ttnn.distributed_context_get_rank()
    size = ttnn.distributed_context_get_size()

    total_rows = mesh_shape[0]
    rank_row_start = int(rank) * total_rows // int(size)
    rank_row_end = rank_row_start + total_rows // int(size)

    logger.info(f"Rank: {rank}, Size: {size}, Row start: {rank_row_start}, Row end: {rank_row_end}")

    num_layers = config.num_layers
    print(f"Num layers is: ", num_layers)

    all_fabric_node_ids = []
    for row in range(rank_row_start, rank_row_end):
        fabric_node_ids = []
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            fabric_node_id = mesh_device.get_fabric_node_id(coord)
            fabric_node_ids.append(fabric_node_id)

        all_fabric_node_ids.extend(fabric_node_ids)
        group_idx = lookup_table.add_device_group(fabric_node_ids)
        logger.info(f"Device group {int(group_idx)}: {len(fabric_node_ids)} nodes")
        for idx, fid in enumerate(fabric_node_ids):
            mesh_id = int(fid.mesh_id)
            chip_id = int(fid.chip_id)
            logger.info(f"  Node {idx}: mesh_id={mesh_id}, chip_id={chip_id}")

        device_group_idx_per_row.append(group_idx)

    for fid in all_fabric_node_ids:
        lookup_table.set_fabric_node_host(fid, host_name=host_name)
        logger.info(
            f"Set host name for fabric node id: mesh_id={int(fid.mesh_id)}, chip_id={int(fid.chip_id)} to {host_name}"
        )

    num_tokens_in_strip = seq_len // (mesh_shape[sp_axis] * 2)
    num_chunks_in_strip = num_tokens_in_strip // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    logger.info(f"Num tokens in strip is: {num_tokens_in_strip} num_chunks in strip is: {num_chunks_in_strip}")

    # describes high and low sequence length per rank
    seq_len_per_rank = seq_len // (int(size) * 2)

    device_position_indices_low_strip = []
    device_position_indices_high_strip = []
    low_strip_start_idx = seq_len_per_rank * int(rank)
    high_strip_end_idx = seq_len_per_rank * (int(size) - int(rank)) - 1 + seq_len // 2
    for row in range(len(device_group_idx_per_row)):
        low_strip_end_idx = low_strip_start_idx + num_chunks_in_strip * NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK - 1
        device_position_indices_low_strip.append((low_strip_start_idx, low_strip_end_idx))
        high_strip_start_idx = high_strip_end_idx - (num_chunks_in_strip * NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK - 1)
        device_position_indices_high_strip.append((high_strip_start_idx, high_strip_end_idx))

        low_strip_start_idx = low_strip_end_idx + 1
        high_strip_end_idx = high_strip_start_idx - 1
        logger.info(
            f"Token positions for device group index: Rank = {rank}, Device group index = {device_group_idx_per_row[row]} are {device_position_indices_low_strip[row]} and {device_position_indices_high_strip[row]}"
        )

    slot = 0
    current_position = 0  # Must be chunk-aligned
    chunks_per_device_group = num_chunks_in_strip * 2
    logger.info("chunks_per_device_group = ", chunks_per_device_group)

    logger.info(f"kvpe cache shape is: {tt_kvpe_cache.shape}")
    dram_bank_base_addr = tt_kvpe_cache.buffer_address()
    for row in range(len(device_group_idx_per_row)):
        group_idx = device_group_idx_per_row[row]
        curr_bank_id = 0
        curr_bank_offset = 0

        logger.info(
            f"Rank: {rank} Populating device_group_index: {group_idx} with positions: {device_position_indices_low_strip[row]} and {device_position_indices_high_strip[row]}"
        )
        (current_position, max_position) = device_position_indices_low_strip[row]
        for layer in range(num_layers):
            layer_current_position = current_position
            layer_max_position = max_position
            for chunk in range(chunks_per_device_group):
                location = ttnn.experimental.disaggregation.KvCacheLocation()

                noc_addr = (curr_bank_id << 32) | (dram_bank_base_addr + curr_bank_offset)
                location.noc_addr = noc_addr
                location.size_bytes = chunk_size_bytes
                location.device_group_index = group_idx
                lookup_table.set(layer, layer_current_position, slot, location)
                logger.info(
                    f"Rank: {rank} Set location for (layer={layer}, pos={layer_current_position}, slot={slot}, bank_id={curr_bank_id}, curr_bank_offset = {curr_bank_offset} noc_addr = 0x{noc_addr:X})"
                )

                curr_bank_id = (curr_bank_id + 1) % BH_NUM_DRAM_BANKS
                # move to next chunk offset
                if curr_bank_id == 0:
                    curr_bank_offset += chunk_size_bytes
                layer_current_position += NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
                if chunk == num_chunks_in_strip - 1:
                    # switch to high chunk
                    assert (
                        layer_current_position == layer_max_position + 1
                    ), f"Missmatch in position calculation. Expected layer current_position to be {layer_max_position + 1}, but it is: {layer_current_position}."
                    (layer_current_position, layer_max_position) = device_position_indices_high_strip[row]

    return lookup_table


def init_kvpe_cache(kvpe_cache_head_dim, mesh_device, seq_len, mesh_shape, sp_axis, num_kvpe_cache_layers):
    """
    Initialize KVPE cache for MLA.

    Args:
        kvpe_cache_head_dim: Head dimension for KVPE cache (qk_rope_head_dim + kv_lora_rank)
        mesh_device: Mesh device for TT
        seq_len: Sequence length
        mesh_shape: Shape of mesh device
        sp_axis: Sequence parallel axis
        num_kvpe_cache_layers: Number of layers for KVPE cache

    Returns:
        tt_kvpe_cache: Initialized KVPE cache on device
    """
    # hack in num_layers into batch size, so they are contiguous in memory
    num_layers = num_kvpe_cache_layers
    seq_len_local = seq_len // mesh_shape[sp_axis]
    torch_kvpe_cache = torch.zeros(num_layers, 1, seq_len_local, kvpe_cache_head_dim)

    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    grid = ttnn.CoreRangeSet(core_ranges)

    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, kvpe_cache_head_dim],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=kv_nd_shard_spec,
    )

    tt_kvpe_cache = ttnn.from_torch(
        torch_kvpe_cache,
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=kv_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    return tt_kvpe_cache
