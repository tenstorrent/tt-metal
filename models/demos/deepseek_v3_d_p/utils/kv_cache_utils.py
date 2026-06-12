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
PREFILL_CHUNK_OUTPUT_TOKENS = 5 * 1024


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
    logger.debug(f"Host name: {host_name}")

    # Create device groups that contain replicated data
    # Data is replicated on each column of the mesh
    device_group_idx_per_row = []

    rank = ttnn.distributed_context_get_rank()
    size = ttnn.distributed_context_get_size()

    total_rows = mesh_shape[0]
    rank_row_start = int(rank) * total_rows // int(size)
    rank_row_end = rank_row_start + total_rows // int(size)

    logger.debug(f"Rank: {rank}, Size: {size}, Row start: {rank_row_start}, Row end: {rank_row_end}")

    num_layers = config.num_layers
    logger.debug(f"Num layers is: {num_layers}")

    all_fabric_node_ids = []
    for row in range(rank_row_start, rank_row_end):
        fabric_node_ids = []
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            fabric_node_id = mesh_device.get_fabric_node_id(coord)
            fabric_node_ids.append(fabric_node_id)

        all_fabric_node_ids.extend(fabric_node_ids)
        group_idx = lookup_table.add_device_group(fabric_node_ids)
        logger.debug(f"Device group {int(group_idx)}: {len(fabric_node_ids)} nodes")
        for idx, fid in enumerate(fabric_node_ids):
            mesh_id = int(fid.mesh_id)
            chip_id = int(fid.chip_id)
            logger.debug(f"  Node {idx}: mesh_id={mesh_id}, chip_id={chip_id}")

        device_group_idx_per_row.append(group_idx)

    for fid in all_fabric_node_ids:
        lookup_table.set_fabric_node_host(fid, host_name=host_name)
        logger.debug(
            f"Set host name for fabric node id: mesh_id={int(fid.mesh_id)}, chip_id={int(fid.chip_id)} to {host_name}"
        )

    num_tokens_in_strip = seq_len // (mesh_shape[sp_axis] * 2)
    num_chunks_in_strip = num_tokens_in_strip // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    logger.debug(f"Num tokens in strip is: {num_tokens_in_strip} num_chunks in strip is: {num_chunks_in_strip}")

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
        logger.debug(
            f"Token positions for device group index: Rank = {rank}, Device group index = {device_group_idx_per_row[row]} are {device_position_indices_low_strip[row]} and {device_position_indices_high_strip[row]}"
        )

    slot = 0
    current_position = 0  # Must be chunk-aligned
    chunks_per_device_group = num_chunks_in_strip * 2
    logger.debug("chunks_per_device_group = ", chunks_per_device_group)

    logger.debug(f"kvpe cache shape is: {tt_kvpe_cache.shape}")
    dram_bank_base_addr = tt_kvpe_cache.buffer_address()
    for row in range(len(device_group_idx_per_row)):
        group_idx = device_group_idx_per_row[row]
        curr_bank_id = 0
        curr_bank_offset = 0

        logger.debug(
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
                logger.debug(
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


def create_kv_chunk_address_table_kimi(
    config, mesh_device, mesh_shape, seq_len, sp_axis, tt_kvpe_cache, chunk_size_bytes, num_users
):
    """
    Create and populate a KV chunk address table for disaggregation (Kimi K2.6 model - non-balanced).

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
    assert seq_len % (mesh_shape[sp_axis] * NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK) == 0, (
        f"seq_len {seq_len} must be divisible by sp_factor({mesh_shape[sp_axis]}) * "
        f"{NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK} for the sequential layout"
    )

    lookup_table = ttnn.experimental.disaggregation.KvChunkAddressTable(config)
    host_name = socket.gethostname()

    rank = ttnn.distributed_context_get_rank()
    size = ttnn.distributed_context_get_size()
    total_rows = mesh_shape[0]
    rank_row_start = int(rank) * total_rows // int(size)
    rank_row_end = rank_row_start + total_rows // int(size)

    num_layers = config.num_layers

    # Data is replicated across each column of the mesh, so one device group per row.
    device_group_idx_per_row = []
    all_fabric_node_ids = []
    for row in range(rank_row_start, rank_row_end):
        fabric_node_ids = []
        for col in range(mesh_shape[1]):
            fabric_node_ids.append(mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col)))
        all_fabric_node_ids.extend(fabric_node_ids)
        device_group_idx_per_row.append(lookup_table.add_device_group(fabric_node_ids))

    for fid in all_fabric_node_ids:
        lookup_table.set_fabric_node_host(fid, host_name=host_name)
        logger.debug(
            f"Set host name for fabric node id: mesh_id={int(fid.mesh_id)}, chip_id={int(fid.chip_id)} to {host_name}"
        )

    seq_len_local = seq_len // mesh_shape[sp_axis]

    dram_bank_base_addr = tt_kvpe_cache.buffer_address()
    for local_idx, global_row in enumerate(range(rank_row_start, rank_row_end)):
        group_idx = device_group_idx_per_row[local_idx]
        curr_bank_id = 0
        curr_bank_offset = 0
        device_token_start = global_row * seq_len_local
        device_token_end = device_token_start + seq_len_local
        for slot in range(num_users):
            for layer in range(num_layers):
                for position in range(device_token_start, device_token_end, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                    location = ttnn.experimental.disaggregation.KvCacheLocation()
                    location.noc_addr = (curr_bank_id << 32) | (dram_bank_base_addr + curr_bank_offset)
                    location.size_bytes = chunk_size_bytes
                    location.device_group_index = group_idx
                    lookup_table.set(layer, position, slot, location)

                    curr_bank_id = (curr_bank_id + 1) % BH_NUM_DRAM_BANKS
                    if curr_bank_id == 0:
                        curr_bank_offset += chunk_size_bytes

    return lookup_table


def create_kv_chunk_address_table_deepseek(
    config, mesh_device, mesh_shape, seq_len, sp_axis, tt_kvpe_cache, chunk_size_bytes, num_users, chunk_size_global
):
    """Block-cyclic-aware KV chunk address table for the DeepSeek non-balanced prefill cache.

    The DeepSeek prefill KV cache stores positions in BLOCK-CYCLIC order across the SP shards
    (see tt.mla.utils.blockcyclic_positions / update_padded_kv_cache): natural position P lives on
    SP-row chip ``c = (P % chunk_size_global) // chunk_local`` at local row
    ``lr = (P // chunk_size_global) * chunk_local + (P % chunk_size_global) % chunk_local``,
    where ``chunk_local = chunk_size_global // sp``.

    The Kimi builder instead assumes a CONTIGUOUS-block layout (position P on chip
    P // seq_len_local), which mismaps every position: a partial migration of natural range [0, N)
    then copies contiguous storage chunk 0..N/32 — mostly UN-prefilled storage — instead of the
    block-cyclic-scattered chunks that actually hold the first N tokens (observed: AFTER-migration
    KV PCC ~0.35 because only ~1/sp of the data lands correctly). This builder maps each natural
    position to its true block-cyclic storage chip + local DRAM-bank offset, reusing the same
    ROUND_ROBIN_1D bank/offset math as init_kvpe_cache's NdShardSpec
    (linear = batch * chunks_per_layer_local + local_chunk; bank = linear % banks;
    offset = (linear // banks) * chunk_size_bytes), so a [0, N) migration moves exactly the
    prefilled KV. ``slot`` is the user index; cache batch index = slot * num_layers + layer.
    """
    sp = mesh_shape[sp_axis]
    assert (
        seq_len % (sp * NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK) == 0
    ), f"seq_len {seq_len} must be divisible by sp({sp}) * {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}"
    assert chunk_size_global % sp == 0, f"chunk_size_global {chunk_size_global} must be divisible by sp {sp}"
    chunk_local = chunk_size_global // sp
    assert (
        chunk_local % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0
    ), f"chunk_local {chunk_local} must be divisible by {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}"
    seq_len_local = seq_len // sp
    chunks_per_layer_local = seq_len_local // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

    lookup_table = ttnn.experimental.disaggregation.KvChunkAddressTable(config)
    host_name = socket.gethostname()
    num_layers = config.num_layers

    # One device group per SP row (the row's TP columns hold replicated KV). Single-process
    # runner => rank 0 of size 1 => covers all rows; multi-rank slices rows per rank.
    rank = ttnn.distributed_context_get_rank()
    size = ttnn.distributed_context_get_size()
    total_rows = mesh_shape[0]
    rank_row_start = int(rank) * total_rows // int(size)
    rank_row_end = rank_row_start + total_rows // int(size)
    device_group_for_row = {}
    all_fabric_node_ids = []
    for row in range(rank_row_start, rank_row_end):
        fabric_node_ids = [
            mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col)) for col in range(mesh_shape[1])
        ]
        all_fabric_node_ids.extend(fabric_node_ids)
        device_group_for_row[row] = lookup_table.add_device_group(fabric_node_ids)
    for fid in all_fabric_node_ids:
        lookup_table.set_fabric_node_host(fid, host_name=host_name)

    dram_bank_base_addr = tt_kvpe_cache.buffer_address()
    for slot in range(num_users):
        for layer in range(num_layers):
            batch = slot * num_layers + layer
            for position in range(0, seq_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                slab = position // chunk_size_global
                rem = position % chunk_size_global
                c = rem // chunk_local
                if c < rank_row_start or c >= rank_row_end:
                    continue  # this SP-row chip is owned by another rank
                off = rem % chunk_local
                local_row = slab * chunk_local + off
                local_chunk = local_row // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
                linear = batch * chunks_per_layer_local + local_chunk
                bank_id = linear % BH_NUM_DRAM_BANKS
                bank_offset = (linear // BH_NUM_DRAM_BANKS) * chunk_size_bytes
                location = ttnn.experimental.disaggregation.KvCacheLocation()
                location.noc_addr = (bank_id << 32) | (dram_bank_base_addr + bank_offset)
                location.size_bytes = chunk_size_bytes
                location.device_group_index = device_group_for_row[c]
                lookup_table.set(layer, position, slot, location)

    logger.info(
        f"[migration] DeepSeek block-cyclic KV chunk table: sp={sp} chunk_local={chunk_local} "
        f"seq_len_local={seq_len_local} entries={lookup_table.total_entries()}"
    )
    return lookup_table


def init_kvpe_cache(kvpe_cache_head_dim, mesh_device, seq_len, mesh_shape, sp_axis, num_kvpe_cache_layers, num_users=1):
    """
    Initialize KVPE cache for MLA.

    Args:
        kvpe_cache_head_dim: Head dimension for KVPE cache (qk_rope_head_dim + kv_lora_rank)
        mesh_device: Mesh device for TT
        seq_len: Sequence length
        mesh_shape: Shape of mesh device
        sp_axis: Sequence parallel axis
        num_kvpe_cache_layers: Number of layers per user in the cache.
        num_users: Number of independent users sharing the cache. The batch dim
            is laid out user-major: slot index = user_id * num_kvpe_cache_layers + layer_idx,
            so each user's layers stay contiguous.

    Returns:
        tt_kvpe_cache: Initialized KVPE cache on device
    """
    # hack in num_users * num_layers into batch size, so each user's layers are contiguous in memory
    num_layers = num_kvpe_cache_layers
    seq_len_local = seq_len // mesh_shape[sp_axis]
    torch_kvpe_cache = torch.zeros(num_users * num_layers, 1, seq_len_local, kvpe_cache_head_dim)

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
