# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test for KvChunkAddressTable Python bindings.
This test verifies that the disaggregation APIs are accessible from Python
and work correctly.
"""

import socket

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    ids=["8x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_kv_chunk_address_table(mesh_device):
    """Test KvChunkAddressTable Python bindings"""

    logger.info("Testing KvChunkAddressTable Python bindings")

    # 1. Create configuration
    config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    config.num_layers = 1
    config.max_sequence_length = 8192
    config.num_slots = 8
    config.chunk_n_tokens = 32
    config.chunk_size_bytes = 19584

    logger.info(
        f"Config: layers={config.num_layers}, seq_len={config.max_sequence_length}, "
        f"slots={config.num_slots}, chunk_tokens={config.chunk_n_tokens}"
    )

    # 2. Initialize table
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(config)

    logger.info(
        f"Table created: {table.num_position_chunks()} position chunks, " f"{table.total_entries()} total entries"
    )

    # 3. Add device groups using FabricNodeIds from mesh
    mesh_shape = list(mesh_device.shape)
    device_groups = []

    # Create one device group per mesh column
    for col in range(mesh_shape[1]):
        fabric_node_ids = []
        for row in range(mesh_shape[0]):
            coord = ttnn.MeshCoordinate(row, col)
            fabric_node_id = mesh_device.get_fabric_node_id(coord)
            fabric_node_ids.append(fabric_node_id)

        group_idx = table.add_device_group(fabric_node_ids)
        device_groups.append(group_idx)
        logger.info(f"Device group {int(group_idx)}: {len(fabric_node_ids)} nodes")

    logger.info(f"Total device groups: {table.num_device_groups()}")

    # 4. Set KV cache locations
    layer = 0
    slot = 0
    position = 0  # Must be chunk-aligned

    location = ttnn.experimental.disaggregation.KvCacheLocation()
    location.noc_addr = 0xDEADBEEF
    location.size_bytes = 19584
    location.device_group_index = device_groups[0]

    table.set(layer, position, slot, location)
    logger.info(f"Set location for (layer={layer}, pos={position}, slot={slot})")

    # 5. Lookup the location
    retrieved = table.lookup(layer, position, slot)
    logger.info(
        f"Retrieved: noc_addr=0x{retrieved.noc_addr:X}, "
        f"size={retrieved.size_bytes}, group_idx={int(retrieved.device_group_index)}"
    )

    assert retrieved.noc_addr == 0xDEADBEEF
    assert retrieved.size_bytes == 19584
    assert int(retrieved.device_group_index) == int(device_groups[0])

    # 6. Test range lookup
    start_pos = 0
    end_pos = 64  # 2 chunks
    locations = table.lookup_range(layer, start_pos, end_pos, slot)
    logger.info(f"Range lookup [{start_pos}, {end_pos}): {len(locations)} entries")

    # 7. Test device group retrieval
    group = table.get_device_group(device_groups[0])
    logger.info(f"Device group 0 has {len(group.fabric_node_ids)} fabric nodes")

    # 8. Print fabric node IDs in the first group
    for idx, fid in enumerate(group.fabric_node_ids):
        mesh_id = int(fid.mesh_id)
        chip_id = int(fid.chip_id)
        logger.info(f"  Node {idx}: mesh_id={mesh_id}, chip_id={chip_id}")

    logger.success("KvChunkAddressTable bindings test passed!")


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4), (32, 4), (1, 1)],
    ids=["8x4", "32x4", "1x1"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", [3 * 1024, 100 * 1024], ids=["seq3k", "seq100k"])
def test_kv_cache_address_table(mesh_device, seq_len):
    sp_axis = 0
    kvpe_cache_head_dim = 576

    mesh_shape = list(mesh_device.shape)
    seq_len_local = seq_len // mesh_shape[sp_axis]

    torch_kvpe_cache = torch.zeros(1, 1, seq_len_local, kvpe_cache_head_dim)

    BH_NUM_DRAM_BANKS = 8
    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    grid = ttnn.CoreRangeSet(core_ranges)

    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32  # this is a predefined constant
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

    CHUNK_SIZE_BYTES = 19584  # [1, 1, 32, 576] bfp8
    config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    config.num_layers = 1
    config.max_sequence_length = seq_len
    config.num_slots = 1
    config.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    config.chunk_size_bytes = CHUNK_SIZE_BYTES

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

    device_position_indices_low_strip = []
    device_position_indices_high_strip = []
    low_strip_start_idx = 0
    high_strip_end_idx = seq_len - 1
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

    # layer = 0
    # slot = 0
    # current_position = 0  # Must be chunk-aligned
    # chunks_per_device_group = num_chunks_in_strip * 2
    # logger.info("chunks_per_device_group = ", chunks_per_device_group)

    # dram_bank_0_addr = tt_kvpe_cache.buffer_address()
    # for row in range(len(device_group_idx_per_row)):
    #     group_idx = device_group_idx_per_row[row]
    #     curr_bank_id = 0
    #     curr_bank_offset = 0

    #     logger.info(
    #         f"Populating device_group_index: {group_idx} with positions: {device_position_indices_low_strip[row]} and {device_position_indices_high_strip[row]}"
    #     )
    #     (current_position, max_position) = device_position_indices_low_strip[row]
    #     for chunk in range(chunks_per_device_group):
    #         location = ttnn.experimental.disaggregation.KvCacheLocation()

    #         # This needs proper handling in KvCacheLocation(), just add it up atm
    #         noc_addr = dram_bank_0_addr + curr_bank_id + curr_bank_offset
    #         location.noc_addr = noc_addr
    #         location.size_bytes = CHUNK_SIZE_BYTES
    #         location.device_group_index = group_idx
    #         lookup_table.set(layer, current_position, slot, location)
    #         logger.info(
    #             f"Set location for (layer={layer}, pos={current_position}, slot={slot}, bank_id={curr_bank_id}, curr_bank_offset = {curr_bank_offset} noc_addr = 0x{noc_addr:X})"
    #         )

    #         curr_bank_id = (curr_bank_id + 1) % BH_NUM_DRAM_BANKS
    #         # move to next chunk offset
    #         if curr_bank_id == 0:
    #             curr_bank_offset += CHUNK_SIZE_BYTES
    #         current_position += NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    #         if chunk == num_chunks_in_strip - 1:
    #             # switch to high chunk
    #             assert (
    #                 current_position == max_position + 1
    #             ), f"Missmatch in position calculation. Expected current_position to be {max_position + 1}, but it is: {current_position}"
    #             (current_position, max_position) = device_position_indices_high_strip[row]

    # # 5. Lookup the location
    # for position in range(0, seq_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
    #     retrieved = lookup_table.lookup(layer, position, slot)
    #     logger.info(
    #         f"Retrieved: position={position}, noc_addr=0x{retrieved.noc_addr:X}, "
    #         f"size={retrieved.size_bytes}, group_idx={int(retrieved.device_group_index)}"
    #     )


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4), (32, 4), (1, 2)],
    ids=["8x4", "32x4", "1x2"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_fnids(mesh_device):
    mesh_shape = list(mesh_device.shape)

    host_name = socket.gethostname()
    logger.info(f"Host name: {host_name}")

    rank = ttnn.distributed_context_get_rank()
    size = ttnn.distributed_context_get_size()

    total_rows = mesh_shape[0]
    total_cols = mesh_shape[1]
    rank_row_start = int(rank) * total_rows // int(size)
    rank_row_end = rank_row_start + total_rows // int(size)

    logger.info(f"Rank: {rank}, Size: {size}, Row start: {rank_row_start}, Row end: {rank_row_end}")

    all_fabric_node_ids = []
    for row in range(rank_row_start, rank_row_end):
        fabric_node_ids = []
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            fabric_node_id = mesh_device.get_fabric_node_id(coord)
            fabric_node_ids.append(fabric_node_id)

        all_fabric_node_ids.extend(fabric_node_ids)
        # group_idx = lookup_table.add_device_group(fabric_node_ids)
        # logger.info(f"Device group {int(group_idx)}: {len(fabric_node_ids)} nodes")
        for idx, fid in enumerate(fabric_node_ids):
            mesh_id = int(fid.mesh_id)
            chip_id = int(fid.chip_id)
            logger.info(
                f"  Node {idx} row={row}, col={col}: mesh_id={mesh_id},  chip_id={chip_id} host_name={host_name}"
            )
        # device_group_idx_per_row.append(group_idx)
