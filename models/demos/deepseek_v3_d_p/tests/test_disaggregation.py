# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test for KvChunkAddressTable Python bindings.
This test verifies that the disaggregation APIs are accessible from Python
and work correctly.
"""

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
@pytest.mark.parametrize("seq_len", [32 * 8], ids=["seq32x8"])
def test_kv_cache_address_table(mesh_device, seq_len):
    sp_axis = 0
    kvpe_cache_head_dim = 576

    mesh_shape = list(mesh_device.shape)
    seq_len_local = seq_len // mesh_shape[sp_axis]

    num_layers = 1
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

    config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    config.num_layers = 1
    config.max_sequence_length = seq_len
    config.num_slots = 1
    config.chunk_n_tokens = 32
    config.chunk_size_bytes = 19584

    lookup_table = ttnn.experimental.disaggregation.KvChunkAddressTable(config)

    # Create device groups that contain replicated data
    # Data is replicated on each column of the mesh

    device_group_idx_per_row = []

    for row in range(mesh_shape[0]):
        fabric_node_ids = []
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            fabric_node_id = mesh_device.get_fabric_node_id(coord)
            # table.set_fabric_node_host(fabric_node_id, host_name="abc")
            fabric_node_ids.append(fabric_node_id)

        group_idx = lookup_table.add_device_group(fabric_node_ids)
        logger.info(f"Device group {int(group_idx)}: {len(fabric_node_ids)} nodes")
        for idx, fid in enumerate(fabric_node_ids):
            mesh_id = int(fid.mesh_id)
            chip_id = int(fid.chip_id)
            logger.info(f"  Node {idx}: mesh_id={mesh_id}, chip_id={chip_id}")

        device_group_idx_per_row.append(group_idx)

    layer = 0
    slot = 0
    position = 0  # Must be chunk-aligned

    for row in range(len(device_group_idx_per_row)):
        group_idx = device_group_idx_per_row[row]
        location = ttnn.experimental.disaggregation.KvCacheLocation()
        location.noc_addr = tt_kvpe_cache.buffer_address()
        location.size_bytes = 19584
        location.device_group_index = group_idx
        lookup_table.set(layer, position, slot, location)
        logger.info(f"Set location for (layer={layer}, pos={position}, slot={slot})")
        position += 32

    # 5. Lookup the location
    for position in range(0, seq_len, 32):
        retrieved = lookup_table.lookup(layer, position, slot)
        logger.info(
            f"Retrieved: position={position}, noc_addr=0x{retrieved.noc_addr:X}, "
            f"size={retrieved.size_bytes}, group_idx={int(retrieved.device_group_index)}"
        )
