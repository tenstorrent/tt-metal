# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test for KvChunkAddressTable Python bindings.
This test verifies that the disaggregation APIs are accessible from Python
and work correctly.
"""

import pytest
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
