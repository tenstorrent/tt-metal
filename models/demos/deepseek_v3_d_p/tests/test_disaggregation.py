# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test for KvChunkAddressTable Python bindings.
This test verifies that the disaggregation APIs are accessible from Python
and work correctly.
"""

import socket

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    create_kv_chunk_address_table,
    init_kvpe_cache,
)


@pytest.mark.parametrize(
    "mesh_device",
    [(32, 4), (8, 4), (1, 2)],
    ids=["32x4", "8x4", "1x2"],
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
@pytest.mark.parametrize("seq_len", [3 * 1024, 4 * 3 * 1024, 100 * 1024], ids=["seq3k", "seq12k", "seq100k"])
def test_kv_cache_address_table(mesh_device, seq_len):
    sp_axis = 0
    kvpe_cache_head_dim = 576

    mesh_shape = list(mesh_device.shape)
    if mesh_shape[0] == 32 and mesh_shape[1] == 4 and seq_len == 3 * 1024:
        pytest.skip("Skipping test for 32x4 mesh and seq3k")

    num_kvpe_cache_layers = 2
    # Initialize KVPE cache using utility function
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_cache_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_kvpe_cache_layers,
    )

    # Create and populate KV chunk address table using utility function
    CHUNK_SIZE_BYTES = 19584  # [1, 1, 32, 576] bfp8
    config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    config.num_layers = num_kvpe_cache_layers
    config.max_sequence_length = seq_len
    config.num_slots = 1
    config.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    config.chunk_size_bytes = CHUNK_SIZE_BYTES

    lookup_table = create_kv_chunk_address_table(
        config=config,
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tt_kvpe_cache=tt_kvpe_cache,
        chunk_size_bytes=CHUNK_SIZE_BYTES,
    )

    # Test lookup functionality
    rank = ttnn.distributed_context_get_rank()
    size = ttnn.distributed_context_get_size()
    seq_len_per_rank = seq_len // (int(size) * 2)
    low_strip_start_idx = seq_len_per_rank * int(rank)
    high_strip_end_idx = seq_len_per_rank * (int(size) - int(rank)) - 1 + seq_len // 2
    layer = 0
    slot = 0

    for position in range(low_strip_start_idx, high_strip_end_idx, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
        for layer in range(0, num_kvpe_cache_layers):
            retrieved = lookup_table.lookup(layer, position, slot)
            logger.info(
                f"Rank: {rank} Retrieved: layer={layer} position={position}, noc_addr=0x{retrieved.noc_addr:X}, "
                f"size={retrieved.size_bytes}, group_idx={int(retrieved.device_group_index)}"
            )


@pytest.mark.parametrize(
    "mesh_device",
    [(32, 4), (8, 4), (1, 2)],
    ids=["32x4", "8x4", "1x2"],
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
        for idx, fid in enumerate(fabric_node_ids):
            mesh_id = int(fid.mesh_id)
            chip_id = int(fid.chip_id)
            logger.info(
                f"  Node {idx} row={row}, col={col}: mesh_id={mesh_id},  chip_id={chip_id} host_name={host_name}"
            )
