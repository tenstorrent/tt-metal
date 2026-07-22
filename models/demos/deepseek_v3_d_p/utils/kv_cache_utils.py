# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for KVPE cache initialization and management.
"""

import socket

from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

# This is a predefined constant for the number of contiguous tokens in a DRAM bank
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32
# Nominal DRAM bank count for a full (unharvested) Blackhole part. Prefer get_num_dram_banks(device)
# at runtime: harvested parts expose fewer banks (e.g. 7), and the cache ND-shard grid + the
# disaggregation address-table striding must both use the device's actual count to stay consistent.
BH_NUM_DRAM_BANKS = 8
PREFILL_CHUNK_OUTPUT_TOKENS = 5 * 1024


def get_num_dram_banks(mesh_device):
    """Usable DRAM banks on this device. Full Blackhole = 8; harvested parts expose fewer (e.g. 7).

    The KV cache ND-shards round-robin across these banks and the disaggregation address table replays
    that exact striping (`curr_bank_id = (curr_bank_id + 1) % num_banks`), so both MUST derive the count
    from the same device. dram_grid_size().x is the number of DRAM cores/banks the device exposes."""
    return mesh_device.dram_grid_size().x


def create_kv_chunk_address_table(
    config, mesh_device, mesh_shape, seq_len, sp_axis, tt_kvpe_cache, chunk_size_bytes, num_users=1
):
    """
    Create and populate a KV chunk address table for disaggregation.

    Block-cyclic storage layout, model-agnostic: chunked prefill stripes KV positions block-cyclically
    across the SP shards, and this maps each natural position to its storage chip + DRAM offset.

    Args:
        config: KvChunkAddressTableConfig
        mesh_device: Mesh device for TT
        mesh_shape: Shape of mesh device
        seq_len: Sequence length
        sp_axis: Sequence parallel axis
        tt_kvpe_cache: Initialized KVPE cache on device
        chunk_size_bytes: Size of each chunk in bytes
        num_users: Number of users (slots) sharing the buffer; cache batch dim folds them as user * num_layers + layer

    Returns:
        lookup_table: Populated KvChunkAddressTable
    """
    lookup_table = ttnn.experimental.disaggregation.KvChunkAddressTable(config)
    return populate_kv_chunk_address_table(
        lookup_table=lookup_table,
        config=config,
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tt_kvpe_cache=tt_kvpe_cache,
        chunk_size_bytes=chunk_size_bytes,
        num_users=num_users,
        config_id=0,
    )


def populate_kv_chunk_address_table(
    lookup_table,
    config,
    mesh_device,
    mesh_shape,
    seq_len,
    sp_axis,
    tt_kvpe_cache,
    chunk_size_bytes,
    num_users=1,
    config_id=0,
):
    """
    Populate ONE config (``config_id``) of an existing KvChunkAddressTable from a device cache tensor.

    Factored out of create_kv_chunk_address_table so a single multi-config table can hold several
    caches at once (the serving convention is config 0 = the MLA KVPE cache, config 1 = the block-cyclic
    index-key cache); each config carries its own grid + chunk_size_bytes and is addressed by config_id.
    The device-group
    side table and fabric-node host map are SHARED across configs — re-registering them here per config is
    safe (add_device_group dedups identical replica sets; set_fabric_node_host is idempotent).

    Args:
        lookup_table: an existing KvChunkAddressTable (single- or multi-config).
        config: the KvChunkAddressTableConfig for THIS config_id (read for num_layers).
        config_id: which config of the table to populate (default 0, the single-config case).
        (remaining args as in create_kv_chunk_address_table)

    Returns:
        lookup_table: the same table, with config_id populated.
    """
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

    tokens_per_chunk_local = PREFILL_CHUNK_OUTPUT_TOKENS // mesh_shape[sp_axis]  # 640 for 5k chunks
    num_chunks_per_seq_len = (
        seq_len // PREFILL_CHUNK_OUTPUT_TOKENS
    )  # number of 5k chunks contained in the sequence length

    assert (
        seq_len % PREFILL_CHUNK_OUTPUT_TOKENS == 0
    ), f"seq_len {seq_len} must be a multiple of {PREFILL_CHUNK_OUTPUT_TOKENS}"

    assert tokens_per_chunk_local % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0, (
        f"{PREFILL_CHUNK_OUTPUT_TOKENS} tokens / sp({mesh_shape[sp_axis]}) = {tokens_per_chunk_local}, "
        f"not a multiple of {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}"
    )

    assert (
        tt_kvpe_cache.shape[0] == num_users * num_layers
    ), f"cache batch dim {tt_kvpe_cache.shape[0]} != num_users({num_users}) * num_layers({num_layers})"

    dram_bank_base_addr = tt_kvpe_cache.buffer_address()
    # Must match the bank count the cache was ND-sharded across (see get_num_dram_banks).
    num_dram_banks = get_num_dram_banks(mesh_device)
    for local_idx, global_row in enumerate(range(rank_row_start, rank_row_end)):
        group_idx = device_group_idx_per_row[local_idx]
        curr_bank_id = 0
        curr_bank_offset = 0

        for slot in range(num_users):
            for layer in range(num_layers):
                for seq_chunk in range(num_chunks_per_seq_len):
                    chunk_token_start = seq_chunk * PREFILL_CHUNK_OUTPUT_TOKENS + global_row * tokens_per_chunk_local
                    chunk_token_end = chunk_token_start + tokens_per_chunk_local
                    for position in range(chunk_token_start, chunk_token_end, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                        location = ttnn.experimental.disaggregation.KvCacheLocation()
                        location.noc_addr = (curr_bank_id << 32) | (dram_bank_base_addr + curr_bank_offset)
                        location.size_bytes = chunk_size_bytes
                        location.device_group_index = group_idx
                        lookup_table.set(layer, position, slot, location, config_id)

                        curr_bank_id = (curr_bank_id + 1) % num_dram_banks
                        if curr_bank_id == 0:
                            curr_bank_offset += chunk_size_bytes

    return lookup_table


def init_kvpe_cache(
    kvpe_cache_head_dim,
    mesh_device,
    seq_len,
    mesh_shape,
    sp_axis,
    num_kvpe_cache_layers,
    num_users=1,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
):
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
        dtype: Cache element dtype (default bfloat8_b). Use fp8_e4m3 with ROW_MAJOR.
        layout: Cache layout (default TILE_LAYOUT). ROW_MAJOR required for fp8_e4m3.

    Returns:
        tt_kvpe_cache: Initialized KVPE cache on device
    """
    # hack in num_users * num_layers into batch size, so each user's layers are contiguous in memory
    num_layers = num_kvpe_cache_layers
    seq_len_local = seq_len // mesh_shape[sp_axis]

    num_dram_banks = get_num_dram_banks(mesh_device)
    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(num_dram_banks)
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

    # Allocate + zero on device. The host from_torch path packs the full replicated
    # cache as bfp8 on host, overflowing pack_as_bfp8_tiles' 32-bit page index at high
    # num_users; a device kernel zeros it instead with no host transfer. Allocating
    # directly in the requested dtype/layout also sidesteps the mesh-mapper from_torch
    # path that forces TILE for fp8_e4m3 (so fp8 rides on ROW_MAJOR).
    #
    # fp8_e4m3 can't be DRAMZeroFill'd directly: its compute kernel needs fp32_dest_acc_en
    # whenever an 8-bit-float CB is on-core (Blackhole TT_FATAL), so for fp8 we allocate +
    # zero in bf16 and typecast on device (still no host transfer; typecast keeps the
    # ROW_MAJOR layout and the ND shard spec).
    is_fp8 = dtype == ttnn.fp8_e4m3
    tt_kvpe_cache = ttnn.allocate_tensor_on_device(
        ttnn.Shape([num_users * num_layers, 1, seq_len_local, kvpe_cache_head_dim]),
        ttnn.bfloat16 if is_fp8 else dtype,
        layout,
        mesh_device,
        kv_mem_config,
    )
    DRAMZeroFill.op(tt_kvpe_cache)
    if is_fp8:
        tt_kvpe_cache = ttnn.typecast(tt_kvpe_cache, ttnn.fp8_e4m3)

    # allocate_tensor_on_device assigns a default 2D fully-replicated topology, but the rest
    # of the model produces replicated tensors via ReplicateTensorToMesh, which is a 1D
    # MeshShape(num_devices) with a single Replicate placement. Reproduce that exactly: a 1D
    # distribution_shape + single Replicate, with mesh_coords being the 2D physical device
    # coordinates (row-major), matching what the ReplicateTensorToMesh mapper emits.
    num_devices = mesh_device.shape[0] * mesh_device.shape[1]
    dist_shape = ttnn.MeshShape([num_devices])
    placements = [ttnn.PlacementReplicate()]
    physical_mesh_shape = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
    coords = list(ttnn.MeshCoordinateRange(physical_mesh_shape))
    tt_kvpe_cache.update_tensor_topology(ttnn.TensorTopology(dist_shape, placements, coords))

    return tt_kvpe_cache


def allocate_mla_kvpe_cache(*, mesh_device, hf_config, max_seq_len, mesh_shape, sp_axis, num_layers, num_users):
    """Allocate the MLA KVPE cache for one runtime from the HF config.

    The MLA per-token cache row is ``qk_rope_head_dim + kv_lora_rank`` wide; ONE
    shared cache holds ``num_users * num_layers`` user-major slots of
    ``max_seq_len`` each. Shared by ``TtPrefillRuntime`` (its default allocator)
    and the MLA model adapter, so the MLA KV layout has one definition.
    """
    kvpe_head_dim = hf_config.qk_rope_head_dim + hf_config.kv_lora_rank
    return init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_head_dim,
        mesh_device=mesh_device,
        seq_len=max_seq_len,
        mesh_shape=list(mesh_shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
    )
