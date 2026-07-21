# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for KVPE cache initialization and management.
"""

import socket
from dataclasses import dataclass
from enum import Enum

import torch
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


class MlaKvCacheFormat(str, Enum):
    """Physical encodings supported by the persistent MLA cache."""

    BFP8_TILE = "bfp8_tile"
    BF16_RM = "bf16_rm"
    SCALED_FP8 = "scaled_fp8"


@dataclass(frozen=True)
class MlaKvCache:
    """Persistent storage paired with the encoding of its physical rows.

    Logical MLA values are ``[latent || RoPE]``. Homogeneous formats store that
    row directly; scaled FP8 stores latent bytes, FP32 scales, and BF16 RoPE in
    one mixed-format row. Physical operations use ``storage`` as a bare tensor.
    """

    format: MlaKvCacheFormat
    storage: ttnn.Tensor

    LATENT_DIM = 512
    SCALE_BLOCK_SIZE = 128
    NUM_SCALES = LATENT_DIM // SCALE_BLOCK_SIZE
    ROPE_DIM = 64
    LATENT_BYTES = LATENT_DIM
    SCALE_OFFSET_BYTES = LATENT_BYTES
    SCALE_BYTES = NUM_SCALES * 4
    ROPE_OFFSET_BYTES = SCALE_OFFSET_BYTES + SCALE_BYTES
    ROPE_BYTES = ROPE_DIM * 2
    PACKED_ROW_BYTES = ROPE_OFFSET_BYTES + ROPE_BYTES

    def __post_init__(self) -> None:
        specs = {
            MlaKvCacheFormat.BFP8_TILE: (ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 576),
            MlaKvCacheFormat.BF16_RM: (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, 576),
            MlaKvCacheFormat.SCALED_FP8: (ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT, self.PACKED_ROW_BYTES),
        }
        try:
            dtype, layout, width = specs[self.format]
        except KeyError as error:
            raise ValueError(f"unsupported MLA KV cache format: {self.format}") from error
        if self.storage.dtype != dtype:
            raise ValueError(f"{self.format} cache must use {dtype}, got {self.storage.dtype}")
        if self.storage.layout != layout:
            raise ValueError(f"{self.format} cache must use {layout}, got {self.storage.layout}")
        if self.storage.shape[-1] != width:
            raise ValueError(f"{self.format} cache width must be {width}, got {self.storage.shape[-1]}")

    def pack(
        self,
        latent: ttnn.Tensor,
        rope: ttnn.Tensor,
        *,
        intermediates: dict[str, ttnn.Tensor] | None = None,
    ) -> ttnn.Tensor:
        """Encode logical values for a physical cache write without mutating storage."""
        if self.format == MlaKvCacheFormat.SCALED_FP8:
            return self._pack_scaled_fp8(latent, rope, intermediates=intermediates)
        logical = ttnn.concat([latent, rope], dim=-1)
        packed = logical
        if packed.layout != self.storage.layout:
            packed = ttnn.to_layout(packed, self.storage.layout)
        if packed.dtype != self.storage.dtype:
            cast = ttnn.typecast(packed, self.storage.dtype)
            if packed is not logical:
                ttnn.deallocate(packed)
            packed = cast
        if packed is not logical:
            ttnn.deallocate(logical)
        if intermediates is not None:
            intermediates["tt_kvpe"] = ttnn.clone(packed)
        return packed

    def _pack_scaled_fp8(
        self, latent: ttnn.Tensor, rope: ttnn.Tensor, *, intermediates: dict[str, ttnn.Tensor] | None
    ) -> ttnn.Tensor:
        latent_rm = ttnn.to_layout(latent, ttnn.ROW_MAJOR_LAYOUT)
        latent_fp8, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
            latent_rm, round_scale_to_power_of_two=True
        )
        if latent_rm is not latent:
            ttnn.deallocate(latent_rm)
        rope_rm = ttnn.to_layout(rope, ttnn.ROW_MAJOR_LAYOUT)
        packed = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(latent_fp8, scales, rope_rm)
        if intermediates is not None:
            reconstructed = ttnn.experimental.deepseek_prefill.per_token_cast_back(
                latent_fp8, scales, output_dtype=ttnn.bfloat16
            )
            intermediates["tt_kvpe"] = ttnn.concat([reconstructed, rope_rm], dim=-1)
            ttnn.deallocate(reconstructed)
            intermediates["tt_kvpe_latent"] = ttnn.clone(latent_fp8)
            intermediates["tt_kvpe_scales"] = ttnn.clone(scales)
            intermediates["tt_kvpe_rope"] = ttnn.clone(rope_rm)
            intermediates["tt_kvpe_packed"] = ttnn.clone(packed)
        ttnn.deallocate(latent_fp8)
        ttnn.deallocate(scales)
        if rope_rm is not rope:
            ttnn.deallocate(rope_rm)
        return packed

    def unpack_host(self, physical: torch.Tensor) -> torch.Tensor:
        """Decode host physical rows into logical BF16 [latent || RoPE] values."""
        if self.format == MlaKvCacheFormat.SCALED_FP8:
            return reconstruct_scaled_fp8_kv_cache(physical)
        return physical.to(torch.bfloat16)


def unpack_scaled_fp8_kv_cache(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode a host copy of the packed sparse-MLA cache without interpreting its mixed fields as FP8.

    ``ttnn.to_torch`` preserves the packed tensor's FP8 bytes. Re-viewing that storage as uint8 lets the
    scale and RoPE fields be reconstructed using their native dtypes. Returns FP8 latent values widened
    to float32, FP32 scales, and BF16 RoPE values.
    """
    if packed.shape[-1] != MlaKvCache.PACKED_ROW_BYTES:
        raise ValueError(f"packed sparse KV width must be {MlaKvCache.PACKED_ROW_BYTES}, got {packed.shape[-1]}")

    prefix = packed.shape[:-1]
    raw = packed.contiguous().view(torch.uint8)
    latent = (
        raw[..., : MlaKvCache.SCALE_OFFSET_BYTES]
        .contiguous()
        .view(packed.dtype)
        .reshape(*prefix, MlaKvCache.LATENT_DIM)
        .float()
    )
    scales = (
        raw[..., MlaKvCache.SCALE_OFFSET_BYTES : MlaKvCache.ROPE_OFFSET_BYTES]
        .contiguous()
        .view(torch.float32)
        .reshape(*prefix, MlaKvCache.NUM_SCALES)
    )
    rope = (
        raw[..., MlaKvCache.ROPE_OFFSET_BYTES :].contiguous().view(torch.bfloat16).reshape(*prefix, MlaKvCache.ROPE_DIM)
    )
    return latent, scales, rope


def reconstruct_scaled_fp8_kv_cache(packed: torch.Tensor) -> torch.Tensor:
    """Reconstruct the logical BF16 ``[scaled latent || RoPE]`` cache from packed host bytes."""
    latent, scales, rope = unpack_scaled_fp8_kv_cache(packed)
    scaled = latent * scales.repeat_interleave(MlaKvCache.SCALE_BLOCK_SIZE, dim=-1)
    return torch.cat((scaled.to(torch.bfloat16), rope), dim=-1)


def get_num_dram_banks(mesh_device):
    """Usable DRAM banks on this device. Full Blackhole = 8; harvested parts expose fewer (e.g. 7).

    The KV cache ND-shards round-robin across these banks and the disaggregation address table replays
    that exact striping (`curr_bank_id = (curr_bank_id + 1) % num_banks`), so both MUST derive the count
    from the same device. dram_grid_size().x is the number of DRAM cores/banks the device exposes."""
    return mesh_device.dram_grid_size().x


def create_kv_chunk_address_table_ds(
    config, mesh_device, mesh_shape, seq_len, sp_axis, tt_kvpe_cache, chunk_size_bytes, num_users=1
):
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
        num_users: number of per-user cache slots (multi-user balanced layout is a follow-up;
            only num_users == 1 is supported here)

    Returns:
        lookup_table: Populated KvChunkAddressTable
    """
    assert num_users == 1, "create_kv_chunk_address_table_ds (balanced) supports only num_users == 1"
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
    # Must match the bank count the cache was ND-sharded across (see get_num_dram_banks).
    num_dram_banks = get_num_dram_banks(mesh_device)
    for row in range(len(device_group_idx_per_row)):
        group_idx = device_group_idx_per_row[row]
        curr_bank_id = 0
        curr_bank_offset = 0

        logger.debug(
            f"Rank: {rank} Populating device_group_index: {group_idx} with positions: {device_position_indices_low_strip[row]} and {device_position_indices_high_strip[row]}"
        )
        current_position, max_position = device_position_indices_low_strip[row]
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

                curr_bank_id = (curr_bank_id + 1) % num_dram_banks
                # move to next chunk offset
                if curr_bank_id == 0:
                    curr_bank_offset += chunk_size_bytes
                layer_current_position += NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
                if chunk == num_chunks_in_strip - 1:
                    # switch to high chunk
                    assert (
                        layer_current_position == layer_max_position + 1
                    ), f"Missmatch in position calculation. Expected layer current_position to be {layer_max_position + 1}, but it is: {layer_current_position}."
                    layer_current_position, layer_max_position = device_position_indices_high_strip[row]

    return lookup_table


def create_kv_chunk_address_table_kimi(
    config, mesh_device, mesh_shape, seq_len, sp_axis, tt_kvpe_cache, chunk_size_bytes, num_users=1
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
        num_users: Number of users (slots) sharing the buffer; cache batch dim folds them as user * num_layers + layer

    Returns:
        lookup_table: Populated KvChunkAddressTable
    """
    lookup_table = ttnn.experimental.disaggregation.KvChunkAddressTable(config)
    return populate_kv_chunk_address_table_kimi(
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


def populate_kv_chunk_address_table_kimi(
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

    Factored out of create_kv_chunk_address_table_kimi so a single multi-config table can hold several
    caches at once (the serving convention is config 0 = the MLA KVPE cache, config 1 = the block-cyclic
    index-key cache); each config carries its own grid + chunk_size_bytes and is addressed by config_id.
    The device-group
    side table and fabric-node host map are SHARED across configs — re-registering them here per config is
    safe (add_device_group dedups identical replica sets; set_fabric_node_host is idempotent).

    Args:
        lookup_table: an existing KvChunkAddressTable (single- or multi-config).
        config: the KvChunkAddressTableConfig for THIS config_id (read for num_layers).
        config_id: which config of the table to populate (default 0, the single-config case).
        (remaining args as in create_kv_chunk_address_table_kimi)

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
    tt_kvpe_cache = ttnn.allocate_tensor_on_device(
        ttnn.Shape([num_users * num_layers, 1, seq_len_local, kvpe_cache_head_dim]),
        dtype,
        layout,
        mesh_device,
        kv_mem_config,
    )
    DRAMZeroFill.op(tt_kvpe_cache)

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


def init_mla_kv_cache(
    *,
    cache_format: MlaKvCacheFormat,
    mesh_device: ttnn.MeshDevice,
    seq_len: int,
    mesh_shape: list[int] | tuple[int, ...],
    sp_axis: int,
    num_kvpe_cache_layers: int,
    num_users: int = 1,
) -> MlaKvCache:
    """Allocate and zero a persistent MLA cache in the selected physical format.

    BFP8 tile and BF16 row-major store one homogeneous 576-wide row. Scaled
    FP8 stores one ND-sharded, 656-byte mixed-format row per token. Physical DRAM usage is
    derived from the tensor's aligned page size, not the logical row width.
    """
    cache_format = MlaKvCacheFormat(cache_format)

    layout = ttnn.TILE_LAYOUT if cache_format == MlaKvCacheFormat.BFP8_TILE else ttnn.ROW_MAJOR_LAYOUT
    common = dict(
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_kvpe_cache_layers,
        num_users=num_users,
        layout=layout,
    )
    if cache_format in (MlaKvCacheFormat.BFP8_TILE, MlaKvCacheFormat.BF16_RM):
        return MlaKvCache(
            format=cache_format,
            storage=init_kvpe_cache(
                kvpe_cache_head_dim=MlaKvCache.LATENT_DIM + MlaKvCache.ROPE_DIM,
                dtype=ttnn.bfloat8_b if cache_format == MlaKvCacheFormat.BFP8_TILE else ttnn.bfloat16,
                **common,
            ),
        )
    if cache_format != MlaKvCacheFormat.SCALED_FP8:
        raise ValueError(f"unsupported MLA KV cache format: {cache_format}")

    packed = init_kvpe_cache(
        kvpe_cache_head_dim=MlaKvCache.PACKED_ROW_BYTES,
        dtype=ttnn.fp8_e4m3,
        **common,
    )
    return MlaKvCache(format=cache_format, storage=packed)


def allocate_mla_kvpe_cache(
    *, mesh_device, hf_config, max_seq_len, mesh_shape, sp_axis, num_layers, num_users
) -> MlaKvCache:
    """Allocate the MLA KVPE cache for one runtime from the HF config.

    The MLA per-token cache row is ``qk_rope_head_dim + kv_lora_rank`` wide; ONE
    shared cache holds ``num_users * num_layers`` user-major slots of
    ``max_seq_len`` each. Shared by ``TtPrefillRuntime`` (its default allocator)
    and the MLA model adapter, so the MLA KV layout has one definition.
    """
    return init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,
        mesh_device=mesh_device,
        seq_len=max_seq_len,
        mesh_shape=list(mesh_shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
    )
