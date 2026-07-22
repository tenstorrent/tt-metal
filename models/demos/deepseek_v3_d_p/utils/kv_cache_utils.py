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
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import DFlashDrafterConfig

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

    @property
    def storage_dtype(self):
        return {
            MlaKvCacheFormat.BFP8_TILE: ttnn.bfloat8_b,
            MlaKvCacheFormat.BF16_RM: ttnn.bfloat16,
            MlaKvCacheFormat.SCALED_FP8: ttnn.fp8_e4m3,
        }[self]

    @property
    def storage_layout(self):
        return ttnn.TILE_LAYOUT if self == MlaKvCacheFormat.BFP8_TILE else ttnn.ROW_MAJOR_LAYOUT

    def storage_width(self, geometry: "MlaKvCacheGeometry") -> int:
        return geometry.packed_row_bytes if self == MlaKvCacheFormat.SCALED_FP8 else geometry.logical_width

    @property
    def sparse_sdpa_format(self):
        try:
            return {
                MlaKvCacheFormat.BF16_RM: ttnn.transformer.SparseKVFormat.BF16,
                MlaKvCacheFormat.SCALED_FP8: ttnn.transformer.SparseKVFormat.SCALED_FP8,
            }[self]
        except KeyError as error:
            raise ValueError(f"{self} is not a sparse-SDPA cache format") from error


@dataclass(frozen=True)
class MlaKvCacheGeometry:
    """Logical MLA dimensions and the packed scaled-FP8 row they imply."""

    latent_dim: int
    rope_dim: int

    SCALE_BLOCK_SIZE = 128
    SCALE_ELEMENT_BYTES = 4
    ROPE_ELEMENT_BYTES = 2
    PACKED_FIELD_ADDRESS_UNIT_BYTES = 16

    @classmethod
    def from_config(cls, config) -> "MlaKvCacheGeometry":
        return cls(latent_dim=config.kv_lora_rank, rope_dim=config.qk_rope_head_dim)

    def __post_init__(self) -> None:
        if self.latent_dim <= 0 or self.rope_dim <= 0:
            raise ValueError("MLA KV cache dimensions must be positive")

    @property
    def logical_width(self) -> int:
        return self.latent_dim + self.rope_dim

    @property
    def num_scales(self) -> int:
        block_size = self.SCALE_BLOCK_SIZE
        if self.latent_dim % block_size != 0:
            raise ValueError(f"scaled MLA KV latent dimension {self.latent_dim} must be a multiple of {block_size}")
        return self.latent_dim // block_size

    @property
    def scale_bytes(self) -> int:
        return self.num_scales * self.SCALE_ELEMENT_BYTES

    @property
    def rope_offset_bytes(self) -> int:
        return self.latent_dim + self.scale_bytes

    @property
    def packed_row_bytes(self) -> int:
        return self.rope_offset_bytes + self.rope_dim * self.ROPE_ELEMENT_BYTES

    def validate_scaled(self) -> None:
        if self.rope_offset_bytes % self.PACKED_FIELD_ADDRESS_UNIT_BYTES != 0:
            raise ValueError(
                f"scaled MLA KV RoPE offset {self.rope_offset_bytes} must be "
                f"{self.PACKED_FIELD_ADDRESS_UNIT_BYTES}-byte aligned"
            )


@dataclass(frozen=True)
class MlaKvCache:
    """Persistent storage paired with the encoding of its physical rows.

    Logical MLA values are ``[latent || RoPE]``. Homogeneous formats store that
    row directly; scaled FP8 stores latent bytes, FP32 scales, and BF16 RoPE in
    one mixed-format row. Physical operations use ``storage`` as a bare tensor.
    """

    format: MlaKvCacheFormat
    storage: ttnn.Tensor
    geometry: MlaKvCacheGeometry

    def __post_init__(self) -> None:
        if self.format == MlaKvCacheFormat.SCALED_FP8:
            self.geometry.validate_scaled()
        dtype = self.format.storage_dtype
        layout = self.format.storage_layout
        width = self.format.storage_width(self.geometry)
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
        if latent.shape[-1] != self.geometry.latent_dim or rope.shape[-1] != self.geometry.rope_dim:
            raise ValueError(
                f"MLA KV inputs must be latent width {self.geometry.latent_dim} and RoPE width "
                f"{self.geometry.rope_dim}, got {latent.shape[-1]} and {rope.shape[-1]}"
            )
        if self.format == MlaKvCacheFormat.SCALED_FP8:
            return self._pack_scaled_fp8(latent, rope, intermediates=intermediates)
        packed = ttnn.concat([latent, rope], dim=-1)
        if packed.layout != self.storage.layout:
            converted = ttnn.to_layout(packed, self.storage.layout)
            ttnn.deallocate(packed)
            packed = converted
        if packed.dtype != self.storage.dtype:
            converted = ttnn.typecast(packed, self.storage.dtype)
            ttnn.deallocate(packed)
            packed = converted
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
            return reconstruct_scaled_fp8_kv_cache(physical, self.geometry)
        return physical.to(torch.bfloat16)


def unpack_scaled_fp8_kv_cache(
    packed: torch.Tensor, geometry: MlaKvCacheGeometry
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode a host copy of the packed sparse-MLA cache without interpreting its mixed fields as FP8.

    ``ttnn.to_torch`` preserves the packed tensor's FP8 bytes. Re-viewing that storage as uint8 lets the
    scale and RoPE fields be reconstructed using their native dtypes. Returns FP8 latent values widened
    to float32, FP32 scales, and BF16 RoPE values.
    """
    geometry.validate_scaled()
    if packed.shape[-1] != geometry.packed_row_bytes:
        raise ValueError(f"packed sparse KV width must be {geometry.packed_row_bytes}, got {packed.shape[-1]}")

    prefix = packed.shape[:-1]
    raw = packed.contiguous().view(torch.uint8)
    latent = (
        raw[..., : geometry.latent_dim].contiguous().view(packed.dtype).reshape(*prefix, geometry.latent_dim).float()
    )
    scales = (
        raw[..., geometry.latent_dim : geometry.rope_offset_bytes]
        .contiguous()
        .view(torch.float32)
        .reshape(*prefix, geometry.num_scales)
    )
    rope = raw[..., geometry.rope_offset_bytes :].contiguous().view(torch.bfloat16).reshape(*prefix, geometry.rope_dim)
    return latent, scales, rope


def reconstruct_scaled_fp8_kv_cache(packed: torch.Tensor, geometry: MlaKvCacheGeometry) -> torch.Tensor:
    """Reconstruct the logical BF16 ``[scaled latent || RoPE]`` cache from packed host bytes."""
    latent, scales, rope = unpack_scaled_fp8_kv_cache(packed, geometry)
    scaled = latent * scales.repeat_interleave(geometry.SCALE_BLOCK_SIZE, dim=-1)
    return torch.cat((scaled.to(torch.bfloat16), rope), dim=-1)


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
    hf_config,
    mesh_device,
    seq_len,
    mesh_shape,
    sp_axis,
    num_kvpe_cache_layers,
    num_users=1,
) -> MlaKvCache:
    """Allocate and zero a persistent MLA cache in the selected physical format.

    Homogeneous formats store the config-derived logical row directly. Scaled FP8 owns one
    ND-sharded mixed-format row per token. Physical DRAM usage is derived from the tensor's
    aligned page size, not the logical row width.
    """
    cache_format = MlaKvCacheFormat(cache_format)
    geometry = MlaKvCacheGeometry.from_config(hf_config)
    if cache_format == MlaKvCacheFormat.SCALED_FP8:
        geometry.validate_scaled()
    storage = init_kvpe_cache(
        kvpe_cache_head_dim=cache_format.storage_width(geometry),
        dtype=cache_format.storage_dtype,
        layout=cache_format.storage_layout,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_kvpe_cache_layers,
        num_users=num_users,
    )
    return MlaKvCache(format=cache_format, storage=storage, geometry=geometry)


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
        hf_config=hf_config,
        mesh_device=mesh_device,
        seq_len=max_seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
    )


def allocate_dflash_kv_cache(
    mesh_device: ttnn.MeshDevice,
    config: DFlashDrafterConfig,
    cache_seq: int,
    *,
    sp_axis: int = 0,
    tp_axis: int = 1,
    dtype: ttnn.DataType = ttnn.bfloat8_b,  # align w/ decode KV cache (init_kvpe_cache default is bf8); bf8/TILE
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Allocate the DFlash drafter's separate K and V context caches, owned OUTSIDE the module by the
    caller (prefill runner / test) and passed into ``TtDFlashDrafter.write_kv_cache`` — the drafter analog
    of ``allocate_mla_kvpe_cache`` above: one file owns each model's KV layout, and the model module only
    consumes the cache handed in (like the MLA model's ``forward(..., kvpe_cache=...)``). Keeping ownership
    with the caller lets it drive cache lifecycle (the migration hand-off to the decode mesh) and dtype
    (default bf8/``bfloat8_b`` to match the decode KV cache; see ``init_kvpe_cache``).

    Global (logical) shape ``[num_hidden_layers, num_key_value_heads, cache_seq, head_dim]``, TP-sharded on
    kv-head (dim 1) and SP-sharded on seq (dim 2) so each SP chip owns ``cache_seq/sp`` tokens (the
    decode/migration layout, no redundant per-SP copies). Allocated + zeroed on device (DRAMZeroFill) with
    no host tensor / H2D copy. Returns ``(k_cache, v_cache)``.

    NOTE: the interleaved DRAM format here is provisional pending decode alignment — see the
    ND_DRAM_SHARDED ``TODO`` in the body."""
    sp = mesh_device.shape[sp_axis]
    tp = mesh_device.shape[tp_axis]
    assert (
        config.num_key_value_heads % tp == 0
    ), f"num_key_value_heads ({config.num_key_value_heads}) must divide across tp ({tp})"
    assert cache_seq % sp == 0, f"cache_seq ({cache_seq}) must divide across sp ({sp})"

    # Per-device (post-shard) shape: kv-head split across TP (dim 1), seq split across SP (dim 2).
    local_shape = ttnn.Shape(
        [config.num_hidden_layers, config.num_key_value_heads // tp, cache_seq // sp, config.head_dim]
    )
    # 2D-shard topology matching ShardTensor2dMesh (seq on sp_axis -> dim 2, kv-head on tp_axis -> dim 1) so
    # the readback composer (ConcatMesh2dToTensor, read_dims=(2,1)) reconstructs the global cache — the same
    # layout the old from_torch(mesh_mapper=…) path produced (cf. DRAMZeroFill.allocate_kv_cache_on_device).
    dist_shape = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
    placements = [None, None]
    placements[sp_axis] = ttnn.PlacementShard(2)  # seq dim across SP
    placements[tp_axis] = ttnn.PlacementShard(1)  # kv-head dim across TP
    coords = [
        ttnn.MeshCoordinate([coord[i] for i in range(coord.dims())]) for coord in ttnn.MeshCoordinateRange(dist_shape)
    ]

    def _alloc_zeroed() -> ttnn.Tensor:
        # Allocate + zero on device (DRAMZeroFill) instead of a host torch.zeros + H2D copy: the drafter
        # cache scales with the sequence and the host pack/transfer of the full cache is slow (mirrors
        # init_kvpe_cache above). The shard topology is stamped after the fill.
        # TODO: switch this interleaved DRAM (DRAM_MEMORY_CONFIG) to ND_DRAM_SHARDED to align with the
        # decode-side drafter KV-cache layout for the migration hand-off (cf. init_kvpe_cache's NdShardSpec
        # + create_kv_chunk_address_table).
        cache = ttnn.allocate_tensor_on_device(
            local_shape, dtype, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        DRAMZeroFill.op(cache)
        cache.update_tensor_topology(ttnn.TensorTopology(dist_shape, placements, coords))
        return cache

    # K and V are independent caches → two separate on-device zero-fills (the old path shared one host
    # buffer across two from_torch copies).
    return _alloc_zeroed(), _alloc_zeroed()
