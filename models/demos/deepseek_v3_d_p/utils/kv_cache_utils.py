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
from models.demos.common.prefill.runners.migration import allgather_kv_stage_layout, get_num_dram_banks
from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import DFlashDrafterConfig

# This is a predefined constant for the number of contiguous tokens in a DRAM bank
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32
# Nominal DRAM bank count for a full (unharvested) Blackhole part. Prefer get_num_dram_banks(device)
# at runtime: harvested parts expose fewer banks (e.g. 7), and the cache ND-shard grid + the
# disaggregation address-table striding must both use the device's actual count to stay consistent.
BH_NUM_DRAM_BANKS = 8
PREFILL_CHUNK_OUTPUT_TOKENS = 5 * 1024


def create_sequence_cache_mesh_composer(mesh_device, sp_axis: int = 0, full_mesh: bool = False):
    """Compose canonical sequence shards while dropping only true replicas.

    Axis mode concatenates the named SP axis and collapses TP replicas. Full-mesh mode flattens all
    2D coordinates in canonical row-major order and concatenates every device shard.
    """
    assert full_mesh or sp_axis == 0, "axis-mode sequence cache composition currently assumes sp_axis=0"
    sequence_factor = mesh_device.get_num_devices() if full_mesh else mesh_device.shape[sp_axis]
    return ttnn.create_mesh_composer(
        mesh_device,
        config=ttnn.MeshComposerConfig(
            dims=(2, -1),
            mesh_shape_override=ttnn.MeshShape(sequence_factor, 1),
        ),
    )


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


def create_kv_chunk_address_table_ds(
    config, mesh_device, mesh_shape, seq_len, sp_axis, kvpe_cache, chunk_size_bytes, num_users=1
):
    """
    Create and populate a KV chunk address table for disaggregation.

    Args:
        config: KvChunkAddressTableConfig
        mesh_device: Mesh device for TT
        mesh_shape: Shape of mesh device
        seq_len: Sequence length
        sp_axis: Sequence parallel axis
        kvpe_cache: Initialized KVPE cache on device
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

    logger.debug(f"kvpe cache shape is: {kvpe_cache.shape}")
    dram_bank_base_addr = kvpe_cache.buffer_address()
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


def merged_num_layers(stage_layout):
    """Layers a merged (all-stage) table config spans: the sum of every stage's owned count. Also
    enforces tt-blaze's missing-layer guard -- the stages must tile ``[0, total)`` with no gaps or
    overlaps, which compute_layer_split's contiguous partition satisfies."""
    total = sum(s["count"] for s in stage_layout)
    expected = 0
    for s in sorted(stage_layout, key=lambda s: s["first_layer"]):
        if s["first_layer"] != expected:
            raise RuntimeError(
                f"gathered layer ranges are not contiguous: expected next stage at layer {expected} but got "
                f"first_layer={s['first_layer']} (stages={[(x['first_layer'], x['count']) for x in stage_layout]})"
            )
        expected += s["count"]
    return total


def create_kv_chunk_address_table_kimi(
    config,
    mesh_device,
    mesh_shape,
    seq_len,
    sp_axis,
    kvpe_cache,
    chunk_size_bytes,
    num_users=1,
    first_layer_idx=0,
    num_my_layers=None,
    stage_layout=None,
):
    """
    Create and populate a KV chunk address table for disaggregation (Kimi K2.6 model - non-balanced).

    Builds ONE table spanning every pipeline stage's layers, following tt-blaze's layer->mesh merge:
    each rank owns a contiguous LAYER range on its full mesh, and the table places each global layer's
    chunks on its OWNING stage's devices / KV base address. The per-(slot, layer, chunk) address math
    is unchanged from the original single-stage builder (a bank round-robin from the stage's base
    addr, per SP row); only the layer index is offset to the global range and the base/mesh/host come
    from the owning stage.

    Args:
        config: KvChunkAddressTableConfig (its num_layers is overwritten with the gathered global total)
        mesh_device: this rank's MeshDevice (its full SP x TP mesh)
        mesh_shape: (rows, cols) of that mesh; rows == SP, cols == TP
        seq_len, sp_axis, kvpe_cache, chunk_size_bytes, num_users: as before
        first_layer_idx: this rank's first global layer id (from compute_layer_split)
        num_my_layers: this rank's layer count (defaults to config.num_layers for single-stage callers)
        stage_layout: optional pre-gathered per-rank stage layout from allgather_kv_stage_layout().
            Pass it when the COLLECTIVE all-gather has already run on all ranks (so only rank 0 builds);
            leave None to run the all-gather inline (single-rank / tests).

    Returns:
        lookup_table: Populated KvChunkAddressTable
    """
    num_my_layers = num_my_layers if num_my_layers is not None else config.num_layers

    # COLLECTIVE (all ranks) unless already gathered: each rank reports its layer range + full mesh +
    # KV base + host. The merge below then covers every layer across every stage. The publish path
    # hoists this so all ranks participate while only rank 0 builds; tests/single-rank run it inline.
    if stage_layout is None:
        stage_layout = allgather_kv_stage_layout(
            mesh_device, int(kvpe_cache.buffer_address()), mesh_shape, first_layer_idx, num_my_layers
        )

    rows = mesh_shape[0]

    # This (building) rank's cache must hold exactly its own stage's layers, folded with num_users.
    assert (
        kvpe_cache.shape[0] == num_users * num_my_layers
    ), f"cache batch dim {kvpe_cache.shape[0]} != num_users({num_users}) * num_my_layers({num_my_layers})"

    # The merged table spans ALL layers (not just this rank's), so size the table to the global total.
    config.num_layers = merged_num_layers(stage_layout)
    lookup_table = ttnn.experimental.disaggregation.KvChunkAddressTable(config)
    return populate_kv_chunk_address_table_kimi(
        lookup_table=lookup_table,
        config=config,
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tt_kvpe_cache=kvpe_cache,
        chunk_size_bytes=chunk_size_bytes,
        num_users=num_users,
        config_id=0,
        stage_layout=stage_layout,
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
    stage_layout=None,
    layer_rows=None,
    tp_axis=None,
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
        layer_rows: table row to publish each dense cache layer at; None (default) means row == layer.
            A compacted cache passes its dense -> global layer map here; DRAM addresses do not move.
        tp_axis: None (default) = TP-REPLICATED, one device group per row. When set (KV dedup), each
            (row, col) device holds a distinct sub-slice, so the table uses per-device singleton groups:
            linear chip row*tp + col owns tokens [seq_chunk*5120 + row*640 + col*(640/tp), +640/tp).
        (remaining args as in create_kv_chunk_address_table_kimi)

    Returns:
        lookup_table: the same table, with config_id populated.
    """

    # Row a dense cache layer is published at; identity unless the cache is compacted.
    def _table_row(dense_layer):
        if layer_rows is None:
            return dense_layer
        assert dense_layer < len(layer_rows), (
            f"layer_rows has {len(layer_rows)} entries but the cache holds dense layer {dense_layer}; "
            f"the map must cover every layer the cache physically stores"
        )
        return layer_rows[dense_layer]

    if tp_axis is None and stage_layout is not None:
        # ---- SP-only / TP-replicated, stage_layout-driven path (PP-capable, #48826). ----
        # Per-stage device groups + host tags are built inside the stage loop below (one group per
        # (stage, SP row)); no rank-local single-stage device-group pass here — the multi-stage merge
        # supersedes it, and a rank-local set_fabric_node_host(localhost) would fight the per-stage host.
        rows = mesh_shape[0]

        tokens_per_chunk_local = PREFILL_CHUNK_OUTPUT_TOKENS // mesh_shape[sp_axis]  # 640 for 5k chunks
        num_chunks_per_seq_len = seq_len // PREFILL_CHUNK_OUTPUT_TOKENS  # number of 5k chunks in the seq len

        assert (
            seq_len % PREFILL_CHUNK_OUTPUT_TOKENS == 0
        ), f"seq_len {seq_len} must be a multiple of {PREFILL_CHUNK_OUTPUT_TOKENS}"

        assert tokens_per_chunk_local % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0, (
            f"{PREFILL_CHUNK_OUTPUT_TOKENS} tokens / sp({mesh_shape[sp_axis]}) = {tokens_per_chunk_local}, "
            f"not a multiple of {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}"
        )

        # tt-blaze-style merge: for every STAGE place its layers' chunks on ITS mesh at ITS base addr.
        # Within a stage we replay the original single-stage build exactly (one device group per SP row, an
        # independent bank round-robin per row sequencing slot -> local layer -> chunk), but write to the
        # GLOBAL layer index (first_layer + local_layer) so every stage lands in one table.
        for stage in stage_layout:
            dram_bank_base_addr = stage["base_addr"]
            num_dram_banks = stage["num_banks"]
            host_name = f"host-{stage['host_tag']:08x}"  # crc32 tag rebuilt to a string (int-only allgather)
            first = stage["first_layer"]
            count = stage["count"]
            stage_fnids = stage["fnids"]
            for row in range(rows):
                # Data is replicated across each TP column, so one device group per (stage, SP row).
                fnids_row = stage_fnids[row]
                group_idx = lookup_table.add_device_group(fnids_row)
                for fid in fnids_row:
                    lookup_table.set_fabric_node_host(fid, host_name=host_name)
                curr_bank_id = 0
                curr_bank_offset = 0
                for slot in range(num_users):
                    for local_layer in range(count):
                        global_layer = first + local_layer
                        for seq_chunk in range(num_chunks_per_seq_len):
                            chunk_token_start = seq_chunk * PREFILL_CHUNK_OUTPUT_TOKENS + row * tokens_per_chunk_local
                            chunk_token_end = chunk_token_start + tokens_per_chunk_local
                            for position in range(
                                chunk_token_start, chunk_token_end, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
                            ):
                                location = ttnn.experimental.disaggregation.KvCacheLocation()
                                location.noc_addr = (curr_bank_id << 32) | (dram_bank_base_addr + curr_bank_offset)
                                location.size_bytes = chunk_size_bytes
                                location.device_group_index = group_idx
                                lookup_table.set(_table_row(global_layer), position, slot, location, config_id)

                                curr_bank_id = (curr_bank_id + 1) % num_dram_banks
                                if curr_bank_id == 0:
                                    curr_bank_offset += chunk_size_bytes
        return lookup_table

    # ---- Legacy single-stage path (direct call, stage_layout is None). ----
    # The pre-#48826 behavior, still exercised by direct callers that don't build a stage_layout
    # (e.g. test_glm52_kv_cache_table and the kv_chunk_table runner): base addr / bank count derived
    # from the cache itself. tp_axis=None (TP-replicated) is tp_factor == 1 below, so both layouts run
    # the same loop: one device group per row, spanning the whole row's slice.
    host_name = socket.gethostname()

    rank = ttnn.distributed_context_get_rank()
    size = ttnn.distributed_context_get_size()
    total_rows = mesh_shape[0]
    rank_row_start = int(rank) * total_rows // int(size)
    rank_row_end = rank_row_start + total_rows // int(size)

    # DENSE rows the cache physically holds. With layer_rows the caller has WIDENED the published layer
    # axis to the global count (config.num_layers) while the cache still holds only its compacted rows,
    # so both the loop bound and the batch check below must use the dense count -- otherwise a compacted
    # cache is rejected for not having num_users * global_layers slots. The stage_layout branch above is
    # immune because it loops over each stage's own `count`.
    num_layers = len(layer_rows) if layer_rows is not None else config.num_layers

    assert (
        seq_len % PREFILL_CHUNK_OUTPUT_TOKENS == 0
    ), f"seq_len {seq_len} must be a multiple of {PREFILL_CHUNK_OUTPUT_TOKENS}"
    num_chunks_per_seq_len = seq_len // PREFILL_CHUNK_OUTPUT_TOKENS

    assert (
        tt_kvpe_cache.shape[0] == num_users * num_layers
    ), f"cache batch dim {tt_kvpe_cache.shape[0]} != num_users({num_users}) * num_layers({num_layers})"

    tokens_per_chunk_local = PREFILL_CHUNK_OUTPUT_TOKENS // mesh_shape[sp_axis]  # 640 for 5k chunks
    assert tokens_per_chunk_local % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0, (
        f"{PREFILL_CHUNK_OUTPUT_TOKENS} tokens / sp({mesh_shape[sp_axis]}) = {tokens_per_chunk_local}, "
        f"not a multiple of {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}"
    )

    dram_bank_base_addr = tt_kvpe_cache.buffer_address()
    # Must match the bank count the cache was ND-sharded across (see get_num_dram_banks).
    num_dram_banks = get_num_dram_banks(mesh_device)

    if tp_axis is not None:
        assert sp_axis == 0 and tp_axis == 1, (
            f"TP-sharded KV chunk table requires the production (sp_axis=0, tp_axis=1) layout, "
            f"got sp_axis={sp_axis}, tp_axis={tp_axis}"
        )
        # This path ignores stage_layout and derives everything from THIS rank's cache/mesh, so a
        # multi-stage (PP) layout would silently produce a table covering only one rank's own layers.
        assert stage_layout is None or len(stage_layout) == 1, (
            f"TP-sharded KV chunk table is single-stage only: got a {len(stage_layout)}-stage layout. "
            f"The TP-sharded branch does not merge per-stage layer ranges / base addresses."
        )
    tp_factor = mesh_shape[tp_axis] if tp_axis is not None else 1
    tokens_per_device = tokens_per_chunk_local // tp_factor  # 160 for 5k chunks on tp=4
    assert (
        tokens_per_chunk_local % tp_factor == 0
    ), f"tokens_per_chunk_local ({tokens_per_chunk_local}) must be divisible by tp ({tp_factor})"
    assert (
        tokens_per_device % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0
    ), f"tokens_per_device ({tokens_per_device}) must be a multiple of {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}"

    # TP-replicated: one group per row spanning every col. TP-sharded: a singleton per (row, col), since
    # each device holds a DISTINCT sub-slice and must be addressed individually.
    device_group_idx = {}  # (row, col) -> group_idx
    all_fabric_node_ids = []
    for row in range(rank_row_start, rank_row_end):
        for col in range(tp_factor):
            if tp_axis is None:
                fids = [mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, c)) for c in range(mesh_shape[1])]
            else:
                fids = [mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col))]
            all_fabric_node_ids.extend(fids)
            device_group_idx[(row, col)] = lookup_table.add_device_group(fids)

    for fid in all_fabric_node_ids:
        lookup_table.set_fabric_node_host(fid, host_name=host_name)
        logger.debug(
            f"Set host name for fabric node id: mesh_id={int(fid.mesh_id)}, chip_id={int(fid.chip_id)} "
            f"to {host_name}"
        )

    for global_row in range(rank_row_start, rank_row_end):
        for col in range(tp_factor):
            group_idx = device_group_idx[(global_row, col)]
            # Bank round-robin restarts per device: each ND-shards its own tokens_per_device rows from
            # bank 0 / offset 0, and the base address is identical on every device (co-located shards).
            curr_bank_id = 0
            curr_bank_offset = 0
            for slot in range(num_users):
                for layer in range(num_layers):
                    for seq_chunk in range(num_chunks_per_seq_len):
                        chunk_token_start = (
                            seq_chunk * PREFILL_CHUNK_OUTPUT_TOKENS
                            + global_row * tokens_per_chunk_local
                            + col * tokens_per_device
                        )
                        chunk_token_end = chunk_token_start + tokens_per_device
                        for position in range(chunk_token_start, chunk_token_end, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                            location = ttnn.experimental.disaggregation.KvCacheLocation()
                            location.noc_addr = (curr_bank_id << 32) | (dram_bank_base_addr + curr_bank_offset)
                            location.size_bytes = chunk_size_bytes
                            location.device_group_index = group_idx
                            lookup_table.set(_table_row(layer), position, slot, location, config_id)

                            curr_bank_id = (curr_bank_id + 1) % num_dram_banks
                            if curr_bank_id == 0:
                                curr_bank_offset += chunk_size_bytes

    return lookup_table


def populate_kv_chunk_address_table_dflash(
    lookup_table,
    config,
    mesh_device,
    mesh_shape,
    seq_len,
    sp_axis,
    tp_axis,
    kv_cache,
    chunk_size_bytes,
    num_kv_heads,
    head_idx,
    num_users=1,
    config_id=0,
    chunk_size_global=PREFILL_CHUNK_OUTPUT_TOKENS,
):
    """
    Populate ONE config (``config_id``) of an existing KvChunkAddressTable from ONE HEAD of the DFlash
    drafter's K or V cache (see ``allocate_dflash_kv_cache`` for the layout being described).

    The drafter analog of ``populate_kv_chunk_address_table_kimi``, differing in the two ways the
    drafter's cache differs from MLA's KVPE cache:

      * **TP carries heads, not replicas.** The MLA latent cache is one head (``shape[1] == 1``)
        replicated across every TP column, so a whole SP row is a replica set and one device group
        covers it. The drafter shards its kv-heads over the TP columns, so column
        ``head_idx // heads_per_chip`` is the ONLY device holding this head — and the migration worker
        treats a device group as REPLICAS (it reads one member and writes those bytes to every
        destination), so a row-spanning group would migrate one column's heads as if they were the
        whole row. Hence a SINGLE-MEMBER group per (config, SP row), the same shape
        ``models/demos/minimax_m3/tt/runners/kv_chunk_table.py`` uses for its head-sharded K/V.

      * **Several heads per chip, so the bank walk is strided.** The table key is
        (layer, position, slot) with no head axis, so each head needs its own config — and a per-head
        config visits only every ``heads_per_chip``-th run of that chip's ND shards. The incremental
        ``curr_bank_id += 1`` counter the kimi/M3 builders use is correct only for a contiguous walk,
        so this one computes ``dram_shard_idx`` outright. The two agree exactly at heads_per_chip == 1.

    Address math. One ND shard — the same 32-token object ``blocks_local`` / ``blocks_per_chunk_local``
    count as a "block" — is ``[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim]``: 32 tokens of ONE
    head of ONE layer. Shards are laid out ROUND_ROBIN_1D over the DRAM banks, so a chip's shard ``s``
    lives in bank ``s % num_banks`` at ``base + (s // num_banks) * chunk_size_bytes``. Enumerating the
    per-chip shard grid ``[num_users*num_layers, heads_per_chip, seq_local/32, 1]`` row-major gives ``s``:

        dram_shard_idx = ((slot * num_layers + layer) * heads_per_chip + head_idx % heads_per_chip)
                         * blocks_local + seq_chunk * blocks_per_chunk_local + block_in_chunk

    for the chip's ``block_in_chunk``-th local 32-token block of block-cyclic chunk ``seq_chunk``, which
    holds global position ``seq_chunk * chunk_size_global + row * tokens_per_chunk_local
    + block_in_chunk * 32``. That row-major ordering — dim 1 between dim 0 and dim 2 — is the one
    assumption here the working MLA builder does NOT already prove, since its dim 1 has extent 1; a
    per-head write/readback settles it, and getting it backwards swaps head and layer strides rather
    than failing loudly.

    Single-stage only (no ``stage_layout``): the drafter is written on the last pipeline rank alone
    (``tt_prefill_runtime`` guards the write with ``is_last_rank``), so there is no cross-stage layer
    range to merge and the base address comes from this rank's own tensor instead of an all-gather.

    Args:
        lookup_table: an existing KvChunkAddressTable (single- or multi-config).
        config: the KvChunkAddressTableConfig for THIS config_id (read for num_layers).
        kv_cache: the drafter K or V device tensor this config describes (read for its base address).
        chunk_size_bytes: bytes of one 32-token block (``kv_chunk_table._dram_chunk_size_bytes``).
        num_kv_heads: the drafter's GLOBAL kv-head count (8 for Kimi-K2.6-DFlash).
        head_idx: which global head in [0, num_kv_heads) this config describes.
        config_id: which config of the table to populate.
        chunk_size_global: the block-cyclic period, i.e. the runtime's prefill chunk size. Explicit
            rather than read from the module constant because a cache written at one period and
            addressed at another yields plausible-looking, wholly wrong addresses.
        (remaining args as in populate_kv_chunk_address_table_kimi)

    Returns:
        lookup_table: the same table, with config_id populated.
    """
    assert sp_axis != tp_axis, f"sp_axis and tp_axis must differ; both are {sp_axis}"
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[tp_axis]
    num_layers = config.num_layers

    assert (
        num_kv_heads % tp == 0
    ), f"num_kv_heads ({num_kv_heads}) must divide across tp ({tp}) — allocate_dflash_kv_cache asserts the same"
    heads_per_chip = num_kv_heads // tp
    assert 0 <= head_idx < num_kv_heads, f"head_idx ({head_idx}) out of range for num_kv_heads ({num_kv_heads})"

    # allocate_dflash_kv_cache builds this tensor from its PER-DEVICE shape (allocate_tensor_on_device
    # over local_shape, then update_tensor_topology), so both dims below are local: dim 0 is the
    # unsharded user-major (slot, layer) fold and dim 1 is this chip's slice of the kv-heads.
    assert kv_cache.shape[0] == num_users * num_layers, (
        f"drafter cache batch dim {kv_cache.shape[0]} != num_users({num_users}) * num_layers({num_layers}); "
        f"the walk below assumes the user-major slot*num_layers+layer fold"
    )
    assert kv_cache.shape[1] == heads_per_chip, (
        f"drafter cache head dim {kv_cache.shape[1]} != num_kv_heads({num_kv_heads}) / tp({tp}) = "
        f"{heads_per_chip}; the per-head shard stride below would land on the wrong head"
    )

    assert (
        seq_len % chunk_size_global == 0
    ), f"seq_len ({seq_len}) must be a whole number of block-cyclic chunks ({chunk_size_global})"
    assert seq_len % sp == 0, f"seq_len ({seq_len}) must be divisible by sp ({sp})"
    seq_local = seq_len // sp
    assert seq_local % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0, (
        f"per-chip seq ({seq_local} = seq_len {seq_len} / sp {sp}) must be a multiple of "
        f"{NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}; the shard grid is whole 32-token blocks"
    )
    blocks_local = seq_local // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

    tokens_per_chunk_local = chunk_size_global // sp  # 640 for 5k chunks on sp=8
    assert tokens_per_chunk_local % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0, (
        f"{chunk_size_global} tokens / sp({sp}) = {tokens_per_chunk_local}, "
        f"not a multiple of {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}"
    )
    blocks_per_chunk_local = tokens_per_chunk_local // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    num_chunks_per_seq_len = seq_len // chunk_size_global

    host_name = socket.gethostname()
    dram_bank_base_addr = int(kv_cache.buffer_address())
    # Must match the bank count the cache was ND-sharded across (see get_num_dram_banks).
    num_dram_banks = get_num_dram_banks(mesh_device)

    col = head_idx // heads_per_chip  # the TP column that owns this head
    h_local = head_idx % heads_per_chip  # its index within that chip's head slice

    for row in range(sp):
        # One device, not one row: this head lives on exactly (row, col), and the worker would read a
        # multi-member group as replicas. Written axis-generically so sp_axis/tp_axis can swap.
        coord = [0, 0]
        coord[sp_axis] = row
        coord[tp_axis] = col
        fabric_node_id = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
        group_idx = lookup_table.add_device_group([fabric_node_id])
        lookup_table.set_fabric_node_host(fabric_node_id, host_name=host_name)

        for slot in range(num_users):
            for layer in range(num_layers):
                # DRAM shard index of the FIRST block of this (slot, layer, local head) plane. A plane's
                # seq blocks are contiguous in the shard grid, so the per-position term below is an add.
                head_base_dram_shard = ((slot * num_layers + layer) * heads_per_chip + h_local) * blocks_local
                for seq_chunk in range(num_chunks_per_seq_len):
                    chunk_token_start = seq_chunk * chunk_size_global + row * tokens_per_chunk_local
                    chunk_token_end = chunk_token_start + tokens_per_chunk_local
                    # Same loop shape as populate_kv_chunk_address_table_kimi — position IS the table key
                    # rather than a value derived after the fact. enumerate recovers the block index the
                    # shard walk needs; the tokens_per_chunk_local assert above makes block_in_chunk
                    # exactly 0..blocks_per_chunk_local-1.
                    for block_in_chunk, position in enumerate(
                        range(chunk_token_start, chunk_token_end, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK)
                    ):
                        dram_shard_idx = head_base_dram_shard + seq_chunk * blocks_per_chunk_local + block_in_chunk
                        # No curr_ prefix on purpose: derived per entry, not a carried cursor as in kimi/M3.
                        bank_id = dram_shard_idx % num_dram_banks
                        bank_offset = (dram_shard_idx // num_dram_banks) * chunk_size_bytes
                        location = ttnn.experimental.disaggregation.KvCacheLocation()
                        location.noc_addr = (bank_id << 32) | (dram_bank_base_addr + bank_offset)
                        location.size_bytes = chunk_size_bytes
                        location.device_group_index = group_idx
                        lookup_table.set(layer, position, slot, location, config_id)
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
    full_mesh=False,
    tp_axis=None,
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
        tp_axis: KV dedup. None (default) = SP-sharded, TP-replicated (per-device rows = seq_len / sp).
            When set, also sharded across TP (rows = seq_len / (sp*tp)); the container topology is
            unchanged (the op defines the per-device layout) — only the per-device seq width shrinks.

    Returns:
        kvpe_cache: Initialized KVPE cache on device
    """
    # hack in num_users * num_layers into batch size, so each user's layers are contiguous in memory
    num_layers = num_kvpe_cache_layers
    # full_mesh and tp_axis express the SAME striping by different means -- row-major over the complete
    # mesh is exactly sp*tp -- so allowing both would divide by the shard extent twice.
    assert not (
        full_mesh and tp_axis is not None
    ), f"full_mesh already shards across every mesh coordinate; tp_axis ({tp_axis}) has nothing left to split"
    assert tp_axis is None or tp_axis != sp_axis, (
        f"tp_axis ({tp_axis}) must differ from sp_axis ({sp_axis}): the same physical axis cannot carry both "
        f"shardings, and dividing by its extent twice would under-allocate the cache"
    )
    tp_factor = mesh_shape[tp_axis] if tp_axis is not None else 1
    stripes = mesh_shape[0] * mesh_shape[1] if full_mesh else mesh_shape[sp_axis] * tp_factor
    # Floor division would silently allocate fewer than seq_len rows in total, so the cache would be smaller
    # than the global capacity it declares and the block-cyclic writes would run off the end of the last stripe.
    assert seq_len % stripes == 0, (
        f"seq_len ({seq_len}) must be divisible by the shard extent ({stripes}); a partial stripe would "
        f"under-allocate the cache"
    )
    seq_len_local = seq_len // stripes

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
    full_mesh_topology = None
    if full_mesh:
        dist_shape = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
        physical_mesh_shape = dist_shape
        coords = [
            ttnn.MeshCoordinate([coord[i] for i in range(coord.dims())])
            for coord in ttnn.MeshCoordinateRange(physical_mesh_shape)
        ]
        full_mesh_topology = ttnn.TensorTopology(dist_shape, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)

    kvpe_cache = ttnn.allocate_tensor_on_device(
        ttnn.Shape([num_users * num_layers, 1, seq_len_local, kvpe_cache_head_dim]),
        dtype,
        layout,
        mesh_device,
        kv_mem_config,
    )
    DRAMZeroFill.op(kvpe_cache)

    if full_mesh:
        # Full-mesh ring mode assigns one canonical row-major sequence shard to every coordinate.
        # DRAMZeroFill is an in-place generic op whose output follows the allocator's default replicated
        # topology, so stamp the intended full-mesh distribution after the fill.
        kvpe_cache.update_tensor_topology(full_mesh_topology)
        return kvpe_cache

    # allocate_tensor_on_device assigns a default 2D fully-replicated topology, but the rest
    # of the model produces replicated tensors via ReplicateTensorToMesh, which is a 1D
    # MeshShape(num_devices) with a single Replicate placement. Reproduce that exactly: a 1D
    # distribution_shape + single Replicate, with mesh_coords being the 2D physical device
    # coordinates (row-major), matching what the ReplicateTensorToMesh mapper emits.
    num_devices = mesh_device.shape[0] * mesh_device.shape[1]
    dist_shape = ttnn.MeshShape([num_devices])
    placements = [ttnn.PlacementReplicate()]
    physical_mesh_shape = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
    coords = [
        ttnn.MeshCoordinate([coord[i] for i in range(coord.dims())])
        for coord in ttnn.MeshCoordinateRange(physical_mesh_shape)
    ]
    kvpe_cache.update_tensor_topology(ttnn.TensorTopology(dist_shape, placements, coords))

    return kvpe_cache


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
    full_mesh=False,
    tp_axis=None,
) -> MlaKvCache:
    """Allocate and zero a persistent MLA cache in the selected physical format.

    Homogeneous formats store the config-derived logical row directly. Scaled FP8 owns one
    ND-sharded mixed-format row per token. Physical DRAM usage is derived from the tensor's
    aligned page size, not the logical row width.

    tp_axis: KV dedup, forwarded to init_kvpe_cache -- see its docstring.
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
        full_mesh=full_mesh,
        tp_axis=tp_axis,
    )
    return MlaKvCache(format=cache_format, storage=storage, geometry=geometry)


def allocate_mla_kvpe_cache(
    *, mesh_device, hf_config, max_seq_len, mesh_shape, sp_axis, num_layers, num_users, full_mesh=False
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
        full_mesh=full_mesh,
    )


def allocate_dflash_kv_cache(
    mesh_device: ttnn.MeshDevice,
    config: DFlashDrafterConfig,
    cache_seq: int,
    *,
    sp_axis: int = 0,
    tp_axis: int = 1,
    num_users: int = 1,
    dtype: ttnn.DataType = ttnn.bfloat8_b,  # align w/ decode KV cache (init_kvpe_cache default is bf8); bf8/TILE
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Allocate the DFlash drafter's separate K and V context caches, owned OUTSIDE the module by the
    caller (prefill runner / test) and passed into ``TtDFlashDrafter.forward`` — the drafter analog
    of ``allocate_mla_kvpe_cache`` above: one file owns each model's KV layout, and the model module only
    consumes the cache handed in (like the MLA model's ``forward(..., kvpe_cache=...)``). Keeping ownership
    with the caller lets it drive cache lifecycle (the migration hand-off to the decode mesh) and dtype
    (default bf8/``bfloat8_b`` to match the decode KV cache; see ``init_kvpe_cache``).

    Global (logical) shape ``[num_users * num_hidden_layers, num_key_value_heads, cache_seq, head_dim]``,
    TP-sharded on kv-head (dim 1) and SP-sharded on seq (dim 2) so each SP chip owns ``cache_seq/sp`` tokens
    (the decode/migration layout, no redundant per-SP copies). Allocated + zeroed on device (DRAMZeroFill)
    with no host tensor / H2D copy. Returns ``(k_cache, v_cache)``.

    ``num_users`` independent cache slots share one cache, laid out **user-major** exactly as
    ``init_kvpe_cache`` documents for the verifier's KVPE cache: ``slot = user_id * num_hidden_layers +
    layer_idx``, so each user's draft layers stay contiguous. This is the linearization
    ``update_padded_kv_cache`` computes on device (``batch_idx = slot_idx * num_layers + layer_idx``), and
    the one ``kv_chunk_table._num_layers_from_cache`` assumes when it derives a cache's layer count as
    ``shape[0] // num_users``. ``num_users=1`` (the default) reproduces the single-slot shape.

    ND-DRAM-sharded with the same spec ``init_kvpe_cache`` uses — ``[1, 1,
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim]``, round-robin over the DRAM-bank grid — because the
    migration address table REQUIRES it: ``populate_kv_chunk_address_table_kimi`` emits one address per
    (bank, 32-token chunk) off ``dram_bank_base_addr``, which describes the buffer only if each 32-token
    chunk is contiguous within a single bank. Under interleaved DRAM that chunk is ``head_dim/32`` bfp8
    tiles striped across as many banks, so every table entry would point at the wrong bytes. The write op
    is layout-agnostic (generic ``TensorAccessor``, and it hashes ``cache.memory_config()``), so this
    changes only the compiled program, not the call."""
    sp = mesh_device.shape[sp_axis]
    tp = mesh_device.shape[tp_axis]
    assert (
        config.num_key_value_heads % tp == 0
    ), f"num_key_value_heads ({config.num_key_value_heads}) must divide across tp ({tp})"
    assert cache_seq % sp == 0, f"cache_seq ({cache_seq}) must divide across sp ({sp})"
    assert num_users >= 1, f"num_users ({num_users}) must be >= 1"
    # update_padded_kv_cache requires the per-chip cache seq to be a whole number of tiles, and a whole
    # number of chunk writes (it FATALs on cache_seq % input_seq != 0), so keep the tile check here where
    # the layout is decided rather than discovering it inside the op.
    assert (cache_seq // sp) % ttnn.TILE_SIZE == 0, (
        f"per-chip cache seq ({cache_seq // sp} = cache_seq {cache_seq} / sp {sp}) must be a multiple of "
        f"TILE_SIZE ({ttnn.TILE_SIZE})"
    )
    # ND shard / migration-table constraints. These are a DIFFERENT requirement from the tile check above
    # (the two constants both happen to be 32), so assert them separately: the address table walks whole
    # NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK-token chunks, and a bfp8 chunk must be a whole number of tiles wide.
    assert (cache_seq // sp) % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0, (
        f"per-chip cache seq ({cache_seq // sp}) must be a multiple of NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK "
        f"({NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}); the migration address table addresses whole bank chunks"
    )
    assert config.head_dim % ttnn.TILE_SIZE == 0, (
        f"head_dim ({config.head_dim}) must be a multiple of {ttnn.TILE_SIZE} so a bank chunk is a whole "
        f"number of tiles (kv_chunk_table._dram_chunk_size_bytes rejects otherwise)"
    )

    # Per-device (post-shard) shape: slot/layer on dim 0 (user-major, NOT sharded), kv-head split across TP
    # (dim 1), seq split across SP (dim 2).
    local_shape = ttnn.Shape(
        [
            num_users * config.num_hidden_layers,
            config.num_key_value_heads // tp,
            cache_seq // sp,
            config.head_dim,
        ]
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

    # ND DRAM shard spec, identical in form to init_kvpe_cache's: one shard is
    # [1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim] -- a single (slot/layer, kv-head, 32-token) block
    # kept contiguous in ONE bank -- round-robined across the DRAM-bank grid. Dim 1 is sharded at extent 1,
    # so the drafter's 2 kv-heads/chip are simply separate shards (unlike zero_padded_kv_cache, nothing here
    # requires a single head).
    num_dram_banks = get_num_dram_banks(mesh_device)
    bank_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(num_dram_banks)]
    )
    kv_mem_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, config.head_dim],
            grid=bank_grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )

    def _alloc_zeroed() -> ttnn.Tensor:
        # Allocate + zero on device (DRAMZeroFill) instead of a host torch.zeros + H2D copy: the drafter
        # cache scales with the sequence and the host pack/transfer of the full cache is slow (mirrors
        # init_kvpe_cache above). The shard topology is stamped after the fill.
        cache = ttnn.allocate_tensor_on_device(local_shape, dtype, ttnn.TILE_LAYOUT, mesh_device, kv_mem_config)
        DRAMZeroFill.op(cache)
        cache.update_tensor_topology(ttnn.TensorTopology(dist_shape, placements, coords))
        return cache

    # K and V are independent caches → two separate on-device zero-fills (the old path shared one host
    # buffer across two from_torch copies).
    return _alloc_zeroed(), _alloc_zeroed()
