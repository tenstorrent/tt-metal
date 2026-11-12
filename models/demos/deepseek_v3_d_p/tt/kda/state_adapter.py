# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The KDA state migration contract: segment geometry, contract layouts, per-layer export/import.

``ttKDA`` reads and writes its state in the *native* form: ``recurrent`` ``[1, H_local, D, D]`` FP32
TILE and ``convolution`` ``[1, SEG, (K-1)*64]`` BF16 ROW_MAJOR, channel-paged with 64-channel pages (the layer's
``KDAProgramConfig.convolution_page_channels`` set to ``KDA_CONVOLUTION_PAGE_CHANNELS``), both interleaved DRAM, heads
sharded across TP and the complete state replicated across SP. The kernels accept nothing else, so a layer runs on one
native scratch carry, loaded from and stored back to the *contract* form, which is the carries' only store and what
migration reads. The contract form stores every head once on the mesh: SP row ``r`` of TP column ``c`` stores only the
column's local heads ``[r * H_store, (r + 1) * H_store)``, ``H_store = num_heads / (SP * TP)``. A load copies the
chip's shard out of its slab and all-gathers the shards over SP into the native carry; a store copies each chip's slice
of the (SP-replicated) native carry straight into the slab, the copy selecting the slice on device from the chip's SP
coordinate (``KdaStates.sp_rank``).

The contract fixes what one migration segment is and requires every segment to be one contiguous
DRAM span, round-robin over the DRAM banks exactly like the kvpe cache:

* recurrent: one ``[D, 32]`` FP32 V-band per (head, band), ``D/32`` tiles = 16384 bytes at D=128;
  ``num_heads * D/32`` segments per layer (384 for Kimi-K3).
* convolution: one ``[K-1, 64]`` BF16 rectangle per (branch, head, half), one 384-byte page;
  ``3 * num_heads * D/64`` segments per layer (576 for Kimi-K3).

Two physical forms hold identical bytes at identical addresses, so the address table cannot tell
them apart:

* a single-layer ND-sharded tensor whose shard IS the segment (``[1, 1, D, 32]`` / ``[1, 1, (K-1)*64]``),
  the form PR #56443 measured and the one ``to_memory_config`` produces from a native tensor;
* the consolidated per-chip slabs the engine owns (``KdaStates``): the recurrent slab keeps the ND
  form over ``[slots * layers, H_store, D, D]`` and is written per layer with ``indexed_entry_copy``;
  the convolution slab is an INTERLEAVED row-major ``[slots * layers, 3 * H_store * halves, (K-1)*64]``
  tensor whose 384-byte pages are the segments. Interleaved page ``p`` lives at bank ``p % N``, offset
  ``(p // N) * 384``, which is where ND shard ``s = p`` would live, so the bytes and the table math
  are the same while the write becomes a plain whole-row copy. The native convolution tail has the
  same pages, so a slab batch is a byte copy of the chip's shard of the native tail.

(The single-layer form describes one chip's whole native carry; it is not the slabs' per-chip layout.)

Exports take the slot as a replicated one-element UINT32 device tensor (``device_scalar``, or the traced
metadata's slot word) and write slab batch ``slot * L + layer_position`` with ``indexed_entry_copy``, which
resolves the batch on device. A captured export therefore follows the slot written before each replay,
and an import reads the same batch back into a native carry the same way, so a captured load follows it too.

Segment numbering (``g`` is the GLOBAL head ``tp_col * H_local + h_local``, ``branch`` is q/k/v =
0/1/2, the golden's ``[all q | all k | all v]`` order):

    recurrent   G = g * bands + band
    convolution G = (branch * num_heads + g) * halves + half

The numbering, segment bytes and table positions depend only on the model, never on how the mesh
stores the state. A stored head is addressed as ``(sp_row, j)`` with ``h_local = sp_row * H_store + j``, and its
segments live on the one device ``(sp_row, tp_col)``:

    recurrent   local shard s = (batch * H_store + j) * bands + band
    convolution local shard s = batch * shards_per_layer + (branch * H_store + j) * halves + half
    bank = s % num_banks ; offset = base + (s // num_banks) * segment_bytes

``batch`` is the slab's leading index, ``slot * num_layers + layer_position`` (user-major, the kvpe
cache convention). The per-chip native convolution channel order is ``[q_local | k_local | v_local]``
with ``D`` channels per head inside each branch, so native page ``(branch * H_local + h_local) * halves + half``
holds (branch, head, half). A stored convolution layer keeps the same branch-major order over its own
``H_store`` heads, so viewing native ``[1, SEG, 192]`` as ``[1, 3, H_local * halves, 192]`` (a metadata view) and
gathering the SP rows' ``[1, 3, H_store * halves, 192]`` shards on dim 2 rebuilds the native tail with no permute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tt.kda.config import convolution_tail_shape
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.deepseek_v3_d_p.tt.kda.state_contract import KDA_CONVOLUTION_PAGE_CHANNELS, KDA_RECURRENT_BAND_WIDTH

KDA_RECURRENT_DTYPE = ttnn.float32
KDA_CONVOLUTION_DTYPE = ttnn.bfloat16
_RECURRENT_ELEMENT_BYTES = 4
_CONVOLUTION_ELEMENT_BYTES = 2
_TILE = 32
_CONVOLUTION_BRANCHES = 3  # q, k, v: the branch-major groups of the convolution tail's rows
_FACE = 16


@dataclass(frozen=True)
class KdaContractGeometry:
    """Segment geometry of one model's KDA state on one mesh."""

    num_heads: int
    head_dim: int
    conv_history: int
    sequence_parallel_size: int
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.head_dim <= 0 or self.conv_history <= 0:
            raise ValueError("KDA contract geometry needs positive heads, head_dim and conv_history")
        if self.sequence_parallel_size <= 0 or self.tensor_parallel_size <= 0:
            raise ValueError("KDA contract geometry needs positive SP and TP sizes")
        if self.num_heads % self.tensor_parallel_size != 0:
            raise ValueError(f"{self.num_heads} heads cannot be divided over TP{self.tensor_parallel_size}")
        if self.local_heads % self.sequence_parallel_size != 0:
            raise ValueError(
                f"{self.local_heads} heads per TP column cannot be sharded over SP{self.sequence_parallel_size}"
            )
        # Whole convolution pages per head; this also gives whole recurrent bands.
        if self.head_dim % KDA_CONVOLUTION_PAGE_CHANNELS != 0:
            raise ValueError(f"head_dim {self.head_dim} must be a multiple of {KDA_CONVOLUTION_PAGE_CHANNELS}")

    @classmethod
    def from_kda_config(
        cls,
        config: KDAConfig,
        *,
        mesh_shape: tuple[int, int],
        sp_axis: int,
        tp_axis: int,
    ) -> "KdaContractGeometry":
        """The geometry of a GLOBAL ``KDAConfig`` (``num_heads`` is the model's, not the TP-local count)."""
        if sp_axis == tp_axis or sp_axis not in (0, 1) or tp_axis not in (0, 1):
            raise ValueError(f"KDA requires distinct 2D SP/TP axes, got SP={sp_axis}, TP={tp_axis}")
        if config.head_k_dim != config.head_v_dim:
            raise ValueError("the KDA state contract assumes head_k_dim == head_v_dim")
        return cls(
            num_heads=config.num_heads,
            head_dim=config.head_k_dim,
            conv_history=config.conv_kernel_size - 1,
            sequence_parallel_size=mesh_shape[sp_axis],
            tensor_parallel_size=mesh_shape[tp_axis],
        )

    # --- native shapes ---
    @property
    def local_heads(self) -> int:
        return self.num_heads // self.tensor_parallel_size

    @property
    def recurrent_shape(self) -> tuple[int, int, int, int]:
        return (1, self.local_heads, self.head_dim, self.head_dim)

    @property
    def convolution_width(self) -> int:
        return 3 * self.local_heads * self.head_dim

    @property
    def convolution_shape(self) -> tuple[int, int, int]:
        """The native tail, channel-paged with one segment per page.

        ``TtK3KdaAttention`` rejects a layer built with other ``convolution_page_channels``.
        """
        return convolution_tail_shape(self.conv_history, self.convolution_width, KDA_CONVOLUTION_PAGE_CHANNELS)

    # --- segments ---
    @property
    def bands(self) -> int:
        return self.head_dim // KDA_RECURRENT_BAND_WIDTH

    @property
    def halves(self) -> int:
        return self.head_dim // KDA_CONVOLUTION_PAGE_CHANNELS

    @property
    def recurrent_segment_bytes(self) -> int:
        return self.head_dim * KDA_RECURRENT_BAND_WIDTH * _RECURRENT_ELEMENT_BYTES

    @property
    def convolution_segment_bytes(self) -> int:
        return self.convolution_slab_row_width * _CONVOLUTION_ELEMENT_BYTES

    @property
    def recurrent_segments_per_layer(self) -> int:
        """Unique segments per layer across the whole mesh, however it stores them."""
        return self.num_heads * self.bands

    @property
    def convolution_segments_per_layer(self) -> int:
        return 3 * self.num_heads * self.halves

    @property
    def recurrent_shards_per_layer(self) -> int:
        """Segments one device stores per layer."""
        return self.stored_heads * self.bands

    @property
    def convolution_shards_per_layer(self) -> int:
        return 3 * self.stored_heads * self.halves

    @property
    def convolution_slab_row_width(self) -> int:
        """Elements per convolution slab row: one segment, the native tail's page."""
        return self.convolution_shape[-1]

    # --- storage: each head stored once on the mesh ---
    @property
    def stored_heads(self) -> int:
        """``H_store``: heads one device stores per layer, its SP row's share of the TP column's ``H_local``."""
        return self.local_heads // self.sequence_parallel_size

    @property
    def stored_recurrent_shape(self) -> tuple[int, int, int, int]:
        """One layer of one device's recurrent store."""
        return (1, self.stored_heads, self.head_dim, self.head_dim)

    @property
    def stored_convolution_shape(self) -> tuple[int, int, int]:
        """One layer of one device's convolution store: one segment per row."""
        return (1, self.convolution_shards_per_layer, self.convolution_slab_row_width)

    def stored_head(self, h_local: int) -> tuple[int, int]:
        """``h_local`` -> ``(sp_row, j)``: the SP row of its TP column storing it, and where."""
        self._check(h_local, self.local_heads, "h_local")
        return divmod(h_local, self.stored_heads)

    def local_head(self, sp_row: int, j: int) -> int:
        """Inverse of ``stored_head``."""
        self._check(sp_row, self.sequence_parallel_size, "sp_row")
        self._check(j, self.stored_heads, "j")
        return sp_row * self.stored_heads + j

    # --- segment maps ---
    def recurrent_segment(self, tp_col: int, h_local: int, band: int) -> int:
        self._check(tp_col, self.tensor_parallel_size, "tp_col")
        self._check(h_local, self.local_heads, "h_local")
        self._check(band, self.bands, "band")
        return (tp_col * self.local_heads + h_local) * self.bands + band

    def decompose_recurrent(self, segment: int) -> tuple[int, int, int]:
        self._check(segment, self.recurrent_segments_per_layer, "segment")
        head, band = divmod(segment, self.bands)
        tp_col, h_local = divmod(head, self.local_heads)
        return tp_col, h_local, band

    def convolution_segment(self, branch: int, tp_col: int, h_local: int, half: int) -> int:
        self._check(branch, 3, "branch")
        self._check(tp_col, self.tensor_parallel_size, "tp_col")
        self._check(h_local, self.local_heads, "h_local")
        self._check(half, self.halves, "half")
        return (branch * self.num_heads + tp_col * self.local_heads + h_local) * self.halves + half

    def decompose_convolution(self, segment: int) -> tuple[int, int, int, int]:
        self._check(segment, self.convolution_segments_per_layer, "segment")
        head, half = divmod(segment, self.halves)
        branch, global_head = divmod(head, self.num_heads)
        tp_col, h_local = divmod(global_head, self.local_heads)
        return branch, tp_col, h_local, half

    def recurrent_local_shard(self, batch: int, j: int, band: int) -> int:
        """Shard index of stored head ``j``'s ``band`` in slab batch ``batch``."""
        self._check(j, self.stored_heads, "j")
        self._check(band, self.bands, "band")
        return (batch * self.stored_heads + j) * self.bands + band

    def convolution_native_page(self, branch: int, h_local: int, half: int) -> int:
        """The page (64-channel column slice) of a chip's native convolution tail holding (branch, head, half)."""
        self._check(branch, 3, "branch")
        self._check(h_local, self.local_heads, "h_local")
        self._check(half, self.halves, "half")
        return (branch * self.local_heads + h_local) * self.halves + half

    def convolution_stored_page(self, branch: int, j: int, half: int) -> int:
        """The row of one stored convolution layer holding (branch, stored head ``j``, half)."""
        self._check(branch, 3, "branch")
        self._check(j, self.stored_heads, "j")
        self._check(half, self.halves, "half")
        return (branch * self.stored_heads + j) * self.halves + half

    def convolution_local_shard(self, batch: int, branch: int, j: int, half: int) -> int:
        return batch * self.convolution_shards_per_layer + self.convolution_stored_page(branch, j, half)

    @staticmethod
    def _check(value: int, limit: int, name: str) -> None:
        if not 0 <= value < limit:
            raise ValueError(f"{name}={value} out of range [0, {limit})")


@dataclass(frozen=True)
class KdaContractMemoryConfigs:
    recurrent: ttnn.MemoryConfig
    convolution: ttnn.MemoryConfig


def dram_bank_grid(device: ttnn.Device | ttnn.MeshDevice) -> ttnn.CoreRangeSet:
    num_banks = device.dram_grid_size().x
    if num_banks <= 0:
        raise ValueError("KDA state contract requires at least one DRAM bank")
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0)) for bank in range(num_banks)}
    )


def nd_dram_memory_config(device: ttnn.Device | ttnn.MeshDevice, shard_shape: list[int]) -> ttnn.MemoryConfig:
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=shard_shape,
            grid=dram_bank_grid(device),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def contract_memory_configs(
    device: ttnn.Device | ttnn.MeshDevice, geometry: KdaContractGeometry
) -> KdaContractMemoryConfigs:
    """ND-DRAM layouts whose shards are exactly the migration segments (single-layer tensors)."""
    return KdaContractMemoryConfigs(
        recurrent=nd_dram_memory_config(device, [1, 1, geometry.head_dim, KDA_RECURRENT_BAND_WIDTH]),
        convolution=nd_dram_memory_config(device, [1, 1, geometry.convolution_slab_row_width]),
    )


def validate_native_state(state: KdaState, geometry: KdaContractGeometry) -> None:
    if tuple(state.recurrent.shape) != geometry.recurrent_shape:
        raise ValueError(f"recurrent shape {tuple(state.recurrent.shape)} != {geometry.recurrent_shape}")
    if tuple(state.convolution.shape) != geometry.convolution_shape:
        raise ValueError(f"convolution shape {tuple(state.convolution.shape)} != {geometry.convolution_shape}")
    if state.recurrent.dtype != KDA_RECURRENT_DTYPE or state.recurrent.layout != ttnn.TILE_LAYOUT:
        raise ValueError("native recurrent state must be FP32 tile layout")
    if state.convolution.dtype != KDA_CONVOLUTION_DTYPE or state.convolution.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("native convolution state must be BF16 row-major layout")
    if state.recurrent.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("native recurrent state must use interleaved DRAM")
    if state.convolution.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("native convolution state must use interleaved DRAM")


def validate_contract_state(state: KdaState, geometry: KdaContractGeometry, configs: KdaContractMemoryConfigs) -> None:
    if tuple(state.recurrent.shape) != geometry.recurrent_shape:
        raise ValueError(f"contract recurrent shape {tuple(state.recurrent.shape)} != {geometry.recurrent_shape}")
    if tuple(state.convolution.shape) != geometry.convolution_shape:
        raise ValueError(f"contract convolution shape {tuple(state.convolution.shape)} != {geometry.convolution_shape}")
    if state.recurrent.dtype != KDA_RECURRENT_DTYPE or state.recurrent.layout != ttnn.TILE_LAYOUT:
        raise ValueError("contract recurrent state must be FP32 tile layout")
    if state.convolution.dtype != KDA_CONVOLUTION_DTYPE or state.convolution.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("contract convolution state must be BF16 row-major layout")
    if state.recurrent.memory_config() != configs.recurrent:
        raise ValueError("contract recurrent state has the wrong ND-DRAM layout")
    if state.convolution.memory_config() != configs.convolution:
        raise ValueError("contract convolution state has the wrong ND-DRAM layout")


def allocate_contract_state(device: ttnn.Device | ttnn.MeshDevice, geometry: KdaContractGeometry) -> KdaState:
    configs = contract_memory_configs(device, geometry)
    return KdaState(
        recurrent=ttnn.allocate_tensor_on_device(
            ttnn.Shape(geometry.recurrent_shape), KDA_RECURRENT_DTYPE, ttnn.TILE_LAYOUT, device, configs.recurrent
        ),
        convolution=ttnn.allocate_tensor_on_device(
            ttnn.Shape(geometry.convolution_shape),
            KDA_CONVOLUTION_DTYPE,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            configs.convolution,
        ),
    )


def allocate_native_state(device: ttnn.Device | ttnn.MeshDevice, geometry: KdaContractGeometry) -> KdaState:
    return KdaState(
        recurrent=ttnn.allocate_tensor_on_device(
            ttnn.Shape(geometry.recurrent_shape),
            KDA_RECURRENT_DTYPE,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        convolution=ttnn.allocate_tensor_on_device(
            ttnn.Shape(geometry.convolution_shape),
            KDA_CONVOLUTION_DTYPE,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    )


def relayout(source: ttnn.Tensor, destination: ttnn.Tensor) -> ttnn.Tensor:
    """Whole-tensor conversion between the native and the single-layer contract form, in place."""
    ttnn.to_memory_config(source, destination.memory_config(), output_tensor=destination)
    return destination


def export_state(source: KdaState, destination: KdaState, geometry: KdaContractGeometry) -> KdaState:
    """Native -> single-layer contract tensors."""
    validate_native_state(source, geometry)
    validate_contract_state(destination, geometry, contract_memory_configs(source.recurrent.device(), geometry))
    return KdaState(
        recurrent=relayout(source.recurrent, destination.recurrent),
        convolution=relayout(source.convolution, destination.convolution),
    )


def import_state(source: KdaState, destination: KdaState, geometry: KdaContractGeometry) -> KdaState:
    """Single-layer contract tensors -> native."""
    validate_contract_state(source, geometry, contract_memory_configs(source.recurrent.device(), geometry))
    validate_native_state(destination, geometry)
    return KdaState(
        recurrent=relayout(source.recurrent, destination.recurrent),
        convolution=relayout(source.convolution, destination.convolution),
    )


def deallocate_state(state: KdaState) -> None:
    ttnn.deallocate(state.recurrent)
    ttnn.deallocate(state.convolution)


# --- consolidated slabs and the per-layer adapter ---


def device_scalar(device: ttnn.Device | ttnn.MeshDevice, value: int) -> ttnn.Tensor:
    """A host integer as the replicated one-element UINT32 row-major DRAM tensor the KDA ops and copies read on device."""
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def _copy_entry(
    source: ttnn.Tensor,
    destination: ttnn.Tensor,
    slot: ttnn.Tensor,
    num_layers: int,
    position: int,
    zero_fill_at_zero: ttnn.Tensor | None = None,
    source_part: ttnn.Tensor | None = None,
    source_part_groups: int = 1,
) -> None:
    """Copy one layer's state between a native carry and slab batch ``slot * num_layers + position`` (captured).

    Native recurrent tiles land in the ND slab's segments through logical page ids, as ``fill_cache_for_user_`` put
    them; a native convolution page is one slab segment. When ``zero_fill_at_zero`` reads 0 on device the
    destination is zero-filled and the source is not read; only loads pass it. ``source_part`` (the chip's SP
    coordinate, read on device) selects the chip's heads of a native carry; only stores pass it.
    """
    ttnn.experimental.kda.indexed_entry_copy(
        source,
        destination,
        slot,
        index_scale=num_layers,
        index_offset=position,
        zero_fill_at_zero=zero_fill_at_zero,
        source_part=source_part,
        source_part_groups=source_part_groups,
    )


def _convolution_branches(tail: ttnn.Tensor) -> ttnn.Tensor:
    """``[1, rows, page]`` -> ``[1, 3, rows / 3, page]``: the branch-major tail with branches split off (a view)."""
    _, rows, page = tuple(tail.shape)
    return ttnn.reshape(tail, [1, _CONVOLUTION_BRANCHES, rows // _CONVOLUTION_BRANCHES, page])


@dataclass
class KdaStates:
    """The engine-owned store of every KDA layer's state on this mesh, in the contract form.

    ``recurrent`` and ``convolution`` are the two slabs described in the module docstring; both are
    single allocations so a ``KvCacheStage`` can name each by one base address. Each chip's slabs hold its
    ``H_store`` heads. ``layer_ids`` are this rank's GLOBAL KDA layer indices in ascending order and fix the
    layer axis of both slabs. ``sp_axis`` is the mesh axis the SP rows lie on: the axis of the import's
    all-gather. ``staging`` holds one layer of one chip's stored heads, the gather's input at a fixed address so a
    captured import replays. ``sp_rank`` holds each chip's SP coordinate as a one-element UINT32 device tensor at a
    fixed address; the export's copy reads it on device to select the chip's heads.

    The slabs are never cleared; ``import_layer``'s ``zero_fill_at_zero`` loads zeros instead of a slot.
    """

    recurrent: ttnn.Tensor
    convolution: ttnn.Tensor
    staging: KdaState
    sp_rank: ttnn.Tensor
    geometry: KdaContractGeometry
    sp_axis: int
    layer_ids: tuple[int, ...]
    num_slots: int

    def __post_init__(self) -> None:
        if not self.layer_ids or list(self.layer_ids) != sorted(set(self.layer_ids)):
            raise ValueError(f"layer_ids must be non-empty, unique and ascending, got {self.layer_ids}")
        if self.num_slots < 1:
            raise ValueError(f"num_slots must be at least 1, got {self.num_slots}")
        batches = self.num_slots * self.num_layers
        expected_recurrent = (batches, *self.geometry.stored_recurrent_shape[1:])
        expected_convolution = (batches, *self.geometry.stored_convolution_shape[1:])
        if tuple(self.recurrent.shape) != expected_recurrent:
            raise ValueError(f"recurrent slab shape {tuple(self.recurrent.shape)} != {expected_recurrent}")
        if tuple(self.convolution.shape) != expected_convolution:
            raise ValueError(f"convolution slab shape {tuple(self.convolution.shape)} != {expected_convolution}")
        if tuple(self.staging.recurrent.shape) != self.geometry.stored_recurrent_shape:
            raise ValueError(f"recurrent staging shape {tuple(self.staging.recurrent.shape)} is not one stored layer")
        if tuple(self.staging.convolution.shape) != self.geometry.stored_convolution_shape:
            raise ValueError(
                f"convolution staging shape {tuple(self.staging.convolution.shape)} is not one stored layer"
            )
        mesh_shape = tuple(self.recurrent.device().shape)
        if mesh_shape[self.sp_axis] != self.geometry.sequence_parallel_size:
            raise ValueError(
                f"mesh axis {self.sp_axis} of {mesh_shape} is not the geometry's SP{self.geometry.sequence_parallel_size}"
            )

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)

    def layer_position(self, layer_idx: int) -> int:
        try:
            return self.layer_ids.index(layer_idx)
        except ValueError:
            raise KeyError(f"layer {layer_idx} is not one of this rank's KDA layers {self.layer_ids}") from None

    def batch_index(self, slot: int, layer_idx: int) -> int:
        self._check_slot(slot)
        return slot * self.num_layers + self.layer_position(layer_idx)

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self.num_slots:
            raise ValueError(f"slot {slot} out of range [0, {self.num_slots})")

    @classmethod
    def allocate(
        cls,
        mesh_device: ttnn.MeshDevice,
        geometry: KdaContractGeometry,
        *,
        sp_axis: int,
        layer_ids: tuple[int, ...],
        num_slots: int,
    ) -> "KdaStates":
        """Allocate both slabs, uninitialized, and the staging layer. The recurrent slab keeps the ND contract
        form; the convolution slab is the interleaved row form (see the module docstring). None is ever
        reallocated: a trace capture bakes their addresses in."""
        batches = num_slots * len(layer_ids)
        recurrent = ttnn.allocate_tensor_on_device(
            ttnn.Shape([batches, *geometry.stored_recurrent_shape[1:]]),
            KDA_RECURRENT_DTYPE,
            ttnn.TILE_LAYOUT,
            mesh_device,
            nd_dram_memory_config(mesh_device, [1, 1, geometry.head_dim, KDA_RECURRENT_BAND_WIDTH]),
        )
        convolution = ttnn.allocate_tensor_on_device(
            ttnn.Shape([batches, *geometry.stored_convolution_shape[1:]]),
            KDA_CONVOLUTION_DTYPE,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        staging = KdaState(
            recurrent=ttnn.allocate_tensor_on_device(
                ttnn.Shape(geometry.stored_recurrent_shape),
                KDA_RECURRENT_DTYPE,
                ttnn.TILE_LAYOUT,
                mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            ),
            convolution=ttnn.allocate_tensor_on_device(
                ttnn.Shape(geometry.stored_convolution_shape),
                KDA_CONVOLUTION_DTYPE,
                ttnn.ROW_MAJOR_LAYOUT,
                mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            ),
        )
        mesh_shape = tuple(mesh_device.shape)
        sp_rank = ttnn.from_torch(
            torch.arange(mesh_shape[sp_axis], dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=mesh_shape, dims=(0, None) if sp_axis == 0 else (None, 0)
            ),
        )
        states = cls(
            recurrent=recurrent,
            convolution=convolution,
            staging=staging,
            sp_rank=sp_rank,
            geometry=geometry,
            sp_axis=sp_axis,
            layer_ids=tuple(layer_ids),
            num_slots=num_slots,
        )
        states.validate_physical_layout()
        return states

    def validate_physical_layout(self) -> None:
        """The address table assumes these strides; refuse to be described otherwise."""
        recurrent_page = self.recurrent.buffer_aligned_page_size()
        if self.geometry.recurrent_segment_bytes % recurrent_page != 0:
            raise ValueError(
                f"recurrent slab page {recurrent_page} does not tile a {self.geometry.recurrent_segment_bytes}-byte segment"
            )
        convolution_page = self.convolution.buffer_aligned_page_size()
        if convolution_page != self.geometry.convolution_segment_bytes:
            raise ValueError(
                f"convolution slab page {convolution_page} != segment {self.geometry.convolution_segment_bytes} bytes; "
                "the interleaved row form only matches the contract when one row is one aligned page"
            )

    def export_layer(self, state: KdaState, slot: ttnn.Tensor, layer_idx: int) -> None:
        """Write one layer's native state into its slot's batch; ``slot`` is read on device (see ``device_scalar``).

        ``state`` must be the same on every SP row of a TP column, as the layer's outgoing carry is: each chip
        stores its own heads' slice of it, selected on device by ``sp_rank`` (the whole carry at SP1). The recurrent
        heads are contiguous; the convolution tail's rows are branch-major, so each branch holds every chip's heads.
        """
        position = self.layer_position(layer_idx)
        _copy_entry(state.recurrent, self.recurrent, slot, self.num_layers, position, source_part=self.sp_rank)
        _copy_entry(
            state.convolution,
            self.convolution,
            slot,
            self.num_layers,
            position,
            source_part=self.sp_rank,
            source_part_groups=_CONVOLUTION_BRANCHES,
        )

    def import_layer(
        self,
        destination: KdaState,
        slot: ttnn.Tensor,
        layer_idx: int,
        *,
        zero_fill_at_zero: ttnn.Tensor | None = None,
    ) -> None:
        """Read one layer's state at ``slot`` into a native carry; ``slot`` is read on device (see ``device_scalar``).

        ``zero_fill_at_zero``: device scalar; 0 loads zeros instead of the slot (K3 passes the chunk start).
        """
        position = self.layer_position(layer_idx)
        if self.geometry.sequence_parallel_size == 1:
            # One SP row stores every local head: the shard is the native carry.
            _copy_entry(self.recurrent, destination.recurrent, slot, self.num_layers, position, zero_fill_at_zero)
            _copy_entry(self.convolution, destination.convolution, slot, self.num_layers, position, zero_fill_at_zero)
            return
        staging = self.staging
        _copy_entry(self.recurrent, staging.recurrent, slot, self.num_layers, position, zero_fill_at_zero)
        _copy_entry(self.convolution, staging.convolution, slot, self.num_layers, position, zero_fill_at_zero)
        ttnn.all_gather(staging.recurrent, dim=1, cluster_axis=self.sp_axis, output_tensor=destination.recurrent)
        ttnn.all_gather(
            _convolution_branches(staging.convolution),
            dim=2,
            cluster_axis=self.sp_axis,
            output_tensor=_convolution_branches(destination.convolution),
        )

    def deallocate(self) -> None:
        ttnn.deallocate(self.recurrent)
        ttnn.deallocate(self.convolution)
        ttnn.deallocate(self.staging.recurrent)
        ttnn.deallocate(self.staging.convolution)
        ttnn.deallocate(self.sp_rank)


# --- host-side decoding of raw segments (read-back through the address table) ---


def recurrent_segment_to_torch(raw: bytes, geometry: KdaContractGeometry) -> torch.Tensor:
    """16384 raw bytes of one V-band -> ``[D, 32]`` FP32, undoing the tile face order."""
    if len(raw) != geometry.recurrent_segment_bytes:
        raise ValueError(f"expected {geometry.recurrent_segment_bytes} bytes, got {len(raw)}")
    tiles = geometry.head_dim // _TILE
    values = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    # A 32x32 tile is stored as four 16x16 faces: (0,0), (0,1), (1,0), (1,1), each row-major.
    return values.reshape(tiles, 2, 2, _FACE, _FACE).permute(0, 1, 3, 2, 4).reshape(geometry.head_dim, _TILE)


def convolution_segment_to_torch(raw: bytes, geometry: KdaContractGeometry) -> torch.Tensor:
    """384 raw bytes of one rectangle -> ``[K-1, 64]`` BF16."""
    if len(raw) != geometry.convolution_segment_bytes:
        raise ValueError(f"expected {geometry.convolution_segment_bytes} bytes, got {len(raw)}")
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(
        geometry.conv_history, KDA_CONVOLUTION_PAGE_CHANNELS
    )


def assemble_recurrent(segments: Mapping[int, torch.Tensor], geometry: KdaContractGeometry) -> torch.Tensor:
    """Global segments -> ``[num_heads, D, D]`` in the layer's ``[k, v]`` orientation."""
    out = torch.zeros(geometry.num_heads, geometry.head_dim, geometry.head_dim, dtype=torch.float32)
    for segment, band_values in segments.items():
        tp_col, h_local, band = geometry.decompose_recurrent(segment)
        head = tp_col * geometry.local_heads + h_local
        start = band * KDA_RECURRENT_BAND_WIDTH
        out[head, :, start : start + KDA_RECURRENT_BAND_WIDTH] = band_values
    return out


def assemble_convolution(segments: Mapping[int, torch.Tensor], geometry: KdaContractGeometry) -> torch.Tensor:
    """Global segments -> ``[K-1, 3 * num_heads * D]`` in the golden's ``[all q | all k | all v]`` order."""
    out = torch.zeros(geometry.conv_history, 3 * geometry.num_heads * geometry.head_dim, dtype=torch.bfloat16)
    for segment, rectangle in segments.items():
        branch, tp_col, h_local, half = geometry.decompose_convolution(segment)
        head = tp_col * geometry.local_heads + h_local
        start = (branch * geometry.num_heads + head) * geometry.head_dim + half * KDA_CONVOLUTION_PAGE_CHANNELS
        out[:, start : start + KDA_CONVOLUTION_PAGE_CHANNELS] = rectangle
    return out
