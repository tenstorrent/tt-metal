# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The KDA state migration contract: segment geometry, contract layouts, per-layer export/import.

``ttKDA`` reads and writes its state in the *native* form: ``recurrent`` ``[1, H_local, D, D]`` FP32
TILE and ``convolution`` ``[1, K-1, 3*H_local*D]`` BF16 ROW_MAJOR, both interleaved DRAM, heads
sharded across TP and the complete state replicated across SP. The kernels accept nothing else, so
the native carries stay as they are and migration reads a second copy in the *contract* form.

The contract fixes what one migration segment is and requires every segment to be one contiguous
DRAM span, round-robin over the DRAM banks exactly like the kvpe cache:

* recurrent: one ``[D, 32]`` FP32 V-band per (head, band), ``D/32`` tiles = 16384 bytes at D=128;
  ``num_heads * D/32`` segments per layer (384 for Kimi-K3).
* convolution: one ``[K-1, 64]`` BF16 rectangle per (branch, head, half), three 128-byte pages =
  384 bytes; ``3 * num_heads * D/64`` segments per layer (576 for Kimi-K3).

Two physical forms hold identical bytes at identical addresses, so the address table cannot tell
them apart:

* a single-layer ND-sharded tensor whose shard IS the segment (``[1, 1, D, 32]`` / ``[1, K-1, 64]``),
  the form PR #56443 measured and the one ``to_memory_config`` produces from a native tensor;
* the consolidated per-rank slabs the engine owns (``KdaStates``): the recurrent slab keeps the ND
  form over ``[slots * layers, H_local, D, D]`` and is written per layer with ``fill_cache_for_user_``;
  the convolution slab is an INTERLEAVED row-major ``[slots * layers, SEG, (K-1)*64]`` tensor whose
  384-byte pages are the segments (``row0[64c:64c+64] | row1[...] | row2[...]``). Interleaved page
  ``p`` lives at bank ``p % N``, offset ``(p // N) * 384``, which is where ND shard ``s = p`` would
  live, so the bytes and the table math are the same while the write becomes a plain ``slice_write``
  of whole rows (``slice_write`` refuses sharded outputs, and ``fill_cache`` refuses row-major).

Segment numbering (``g`` is the GLOBAL head ``tp_col * H_local + h_local``, ``branch`` is q/k/v =
0/1/2, the golden's ``[all q | all k | all v]`` order):

    recurrent   G = g * bands + band          local shard s = (batch * H_local + h_local) * bands + band
    convolution G = (branch * num_heads + g) * halves + half
                local shard s = batch * shards_per_layer + (branch * H_local + h_local) * halves + half
    bank = s % num_banks ; offset = base + (s // num_banks) * segment_bytes
    owner = TP column g // H_local ; replicas = every SP row of that column

``batch`` is the slab's leading index, ``slot * num_layers + layer_position`` (user-major, the kvpe
cache convention). The per-chip convolution channel order is ``[q_local | k_local | v_local]`` with
``D`` channels per head inside each branch, so the ``64``-wide column slice ``j`` of a chip's row is
``(branch * H_local + h_local) * halves + half`` -- the same expression as the local shard index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState

KDA_RECURRENT_BAND_WIDTH = 32
KDA_CONVOLUTION_HALF_WIDTH = 64
KDA_RECURRENT_DTYPE = ttnn.float32
KDA_CONVOLUTION_DTYPE = ttnn.bfloat16
_RECURRENT_ELEMENT_BYTES = 4
_CONVOLUTION_ELEMENT_BYTES = 2
_TILE = 32
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
        if self.head_dim % KDA_CONVOLUTION_HALF_WIDTH != 0:
            raise ValueError(f"head_dim {self.head_dim} must be a multiple of {KDA_CONVOLUTION_HALF_WIDTH}")

    @classmethod
    def from_kda_config(
        cls, config: KDAConfig, *, mesh_shape: tuple[int, int], sp_axis: int, tp_axis: int
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
        return (1, self.conv_history, self.convolution_width)

    # --- segments ---
    @property
    def bands(self) -> int:
        return self.head_dim // KDA_RECURRENT_BAND_WIDTH

    @property
    def halves(self) -> int:
        return self.head_dim // KDA_CONVOLUTION_HALF_WIDTH

    @property
    def recurrent_segment_bytes(self) -> int:
        return self.head_dim * KDA_RECURRENT_BAND_WIDTH * _RECURRENT_ELEMENT_BYTES

    @property
    def convolution_segment_bytes(self) -> int:
        return self.conv_history * KDA_CONVOLUTION_HALF_WIDTH * _CONVOLUTION_ELEMENT_BYTES

    @property
    def recurrent_segments_per_layer(self) -> int:
        """Unique segments per layer across the whole mesh (SP replicas not counted)."""
        return self.num_heads * self.bands

    @property
    def convolution_segments_per_layer(self) -> int:
        return 3 * self.num_heads * self.halves

    @property
    def recurrent_shards_per_layer(self) -> int:
        """Segments one device holds per layer."""
        return self.local_heads * self.bands

    @property
    def convolution_shards_per_layer(self) -> int:
        return 3 * self.local_heads * self.halves

    @property
    def convolution_slab_row_width(self) -> int:
        """Elements per convolution slab row: one segment, ``conv_history`` history rows of 64."""
        return self.conv_history * KDA_CONVOLUTION_HALF_WIDTH

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

    def recurrent_local_shard(self, batch: int, h_local: int, band: int) -> int:
        return (batch * self.local_heads + h_local) * self.bands + band

    def convolution_local_column(self, branch: int, h_local: int, half: int) -> int:
        """The 64-wide column slice of a chip's convolution row holding (branch, head, half)."""
        return (branch * self.local_heads + h_local) * self.halves + half

    def convolution_local_shard(self, batch: int, branch: int, h_local: int, half: int) -> int:
        return batch * self.convolution_shards_per_layer + self.convolution_local_column(branch, h_local, half)

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
        convolution=nd_dram_memory_config(device, [1, geometry.conv_history, KDA_CONVOLUTION_HALF_WIDTH]),
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


def _free_unless_shared(tensor: ttnn.Tensor, *keep: ttnn.Tensor) -> None:
    """Free an intermediate unless a reshape handed back a view over the same buffer."""
    if all(tensor.buffer_address() != other.buffer_address() for other in keep):
        ttnn.deallocate(tensor)


def convolution_to_slab_rows(convolution: ttnn.Tensor, geometry: KdaContractGeometry) -> ttnn.Tensor:
    """``[1, K-1, W]`` -> ``[1, SEG, (K-1)*64]``: row ``j`` is segment ``j`` in contract byte order."""
    history, width, half = geometry.conv_history, geometry.convolution_width, KDA_CONVOLUTION_HALF_WIDTH
    split = ttnn.reshape(convolution, (1, history, width // half, half))
    swapped = ttnn.permute(split, (0, 2, 1, 3))
    _free_unless_shared(split, convolution, swapped)
    rows = ttnn.reshape(swapped, (1, width // half, history * half))
    _free_unless_shared(swapped, rows)
    return rows


def slab_rows_to_convolution(rows: ttnn.Tensor, geometry: KdaContractGeometry) -> ttnn.Tensor:
    """Inverse of :func:`convolution_to_slab_rows`."""
    history, width, half = geometry.conv_history, geometry.convolution_width, KDA_CONVOLUTION_HALF_WIDTH
    split = ttnn.reshape(rows, (1, width // half, history, half))
    swapped = ttnn.permute(split, (0, 2, 1, 3))
    _free_unless_shared(split, rows, swapped)
    convolution = ttnn.reshape(swapped, (1, history, width))
    _free_unless_shared(swapped, convolution)
    return convolution


def export_recurrent(slab: ttnn.Tensor, recurrent: ttnn.Tensor, batch: int) -> None:
    """Write one layer's native recurrent carry into slab batch ``batch`` (a captured device op)."""
    ttnn.kv_cache.fill_cache_for_user_(slab, recurrent, batch)


def export_convolution(slab: ttnn.Tensor, convolution: ttnn.Tensor, batch: int, geometry: KdaContractGeometry) -> None:
    """Write one layer's native convolution tail into slab batch ``batch`` (captured device ops)."""
    rows = convolution_to_slab_rows(convolution, geometry)
    _, seg, width = rows.shape
    ttnn.experimental.slice_write(rows, slab, [batch, 0, 0], [batch + 1, seg, width], [1, 1, 1])
    ttnn.deallocate(rows)


def import_recurrent(slab: ttnn.Tensor, batch: int, destination: ttnn.Tensor) -> None:
    """Copy slab batch ``batch`` into a native recurrent carry."""
    _, heads, rows, columns = slab.shape
    view = ttnn.slice(slab, [batch, 0, 0, 0], [batch + 1, heads, rows, columns], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy(view, destination)
    ttnn.deallocate(view)


def import_convolution(slab: ttnn.Tensor, batch: int, destination: ttnn.Tensor, geometry: KdaContractGeometry) -> None:
    """Copy slab batch ``batch`` into a native convolution tail."""
    _, seg, width = slab.shape
    rows = ttnn.slice(slab, [batch, 0, 0], [batch + 1, seg, width], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    convolution = slab_rows_to_convolution(rows, geometry)
    ttnn.copy(convolution, destination)
    _free_unless_shared(rows, convolution)
    ttnn.deallocate(convolution)


@dataclass
class KdaStates:
    """The engine-owned contract copy of every KDA layer's state on this rank.

    ``recurrent`` and ``convolution`` are the two slabs described in the module docstring; both are
    single allocations so a ``KvCacheStage`` can name each by one base address. ``layer_ids`` are
    this rank's GLOBAL KDA layer indices in ascending order and fix the layer axis of both slabs.
    """

    recurrent: ttnn.Tensor
    convolution: ttnn.Tensor
    geometry: KdaContractGeometry
    layer_ids: tuple[int, ...]
    num_slots: int

    def __post_init__(self) -> None:
        if not self.layer_ids or list(self.layer_ids) != sorted(set(self.layer_ids)):
            raise ValueError(f"layer_ids must be non-empty, unique and ascending, got {self.layer_ids}")
        if self.num_slots < 1:
            raise ValueError(f"num_slots must be at least 1, got {self.num_slots}")
        expected_recurrent = (self.num_slots * self.num_layers, *self.geometry.recurrent_shape[1:])
        expected_convolution = (
            self.num_slots * self.num_layers,
            self.geometry.convolution_shards_per_layer,
            self.geometry.convolution_slab_row_width,
        )
        if tuple(self.recurrent.shape) != expected_recurrent:
            raise ValueError(f"recurrent slab shape {tuple(self.recurrent.shape)} != {expected_recurrent}")
        if tuple(self.convolution.shape) != expected_convolution:
            raise ValueError(f"convolution slab shape {tuple(self.convolution.shape)} != {expected_convolution}")

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)

    def layer_position(self, layer_idx: int) -> int:
        try:
            return self.layer_ids.index(layer_idx)
        except ValueError:
            raise KeyError(f"layer {layer_idx} is not one of this rank's KDA layers {self.layer_ids}") from None

    def batch_index(self, slot: int, layer_idx: int) -> int:
        if not 0 <= slot < self.num_slots:
            raise ValueError(f"slot {slot} out of range [0, {self.num_slots})")
        return slot * self.num_layers + self.layer_position(layer_idx)

    @classmethod
    def allocate(
        cls,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        geometry: KdaContractGeometry,
        *,
        layer_ids: tuple[int, ...],
        num_slots: int,
    ) -> "KdaStates":
        """Allocate both slabs zeroed. The recurrent slab keeps the ND contract form; the convolution
        slab is the interleaved row form (see the module docstring). Neither is ever reallocated:
        a trace capture bakes their addresses in."""
        from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

        batches = num_slots * len(layer_ids)
        recurrent = ttnn.allocate_tensor_on_device(
            ttnn.Shape([batches, *geometry.recurrent_shape[1:]]),
            KDA_RECURRENT_DTYPE,
            ttnn.TILE_LAYOUT,
            mesh_device,
            nd_dram_memory_config(mesh_device, [1, 1, geometry.head_dim, KDA_RECURRENT_BAND_WIDTH]),
        )
        DRAMZeroFill.op(recurrent)
        convolution = ttnn.zeros(
            (batches, geometry.convolution_shards_per_layer, geometry.convolution_slab_row_width),
            dtype=KDA_CONVOLUTION_DTYPE,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        states = cls(
            recurrent=recurrent,
            convolution=convolution,
            geometry=geometry,
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

    def export_layer(self, state: KdaState, slot: int, layer_idx: int) -> None:
        batch = self.batch_index(slot, layer_idx)
        export_recurrent(self.recurrent, state.recurrent, batch)
        export_convolution(self.convolution, state.convolution, batch, self.geometry)

    def import_layer(self, destination: KdaState, slot: int, layer_idx: int) -> None:
        batch = self.batch_index(slot, layer_idx)
        import_recurrent(self.recurrent, batch, destination.recurrent)
        import_convolution(self.convolution, batch, destination.convolution, self.geometry)

    def deallocate(self) -> None:
        ttnn.deallocate(self.recurrent)
        ttnn.deallocate(self.convolution)


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
        geometry.conv_history, KDA_CONVOLUTION_HALF_WIDTH
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
        start = (branch * geometry.num_heads + head) * geometry.head_dim + half * KDA_CONVOLUTION_HALF_WIDTH
        out[:, start : start + KDA_CONVOLUTION_HALF_WIDTH] = rectangle
    return out
