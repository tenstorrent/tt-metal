# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 GDN state as the KV manager moves it: Kimi-K3's KDA Snapshot contract with Qwen3.5's head counts.

Kimi-K3 (``models/demos/deepseek_v3_d_p/tt/kda/KDA_STATE_MIGRATION.md``, tt-metal #63958) publishes each linear-attention
layer's state as two extra configs of the KV chunk address table, tagged ``Snapshot``: the KV manager moves a layer's
whole window whatever range a migration names. This module is the same contract for the 24 GDN layers of
Qwen3.5-9B. The only difference to K3 is the convolution: K3's q, k and v branches all have ``num_heads`` heads,
Qwen3.5's q and k have 16 heads and v has 32, so the convolution segment numbering takes per-branch head counts.

Native form (what the qwen36 GDN layer holds after a prefill, B = 1, chip c of a (1, 4) column):
  recurrent    ``[1, 8, 128, 128]`` FP32 TILE, ``[k, v]`` per head, value heads ``[8c, 8c + 8)``
  convolution  the last K-1 = 3 conv inputs ``[3, 2048]`` BF16 (``conv_states[1..3]``), channels
               ``[q 512 | k 512 | v 1024]`` = q / k heads ``[4c, 4c + 4)``, v heads ``[8c, 8c + 8)``, row r = the
               input at position ``L - 3 + r`` (row 2 newest)

Contract form (engine-owned, ``GdnStates``; one slab per kind so a ``KvCacheStage`` names it by one base address):
  recurrent slab    ``[slots * 24, 8, 128, 128]`` FP32 TILE, ND-sharded DRAM, shard ``[1, 1, 128, 32]`` = one
                    ``[128, 32]`` V-band = 16384 B per segment
  convolution slab  ``[slots * 24, 32, 192]`` BF16 ROW_MAJOR, interleaved DRAM; one row = one 384 B segment =
                    ``[3, 64]`` (history rows 0..2 of 64 channels)
leading index ``slot * 24 + layer_position`` (user-major, as the KV cache).

Segment numbering (``g`` global head, ``h`` local head, ``c`` = TP column):
  recurrent    G = g * 4 + band,   g = c * 8 + h                         (128 per layer)
  convolution  G = (branch_base + g) * 2 + half,  branch_base = 0 / 16 / 32 for q / k / v,
               g = c * heads_local(branch) + h                           (128 per layer: (16 + 16 + 32) * 2)
  local shard  recurrent  s = (batch * 8 + h) * 4 + band
               convolution s = batch * 32 + (local_branch_base + h) * 2 + half, local_branch_base = 0 / 4 / 8
  address      bank = s % num_banks, offset = base + (s // num_banks) * segment_bytes; owner = TP column c.

Snapshot window (one per layer, both kinds): ``W = lcm(128, 128) * 32 = 4096`` positions; recurrent segment i at
``i * 32``, convolution segment i at ``i * 32``; ``chunk_n_tokens = 32``, ``max_sequence_length = W``.

Fold rule: the state is a fold over the prefix. Decode must resume at exactly the prefilled length; per the d-gen
multimodal design (recurrent models), decode computes the last prompt token, so the engine prefills ``L - 1`` tokens and
the published state is the state after position ``L - 2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_TILE = 32
_FACE = 16
RECURRENT_BAND_WIDTH = 32
CONVOLUTION_PAGE_CHANNELS = 64
POSITION_QUANTUM = 32
BRANCHES = ("q", "k", "v")


@dataclass(frozen=True)
class GdnGeometry:
    """Segment geometry of one model's GDN state on one (SP, TP) mesh. Global head counts."""

    num_k_heads: int = 16
    num_v_heads: int = 32
    head_dim: int = 128
    conv_history: int = 3
    sequence_parallel_size: int = 1
    tensor_parallel_size: int = 4

    def __post_init__(self):
        tp = self.tensor_parallel_size
        if self.num_k_heads % tp or self.num_v_heads % tp:
            raise ValueError(f"{self.num_k_heads} / {self.num_v_heads} heads do not divide over TP{tp}")
        if self.head_dim % CONVOLUTION_PAGE_CHANNELS:
            raise ValueError(f"head_dim {self.head_dim} is not whole {CONVOLUTION_PAGE_CHANNELS}-channel pages")

    @classmethod
    def from_args(cls, args) -> "GdnGeometry":
        """From ``Qwen36ModelArgs`` (local counts ``gdn_nk_tp`` / ``gdn_nv_tp`` per chip)."""
        tp = int(args.cluster_shape[1])
        assert int(args.gdn_dk) == int(args.gdn_dv), "the contract assumes head_k_dim == head_v_dim"
        return cls(
            num_k_heads=int(args.gdn_nk_tp) * tp,
            num_v_heads=int(args.gdn_nv_tp) * tp,
            head_dim=int(args.gdn_dk),
            conv_history=int(args.gdn_conv_kernel_size) - 1,
            sequence_parallel_size=int(args.cluster_shape[0]),
            tensor_parallel_size=tp,
        )

    # --- shapes ---
    @property
    def local_k_heads(self) -> int:
        return self.num_k_heads // self.tensor_parallel_size

    @property
    def local_v_heads(self) -> int:
        return self.num_v_heads // self.tensor_parallel_size

    def branch_heads(self, branch: int) -> int:
        return (self.num_k_heads, self.num_k_heads, self.num_v_heads)[branch]

    def branch_local_heads(self, branch: int) -> int:
        return self.branch_heads(branch) // self.tensor_parallel_size

    def branch_base(self, branch: int) -> int:
        return sum(self.branch_heads(b) for b in range(branch))

    def branch_local_base(self, branch: int) -> int:
        return sum(self.branch_local_heads(b) for b in range(branch))

    @property
    def recurrent_shape(self) -> tuple[int, int, int, int]:
        return (1, self.local_v_heads, self.head_dim, self.head_dim)

    @property
    def convolution_channels(self) -> int:
        """Per-chip conv channels: ``[q | k | v]``."""
        return (2 * self.local_k_heads + self.local_v_heads) * self.head_dim

    @property
    def bands(self) -> int:
        return self.head_dim // RECURRENT_BAND_WIDTH

    @property
    def halves(self) -> int:
        return self.head_dim // CONVOLUTION_PAGE_CHANNELS

    @property
    def recurrent_segment_bytes(self) -> int:
        return self.head_dim * RECURRENT_BAND_WIDTH * 4

    @property
    def convolution_slab_row_width(self) -> int:
        return self.conv_history * CONVOLUTION_PAGE_CHANNELS

    @property
    def convolution_segment_bytes(self) -> int:
        return self.convolution_slab_row_width * 2

    @property
    def recurrent_segments_per_layer(self) -> int:
        return self.num_v_heads * self.bands

    @property
    def convolution_segments_per_layer(self) -> int:
        return (2 * self.num_k_heads + self.num_v_heads) * self.halves

    @property
    def recurrent_shards_per_layer(self) -> int:
        return self.local_v_heads * self.bands

    @property
    def convolution_shards_per_layer(self) -> int:
        return self.convolution_channels // CONVOLUTION_PAGE_CHANNELS

    # --- Snapshot window ---
    @property
    def window(self) -> int:
        return math.lcm(self.recurrent_segments_per_layer, self.convolution_segments_per_layer) * POSITION_QUANTUM

    def chunk_n_tokens(self, kind: str) -> int:
        return self.window // self.segments_per_layer(kind)

    def segments_per_layer(self, kind: str) -> int:
        return {"gdn_recurrent": self.recurrent_segments_per_layer, "gdn_convolution": self.convolution_segments_per_layer}[
            kind
        ]

    def shards_per_layer(self, kind: str) -> int:
        return {"gdn_recurrent": self.recurrent_shards_per_layer, "gdn_convolution": self.convolution_shards_per_layer}[
            kind
        ]

    def segment_bytes(self, kind: str) -> int:
        return {"gdn_recurrent": self.recurrent_segment_bytes, "gdn_convolution": self.convolution_segment_bytes}[kind]

    # --- maps ---
    def recurrent_segment(self, tp_col: int, h_local: int, band: int) -> int:
        return (tp_col * self.local_v_heads + h_local) * self.bands + band

    def decompose_recurrent(self, segment: int) -> tuple[int, int, int]:
        head, band = divmod(segment, self.bands)
        tp_col, h_local = divmod(head, self.local_v_heads)
        return tp_col, h_local, band

    def recurrent_local_shard(self, h_local: int, band: int) -> int:
        return h_local * self.bands + band

    def convolution_segment(self, branch: int, tp_col: int, h_local: int, half: int) -> int:
        g = tp_col * self.branch_local_heads(branch) + h_local
        return (self.branch_base(branch) + g) * self.halves + half

    def decompose_convolution(self, segment: int) -> tuple[int, int, int, int]:
        head, half = divmod(segment, self.halves)
        for branch in range(3):
            n = self.branch_heads(branch)
            if head < self.branch_base(branch) + n:
                tp_col, h_local = divmod(head - self.branch_base(branch), self.branch_local_heads(branch))
                return branch, tp_col, h_local, half
        raise ValueError(f"segment {segment} out of range")

    def convolution_local_shard(self, branch: int, h_local: int, half: int) -> int:
        """Also the 64-channel page index of the chip's ``[q | k | v]`` row."""
        return (self.branch_local_base(branch) + h_local) * self.halves + half

    def local_segments(self, kind: str, tp_col: int) -> list[tuple[int, int]]:
        """(global segment, shard within the layer) for every segment chip column ``tp_col`` holds."""
        out = []
        if kind == "gdn_recurrent":
            for h in range(self.local_v_heads):
                for band in range(self.bands):
                    out.append((self.recurrent_segment(tp_col, h, band), self.recurrent_local_shard(h, band)))
        elif kind == "gdn_convolution":
            for branch in range(3):
                for h in range(self.branch_local_heads(branch)):
                    for half in range(self.halves):
                        out.append(
                            (
                                self.convolution_segment(branch, tp_col, h, half),
                                self.convolution_local_shard(branch, h, half),
                            )
                        )
        else:
            raise ValueError(kind)
        return out


# --- host side: native <-> contract bytes, HF layout (tests, read-back) -------------------------------------------


def tile_bytes_fp32(mat):
    """[R, C] fp32 (R, C multiples of 32) -> the bytes of its TILE layout (row-major tiles, four 16x16 faces each)."""
    import torch

    r, c = mat.shape
    t = mat.reshape(r // _TILE, 2, _FACE, c // _TILE, 2, _FACE).permute(0, 3, 1, 4, 2, 5)
    return t.contiguous().to(torch.float32).numpy().tobytes()


def recurrent_segment_to_torch(raw: bytes, geometry: GdnGeometry):
    """One 16384 B V-band -> ``[D, 32]`` fp32 (undoes the tile face order; same as K3's decoder)."""
    import torch

    if len(raw) != geometry.recurrent_segment_bytes:
        raise ValueError(f"expected {geometry.recurrent_segment_bytes} bytes, got {len(raw)}")
    tiles = geometry.head_dim // _TILE
    v = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    return v.reshape(tiles, 2, 2, _FACE, _FACE).permute(0, 1, 3, 2, 4).reshape(geometry.head_dim, _TILE)


def convolution_segment_to_torch(raw: bytes, geometry: GdnGeometry):
    """One 384 B page -> ``[K-1, 64]`` bf16."""
    import torch

    if len(raw) != geometry.convolution_segment_bytes:
        raise ValueError(f"expected {geometry.convolution_segment_bytes} bytes, got {len(raw)}")
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(geometry.conv_history, CONVOLUTION_PAGE_CHANNELS)


def native_convolution_to_slab_rows(rows, geometry: GdnGeometry):
    """Host mirror of the device relayout: native ``[K-1, C]`` (one chip) -> slab rows ``[C / 64, (K-1) * 64]``."""
    k1, c = rows.shape
    assert k1 == geometry.conv_history and c == geometry.convolution_channels, (tuple(rows.shape), geometry)
    return (
        rows.reshape(k1, c // CONVOLUTION_PAGE_CHANNELS, CONVOLUTION_PAGE_CHANNELS)
        .permute(1, 0, 2)
        .reshape(c // CONVOLUTION_PAGE_CHANNELS, k1 * CONVOLUTION_PAGE_CHANNELS)
    )


def assemble_recurrent(segments, geometry: GdnGeometry):
    """{global segment: [D, 32]} -> ``[num_v_heads, D, D]`` fp32 in ``[k, v]`` order (HF ``recurrent_state``)."""
    import torch

    out = torch.zeros(geometry.num_v_heads, geometry.head_dim, geometry.head_dim, dtype=torch.float32)
    for seg, band_values in segments.items():
        c, h, band = geometry.decompose_recurrent(seg)
        out[c * geometry.local_v_heads + h, :, band * RECURRENT_BAND_WIDTH : (band + 1) * RECURRENT_BAND_WIDTH] = band_values
    return out


def assemble_convolution(segments, geometry: GdnGeometry):
    """{global segment: [K-1, 64]} -> ``[K-1, 2 * Hk * D + Hv * D]`` bf16 in HF's ``[all q | all k | all v]`` order."""
    import torch

    width = (2 * geometry.num_k_heads + geometry.num_v_heads) * geometry.head_dim
    out = torch.zeros(geometry.conv_history, width, dtype=torch.bfloat16)
    for seg, rect in segments.items():
        branch, c, h, half = geometry.decompose_convolution(seg)
        g = c * geometry.branch_local_heads(branch) + h
        start = (geometry.branch_base(branch) + g) * geometry.head_dim + half * CONVOLUTION_PAGE_CHANNELS
        out[:, start : start + CONVOLUTION_PAGE_CHANNELS] = rect
    return out


def hf_conv_state(assembled):
    """``[K-1, C]`` -> HF ``conv_state`` ``[C, K]`` (column K-1 newest, column 0 the shifted-out input, not migrated)."""
    import torch

    k1, c = assembled.shape
    return torch.cat([torch.zeros(c, 1, dtype=assembled.dtype), assembled.t()], dim=1)


# --- device side --------------------------------------------------------------------------------------------------


@dataclass
class GdnStates:
    """The engine-owned contract store of this rank's GDN layers (``layer_ids``: global GDN layer indices)."""

    recurrent: object
    convolution: object
    geometry: GdnGeometry
    layer_ids: tuple
    num_slots: int

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)

    def layer_position(self, layer_idx: int) -> int:
        return self.layer_ids.index(layer_idx)

    @classmethod
    def allocate(cls, mesh_device, geometry: GdnGeometry, *, layer_ids, num_slots: int) -> "GdnStates":
        import ttnn

        batches = num_slots * len(layer_ids)
        banks = mesh_device.dram_grid_size().x
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)})
        rec_mc = ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.DRAM,
            nd_shard_spec=ttnn.NdShardSpec(
                shard_shape=[1, 1, geometry.head_dim, RECURRENT_BAND_WIDTH],
                grid=grid,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
            ),
        )
        recurrent = ttnn.allocate_tensor_on_device(
            ttnn.Shape([batches, *geometry.recurrent_shape[1:]]), ttnn.float32, ttnn.TILE_LAYOUT, mesh_device, rec_mc
        )
        convolution = ttnn.allocate_tensor_on_device(
            ttnn.Shape([batches, geometry.convolution_shards_per_layer, geometry.convolution_slab_row_width]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        states = cls(recurrent, convolution, geometry, tuple(layer_ids), int(num_slots))
        page = convolution.buffer_aligned_page_size()
        if page != geometry.convolution_segment_bytes:
            raise ValueError(f"convolution slab page {page} B != segment {geometry.convolution_segment_bytes} B")
        if geometry.recurrent_segment_bytes % recurrent.buffer_aligned_page_size():
            raise ValueError("recurrent slab pages do not tile a segment")
        return states

    def export_layer(self, rec_native, conv_rows_native, slot_tensor, layer_idx: int) -> None:
        """One layer's native state -> slab batch ``slot * L + position`` (slot read on device, capturable).

        ``rec_native``: ``[1, Hv_local, D, D]`` fp32 TILE interleaved DRAM. ``conv_rows_native``: ``[1, K-1, C]`` bf16
        (any layout), the last K-1 conv inputs."""
        import ttnn

        pos = self.layer_position(layer_idx)
        ttnn.experimental.kda.indexed_entry_copy(
            rec_native, self.recurrent, slot_tensor, index_scale=self.num_layers, index_offset=pos
        )
        rows = conv_rows_to_slab_rows(conv_rows_native, self.geometry)
        ttnn.experimental.kda.indexed_entry_copy(
            rows, self.convolution, slot_tensor, index_scale=self.num_layers, index_offset=pos
        )
        ttnn.deallocate(rows)

    def deallocate(self) -> None:
        import ttnn

        ttnn.deallocate(self.recurrent)
        ttnn.deallocate(self.convolution)


def conv_rows_to_slab_rows(conv_rows, geometry: GdnGeometry):
    """Device relayout ``[1, K-1, C]`` -> ``[1, C / 64, (K-1) * 64]`` bf16 row-major interleaved DRAM (the slab row
    form: page j = channels ``[64 j, 64 j + 64)`` of history rows 0..K-2)."""
    import ttnn

    k1, c = geometry.conv_history, geometry.convolution_channels
    x = conv_rows
    if x.layout != ttnn.ROW_MAJOR_LAYOUT:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if x.dtype != ttnn.bfloat16:
        x = ttnn.typecast(x, ttnn.bfloat16)
    x = ttnn.reshape(x, [1, k1, c // CONVOLUTION_PAGE_CHANNELS, CONVOLUTION_PAGE_CHANNELS])
    x = ttnn.permute(x, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(x, [1, c // CONVOLUTION_PAGE_CHANNELS, k1 * CONVOLUTION_PAGE_CHANNELS])
