# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exact-shape configuration for the BGE-M3 encoder-only SDPA experiment.

This module deliberately describes only the retained N300 DP=2 path.  Keeping
this contract narrow makes it possible to mirror the production SDPA program
factory in Python before introducing model-local JIT kernel changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

TILE_HEIGHT = 32
TILE_WIDTH = 32
INACTIVE_CB = 0xFFFFFFFF


@dataclass(frozen=True)
class EncoderSDPAConfig:
    batch: int = 6
    num_q_heads: int = 32
    num_kv_heads: int = 16
    q_seq_len: int = 4096
    kv_seq_len: int = 8192
    head_dim: int = 64
    q_chunk_size: int = 128
    k_chunk_size: int = 2048
    grid_x: int = 8
    grid_y: int = 8
    scale: float = 1.0
    # Opt-in: use the production compute_streaming.hpp pipeline (block-streamed
    # QK/softmax/PV, exp_packthread) instead of the legacy standard compute.
    # Host factory normally gates this OFF for fp32 dest; static audit found no
    # hardware invariant forbidding it for this exact shape. Profiling bet only.
    use_streaming: bool = False
    # Destination-accumulate mode. Default matches the retained parity path
    # (fp32 dest, half-sync => DEST capacity 4 tiles). Branch A/B experiments:
    #  - fp32_dest_acc_en=False => DEST=8 (streaming's intended mode);
    #  - dst_full_sync_en=True + fp32 => DEST=8 (Branch B).
    fp32_dest_acc_en: bool = True
    dst_full_sync_en: bool = False

    @property
    def q_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.num_q_heads, self.q_seq_len, self.head_dim)

    @property
    def kv_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.num_kv_heads, self.kv_seq_len, self.head_dim)

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return self.q_shape


@dataclass(frozen=True)
class EncoderSDPAPlan:
    """Derived constants matching ``SDPAProgramFactory`` for the exact path."""

    config: EncoderSDPAConfig

    @property
    def num_cores(self) -> int:
        return self.config.grid_x * self.config.grid_y

    @property
    def sq_tiles(self) -> int:
        return self.config.q_seq_len // TILE_HEIGHT

    @property
    def sk_tiles(self) -> int:
        return self.config.kv_seq_len // TILE_HEIGHT

    @property
    def head_dim_tiles(self) -> int:
        return self.config.head_dim // TILE_WIDTH

    @property
    def q_chunk_tiles(self) -> int:
        return self.config.q_chunk_size // TILE_HEIGHT

    @property
    def k_chunk_tiles(self) -> int:
        return self.config.k_chunk_size // TILE_HEIGHT

    @property
    def q_num_chunks(self) -> int:
        return self.config.q_seq_len // self.config.q_chunk_size

    @property
    def k_num_chunks(self) -> int:
        return self.config.kv_seq_len // self.config.k_chunk_size

    @property
    def total_q_work(self) -> int:
        return self.config.batch * self.config.num_q_heads * self.q_num_chunks

    @property
    def q_work_per_core(self) -> int:
        if self.total_q_work % self.num_cores != 0:
            raise ValueError("encoder SDPA requires uniform Q work across the fixed grid")
        return self.total_q_work // self.num_cores

    @property
    def heads_per_core(self) -> int:
        if self.q_work_per_core % self.q_num_chunks != 0:
            raise ValueError("a Q head would cross a core boundary; chain metadata is required")
        return self.q_work_per_core // self.q_num_chunks

    # ── Fully plan-derived subblock/CB/granularity values, ported verbatim from
    # sdpa_program_factory.cpp so q_chunk/k_chunk can change without hand-tuning.
    # dst_size derived from dest-accumulate mode (mirror dest_helpers.hpp
    # get_dest_limit): full-sync 8(fp32)/16(fp16); half-sync 4(fp32)/8(fp16).
    @property
    def DST_SIZE(self) -> int:
        fp32 = self.config.fp32_dest_acc_en
        full = self.config.dst_full_sync_en
        if full:
            return 8 if fp32 else 16
        return 4 if fp32 else 8

    @staticmethod
    def _largest_subblock(
        bh: int, bw: int, dst_size: int, max_h: int = 1 << 30, max_w: int = 1 << 30
    ) -> tuple[int, int]:
        # Mirror detail::determine_largest_subblock_size.
        subblocks = [
            (2, 4),
            (4, 2),
            (1, 8),
            (8, 1),
            (1, 7),
            (7, 1),
            (2, 3),
            (3, 2),
            (1, 6),
            (6, 1),
            (1, 5),
            (5, 1),
            (2, 2),
            (1, 4),
            (4, 1),
            (1, 3),
            (3, 1),
            (1, 2),
            (2, 1),
            (1, 1),
        ]
        for sh, sw in subblocks:
            if sh * sw > dst_size:
                continue
            if sh > max_h or sw > max_w:
                continue
            if bh % sh != 0 or bw % sw != 0:
                continue
            return sh, sw
        return 1, 1

    @staticmethod
    def _valid_granularity(tile_count: int, max_granularity: int) -> int:
        # Mirror detail::find_valid_granularity.
        g = min(tile_count, max_granularity)
        while g > 1 and tile_count % g != 0:
            g -= 1
        return g

    @property
    def qk_out_subblock(self) -> tuple[int, int]:
        return self._largest_subblock(self.q_chunk_tiles, self.k_chunk_tiles, self.DST_SIZE)

    @property
    def out_out_subblock(self) -> tuple[int, int]:
        # Streaming caps subblock height at 2 (matches factory's max_h arg).
        max_h = 2 if self.config.use_streaming else (1 << 30)
        return self._largest_subblock(self.q_chunk_tiles, self.head_dim_tiles, self.DST_SIZE, max_h)

    @staticmethod
    def _streaming_qktv_h(subblock_h: int, subblock_w: int, dst_size: int, sq_chunk_t: int) -> int:
        # Mirror sdpa_streaming_qktv.hpp:streaming_qktv_h.
        return 2 if (subblock_h == 1 and 2 * subblock_w <= dst_size and sq_chunk_t >= 2) else subblock_h

    @property
    def streaming_cb_out_tiles(self) -> int:
        # detail::streaming_cb_out_tiles = 2 * qktv_h * vDHt.
        sh, sw = self.out_out_subblock
        qh = self._streaming_qktv_h(sh, sw, self.DST_SIZE, self.q_chunk_tiles)
        return 2 * qh * self.head_dim_tiles

    @property
    def qk_in0_num_subblocks(self) -> int:
        return self.q_chunk_tiles // self.qk_out_subblock[0]

    @property
    def qk_in1_num_subblocks(self) -> int:
        return self.k_chunk_tiles // self.qk_out_subblock[1]

    @property
    def out_in0_num_subblocks(self) -> int:
        return self.q_chunk_tiles // self.out_out_subblock[0]

    @property
    def out_in1_num_subblocks(self) -> int:
        return self.head_dim_tiles // self.out_out_subblock[1]

    @property
    def out_im_tiles(self) -> int:
        # cb_out_im_A/B and cb_out both = Sq_chunk_t * vDHt.
        return self.q_chunk_tiles * self.head_dim_tiles

    @property
    def statistics_tiles(self) -> int:
        # cb_max/sum/exp_max_diff = Sq_chunk_t.
        return self.q_chunk_tiles

    @property
    def stats_granularity(self) -> int:
        return self._valid_granularity(self.q_chunk_tiles, self.DST_SIZE)

    @property
    def sub_exp_granularity(self) -> int:
        return self._valid_granularity(self.k_chunk_tiles, self.DST_SIZE)

    @property
    def mul_bcast_granularity(self) -> int:
        return self._valid_granularity(self.q_chunk_tiles * self.k_chunk_tiles, self.DST_SIZE)

    @property
    def dht_granularity(self) -> int:
        # compute_dht_granularity(DHt, vDHt, dst_size); DHt==vDHt here.
        g = min(self.head_dim_tiles, self.head_dim_tiles, self.DST_SIZE)
        while g > 1 and (self.head_dim_tiles % g != 0):
            g -= 1
        return g

    @property
    def reduce_granularity(self) -> int:
        return self._valid_granularity(self.q_chunk_tiles, self.DST_SIZE // 2)

    def global_q_range(self, core_id: int) -> tuple[int, int]:
        if not 0 <= core_id < self.num_cores:
            raise ValueError(f"invalid core_id={core_id}")
        return core_id * self.q_work_per_core, self.q_work_per_core

    def validate_static_contract(self) -> None:
        c = self.config
        if c.q_seq_len % c.q_chunk_size != 0:
            raise ValueError("q_chunk_size must divide q_seq_len exactly")
        if c.kv_seq_len % c.k_chunk_size != 0:
            raise ValueError("k_chunk_size must divide kv_seq_len exactly")
        if c.head_dim % TILE_WIDTH != 0:
            raise ValueError("head_dim must be tile aligned")
        if c.num_q_heads % c.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        # B6*HQ32*32 chunks / 64 cores = 96 chunks = exactly three heads.
        # Therefore no head spans cores and production KV forwarding chains are
        # inactive for this exact work split.
        if self.heads_per_core != 3:
            raise ValueError(f"expected exactly three Q heads/core, got {self.heads_per_core}")


def _shape_tuple(tensor: ttnn.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.padded_shape)


def validate_encoder_sdpa_inputs(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    config: EncoderSDPAConfig,
) -> EncoderSDPAPlan:
    plan = EncoderSDPAPlan(config)
    plan.validate_static_contract()

    if _shape_tuple(q) != config.q_shape:
        raise ValueError(f"expected Q shape {config.q_shape}, got {_shape_tuple(q)}")
    if _shape_tuple(k) != config.kv_shape:
        raise ValueError(f"expected K shape {config.kv_shape}, got {_shape_tuple(k)}")
    if _shape_tuple(v) != config.kv_shape:
        raise ValueError(f"expected V shape {config.kv_shape}, got {_shape_tuple(v)}")
    if q.layout != ttnn.TILE_LAYOUT or k.layout != ttnn.TILE_LAYOUT or v.layout != ttnn.TILE_LAYOUT:
        raise ValueError("encoder SDPA requires TILE_LAYOUT Q/K/V")
    if q.dtype != ttnn.bfloat8_b:
        raise ValueError(f"expected BF8 Q, got {q.dtype}")
    if k.dtype != ttnn.bfloat4_b:
        raise ValueError(f"expected BF4 K, got {k.dtype}")
    if v.dtype != ttnn.bfloat8_b:
        raise ValueError(f"expected BF8 V, got {v.dtype}")
    if q.device() != k.device() or q.device() != v.device():
        raise ValueError("Q/K/V must be on the same device")

    grid = q.device().compute_with_storage_grid_size()
    if (int(grid.x), int(grid.y)) != (config.grid_x, config.grid_y):
        raise ValueError(f"expected {config.grid_x}x{config.grid_y} compute grid, got {int(grid.x)}x{int(grid.y)}")
    return plan


__all__ = [
    "EncoderSDPAConfig",
    "EncoderSDPAPlan",
    "INACTIVE_CB",
    "TILE_HEIGHT",
    "TILE_WIDTH",
    "validate_encoder_sdpa_inputs",
]
