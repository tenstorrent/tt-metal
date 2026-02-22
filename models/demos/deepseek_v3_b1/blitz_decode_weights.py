# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Blitz decode weight overlapping infrastructure.

"Overlapping" means fusing multiple weight tensors into a single
width-sharded tensor so they share the same L1 base address on each core.
For each core the individual shards are stitched together vertically,
with tile-reshape applied when shard widths differ, producing one
contiguous buffer per core.  Kernels locate each sub-weight at a known
row offset within the fused shard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from loguru import logger

import ttnn


def shuffle_weights_for_interleaved_qnope_qrope(
    weights: torch.Tensor,
    num_qnope_heads: int = 64,
    num_qrope_heads: int = 64,
    qnope_head_dim: int = 128,
    qrope_head_dim: int = 64,
    heads_per_row: int = 8,
) -> torch.Tensor:
    """Shuffle matmul2 weight columns for interleaved Qnope/Qrope output layout.

    The matmul2 output needs to be distributed to a grid where each row has:
    - 8 Qnope cores (1 head per core, 128 elements each)
    - 4 Qrope cores (2 heads per core, 64 elements each = 128 total)

    This function reorders the weight columns so the output is interleaved
    by row groups:
    ``[QNOPE_0:8 | QROPE_0:8 | QNOPE_8:16 | QROPE_8:16 | ...]``

    Args:
        weights: Input weight matrix ``[K, N]`` where
            ``N = num_qnope_heads*qnope_head_dim + num_qrope_heads*qrope_head_dim``.
        num_qnope_heads: Number of Qnope heads (default 64).
        num_qrope_heads: Number of Qrope heads (default 64).
        qnope_head_dim: Dimension per Qnope head (default 128).
        qrope_head_dim: Dimension per Qrope head (default 64).
        heads_per_row: Number of heads per grid row for both Qnope and Qrope
            (default 8).

    Returns:
        Shuffled weight matrix ``[K, N]`` with interleaved column order.
    """
    K = weights.shape[0]
    qnope_total = num_qnope_heads * qnope_head_dim
    qrope_total = num_qrope_heads * qrope_head_dim

    qnope_weights = weights[:, :qnope_total]
    qrope_weights = weights[:, qnope_total : qnope_total + qrope_total]

    qnope_heads = qnope_weights.reshape(K, num_qnope_heads, qnope_head_dim)
    qrope_heads = qrope_weights.reshape(K, num_qrope_heads, qrope_head_dim)

    num_rows = num_qnope_heads // heads_per_row

    shuffled_cols = []
    for row in range(num_rows):
        qnope_start = row * heads_per_row
        qnope_row = qnope_heads[:, qnope_start : qnope_start + heads_per_row, :]
        shuffled_cols.append(qnope_row.reshape(K, -1))

        qrope_start = row * heads_per_row
        qrope_row = qrope_heads[:, qrope_start : qrope_start + heads_per_row, :]
        shuffled_cols.append(qrope_row.reshape(K, -1))

    return torch.cat(shuffled_cols, dim=1)


@dataclass(frozen=True)
class QAB_KVA_PROJ_SingleDeviceOverlapSpec:
    """Configuration for the q_a / q_b / kv_a weight overlap.

    Shape tuples follow (height, width) convention.  All shapes describe
    the per-device layout; TP is inferred from the device topology at
    runtime (single-device -> TP=1, 4x2 mesh -> TP=2).
    """

    # Core range sets for q_ab and kv_a regions
    q_ab_core_range_set: ttnn.CoreRangeSet = field(
        default_factory=lambda: ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7))})
    )
    kv_core_range_set: ttnn.CoreRangeSet = field(
        default_factory=lambda: ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(8, 9))})
    )

    # Head configuration
    num_qnope_heads: int = 64
    num_qrope_heads: int = 64
    qnope_head_dim: int = 128
    qrope_head_dim: int = 64
    heads_per_row: int = 8
    knope_dim: int = 512
    krope_dim: int = 64

    # Raw tensor shapes (height, width)
    q_a_proj_shape: tuple[int, int] = (7168, 1536)
    q_b_proj_shape: tuple[int, int] = (1536, 12288)
    kv_a_proj_shape: tuple[int, int] = (7168, 576)

    tile_h: int = 32
    tile_w: int = 32

    # kv_a_proj shard reorder: places the Knope-rope boundary shard so that
    # the physical core layout matches the logical split expected by the
    # KV cache branch kernel.
    kv_a_proj_shard_order: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 16, 8, 9, 10, 11, 12, 13, 14, 15, 17)

    # --- derived properties ---------------------------------------------------

    @property
    def packed_h(self) -> int:
        return self.q_a_proj_shape[0] // 2

    @property
    def packed_w(self) -> int:
        return self.q_a_proj_shape[1] * 2

    @property
    def q_a_tiles_per_shard(self) -> int:
        shard_w = self.q_a_proj_shape[1] * 2 // self.q_ab_core_range_set.num_cores()
        return (self.packed_h // self.tile_h) * (shard_w // self.tile_w)

    @property
    def q_b_tiles_per_shard(self) -> int:
        shard_w = self.q_b_proj_shape[1] // self.q_ab_core_range_set.num_cores()
        return (self.q_b_proj_shape[0] // self.tile_h) * (shard_w // self.tile_w)

    @property
    def kv_tiles_per_shard(self) -> int:
        shard_w = self.kv_a_proj_shape[1] // self.kv_core_range_set.num_cores()
        return (self.kv_a_proj_shape[0] // self.tile_h) * (shard_w // self.tile_w)

    # --- weight shuffles ------------------------------------------------------

    @staticmethod
    def shuffle_q_a(weights: torch.Tensor) -> torch.Tensor:
        """Pack (H, W) -> (H/2, 2W) by interleaving K-halves."""
        H, W = weights.shape
        return weights.reshape(2, H // 2, W).permute(1, 0, 2).reshape(H // 2, 2 * W)

    def shuffle_q_b(self, weights: torch.Tensor) -> torch.Tensor:
        """Shuffle q_b_proj for interleaved Qnope/Qrope layout."""
        return shuffle_weights_for_interleaved_qnope_qrope(
            weights,
            num_qnope_heads=self.num_qnope_heads,
            num_qrope_heads=self.num_qrope_heads,
            qnope_head_dim=self.qnope_head_dim,
            qrope_head_dim=self.qrope_head_dim,
            heads_per_row=self.heads_per_row,
        )

    def get_q_b_slice(self, q_b_proj_weights: torch.Tensor, tp_idx: int, mla_tp: int) -> torch.Tensor:
        """Extract the per-device q_b_proj slice for a given TP index.

        The full q_b_proj tensor is laid out as ``[all_qnope | all_qrope]``
        across ``mla_tp`` devices.  This method splits qnope and qrope,
        takes the ``tp_idx``-th chunk of each, and stitches them into the
        single-device ``(K, qnope_dim + qrope_dim)`` slice.
        """
        per_tp_qnope_dim = self.num_qnope_heads * self.qnope_head_dim
        per_tp_qrope_dim = self.num_qrope_heads * self.qrope_head_dim
        total_qnope_dim = mla_tp * per_tp_qnope_dim

        full_qnope = q_b_proj_weights[:, :total_qnope_dim]
        full_qrope = q_b_proj_weights[:, total_qnope_dim:]

        tp_qnope = full_qnope[:, tp_idx * per_tp_qnope_dim : (tp_idx + 1) * per_tp_qnope_dim]
        tp_qrope = full_qrope[:, tp_idx * per_tp_qrope_dim : (tp_idx + 1) * per_tp_qrope_dim]

        return torch.cat([tp_qnope, tp_qrope], dim=1)

    def shuffle_kv_a(self, weights: torch.Tensor) -> torch.Tensor:
        """Reorder kv_a_proj shards for the KV cache branch core layout."""
        kv_h, kv_w = weights.shape
        kv_num_cores = self.kv_core_range_set.num_cores()
        shards = weights.reshape(kv_h, kv_num_cores, kv_w // kv_num_cores)
        return shards[:, list(self.kv_a_proj_shard_order), :].reshape(kv_h, kv_w)


QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC = QAB_KVA_PROJ_SingleDeviceOverlapSpec()


@dataclass(frozen=True)
class O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec:
    """Configuration for the o_proj / gate_mm / rmsnorm-gamma weight overlap.

    Fuses the following into a single WIDTH_SHARDED raw buffer:

    * **o_proj** — BFP8, (8192, 7168), WIDTH_SHARDED on 112 cores.
    * **gate_mm** — BFP16, (7168, 256), WIDTH_SHARDED on 8 cores.
    * **attn_norm** — BFP16, (1, 7168), on core (12, 9).
      Attention input norm (attn_norm / mla_norm) — applied to hidden
      state before entering MLA.
    * **q_norm** — BFP16, (1, 1536), on core (12, 9).
      Q latent norm (q_norm / q_a_layernorm) — applied after wq_a,
      before wq_b.
    * **kv_norm** — BFP16, (1, 512), on core (0, 8).
      KV latent norm (kv_norm / kv_a_layernorm) — applied after wkv_a,
      before KV cache.
    * **ffn_norm** — BFP16, (1, 7168), on core (12, 9).
      MoE/MLP input norm (ffn_norm / mlp_norm) — applied to hidden
      state before entering FFN.

    Core (0, 8) is a dedicated core for kv_norm.  Core (12, 9) holds
    three gammas (attn_norm, q_norm, ffn_norm) back-to-back.

    Shape tuples follow (height, width) convention.
    """

    # Core range sets for o_proj and gate_mm regions
    o_proj_core_range_set: ttnn.CoreRangeSet = field(
        default_factory=lambda: ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7)),  # 96 cores
                ttnn.CoreRange(ttnn.CoreCoord(1, 8), ttnn.CoreCoord(8, 9)),  # 16 cores
            }
        )
    )
    gate_mm_core_range_set: ttnn.CoreRangeSet = field(
        default_factory=lambda: ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(12, 0), ttnn.CoreCoord(12, 7))})
    )

    # Gamma core (12, 9): hosts attn_norm, q_norm, ffn_norm
    gamma_core: ttnn.CoreCoord = field(default_factory=lambda: ttnn.CoreCoord(12, 9))

    # Raw tensor shapes (height, width)
    o_proj_shape: tuple[int, int] = (8192, 7168)
    gate_mm_shape: tuple[int, int] = (7168, 256)

    # Gamma shapes — all (1, W) bfloat16 with 1×32 tiles
    attn_norm_shape: tuple[int, int] = (1, 7168)
    q_norm_shape: tuple[int, int] = (1, 1536)
    kv_norm_shape: tuple[int, int] = (1, 512)
    ffn_norm_shape: tuple[int, int] = (1, 7168)

    # kv_norm lives on its own dedicated core (0, 8)
    kv_norm_core: ttnn.CoreCoord = field(default_factory=lambda: ttnn.CoreCoord(0, 8))

    tile_h: int = 32
    tile_w: int = 32
    gamma_tile_h: int = 1
    gamma_tile_w: int = 32
    bfp8_tile_bytes: int = 1088
    bfp16_tile_bytes: int = 2048
    bf16_1x32_tile_bytes: int = 64  # 1 × 32 × 2 bytes

    # --- derived properties ---------------------------------------------------

    @property
    def o_proj_tiles_per_shard(self) -> int:
        shard_w = self.o_proj_shape[1] // self.o_proj_core_range_set.num_cores()
        return (self.o_proj_shape[0] // self.tile_h) * (shard_w // self.tile_w)

    @property
    def gate_mm_tiles_per_shard(self) -> int:
        shard_w = self.gate_mm_shape[1] // self.gate_mm_core_range_set.num_cores()
        return (self.gate_mm_shape[0] // self.tile_h) * (shard_w // self.tile_w)

    @property
    def o_proj_shard_bytes(self) -> int:
        return self.o_proj_tiles_per_shard * self.bfp8_tile_bytes

    @property
    def gate_mm_shard_bytes(self) -> int:
        return self.gate_mm_tiles_per_shard * self.bfp16_tile_bytes

    # Gamma byte sizes (bfloat16: 2 bytes per element)
    @property
    def attn_norm_bytes(self) -> int:
        return self.attn_norm_shape[1] * 2

    @property
    def q_norm_bytes(self) -> int:
        return self.q_norm_shape[1] * 2

    @property
    def kv_norm_bytes(self) -> int:
        return self.kv_norm_shape[1] * 2

    @property
    def ffn_norm_bytes(self) -> int:
        return self.ffn_norm_shape[1] * 2

    @property
    def gamma_core_total_bytes(self) -> int:
        """Total bytes for the three gammas on the gamma core (12, 9)."""
        return self.attn_norm_bytes + self.q_norm_bytes + self.ffn_norm_bytes

    # Byte offsets within each core's shard
    @property
    def attn_norm_byte_offset(self) -> int:
        return 0

    @property
    def q_norm_byte_offset(self) -> int:
        return self.attn_norm_bytes

    @property
    def ffn_norm_byte_offset(self) -> int:
        return self.attn_norm_bytes + self.q_norm_bytes

    @property
    def kv_norm_byte_offset(self) -> int:
        """kv_norm is the only occupant of core (0, 8)."""
        return 0

    @property
    def gamma_core_range_set(self) -> ttnn.CoreRangeSet:
        return ttnn.CoreRangeSet([ttnn.CoreRange(self.gamma_core, self.gamma_core)])

    @property
    def kv_norm_core_range_set(self) -> ttnn.CoreRangeSet:
        return ttnn.CoreRangeSet([ttnn.CoreRange(self.kv_norm_core, self.kv_norm_core)])

    @property
    def max_shard_bytes(self) -> int:
        return max(
            self.o_proj_shard_bytes,
            self.gate_mm_shard_bytes,
            self.gamma_core_total_bytes,
            self.kv_norm_bytes,
        )


O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC = O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec()


@dataclass(frozen=True)
class KVB12_PROJ_SingleDeviceOverlapSpec:
    """Configuration for the kv_b1 / kv_b2 weight overlap.

    Stitches ``kv_b1_proj (8192, 512)`` onto 64 cores and
    ``kv_b2_proj (512, 8192)`` onto another 64 cores.
    8192 is 64 heads x 128 head-dim.

    kv_b1_proj is HEIGHT_SHARDED on the 8x8 Qnope grid (same cores
    that run matmul3 in pre-SDPA).  Each core holds ``(128, 512)``.

    kv_b2_proj ``(512, 8192)`` with shard ``(512, 128)`` is placed on
    the 64 remaining cores of the full 10x13 grid (excluding the
    top-left 8x8 and bottom-right 2x1).

    Both sub-tensors are BFP8.  Shape tuples follow (height, width)
    convention.
    """

    # Core range sets
    kv_b1_core_range_set: ttnn.CoreRangeSet = field(
        default_factory=lambda: ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    )
    kv_b2_core_range_set: ttnn.CoreRangeSet = field(
        default_factory=lambda: ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(12, 7)),  # 5×8 = 40 cores
                ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(11, 9)),  # 12×2 = 24 cores
            }
        )
    )

    kv_b1_proj_shape: tuple[int, int] = (8192, 512)
    kv_b2_proj_shape: tuple[int, int] = (512, 8192)

    tile_h: int = 32
    tile_w: int = 32
    bfp8_tile_bytes: int = 1088

    # --- derived properties ---------------------------------------------------

    @property
    def shard_shape(self) -> tuple[int, int]:
        """HEIGHT_SHARDED shard: (128, 512) -- same for both sub-tensors."""
        num_cores = self.kv_b1_core_range_set.num_cores()
        return (self.kv_b1_proj_shape[0] // num_cores, self.kv_b1_proj_shape[1])

    @property
    def tiles_per_shard(self) -> int:
        sh, sw = self.shard_shape
        return (sh // self.tile_h) * (sw // self.tile_w)

    @property
    def shard_bytes(self) -> int:
        return self.tiles_per_shard * self.bfp8_tile_bytes

    # --- weight shuffles ------------------------------------------------------

    def shuffle_kv_b2(self, weights: torch.Tensor) -> torch.Tensor:
        """Reshape (512, 8192) into (8192, 512) for HEIGHT_SHARDED placement.

        Each (128, 512) shard in the result corresponds to one (512, 128)
        column slice of the input.
        """
        kv_dim, n_heads = self.kv_b2_proj_shape
        n_cores = self.kv_b2_core_range_set.num_cores()
        head_dim = n_heads // n_cores
        return weights.reshape(kv_dim, n_cores, head_dim).permute(1, 2, 0).reshape(-1, kv_dim).contiguous()


KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC = KVB12_PROJ_SingleDeviceOverlapSpec()


@dataclass(frozen=True)
class GATE_UP_PROJ_SingleDeviceOverlapSpec:
    """Configuration for the gate / up projection weight overlap.

    Both tensors share the raw shape ``(K_gate, K_down) = (7168, 256)``
    and are **block-sharded** across 64 cores each: the K dimension is
    split among ``k_parallel=8`` partitions and the N dimension among
    ``n_parallel=8`` partitions, giving 64 shards of ``(896, 32)`` per
    weight.

    Gate weights use the A compute cores and up weights the B compute
    cores of the shared-expert dual-matmul layout on a 13×10 grid::

        Rows 0-3: A=cols{0-3,7-9}, B=cols{4-6,10-12}
        Rows 4-9: A=cols{0-2,7-9}, B=cols{3-6,10-12}
        (col 12, rows 8-9 excluded: idle/sender)

    For the fused tensor the block-sharded weights are stitched and
    HEIGHT_SHARDED: ``(num_cores * shard_h, shard_w) = (57344, 32)``
    so each core gets one ``(896, 32)`` shard.

    Shape tuples follow (height, width) convention.
    """

    # Core range sets — gate (A) and up (B) grids, stored as contiguous column chunks
    gate_core_range_set: ttnn.CoreRangeSet = field(
        default_factory=lambda: ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 9)),  # 30 cores  cols 0-2
                ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 3)),  #  4 cores  col 3 top
                ttnn.CoreRange(ttnn.CoreCoord(7, 0), ttnn.CoreCoord(9, 9)),  # 30 cores  cols 7-9
            }
        )
    )
    up_core_range_set: ttnn.CoreRangeSet = field(
        default_factory=lambda: ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(3, 4), ttnn.CoreCoord(3, 9)),  #  6 cores  col 3 bottom
                ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(6, 9)),  # 30 cores  cols 4-6
                ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(11, 9)),  # 20 cores  cols 10-11
                ttnn.CoreRange(ttnn.CoreCoord(12, 0), ttnn.CoreCoord(12, 7)),  #  8 cores  col 12
            }
        )
    )

    # Raw weight shape
    gate_proj_shape: tuple[int, int] = (7168, 256)
    up_proj_shape: tuple[int, int] = (7168, 256)

    # Block-shard parallelism
    k_parallel: int = 8
    n_parallel: int = 8

    tile_h: int = 32
    tile_w: int = 32
    bfp4_tile_bytes: int = 576

    # --- derived properties ---------------------------------------------------

    @property
    def shard_shape(self) -> tuple[int, int]:
        """Per-core shard: (K_gate / k_parallel, K_down / n_parallel)."""
        return (self.gate_proj_shape[0] // self.k_parallel, self.gate_proj_shape[1] // self.n_parallel)

    @property
    def stacked_shape(self) -> tuple[int, int]:
        """Shape of the stacked (per-sub-tensor) representation."""
        sh, sw = self.shard_shape
        return (self.gate_core_range_set.num_cores() * sh, sw)

    @property
    def tiles_per_shard(self) -> int:
        sh, sw = self.shard_shape
        return (sh // self.tile_h) * (sw // self.tile_w)

    @property
    def shard_bytes(self) -> int:
        return self.tiles_per_shard * self.bfp4_tile_bytes

    @property
    def max_shard_bytes(self) -> int:
        return self.shard_bytes

    @staticmethod
    def _crs_shard_permutation(core_range_set: ttnn.CoreRangeSet) -> tuple[int, ...]:
        """Compute shard permutation from CoreRangeSet enumeration to row-major order.

        HEIGHT_SHARDED places shard *j* on the *j*-th core in CoreRangeSet
        enumeration order.  The kernel (SharedExpertOp) assigns
        ``(k_idx, n_idx)`` to each core based on global row-major order
        (y then x).  This method returns a permutation *P* such that
        ``stacked[j] = block_shards[P[j]]`` places the correct shard on
        each physical core.
        """
        crs_cores = []
        for cr in core_range_set.ranges():
            for y in range(cr.start.y, cr.end.y + 1):
                for x in range(cr.start.x, cr.end.x + 1):
                    crs_cores.append((x, y))
        sorted_cores = sorted(crs_cores, key=lambda c: (c[1], c[0]))
        core_to_sorted_idx = {c: i for i, c in enumerate(sorted_cores)}
        return tuple(core_to_sorted_idx[c] for c in crs_cores)

    def reshuffle_block_to_height_sharded(
        self, weights: torch.Tensor, core_range_set: ttnn.CoreRangeSet
    ) -> torch.Tensor:
        """Reorder a (K, N) weight matrix into stacked HEIGHT_SHARDED form.

        The block decomposition splits ``(K, N)`` into
        ``(k_parallel, n_parallel)`` shards of shape ``(sh, sw)``.
        The shards are then permuted so that HEIGHT_SHARDED placement on
        ``core_range_set`` puts each shard on the physical core expected
        by the compute kernel (which assigns shards in global row-major
        core order).

        Returns:
            Tensor of shape :attr:`stacked_shape`.
        """
        sh, sw = self.shard_shape
        num_shards = self.k_parallel * self.n_parallel
        block_shards = (
            weights.reshape(self.k_parallel, sh, self.n_parallel, sw).permute(0, 2, 1, 3).reshape(num_shards, sh, sw)
        )
        perm = self._crs_shard_permutation(core_range_set)
        return block_shards[list(perm)].reshape(-1, sw).contiguous()


GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC = GATE_UP_PROJ_SingleDeviceOverlapSpec()


@dataclass(frozen=True)
class DOWN_PROJ_SingleDeviceSpec:
    """Configuration for the down projection weight layout.

    Down-proj weights are WIDTH_SHARDED on 112 matmul cores from the
    13x10 device grid, excluding 8 DRAM workers, 9 phantom cores
    (col 12, rows 0-8), and 1 mcast/gather core (12,9).

    Per-device layout::

        down_proj (256, 7168) WIDTH_SHARDED on 112 cores
          shard (256, 64)
    """

    DRAM_WORKER_POSITIONS: frozenset = frozenset({(0, 0), (0, 3), (0, 7), (0, 9), (7, 1), (7, 4), (7, 6), (7, 9)})
    GRID_X: int = 13
    GRID_Y: int = 10
    NUM_MATMUL_CORES: int = 112

    def build_matmul_core_grid(self) -> ttnn.CoreRangeSet:
        """Build CoreRangeSet for the 112 matmul cores."""
        excluded = self.DRAM_WORKER_POSITIONS | {(12, row) for row in range(self.GRID_Y)}
        all_cores = [
            (col, row) for row in range(self.GRID_Y) for col in range(self.GRID_X) if (col, row) not in excluded
        ]
        assert len(all_cores) == self.NUM_MATMUL_CORES
        core_ranges = []
        for row in range(self.GRID_Y):
            row_cores = sorted(c for c, r in all_cores if r == row)
            if not row_cores:
                continue
            seg_start = prev_col = row_cores[0]
            for col in row_cores[1:]:
                if col != prev_col + 1:
                    core_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(seg_start, row), ttnn.CoreCoord(prev_col, row)))
                    seg_start = col
                prev_col = col
            core_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(seg_start, row), ttnn.CoreCoord(prev_col, row)))
        return ttnn.CoreRangeSet(core_ranges)


DOWN_PROJ_SINGLE_DEVICE_SPEC = DOWN_PROJ_SingleDeviceSpec()


@dataclass
class OverlappedTensor:
    """A logical view of a sub-tensor within a fused (overlapped) device buffer.

    The fused tensor is a raw byte container whose own tensor properties
    (dtype, layout, shard spec) are generally meaningless for the individual
    sub-tensors.  This class carries the intended per-sub-tensor properties
    alongside a shared reference to the underlying fused buffer.
    """

    fused_tensor: ttnn.Tensor
    tensor_shape: tuple[int, int]
    shard_shape: tuple[int, int]
    core_range_set: ttnn.CoreRangeSet
    dtype: ttnn.DataType
    tile_shape: tuple[int, int]
    byte_offset: int = 0


class BlitzDecodeWeights:
    """Fuses weight tensors to share the same L1 base address per core.

    Methods take raw torch weight tensors, apply any required preprocessing
    (packing, shuffling), stitch per-core shards, and return the result as
    a device-resident ttnn.Tensor with WIDTH_SHARDED placement.

    Args:
        device: The ttnn device (or MeshDevice) to place tensors on.
    """

    def __init__(self, device) -> None:
        self._device = device

        num_devices = device.get_num_devices()
        if num_devices == 1:
            self.mla_tp = 1
            self.moe_tp = 1
        else:
            mesh_shape = (device.shape[0], device.shape[1])
            assert mesh_shape == (
                4,
                2,
            ), f"Only single-device or 4x2 mesh supported, got {mesh_shape[0]}x{mesh_shape[1]}"
            self.mla_tp = 2
            self.moe_tp = 8

    # ------------------------------------------------------------------
    # MLA weight loading
    # ------------------------------------------------------------------

    def get_tt_q_ab_proj_and_kv_a_proj_weights(
        self,
        q_a_proj_weights: torch.Tensor,
        q_b_proj_weights: torch.Tensor,
        kv_a_proj_weights: torch.Tensor,
    ) -> list[OverlappedTensor]:
        """Fuse q_a_proj, q_b_proj, and kv_a_proj into one WIDTH_SHARDED tensor.

        The fused tensor spans two core regions that share the same shard
        width, giving every core a single base address:

        * **Top region** (8x12 = 96 cores): q_a_proj packed + q_b_proj
          shuffled, stitched per core.
        * **Bottom region** (2x9 = 18 cores at offset (8,0)): kv_a_proj
          with shard reordering, zero-padded to the same shard height as
          the top region.

        For multi-device (4x2 mesh, TP=2) the q_b_proj weights span all
        TP devices (width ``mla_tp * per_device_width``).  Per-TP slices are
        shuffled independently and stitched with the (replicated)
        q_a_proj into separate per-TP fused tensors, which are
        concatenated along width and distributed via
        ``ShardTensor2dMesh`` across mesh columns.  q_a_proj and
        kv_a_proj are replicated on every device.

        MLA TP is inferred from the device topology: single-device ->
        mla_tp=1, 4x2 mesh -> mla_tp=2.

        Args:
            q_a_proj_weights: Raw q_a_proj tensor, shape ``(7168, 1536)``.
            q_b_proj_weights: Raw (unshuffled) q_b_proj tensor, shape
                ``(1536, 12288 * mla_tp)``.
            kv_a_proj_weights: Raw kv_a_proj tensor, shape ``(7168, 576)``.

        Returns:
            A list of three :class:`OverlappedTensor` views
            ``[q_a_proj, q_b_proj, kv_a_proj]`` that share the same
            underlying fused device buffer.
        """
        cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
        mla_tp = self.mla_tp

        # -- Validate device grid ----------------------------------------
        device_grid = self._device.compute_with_storage_grid_size()
        q_ab_bb = cfg.q_ab_core_range_set.bounding_box()
        kv_bb = cfg.kv_core_range_set.bounding_box()
        required_rows = max(q_ab_bb.end.y, kv_bb.end.y) + 1
        required_cols = max(q_ab_bb.end.x, kv_bb.end.x) + 1
        assert device_grid.y >= required_rows, f"Device grid needs at least {required_rows} rows, got {device_grid.y}"
        assert device_grid.x >= required_cols, f"Device grid needs at least {required_cols} cols, got {device_grid.x}"

        # -- Validate raw input shapes ----------------------------------
        assert (
            q_a_proj_weights.shape == cfg.q_a_proj_shape
        ), f"q_a_proj_weights must be {cfg.q_a_proj_shape}, got {tuple(q_a_proj_weights.shape)}"
        expected_q_b_shape = (cfg.q_b_proj_shape[0], cfg.q_b_proj_shape[1] * mla_tp)
        assert (
            tuple(q_b_proj_weights.shape) == expected_q_b_shape
        ), f"q_b_proj_weights must be {expected_q_b_shape}, got {tuple(q_b_proj_weights.shape)}"
        assert (
            kv_a_proj_weights.shape == cfg.kv_a_proj_shape
        ), f"kv_a_proj_weights must be {cfg.kv_a_proj_shape}, got {tuple(kv_a_proj_weights.shape)}"

        # -- Step 1: pack q_a_proj  (H, W) -> (H/2, 2W) ----------------
        packed = cfg.shuffle_q_a(q_a_proj_weights)

        # -- Step 2: reorder kv_a_proj shards ---------------------------
        kv_reordered = cfg.shuffle_kv_a(kv_a_proj_weights)
        kv_h, kv_w = kv_a_proj_weights.shape
        kv_num_cores = cfg.kv_core_range_set.num_cores()
        q_ab_num_cores = cfg.q_ab_core_range_set.num_cores()

        # -- Step 3: build per-TP fused tensors -------------------------
        per_tp_combined = []
        for tp_idx in range(mla_tp):
            q_b_slice = cfg.get_q_b_slice(q_b_proj_weights, tp_idx, mla_tp)
            shuffled = cfg.shuffle_q_b(q_b_slice)

            q_ab_fused, q_ab_shard_shape = BlitzDecodeWeights._stitch_width_sharded(packed, shuffled, q_ab_num_cores)
            fused_shard_h, target_w = q_ab_shard_shape

            kv_shard_w = kv_w // kv_num_cores
            assert (
                kv_shard_w == target_w
            ), f"kv_a_proj shard width ({kv_shard_w}) must equal q_ab fused shard width ({target_w})"

            kv_padded = torch.zeros(fused_shard_h, kv_w, dtype=kv_a_proj_weights.dtype)
            kv_padded[:kv_h, :] = kv_reordered

            total_cores = q_ab_num_cores + kv_num_cores
            combined_tp = torch.cat([q_ab_fused, kv_padded], dim=1)
            assert combined_tp.shape == (fused_shard_h, target_w * total_cores)
            per_tp_combined.append(combined_tp)

        combined = torch.cat(per_tp_combined, dim=1) if mla_tp > 1 else per_tp_combined[0]

        # -- Step 4: place on device as WIDTH_SHARDED -------------------
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({q_ab_bb, kv_bb}),
            (fused_shard_h, target_w),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        if mla_tp == 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self._device)
        else:
            mesh_shape = (self._device.shape[0], self._device.shape[1])
            mesh_mapper = ttnn.ShardTensor2dMesh(self._device, mesh_shape=mesh_shape, dims=(None, 1))

        fused = ttnn.from_torch(
            combined,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )

        # -- Build OverlappedTensor views --------------------------------
        tile = fused.get_tile()
        ts = tuple(tile.tile_shape)
        tile_bytes = tile.get_tile_size(ttnn.bfloat8_b)

        q_a_shard_w = cfg.packed_w // q_ab_num_cores
        q_b_shard_w = cfg.q_b_proj_shape[1] // q_ab_num_cores
        kv_shard_w = kv_w // kv_num_cores

        return [
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=(cfg.packed_h, cfg.packed_w),
                shard_shape=(cfg.packed_h, q_a_shard_w),
                core_range_set=cfg.q_ab_core_range_set,
                dtype=ttnn.bfloat8_b,
                tile_shape=ts,
                byte_offset=0,
            ),
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=cfg.q_b_proj_shape,
                shard_shape=(cfg.q_b_proj_shape[0], q_b_shard_w),
                core_range_set=cfg.q_ab_core_range_set,
                dtype=ttnn.bfloat8_b,
                tile_shape=ts,
                byte_offset=cfg.q_a_tiles_per_shard * tile_bytes,
            ),
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=cfg.kv_a_proj_shape,
                shard_shape=(cfg.kv_a_proj_shape[0], kv_shard_w),
                core_range_set=cfg.kv_core_range_set,
                dtype=ttnn.bfloat8_b,
                tile_shape=ts,
                byte_offset=0,
            ),
        ]

    def get_tt_o_proj_and_gate_mm_weights(
        self,
        o_proj_weights: torch.Tensor,
        gate_mm_weights: torch.Tensor,
        attn_norm: torch.Tensor,
        q_norm: torch.Tensor,
        kv_norm: torch.Tensor,
        ffn_norm: torch.Tensor,
    ) -> list[OverlappedTensor]:
        """Fuse o_proj, gate_mm, and 4 RMSNorm gammas into one WIDTH_SHARDED tensor.

        The fused buffer is a UINT32 raw-byte container where each core's
        shard is zero-padded to the same maximum byte size.  The six
        sub-tensors use three distinct formats:

        * **o_proj** — BFP8 (32×32 tiles) on 112 cores.
        * **gate_mm** — BFP16 (32×32 tiles) on 8 cores.
        * **attn_norm, q_norm, ffn_norm** — BFP16
          (1×32 tiles) back-to-back on core (12, 9).
        * **kv_norm** — BFP16 (1×32 tiles) on dedicated core (0, 8).

        Layout::

            -- o_proj region: 112 cores --
            o_proj (8192, 7168) as bfloat8_b
              shard (8192, 64) = 512 tiles × 1088 B = 557 056 B

            -- gate_mm region: 8 cores (col 12, rows 0-7) --
            gate_mm (7168, 256) as bfloat16
              shard (7168, 32) = 224 tiles × 2048 B = 458 752 B

            -- gamma core: 1 core (12, 9) --
            attn_norm (1, 7168)   @ offset 0      → 14 336 B
            q_norm (1, 1536)  @ offset 14 336  →  3 072 B
            ffn_norm (1, 7168) @ offset 17 408 → 14 336 B

            -- kv_norm core: 1 core (0, 8) --
            kv_norm (1, 512) @ offset 0 → 1 024 B

            combined: 122 cores, shard = max_shard_bytes

        Args:
            o_proj_weights:     Raw o_proj tensor, shape
                ``(8192 * mla_tp, 7168)``.  TP-sharded on the inner dim.
            gate_mm_weights:    Raw gate_mm tensor, shape (7168, 256).
                Replicated across TP devices.
            attn_norm:      Pre-SDPA attention-input RMSNorm gamma, shape (1, 7168).
            q_norm:     Pre-SDPA post-matmul1 RMSNorm gamma, shape (1, 1536).
            kv_norm:  Pre-SDPA KV-cache-branch RMSNorm gamma, shape (1, 512).
            ffn_norm:  MoE pre-MLP RMSNorm gamma, shape (1, 7168).

        Returns:
            A list of six :class:`OverlappedTensor` views
            ``[o_proj, gate_mm, attn_norm, q_norm,
            kv_norm, ffn_norm]`` sharing the same
            underlying fused device buffer.
        """
        cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC
        mla_tp = self.mla_tp

        # -- Validate shapes --------------------------------------------
        expected_o_proj_shape = (cfg.o_proj_shape[0] * mla_tp, cfg.o_proj_shape[1])
        assert (
            tuple(o_proj_weights.shape) == expected_o_proj_shape
        ), f"o_proj must be {expected_o_proj_shape}, got {tuple(o_proj_weights.shape)}"
        assert (
            gate_mm_weights.shape == cfg.gate_mm_shape
        ), f"gate_mm must be {cfg.gate_mm_shape}, got {tuple(gate_mm_weights.shape)}"
        assert (
            tuple(attn_norm.shape) == cfg.attn_norm_shape
        ), f"attn_norm must be {cfg.attn_norm_shape}, got {tuple(attn_norm.shape)}"
        assert tuple(q_norm.shape) == cfg.q_norm_shape, f"q_norm must be {cfg.q_norm_shape}, got {tuple(q_norm.shape)}"
        assert (
            tuple(kv_norm.shape) == cfg.kv_norm_shape
        ), f"kv_norm must be {cfg.kv_norm_shape}, got {tuple(kv_norm.shape)}"
        assert (
            tuple(ffn_norm.shape) == cfg.ffn_norm_shape
        ), f"ffn_norm must be {cfg.ffn_norm_shape}, got {tuple(ffn_norm.shape)}"

        o_num_cores = cfg.o_proj_core_range_set.num_cores()
        g_num_cores = cfg.gate_mm_core_range_set.num_cores()
        o_shard_w = cfg.o_proj_shape[1] // o_num_cores
        g_shard_w = cfg.gate_mm_shape[1] // g_num_cores
        max_shard_bytes = cfg.max_shard_bytes
        assert max_shard_bytes % 4 == 0, "shard bytes must be UINT32-aligned"

        # -- Pack shared portion (gate_mm + gammas, replicated) ----------
        shared_packed = bytearray()

        # gate_mm shards (bfloat16) — 8 cores
        for i in range(g_num_cores):
            shard_data = gate_mm_weights[:, i * g_shard_w : (i + 1) * g_shard_w].contiguous()
            shard_raw = BlitzDecodeWeights._tilize_and_pack_bfloat16(shard_data, cfg.tile_h, cfg.tile_w)
            assert len(shard_raw) == cfg.gate_mm_shard_bytes
            shared_packed.extend(shard_raw)
            shared_packed.extend(b"\x00" * (max_shard_bytes - cfg.gate_mm_shard_bytes))

        # Gamma core (12, 9) — attn_norm + q_norm + ffn_norm
        gamma_shard = bytearray(max_shard_bytes)
        offset = 0
        for gamma_tensor, expected_bytes in [
            (attn_norm, cfg.attn_norm_bytes),
            (q_norm, cfg.q_norm_bytes),
            (ffn_norm, cfg.ffn_norm_bytes),
        ]:
            raw = BlitzDecodeWeights._pack_bfloat16_1x32(gamma_tensor)
            assert len(raw) == expected_bytes
            gamma_shard[offset : offset + len(raw)] = raw
            offset += len(raw)
        shared_packed.extend(gamma_shard)

        # kv_norm core (0, 8) — dedicated core
        kv_norm_shard = bytearray(max_shard_bytes)
        kv_norm_raw = BlitzDecodeWeights._pack_bfloat16_1x32(kv_norm)
        assert len(kv_norm_raw) == cfg.kv_norm_bytes
        kv_norm_shard[: len(kv_norm_raw)] = kv_norm_raw
        shared_packed.extend(kv_norm_shard)

        # -- Pack per-TP o_proj shards and combine -----------------------
        total_cores = o_num_cores + g_num_cores + 2  # +1 gamma core, +1 kv_norm core
        uint32_per_shard = max_shard_bytes // 4
        per_device_o_h = cfg.o_proj_shape[0]

        per_tp_raw = []
        for tp_idx in range(mla_tp):
            o_proj_slice = o_proj_weights[tp_idx * per_device_o_h : (tp_idx + 1) * per_device_o_h, :]
            o_packed = bytearray()
            for i in range(o_num_cores):
                shard_data = o_proj_slice[:, i * o_shard_w : (i + 1) * o_shard_w].contiguous()
                shard_raw = BlitzDecodeWeights._tilize_and_pack_bfp8(shard_data, cfg.tile_h, cfg.tile_w)
                assert len(shard_raw) == cfg.o_proj_shard_bytes
                o_packed.extend(shard_raw)
                o_packed.extend(b"\x00" * (max_shard_bytes - cfg.o_proj_shard_bytes))
            per_tp_raw.append(torch.frombuffer(bytes(o_packed + shared_packed), dtype=torch.int32).clone())

        # -- Build UINT32 tensor on device ------------------------------
        if mla_tp == 1:
            combined = per_tp_raw[0].reshape(1, uint32_per_shard * total_cores)
        else:
            combined = torch.cat([t.reshape(1, -1) for t in per_tp_raw], dim=1)

        combined_crs = ttnn.CoreRangeSet(
            list(cfg.o_proj_core_range_set.ranges())
            + list(cfg.gate_mm_core_range_set.ranges())
            + list(cfg.gamma_core_range_set.ranges())
            + list(cfg.kv_norm_core_range_set.ranges())
        )
        shard_spec = ttnn.ShardSpec(
            combined_crs,
            (1, uint32_per_shard),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        if mla_tp == 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self._device)
        else:
            mesh_shape = (self._device.shape[0], self._device.shape[1])
            mesh_mapper = ttnn.ShardTensor2dMesh(self._device, mesh_shape=mesh_shape, dims=(None, 1))

        fused = ttnn.from_torch(
            combined,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self._device,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )

        # -- Build OverlappedTensor views --------------------------------
        tile_32x32 = (cfg.tile_h, cfg.tile_w)
        tile_1x32 = (cfg.gamma_tile_h, cfg.gamma_tile_w)

        return [
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=cfg.o_proj_shape,
                shard_shape=(cfg.o_proj_shape[0], o_shard_w),
                core_range_set=cfg.o_proj_core_range_set,
                dtype=ttnn.bfloat8_b,
                tile_shape=tile_32x32,
                byte_offset=0,
            ),
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=cfg.gate_mm_shape,
                shard_shape=(cfg.gate_mm_shape[0], g_shard_w),
                core_range_set=cfg.gate_mm_core_range_set,
                dtype=ttnn.bfloat16,
                tile_shape=tile_32x32,
                byte_offset=0,
            ),
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=cfg.attn_norm_shape,
                shard_shape=cfg.attn_norm_shape,
                core_range_set=cfg.gamma_core_range_set,
                dtype=ttnn.bfloat16,
                tile_shape=tile_1x32,
                byte_offset=cfg.attn_norm_byte_offset,
            ),
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=cfg.q_norm_shape,
                shard_shape=cfg.q_norm_shape,
                core_range_set=cfg.gamma_core_range_set,
                dtype=ttnn.bfloat16,
                tile_shape=tile_1x32,
                byte_offset=cfg.q_norm_byte_offset,
            ),
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=cfg.kv_norm_shape,
                shard_shape=cfg.kv_norm_shape,
                core_range_set=cfg.kv_norm_core_range_set,
                dtype=ttnn.bfloat16,
                tile_shape=tile_1x32,
                byte_offset=cfg.kv_norm_byte_offset,
            ),
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=cfg.ffn_norm_shape,
                shard_shape=cfg.ffn_norm_shape,
                core_range_set=cfg.gamma_core_range_set,
                dtype=ttnn.bfloat16,
                tile_shape=tile_1x32,
                byte_offset=cfg.ffn_norm_byte_offset,
            ),
        ]

    def get_tt_kv_b12_proj_weights(
        self,
        kv_b1_proj_weights: torch.Tensor,
        kv_b2_proj_weights: torch.Tensor,
    ) -> list[OverlappedTensor]:
        """Fuse kv_b1_proj and kv_b2_proj into one HEIGHT_SHARDED tensor.

        Stitches ``kv_b1_proj (8192, 512)`` onto 64 cores and
        ``kv_b2_proj (512, 8192)`` onto another 64 cores into a
        single HEIGHT_SHARDED buffer with shard ``(128, 512)``.

        Layout::

            -- kv_b1 region: 64 Qnope cores (8x8) --
            kv_b1_proj (8192, 512) as bfloat8_b, shard (128, 512)

            -- kv_b2 region: 64 remaining cores (5x8 + 12x2) --
            kv_b2_proj (512, 8192) as bfloat8_b, shard (512, 128)

            combined: 128 cores, HEIGHT_SHARDED, shard (128, 512)

        Args:
            kv_b1_proj_weights: shape ``(8192 * mla_tp, 512)``.
                TP-sharded on the heads dim.
            kv_b2_proj_weights: shape ``(512, 8192 * mla_tp)``.
                TP-sharded on the heads dim.

        Returns:
            ``[kv_b1_proj, kv_b2_proj]`` as :class:`OverlappedTensor`
            views sharing the same fused device buffer.
        """
        cfg = KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
        mla_tp = self.mla_tp

        expected_b1_shape = (cfg.kv_b1_proj_shape[0] * mla_tp, cfg.kv_b1_proj_shape[1])
        assert (
            tuple(kv_b1_proj_weights.shape) == expected_b1_shape
        ), f"kv_b1 expected {expected_b1_shape}, got {tuple(kv_b1_proj_weights.shape)}"
        expected_b2_shape = (cfg.kv_b2_proj_shape[0], cfg.kv_b2_proj_shape[1] * mla_tp)
        assert (
            tuple(kv_b2_proj_weights.shape) == expected_b2_shape
        ), f"kv_b2 expected {expected_b2_shape}, got {tuple(kv_b2_proj_weights.shape)}"

        per_device_b1_h = cfg.kv_b1_proj_shape[0]
        per_device_b2_w = cfg.kv_b2_proj_shape[1]

        per_tp_combined = []
        for tp_idx in range(mla_tp):
            b1_slice = kv_b1_proj_weights[tp_idx * per_device_b1_h : (tp_idx + 1) * per_device_b1_h, :]
            b2_slice = kv_b2_proj_weights[:, tp_idx * per_device_b2_w : (tp_idx + 1) * per_device_b2_w]
            b2_physical = cfg.shuffle_kv_b2(b2_slice)
            per_tp_combined.append(torch.cat([b1_slice, b2_physical], dim=0))

        combined = torch.cat(per_tp_combined, dim=0) if mla_tp > 1 else per_tp_combined[0]

        # -- Place on device as HEIGHT_SHARDED --------------------------
        combined_crs = ttnn.CoreRangeSet(
            list(cfg.kv_b1_core_range_set.ranges()) + list(cfg.kv_b2_core_range_set.ranges())
        )
        sh, sw = cfg.shard_shape
        shard_spec = ttnn.ShardSpec(
            combined_crs,
            (sh, sw),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        if mla_tp == 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self._device)
        else:
            mesh_shape = (self._device.shape[0], self._device.shape[1])
            mesh_mapper = ttnn.ShardTensor2dMesh(self._device, mesh_shape=mesh_shape, dims=(None, 0))

        fused = ttnn.from_torch(
            combined,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )

        # -- Build OverlappedTensor views --------------------------------
        tile = fused.get_tile()
        tile_shape = tuple(tile.tile_shape)
        num_b2_cores = cfg.kv_b2_core_range_set.num_cores()
        kv_b2_shard = (cfg.kv_b2_proj_shape[0], cfg.kv_b2_proj_shape[1] // num_b2_cores)

        return [
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=cfg.kv_b1_proj_shape,
                shard_shape=cfg.shard_shape,
                core_range_set=cfg.kv_b1_core_range_set,
                dtype=ttnn.bfloat8_b,
                tile_shape=tile_shape,
                byte_offset=0,
            ),
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=cfg.kv_b2_proj_shape,
                shard_shape=kv_b2_shard,
                core_range_set=cfg.kv_b2_core_range_set,
                dtype=ttnn.bfloat8_b,
                tile_shape=tile_shape,
                byte_offset=0,
            ),
        ]

    # ------------------------------------------------------------------
    # MOE weight loading
    # ------------------------------------------------------------------

    def get_tt_moe_shared_expert_weights(
        self,
        gate_proj_weights: torch.Tensor,
        up_proj_weights: torch.Tensor,
        down_proj_weights: torch.Tensor,
    ) -> tuple[OverlappedTensor, OverlappedTensor, ttnn.Tensor]:
        """Create all shared-expert weight tensors in one call.

        **Gate / Up projections** are block-sharded and fused into a single
        HEIGHT_SHARDED L1 tensor.  Gate weights live on the A compute cores,
        up weights on the B compute cores.  Both are BFP4 with shard shape
        ``(896, 32)`` across 64 cores each (128 total).

        **Down projection** is WIDTH_SHARDED on 112 matmul cores as BFP4.

        With ``moe_tp > 1`` the outer (N) dimension of gate/up and the inner
        (K) dimension of down are TP-sharded across devices.

        Per-device layout::

            -- gate region: 64 A cores (non-rectangular) --
            gate_proj (7168, 256) as bfloat4_b, block-sharded
              stacked (57344, 32), shard (896, 32)

            -- up region: 64 B cores (non-rectangular) --
            up_proj (7168, 256) as bfloat4_b, block-sharded
              stacked (57344, 32), shard (896, 32)

            gate+up combined: 128 cores, HEIGHT_SHARDED (114688, 32)

            down_proj (256, 7168) as bfloat4_b, WIDTH_SHARDED on 112 cores
              shard (256, 64)

        Args:
            gate_proj_weights: Raw gate tensor, shape
                ``(7168, 256 * moe_tp)``.  TP-sharded on the outer dim.
            up_proj_weights: Raw up tensor, shape
                ``(7168, 256 * moe_tp)``.  TP-sharded on the outer dim.
            down_proj_weights: Raw down_proj tensor, shape
                ``(256 * moe_tp, 7168)``.  TP-sharded on the inner dim.

        Returns:
            ``(gate_proj, up_proj, down_proj)`` where the first two are
            :class:`OverlappedTensor` views sharing a fused buffer and the
            third is a standalone ``ttnn.Tensor``.
        """
        moe_tp = self.moe_tp

        # ==================================================================
        # Gate + Up (fused HEIGHT_SHARDED in L1)
        # ==================================================================
        cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC

        expected_gate_shape = (cfg.gate_proj_shape[0], cfg.gate_proj_shape[1] * moe_tp)
        assert (
            tuple(gate_proj_weights.shape) == expected_gate_shape
        ), f"gate_proj must be {expected_gate_shape}, got {tuple(gate_proj_weights.shape)}"
        expected_up_shape = (cfg.up_proj_shape[0], cfg.up_proj_shape[1] * moe_tp)
        assert (
            tuple(up_proj_weights.shape) == expected_up_shape
        ), f"up_proj must be {expected_up_shape}, got {tuple(up_proj_weights.shape)}"

        per_device_n = cfg.gate_proj_shape[1]
        gu_per_tp = []
        for tp_idx in range(moe_tp):
            gate_slice = gate_proj_weights[:, tp_idx * per_device_n : (tp_idx + 1) * per_device_n]
            up_slice = up_proj_weights[:, tp_idx * per_device_n : (tp_idx + 1) * per_device_n]
            gate_stacked = cfg.reshuffle_block_to_height_sharded(gate_slice, cfg.gate_core_range_set)
            up_stacked = cfg.reshuffle_block_to_height_sharded(up_slice, cfg.up_core_range_set)
            gu_per_tp.append(torch.cat([gate_stacked, up_stacked], dim=0))

        combined_crs = ttnn.CoreRangeSet(list(cfg.gate_core_range_set.ranges()) + list(cfg.up_core_range_set.ranges()))
        sh, sw = cfg.shard_shape
        gu_shard_spec = ttnn.ShardSpec(combined_crs, (sh, sw), ttnn.ShardOrientation.ROW_MAJOR)
        gu_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gu_shard_spec)

        if moe_tp == 1:
            gu_combined = gu_per_tp[0]
            gu_mapper = ttnn.ReplicateTensorToMesh(self._device)
        else:
            mesh_rows = self._device.shape[0]
            mesh_cols = self._device.shape[1]
            rows = []
            for r in range(mesh_rows):
                rows.append(torch.cat(gu_per_tp[r * mesh_cols : (r + 1) * mesh_cols], dim=1))
            gu_combined = torch.cat(rows, dim=0)
            gu_mapper = ttnn.ShardTensor2dMesh(self._device, mesh_shape=(mesh_rows, mesh_cols), dims=(0, 1))

        fused = ttnn.from_torch(
            gu_combined,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            memory_config=gu_mem,
            mesh_mapper=gu_mapper,
        )

        tile = fused.get_tile()
        ts = tuple(tile.tile_shape)
        gate_ov = OverlappedTensor(
            fused_tensor=fused,
            tensor_shape=cfg.gate_proj_shape,
            shard_shape=cfg.shard_shape,
            core_range_set=cfg.gate_core_range_set,
            dtype=ttnn.bfloat4_b,
            tile_shape=ts,
            byte_offset=0,
        )
        up_ov = OverlappedTensor(
            fused_tensor=fused,
            tensor_shape=cfg.up_proj_shape,
            shard_shape=cfg.shard_shape,
            core_range_set=cfg.up_core_range_set,
            dtype=ttnn.bfloat4_b,
            tile_shape=ts,
            byte_offset=0,
        )

        # ==================================================================
        # Down (WIDTH_SHARDED in L1 on 112 matmul cores)
        # ==================================================================
        dp_spec = DOWN_PROJ_SINGLE_DEVICE_SPEC
        K_down_per_device = 256
        N_per_core = 64
        N_down = N_per_core * dp_spec.NUM_MATMUL_CORES  # 7168

        expected_down_shape = (K_down_per_device * moe_tp, N_down)
        assert (
            tuple(down_proj_weights.shape) == expected_down_shape
        ), f"down_proj_weights must be {expected_down_shape}, got {tuple(down_proj_weights.shape)}"

        matmul_core_grid = dp_spec.build_matmul_core_grid()

        if moe_tp == 1:
            dp_combined = down_proj_weights
            dp_mapper = ttnn.ReplicateTensorToMesh(self._device)
        else:
            mesh_rows = self._device.shape[0]
            mesh_cols = self._device.shape[1]
            dp_combined = (
                down_proj_weights.reshape(mesh_rows, mesh_cols, K_down_per_device, N_down)
                .permute(0, 2, 1, 3)
                .reshape(mesh_rows * K_down_per_device, mesh_cols * N_down)
            )
            dp_mapper = ttnn.ShardTensor2dMesh(self._device, mesh_shape=(mesh_rows, mesh_cols), dims=(0, 1))

        dp_shard_spec = ttnn.ShardSpec(
            matmul_core_grid, (K_down_per_device, N_per_core), ttnn.ShardOrientation.ROW_MAJOR
        )
        dp_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, dp_shard_spec)

        down_tensor = ttnn.from_torch(
            dp_combined,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            memory_config=dp_mem,
            tile=ttnn.Tile([32, 32]),
            mesh_mapper=dp_mapper,
        )

        return gate_ov, up_ov, down_tensor

    def get_tt_moe_routed_expert_weights(
        self,
        gate_proj_weights: torch.Tensor,
        up_proj_weights: torch.Tensor,
        down_proj_weights: torch.Tensor,
    ) -> tuple[list[ttnn.Tensor], list[ttnn.Tensor], list[ttnn.Tensor]]:
        """Create DRAM WIDTH_SHARDED expert weight tensors for routed MoE.

        Each expert projection is uploaded as a separate WIDTH_SHARDED tensor
        across all DRAM banks as ``bfloat4_b``.  Tiles within each bank's
        shard are reordered from row-major to column-major so that K tiles
        stream contiguously.

        Weights are replicated across all devices in a multi-device mesh.

        Per-expert device layout::

            gate_proj_i  (K, N_padded)  WIDTH_SHARDED in DRAM
            up_proj_i    (K, N_padded)  WIDTH_SHARDED in DRAM
            down_proj_i  (K_down, N_down_padded)  WIDTH_SHARDED in DRAM

        Args:
            gate_proj_weights: Stacked gate expert weights, shape
                ``(num_experts, K, N)``.
            up_proj_weights: Stacked up expert weights, shape
                ``(num_experts, K, N)``.
            down_proj_weights: Stacked down expert weights, shape
                ``(num_experts, K_down, N_down)``.

        Returns:
            ``(gate_expert_tensors, up_expert_tensors, down_expert_tensors)``
            — three lists of device-resident ttnn.Tensors, one per expert.
            The first tensor in each list can be used as the base address for
            the op; all tensors must be kept alive to prevent deallocation.
        """
        device = self._device
        tile_w = 32
        num_banks = device.dram_grid_size().x
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)

        def upload(expert_weights: torch.Tensor) -> list[ttnn.Tensor]:
            num_experts, K, N = expert_weights.shape
            N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
            per_core_N = N_padded // num_banks

            dram_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                    )
                }
            )
            shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
            mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

            tensors = []
            for i in range(num_experts):
                w = expert_weights[i]
                if N_padded != N:
                    w = torch.nn.functional.pad(w, (0, N_padded - N))

                w_shuffled = self._shuffle_dram_tiles(w.unsqueeze(0), tile_w, num_banks)
                w_shuffled = w_shuffled.reshape(1, 1, K, N_padded)

                tensors.append(
                    ttnn.from_torch(
                        w_shuffled.contiguous(),
                        dtype=ttnn.bfloat4_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=mem_config,
                        mesh_mapper=mesh_mapper,
                    )
                )
                if (i + 1) % 32 == 0:
                    logger.info(f"  Uploaded {i + 1}/{num_experts} experts")
            return tensors

        return upload(gate_proj_weights), upload(up_proj_weights), upload(down_proj_weights)

    # ------------------------------------------------------------------
    # MLP weight loading
    # ------------------------------------------------------------------

    def get_tt_mlp_shared_expert_weights(
        self,
        gate_proj_weights: torch.Tensor,
        up_proj_weights: torch.Tensor,
        down_proj_weights: torch.Tensor,
    ) -> tuple[OverlappedTensor, OverlappedTensor, ttnn.Tensor | None]:
        """Create MLP shared-expert weights (SRAM) from full projection tensors.

        The full MLP projections contain ``9`` experts of width
        2048.  The first 2048 columns (gate/up) or rows (down) form the
        shared expert, which is TP-sharded across devices and placed in L1
        identically to the MoE shared expert.

        Args:
            gate_proj_weights: Full gate tensor ``(7168, 18432)``.
            up_proj_weights: Full up tensor ``(7168, 18432)``.
            down_proj_weights: Full down tensor ``(18432, 7168)``.

        Returns:
            Same as :meth:`get_tt_moe_shared_expert_weights`.
        """
        shared_n = 2048
        return self.get_tt_moe_shared_expert_weights(
            gate_proj_weights[:, :shared_n],
            up_proj_weights[:, :shared_n],
            down_proj_weights[:shared_n, :],
        )

    def get_tt_mlp_routed_expert_weights(
        self,
        gate_proj_weights: torch.Tensor,
        up_proj_weights: torch.Tensor,
        down_proj_weights: torch.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Create MLP per-device routed expert weights (DRAM).

        After the shared expert (first 2048), the remaining ``8 * 2048``
        columns (gate/up) or rows (down) are split into 8 routed experts,
        one per device.  Each expert is WIDTH_SHARDED in DRAM across all
        banks on its assigned device.

        Per-device layout::

            gate_proj  (7168, N_padded)  WIDTH_SHARDED in DRAM
            up_proj    (7168, N_padded)  WIDTH_SHARDED in DRAM
            down_proj  (2048, N_padded)  WIDTH_SHARDED in DRAM

        Args:
            gate_proj_weights: Full gate tensor ``(7168, 18432)``.
            up_proj_weights: Full up tensor ``(7168, 18432)``.
            down_proj_weights: Full down tensor ``(18432, 7168)``.

        Returns:
            ``(gate_tensor, up_tensor, down_tensor)`` — one device-resident
            ``ttnn.Tensor`` per projection, each containing the single
            routed expert assigned to that device.
        """
        shared_n = 2048
        num_routed = 8
        expert_n = 2048

        K_gate = gate_proj_weights.shape[0]  # 7168
        N_down = down_proj_weights.shape[1]  # 7168

        gate_experts = (
            gate_proj_weights[:, shared_n:].reshape(K_gate, num_routed, expert_n).permute(1, 0, 2).contiguous()
        )  # (8, 7168, 2048)
        up_experts = (
            up_proj_weights[:, shared_n:].reshape(K_gate, num_routed, expert_n).permute(1, 0, 2).contiguous()
        )  # (8, 7168, 2048)
        down_experts = (
            down_proj_weights[shared_n:, :].reshape(num_routed, expert_n, N_down).contiguous()
        )  # (8, 2048, 7168)

        device = self._device
        tile_w = 32
        num_banks = device.dram_grid_size().x
        mesh_rows = device.shape[0]
        mesh_cols = device.shape[1]

        def upload(experts: torch.Tensor) -> ttnn.Tensor:
            n_exp, K, N = experts.shape
            N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
            per_core_N = N_padded // num_banks

            processed = []
            for i in range(n_exp):
                w = experts[i]
                if N_padded != N:
                    w = torch.nn.functional.pad(w, (0, N_padded - N))
                w_shuffled = self._shuffle_dram_tiles(w.unsqueeze(0), tile_w, num_banks)
                processed.append(w_shuffled.reshape(K, N_padded))

            stacked = torch.stack(processed).reshape(mesh_rows, mesh_cols, K, N_padded)

            dram_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                    )
                }
            )
            shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
            mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

            return ttnn.from_torch(
                stacked.contiguous(),
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=mem_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=(mesh_rows, mesh_cols), dims=(0, 1)),
            )

        return upload(gate_experts), upload(up_experts), upload(down_experts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shuffle_dram_tiles(tensor: torch.Tensor, tile_size: int, num_banks: int) -> torch.Tensor:
        """Reorder tiles within each DRAM bank shard from row-major to column-major.

        WIDTH_SHARDED DRAM layout stores tiles row-major, but the streaming
        matmul kernel expects K tiles contiguous for each N column.  This
        function transposes the tile order within each shard so that the
        kernel can linearly read K tiles at a time.

        Args:
            tensor: ``[*, K, N]`` tensor (supports batch dimensions).
            tile_size: Tile dimension (square tiles assumed).
            num_banks: Number of DRAM banks (shards).

        Returns:
            Same-shape tensor with tiles rearranged per shard.
        """
        orig_shape = tensor.shape
        K, N = orig_shape[-2], orig_shape[-1]

        lcm = tile_size * num_banks
        n_padded = ((N + lcm - 1) // lcm) * lcm
        needs_padding = n_padded != N

        tensor = tensor.reshape(-1, K, N)
        batch_size = tensor.shape[0]

        if needs_padding:
            tensor = torch.nn.functional.pad(tensor, (0, n_padded - N))

        K_tiles = K // tile_size
        per_N = n_padded // num_banks
        per_N_tiles = per_N // tile_size
        num_tiles_per_shard = K_tiles * per_N_tiles

        tensor = tensor.reshape(batch_size, K, num_banks, per_N)
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        shards = tensor.reshape(-1, K, per_N)

        tiles = shards.reshape(-1, K_tiles, tile_size, per_N_tiles, tile_size)
        tiles = tiles.permute(0, 1, 3, 2, 4).contiguous()
        tiles = tiles.reshape(-1, num_tiles_per_shard, tile_size, tile_size)

        i = torch.arange(num_tiles_per_shard, device=tensor.device)
        source_idx = (i % K_tiles) * per_N_tiles + (i // K_tiles)
        shuffled_tiles = tiles[:, source_idx, :, :]

        shuffled_tiles = shuffled_tiles.reshape(-1, K_tiles, per_N_tiles, tile_size, tile_size)
        shuffled_tiles = shuffled_tiles.permute(0, 1, 3, 2, 4).contiguous()
        shuffled_shards = shuffled_tiles.reshape(-1, K, per_N)

        shuffled = shuffled_shards.reshape(batch_size, num_banks, K, per_N)
        shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
        shuffled = shuffled.reshape(batch_size, K, n_padded)

        if needs_padding:
            shuffled = shuffled[:, :, :N]

        return shuffled.reshape(*orig_shape)

    @staticmethod
    def _tilize_and_pack_bfp8(data_2d: torch.Tensor, tile_h: int = 32, tile_w: int = 32) -> bytes:
        """Tilize a 2-D tensor and pack as BFP8_b raw bytes.

        Produces the exact byte layout the hardware expects:
        ``[16 exponent uint32 words][256 mantissa uint32 words]`` per tile,
        with tiles in row-major order across the tensor.

        Matches the C++ ``pack_as_bfp_tiles<Bfp8_b>`` with
        ``row_major_input=false, is_exp_a=false``:

        * Shared exponent = max float32 exponent in each 16-element block.
        * Per-element mantissa = 24-bit explicit mantissa (hidden-1 + 23
          fractional bits), right-shifted by the exponent delta *and* by
          17 (``24 - 7``) to yield 7 bits, with round-to-nearest-even.
        * Packed byte = ``sign(1) | mantissa(7)``, zeroed when mantissa
          rounds to 0.

        Tile layout: 4 faces (16x16) in order
        face0 (rows 0-15, cols 0-15), face1 (rows 0-15, cols 16-31),
        face2 (rows 16-31, cols 0-15), face3 (rows 16-31, cols 16-31).
        Each face row of 16 elements forms one BFP8 block.
        """
        H, W = data_2d.shape
        face_h, face_w = tile_h // 2, tile_w // 2
        tr, tc = H // tile_h, W // tile_w
        num_tiles = tr * tc

        data_np = data_2d.contiguous().float().numpy()

        # Reshape into tile grid -> (tr, tc, tile_h, tile_w)
        tiles = data_np.reshape(tr, tile_h, tc, tile_w).transpose(0, 2, 1, 3)
        tiles = tiles.reshape(num_tiles, tile_h, tile_w)

        # Extract 4 faces per tile -> face-ordered (N, 1024)
        face_ordered = np.concatenate(
            [
                tiles[:, :face_h, :face_w].reshape(num_tiles, -1),
                tiles[:, :face_h, face_w:].reshape(num_tiles, -1),
                tiles[:, face_h:, :face_w].reshape(num_tiles, -1),
                tiles[:, face_h:, face_w:].reshape(num_tiles, -1),
            ],
            axis=1,
        )

        # Reshape into BFP8 blocks: (N, 64 blocks, 16 elements)
        blocks = face_ordered.reshape(num_tiles, 64, 16)

        # --- float32 field extraction (vectorised) ---
        float_bits = blocks.view(np.uint32)
        signs = ((float_bits >> 31) & 1).astype(np.uint8)
        exponents = ((float_bits >> 23) & 0xFF).astype(np.int32)
        mantissa23 = (float_bits & 0x007F_FFFF).astype(np.int32)

        # 24-bit explicit mantissa with hidden 1; zero for denormals
        explicit_mant = np.where(exponents == 0, np.int32(0), np.int32(1 << 23) | mantissa23)

        # Shared exponent = max float32 exponent in each 16-element block
        shared_exp = np.max(exponents, axis=2)  # (N, 64)

        # Shift mantissa by exponent delta, then by 17 to get 7-bit result
        delta = shared_exp[:, :, np.newaxis] - exponents
        shifted = explicit_mant >> np.minimum(delta, 31)

        MANT_SHIFT = 17  # 24 - 7
        ROUND_MASK = (1 << MANT_SHIFT) - 1
        TIE = np.int32(1 << (MANT_SHIFT - 1))

        round_value = shifted & ROUND_MASK
        mantissa7 = (shifted >> MANT_SHIFT).astype(np.int32)
        guard_bit = mantissa7 & 1
        round_up = (round_value > TIE) | ((round_value == TIE) & (guard_bit == 1))
        mantissa7 = np.where(round_up, np.minimum(mantissa7 + 1, 127), mantissa7)

        # Zero sign when mantissa is zero
        signs = np.where(mantissa7 == 0, np.uint8(0), signs)
        packed_mant = ((signs.astype(np.int32) << 7) | (mantissa7 & 0x7F)).astype(np.uint8)

        # Assemble per-tile bytes: [exp words][mant words]
        exp_bytes = shared_exp.astype(np.uint8)  # (N, 64)
        exp_words = exp_bytes.view(np.uint32).reshape(num_tiles, 16)

        mant_words = packed_mant.reshape(num_tiles, 1024).view(np.uint32).reshape(num_tiles, 256)

        tile_words = np.concatenate([exp_words, mant_words], axis=1)  # (N, 272)
        return tile_words.tobytes()

    @staticmethod
    def _tilize_and_pack_bfloat16(data_2d: torch.Tensor, tile_h: int = 32, tile_w: int = 32) -> bytes:
        """Tilize a 2-D tensor and pack as bfloat16 (Float16_b) raw bytes.

        Each tile is 2048 bytes: 1024 elements x 2 bytes, stored in
        face order (face0, face1, face2, face3), row-major within each
        face.  bfloat16 is the top 16 bits of IEEE-754 float32.
        """
        H, W = data_2d.shape
        face_h, face_w = tile_h // 2, tile_w // 2
        tr, tc = H // tile_h, W // tile_w
        num_tiles = tr * tc

        data_np = data_2d.contiguous().float().numpy()

        tiles = data_np.reshape(tr, tile_h, tc, tile_w).transpose(0, 2, 1, 3)
        tiles = tiles.reshape(num_tiles, tile_h, tile_w)

        face_ordered = np.concatenate(
            [
                tiles[:, :face_h, :face_w].reshape(num_tiles, -1),
                tiles[:, :face_h, face_w:].reshape(num_tiles, -1),
                tiles[:, face_h:, :face_w].reshape(num_tiles, -1),
                tiles[:, face_h:, face_w:].reshape(num_tiles, -1),
            ],
            axis=1,
        )  # (N, 1024)

        # bfloat16 = top 16 bits of float32
        float_bits = face_ordered.view(np.uint32)
        bf16_bits = (float_bits >> 16).astype(np.uint16)
        return bf16_bits.tobytes()

    @staticmethod
    def _pack_bfloat16_1x32(data: torch.Tensor) -> bytes:
        """Pack a 1-row tensor as raw bfloat16 bytes with 1×32 tile layout.

        For 1×32 tiles there is no face reordering; elements are stored
        sequentially in tile-width chunks.  bfloat16 is the top 16 bits
        of IEEE-754 float32.
        """
        flat = data.contiguous().float().reshape(-1).numpy()
        float_bits = flat.view(np.uint32)
        bf16_bits = (float_bits >> 16).astype(np.uint16)
        return bf16_bits.tobytes()

    @staticmethod
    def _stitch_width_sharded(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        num_cores: int,
        tile_h: int = 32,
        tile_w: int = 32,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Stitch two width-sharded tensors into one fused tensor.

        For every core the two shards are concatenated vertically.  When
        the shard widths differ, the wider shard is tile-reshaped to match
        the narrower one (preserving tile ordering) so the concatenation
        is well-defined.

        Args:
            tensor1: First weight tensor (H1, W1).
            tensor2: Second weight tensor (H2, W2).
            num_cores: Total cores in the width-sharded grid.
            tile_h: Tile height (default 32).
            tile_w: Tile width (default 32).

        Returns:
            (fused_tensor, shard_shape) ready for WIDTH_SHARDED
            placement on num_cores cores.
        """
        H1, W1 = tensor1.shape
        H2, W2 = tensor2.shape

        shard_w1 = W1 // num_cores
        shard_w2 = W2 // num_cores

        # Use the narrower shard width as target; tile-reshape the wider.
        if shard_w1 <= shard_w2:
            target_w = shard_w1
            narrow, wide = tensor1, tensor2
            narrow_h, wide_h = H1, H2
            narrow_sw, wide_sw = shard_w1, shard_w2
        else:
            target_w = shard_w2
            narrow, wide = tensor2, tensor1
            narrow_h, wide_h = H2, H1
            narrow_sw, wide_sw = shard_w2, shard_w1

        # Height of each wide shard after tile-reshape to target_w
        reshaped_h = wide_h * wide_sw // target_w

        fused_shard_h = narrow_h + reshaped_h
        fused = torch.zeros(fused_shard_h, target_w * num_cores, dtype=tensor1.dtype)

        for core_idx in range(num_cores):
            col_start = core_idx * target_w
            col_end = col_start + target_w

            # Narrow shard: already at target width, just copy.
            n_start = core_idx * narrow_sw
            n_end = n_start + narrow_sw
            fused[:narrow_h, col_start:col_end] = narrow[:, n_start:n_end]

            # Wide shard: tile-reshape from (wide_h, wide_sw) to
            # (reshaped_h, target_w), then copy.
            w_start = core_idx * wide_sw
            w_end = w_start + wide_sw
            w_shard = wide[:, w_start:w_end]
            w_reshaped = BlitzDecodeWeights._tile_reshape(
                w_shard,
                src_shape=(wide_h, wide_sw),
                dst_shape=(reshaped_h, target_w),
                tile_h=tile_h,
                tile_w=tile_w,
            )
            fused[narrow_h:, col_start:col_end] = w_reshaped

        shard_shape = (fused_shard_h, target_w)
        return fused, shard_shape

    @staticmethod
    def _tile_reshape(
        tensor: torch.Tensor,
        src_shape: tuple[int, int],
        dst_shape: tuple[int, int],
        tile_h: int = 32,
        tile_w: int = 32,
    ) -> torch.Tensor:
        """Reshape a 2-D tensor while preserving row-major tile ordering.

        Data is stored as a grid of (tile_h x tile_w) tiles in row-major
        order.  A naive torch.reshape changes which values land in each
        tile.  This helper keeps every tile's contents unchanged by:

        1. Splitting into the source tile grid.
        2. Flattening to a 1-D tile sequence (row-major).
        3. Re-gridding into the destination tile dimensions.

        Total tile count must be identical for source and destination.
        """
        src_h, src_w = src_shape
        dst_h, dst_w = dst_shape
        src_tr, src_tc = src_h // tile_h, src_w // tile_w
        dst_tr, dst_tc = dst_h // tile_h, dst_w // tile_w
        assert src_tr * src_tc == dst_tr * dst_tc, f"Tile count mismatch: {src_tr * src_tc} vs {dst_tr * dst_tc}"
        # (H, W) -> (tile_rows, tile_h, tile_cols, tile_w)
        #         -> (tile_rows, tile_cols, tile_h, tile_w)
        tiles = tensor.reshape(src_tr, tile_h, src_tc, tile_w).permute(0, 2, 1, 3)
        # Flatten to 1-D tile sequence, re-grid to destination layout
        tiles = tiles.reshape(-1, tile_h, tile_w).reshape(dst_tr, dst_tc, tile_h, tile_w)
        # (dst_tr, dst_tc, tile_h, tile_w) -> (dst_H, dst_W)
        return tiles.permute(0, 2, 1, 3).reshape(dst_h, dst_w)
