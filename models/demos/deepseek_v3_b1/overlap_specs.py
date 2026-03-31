# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Overlap specification dataclasses for blitz decode weight fusion.

These frozen dataclasses define the core-grid layouts, shard shapes, and
preprocessing transforms that describe how multiple weight tensors are
fused into a single L1 buffer.  They serve as the layout contract between
weight preparation (``prepare_weights``) and runtime fused ops
(``attention_block``, ``post_sdpa``, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlappedShardSpec


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

    IMPORTANT: Input must be in ``[ALL_NOPE | ALL_ROPE]`` column layout, NOT
    HF interleaved ``[h0_nope|h0_rope|h1_nope|h1_rope|...]``. Use
    ``prepare_weights.deinterleave_q_b_proj`` to convert HF weights first.

    Args:
        weights: Input weight matrix ``[K, N]`` in ``[ALL_NOPE | ALL_ROPE]``
            layout where
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

    The three sub-tensor shard specs use :class:`OverlappedShardSpec`
    to compute per-shard tile counts and byte sizes.  ``q_a_shard_spec``
    uses the packed shape ``(H/2, 2W)`` after the ``shuffle_q_a``
    transform.
    """

    # Head configuration
    num_qnope_heads: int = 64
    num_qrope_heads: int = 64
    qnope_head_dim: int = 128
    qrope_head_dim: int = 64
    heads_per_row: int = 8
    # Raw tensor shapes for input validation (height, width)
    q_a_proj_shape: tuple[int, int] = (7168, 1536)
    q_b_proj_shape: tuple[int, int] = (1536, 12288)
    kv_a_proj_shape: tuple[int, int] = (7168, 576)

    # Sub-tensor shard specs — q_a uses the packed shape (H/2, 2W)
    q_a_shard_spec: OverlappedShardSpec = field(
        default_factory=lambda: OverlappedShardSpec(
            core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7))}),
            raw_tensor_shape=(3584, 3072),
            dtype=ttnn.DataType.BFLOAT8_B,
        )
    )
    q_b_shard_spec: OverlappedShardSpec = field(
        default_factory=lambda: OverlappedShardSpec(
            core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7))}),
            raw_tensor_shape=(1536, 12288),
            dtype=ttnn.DataType.BFLOAT8_B,
            tp_dim=(None, 0),
        )
    )
    kv_a_shard_spec: OverlappedShardSpec = field(
        default_factory=lambda: OverlappedShardSpec(
            core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(8, 9))}),
            raw_tensor_shape=(7168, 576),
            dtype=ttnn.DataType.BFLOAT8_B,
        )
    )

    # kv_a_proj shard reorder: places the Knope-rope boundary shard so that
    # the physical core layout matches the logical split expected by the
    # KV cache branch kernel.
    kv_a_proj_shard_order: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 16, 8, 9, 10, 11, 12, 13, 14, 15, 17)

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

    def get_q_b_slice(self, q_b_proj_weights: torch.Tensor, tp_idx: int, mesh_shape: tuple[int, int]) -> torch.Tensor:
        """Extract the per-device q_b_proj slice for a given TP index.

        The full q_b_proj tensor is laid out as ``[all_qnope | all_qrope]``
        across TP devices.  This method splits qnope and qrope, takes the
        ``tp_idx``-th chunk of each, and stitches them into the single-device
        ``(K, qnope_dim + qrope_dim)`` slice.
        """
        per_tp_qnope_dim = self.num_qnope_heads * self.qnope_head_dim
        per_tp_qrope_dim = self.num_qrope_heads * self.qrope_head_dim
        total_qnope_dim = self.q_b_shard_spec.tp(mesh_shape) * per_tp_qnope_dim

        full_qnope = q_b_proj_weights[:, :total_qnope_dim]
        full_qrope = q_b_proj_weights[:, total_qnope_dim:]

        tp_qnope = full_qnope[:, tp_idx * per_tp_qnope_dim : (tp_idx + 1) * per_tp_qnope_dim]
        tp_qrope = full_qrope[:, tp_idx * per_tp_qrope_dim : (tp_idx + 1) * per_tp_qrope_dim]

        return torch.cat([tp_qnope, tp_qrope], dim=1)

    def shuffle_kv_a(self, weights: torch.Tensor) -> torch.Tensor:
        """Reorder kv_a_proj shards for the KV cache branch core layout."""
        kv_h, kv_w = weights.shape
        kv_num_cores = self.kv_a_shard_spec.core_range_set.num_cores()
        shards = weights.reshape(kv_h, kv_num_cores, kv_w // kv_num_cores)
        return shards[:, list(self.kv_a_proj_shard_order), :].reshape(kv_h, kv_w)


QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC = QAB_KVA_PROJ_SingleDeviceOverlapSpec()


_GAMMA_CORE = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(12, 9), ttnn.CoreCoord(12, 9))])
_KV_NORM_CORE = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(0, 8))])


@dataclass(frozen=True)
class O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec:
    """Configuration for the o_proj / gate_mm / rmsnorm-gamma weight overlap.

    Fuses the following into a single WIDTH_SHARDED raw buffer:

    * **o_proj** — BFP8, (16384, 7168) full, TP along mesh columns (tp_dim=(None, 0)),
      WIDTH_SHARDED on 112 cores.
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

    o_proj: OverlappedShardSpec = field(
        default_factory=lambda: OverlappedShardSpec(
            core_range_set=ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7)),  # 96 cores
                    ttnn.CoreRange(ttnn.CoreCoord(1, 8), ttnn.CoreCoord(8, 9)),  # 16 cores
                }
            ),
            raw_tensor_shape=(16384, 7168),
            dtype=ttnn.DataType.BFLOAT8_B,
            tp_dim=(None, 0),
        )
    )
    gate_mm: OverlappedShardSpec = field(
        default_factory=lambda: OverlappedShardSpec(
            core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(12, 0), ttnn.CoreCoord(12, 7))}),
            raw_tensor_shape=(7168, 256),
            dtype=ttnn.DataType.BFLOAT16,
        )
    )
    attn_norm: OverlappedShardSpec = field(
        default_factory=lambda: OverlappedShardSpec(
            core_range_set=_GAMMA_CORE,
            raw_tensor_shape=(1, 7168),
            dtype=ttnn.DataType.BFLOAT16,
            tile_h=1,
            tile_w=32,
        )
    )
    q_norm: OverlappedShardSpec = field(
        default_factory=lambda: OverlappedShardSpec(
            core_range_set=_GAMMA_CORE,
            raw_tensor_shape=(1, 1536),
            dtype=ttnn.DataType.BFLOAT16,
            tile_h=1,
            tile_w=32,
        )
    )
    kv_norm: OverlappedShardSpec = field(
        default_factory=lambda: OverlappedShardSpec(
            core_range_set=_KV_NORM_CORE,
            raw_tensor_shape=(1, 512),
            dtype=ttnn.DataType.BFLOAT16,
            tile_h=1,
            tile_w=32,
        )
    )
    ffn_norm: OverlappedShardSpec = field(
        default_factory=lambda: OverlappedShardSpec(
            core_range_set=_GAMMA_CORE,
            raw_tensor_shape=(1, 7168),
            dtype=ttnn.DataType.BFLOAT16,
            tile_h=1,
            tile_w=32,
        )
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

    # --- weight shuffles ------------------------------------------------------

    def shuffle_kv_b2(self, weights: torch.Tensor) -> torch.Tensor:
        """Tile-level rearrange (512, 8192) into (8192, 512) for HEIGHT_SHARDED.

        Each core's (512, 128) column slice is a 16×4 grid of 32×32 tiles.
        These 64 tiles are laid out contiguously into a (128, 512) shard
        (4×16 tile grid), preserving per-tile element data so that
        ``from_torch`` with TILE_LAYOUT produces byte-identical BFP8 tiles.
        """
        kv_dim, n_heads = self.kv_b2_proj_shape
        n_cores = self.kv_b2_core_range_set.num_cores()
        head_dim = n_heads // n_cores
        t = 32

        k_tiles = kv_dim // t
        n_tiles = head_dim // t

        per_core = weights.reshape(kv_dim, n_cores, head_dim).permute(1, 0, 2).contiguous()
        tiles = per_core.reshape(n_cores, k_tiles, t, n_tiles, t)
        tiles = tiles.permute(0, 1, 3, 2, 4).reshape(n_cores, k_tiles * n_tiles, t, t)
        tiles = tiles.reshape(n_cores, head_dim // t, kv_dim // t, t, t)
        return tiles.permute(0, 1, 3, 2, 4).reshape(n_cores * head_dim, kv_dim).contiguous()


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
