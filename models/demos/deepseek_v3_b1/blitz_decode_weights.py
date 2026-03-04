# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Blitz decode weight overlapping infrastructure.

"Overlapping" means fusing multiple weight tensors into a single
L1 buffer so they share the same base address on each core.
For each core the individual shards are concatenated into one contiguous
buffer, zero-padded to a common maximum byte size.  Kernels locate
each sub-weight at a known byte offset within the fused shard.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlappedShardSpec, OverlappedTensor, overlap_tensors


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


class BlitzDecodeWeights:
    """Fuses weight tensors to share the same L1 base address per core.

    Methods take raw torch weight tensors, apply any required preprocessing
    (packing, shuffling), and fuse them via ``overlap_tensors`` into a
    single L1 buffer.  Each method returns a list of
    ``OverlappedTensor`` views into the fused buffer.

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
        *,
        move_to_device: bool = True,
    ) -> dict[str, OverlappedTensor]:
        """Fuse q_a_proj, q_b_proj, and kv_a_proj via ``overlap_tensors``.

        The fused buffer spans two core regions (lanes):

        * **Lane 0** (8×12 = 96 cores): q_a_proj (packed) and q_b_proj
          (shuffled) back-to-back within each core's shard.
        * **Lane 1** (2×9 = 18 cores at offset (8,0)): kv_a_proj
          with shard reordering, zero-padded to the same max byte size.

        q_b_proj is TP-sharded along mesh columns (``tp_dim=(None, 1)``).
        q_a_proj and kv_a_proj are replicated on every device.

        Args:
            q_a_proj_weights: Raw q_a_proj tensor, shape ``(7168, 1536)``.
            q_b_proj_weights: Raw (unshuffled) q_b_proj tensor, shape
                ``(1536, 12288 * mla_tp)``.
            kv_a_proj_weights: Raw kv_a_proj tensor, shape ``(7168, 576)``.
            move_to_device: If True (default), place the result on the mesh device
                via ``ttnn.from_torch(..., device=...)``. If False, device is
                passed as None so the tensor remains on host; mesh_mapper is still
                provided for layout/sharding metadata.

        Returns:
            A dict of :class:`OverlappedTensor` views keyed by name
            (``q_a_proj``, ``q_b_proj``, ``kv_a_proj``) that share the
            same underlying fused device buffer.
        """
        cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
        mesh_shape = (self._device.shape[0], self._device.shape[1])
        q_b_tp = cfg.q_b_shard_spec.tp(mesh_shape)

        # -- Validate device grid ----------------------------------------
        device_grid = self._device.compute_with_storage_grid_size()
        q_ab_bb = cfg.q_a_shard_spec.core_range_set.bounding_box()
        kv_bb = cfg.kv_a_shard_spec.core_range_set.bounding_box()
        required_rows = max(q_ab_bb.end.y, kv_bb.end.y) + 1
        required_cols = max(q_ab_bb.end.x, kv_bb.end.x) + 1
        assert device_grid.y >= required_rows, f"Device grid needs at least {required_rows} rows, got {device_grid.y}"
        assert device_grid.x >= required_cols, f"Device grid needs at least {required_cols} cols, got {device_grid.x}"

        # -- Validate raw input shapes ----------------------------------
        assert (
            q_a_proj_weights.shape == cfg.q_a_proj_shape
        ), f"q_a_proj_weights must be {cfg.q_a_proj_shape}, got {tuple(q_a_proj_weights.shape)}"
        expected_q_b_shape = (cfg.q_b_proj_shape[0], cfg.q_b_proj_shape[1] * q_b_tp)
        assert (
            tuple(q_b_proj_weights.shape) == expected_q_b_shape
        ), f"q_b_proj_weights must be {expected_q_b_shape}, got {tuple(q_b_proj_weights.shape)}"
        assert (
            kv_a_proj_weights.shape == cfg.kv_a_proj_shape
        ), f"kv_a_proj_weights must be {cfg.kv_a_proj_shape}, got {tuple(kv_a_proj_weights.shape)}"

        # -- Preprocess --------------------------------------------------
        q_a_packed = cfg.shuffle_q_a(q_a_proj_weights)
        kv_reordered = cfg.shuffle_kv_a(kv_a_proj_weights)

        q_b_shuffled_slices = [
            cfg.shuffle_q_b(cfg.get_q_b_slice(q_b_proj_weights, tp_idx, mesh_shape)) for tp_idx in range(q_b_tp)
        ]
        q_b_preprocessed = torch.cat(q_b_shuffled_slices, dim=1) if q_b_tp > 1 else q_b_shuffled_slices[0]

        q_ab_cores = cfg.q_a_shard_spec.core_range_set
        kv_cores = cfg.kv_a_shard_spec.core_range_set

        return overlap_tensors(
            [
                [
                    (
                        "q_a_proj",
                        q_a_packed,
                        OverlappedShardSpec(
                            core_range_set=q_ab_cores,
                            raw_tensor_shape=tuple(q_a_packed.shape),
                            dtype=ttnn.bfloat8_b,
                        ),
                    ),
                    (
                        "q_b_proj",
                        q_b_preprocessed,
                        OverlappedShardSpec(
                            core_range_set=q_ab_cores,
                            raw_tensor_shape=tuple(q_b_preprocessed.shape),
                            dtype=ttnn.bfloat8_b,
                            tp_dim=(None, 1),
                        ),
                    ),
                ],
                [
                    (
                        "kv_a_proj",
                        kv_reordered,
                        OverlappedShardSpec(
                            core_range_set=kv_cores,
                            raw_tensor_shape=tuple(kv_reordered.shape),
                            dtype=ttnn.bfloat8_b,
                        ),
                    ),
                ],
            ],
            device=self._device,
            move_to_device=move_to_device,
        )

    def get_tt_o_proj_and_gate_mm_weights(
        self,
        o_proj_weights: torch.Tensor,
        gate_mm_weights: torch.Tensor,
        attn_norm: torch.Tensor,
        q_norm: torch.Tensor,
        kv_norm: torch.Tensor,
        ffn_norm: torch.Tensor,
        *,
        move_to_device: bool = True,
    ) -> dict[str, OverlappedTensor]:
        """Fuse o_proj, gate_mm, and 4 RMSNorm gammas into one WIDTH_SHARDED tensor.

        The fused buffer is a UINT32 raw-byte container where each core's
        shard is zero-padded to the same maximum byte size.  There are always
        six sub-tensors.

        * **o_proj** — BFP8 (32×32 tiles) on 112 cores.
        * **gate_mm** — BFP16 (32×32 tiles) on 8 cores.
        * **attn_norm, q_norm, ffn_norm** — BFP16 (1×32 tiles) back-to-back on core (12, 9).
        * **kv_norm** — BFP16 (1×32 tiles) on dedicated core (0, 8).

        Args:
            o_proj_weights:     Raw o_proj tensor, shape
                ``(8192 * mla_tp, 7168)``.  TP-sharded on the height
                dim across mesh columns.
            gate_mm_weights:    Raw gate_mm tensor, shape (7168, 256).
                Replicated across TP devices.
            attn_norm:      Pre-SDPA attention-input RMSNorm gamma, shape (1, 7168).
            q_norm:     Pre-SDPA post-matmul1 RMSNorm gamma, shape (1, 1536).
            kv_norm:  Pre-SDPA KV-cache-branch RMSNorm gamma, shape (1, 512).
            ffn_norm:  MoE pre-MLP RMSNorm gamma, shape (1, 7168).

        Returns:
            A dict of six OverlappedTensors keyed by name:
            ``o_proj``, ``gate_mm``, ``attn_norm``, ``q_norm``, ``ffn_norm``, ``kv_norm``.
        """
        cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC

        return overlap_tensors(
            [
                [("o_proj", o_proj_weights, replace(cfg.o_proj, raw_tensor_shape=tuple(o_proj_weights.shape)))],
                [("gate_mm", gate_mm_weights, cfg.gate_mm)],
                [
                    ("attn_norm", attn_norm, cfg.attn_norm),
                    ("q_norm", q_norm, cfg.q_norm),
                    ("ffn_norm", ffn_norm, cfg.ffn_norm),
                ],
                [("kv_norm", kv_norm, cfg.kv_norm)],
            ],
            device=self._device,
            move_to_device=move_to_device,
        )

    def get_tt_kv_b12_proj_weights(
        self,
        kv_b1_proj_weights: torch.Tensor,
        kv_b2_proj_weights: torch.Tensor,
        *,
        move_to_device: bool = True,
    ) -> dict[str, OverlappedTensor]:
        """Fuse kv_b1_proj and kv_b2_proj via ``overlap_tensors``.

        Fuses ``kv_b1_proj (8192, 512)`` onto 64 cores and
        ``kv_b2_proj (512, 8192)`` (pre-transposed to ``(8192, 512)``)
        onto another 64 cores.  Both lanes are HEIGHT_SHARDED with
        shard ``(128, 512)`` as BFP8.

        Layout::

            -- kv_b1 region: 64 Qnope cores (8x8) --
            kv_b1_proj (8192, 512) as bfloat8_b, shard (128, 512)

            -- kv_b2 region: 64 remaining cores (5x8 + 12x2) --
            kv_b2_proj (8192, 512) as bfloat8_b, shard (128, 512)

            combined: 128 cores, fused buffer

        Args:
            kv_b1_proj_weights: shape ``(8192 * mla_tp, 512)``.
                TP-sharded on the heads dim.
            kv_b2_proj_weights: shape ``(512, 8192 * mla_tp)``.
                TP-sharded on the heads dim.

        Returns:
            A dict of :class:`OverlappedTensor` views keyed by name
            (``kv_b1_proj``, ``kv_b2_proj``) sharing the same fused
            device buffer.
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

        per_device_b2_w = cfg.kv_b2_proj_shape[1]
        b2_shuffled = []
        for tp_idx in range(mla_tp):
            b2_slice = kv_b2_proj_weights[:, tp_idx * per_device_b2_w : (tp_idx + 1) * per_device_b2_w]
            b2_shuffled.append(cfg.shuffle_kv_b2(b2_slice))
        kv_b2_preprocessed = torch.cat(b2_shuffled, dim=0) if mla_tp > 1 else b2_shuffled[0]

        return overlap_tensors(
            [
                [
                    (
                        "kv_b1_proj",
                        kv_b1_proj_weights,
                        OverlappedShardSpec(
                            core_range_set=cfg.kv_b1_core_range_set,
                            raw_tensor_shape=tuple(kv_b1_proj_weights.shape),
                            dtype=ttnn.bfloat8_b,
                            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                            tp_dim=(None, 0),
                        ),
                    ),
                ],
                [
                    (
                        "kv_b2_proj",
                        kv_b2_preprocessed,
                        OverlappedShardSpec(
                            core_range_set=cfg.kv_b2_core_range_set,
                            raw_tensor_shape=tuple(kv_b2_preprocessed.shape),
                            dtype=ttnn.bfloat8_b,
                            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                            tp_dim=(None, 0),
                            logical_tensor_shape=cfg.kv_b2_proj_shape,
                        ),
                    ),
                ],
            ],
            device=self._device,
            move_to_device=move_to_device,
        )

    # ------------------------------------------------------------------
    # MOE weight loading
    # ------------------------------------------------------------------

    def get_tt_moe_shared_expert_weights(
        self,
        gate_proj_weights: torch.Tensor,
        up_proj_weights: torch.Tensor,
        down_proj_weights: torch.Tensor,
        *,
        move_to_device: bool = True,
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

        mesh_rows = self._device.shape[0]
        mesh_cols = self._device.shape[1]
        per_device_n = cfg.gate_proj_shape[1]
        stacked_h, stacked_w = cfg.stacked_shape

        gate_stacked_list = []
        up_stacked_list = []
        for tp_idx in range(moe_tp):
            gate_slice = gate_proj_weights[:, tp_idx * per_device_n : (tp_idx + 1) * per_device_n]
            up_slice = up_proj_weights[:, tp_idx * per_device_n : (tp_idx + 1) * per_device_n]
            gate_stacked_list.append(cfg.reshuffle_block_to_height_sharded(gate_slice, cfg.gate_core_range_set))
            up_stacked_list.append(cfg.reshuffle_block_to_height_sharded(up_slice, cfg.up_core_range_set))

        if moe_tp == 1:
            gate_preprocessed = gate_stacked_list[0]
            up_preprocessed = up_stacked_list[0]
        else:
            gate_preprocessed = (
                torch.stack(gate_stacked_list)
                .reshape(mesh_rows, mesh_cols, stacked_h, stacked_w)
                .permute(0, 2, 1, 3)
                .reshape(mesh_rows * stacked_h, mesh_cols * stacked_w)
                .contiguous()
            )
            up_preprocessed = (
                torch.stack(up_stacked_list)
                .reshape(mesh_rows, mesh_cols, stacked_h, stacked_w)
                .permute(0, 2, 1, 3)
                .reshape(mesh_rows * stacked_h, mesh_cols * stacked_w)
                .contiguous()
            )

        gate_up_dict = overlap_tensors(
            [
                [
                    (
                        "gate_proj",
                        gate_preprocessed,
                        OverlappedShardSpec(
                            core_range_set=cfg.gate_core_range_set,
                            raw_tensor_shape=tuple(gate_preprocessed.shape),
                            dtype=ttnn.bfloat4_b,
                            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                            tp_dim=(0, 1),
                            logical_tensor_shape=cfg.gate_proj_shape,
                        ),
                    ),
                ],
                [
                    (
                        "up_proj",
                        up_preprocessed,
                        OverlappedShardSpec(
                            core_range_set=cfg.up_core_range_set,
                            raw_tensor_shape=tuple(up_preprocessed.shape),
                            dtype=ttnn.bfloat4_b,
                            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                            tp_dim=(0, 1),
                            logical_tensor_shape=cfg.up_proj_shape,
                        ),
                    ),
                ],
            ],
            device=self._device,
            move_to_device=move_to_device,
        )
        gate_ov = gate_up_dict["gate_proj"]
        up_ov = gate_up_dict["up_proj"]

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
        device_dp = self._device if move_to_device else None

        dp_shard_spec = ttnn.ShardSpec(
            matmul_core_grid, (K_down_per_device, N_per_core), ttnn.ShardOrientation.ROW_MAJOR
        )
        dp_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, dp_shard_spec)

        down_tensor = ttnn.from_torch(
            dp_combined,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device_dp,
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
        *,
        move_to_device: bool = True,
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
        device_for_torch = device if move_to_device else None

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
                        device=device_for_torch,
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
        *,
        move_to_device: bool = True,
    ) -> tuple[OverlappedTensor, OverlappedTensor, ttnn.Tensor]:
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
            move_to_device=move_to_device,
        )

    def get_tt_mlp_routed_expert_weights(
        self,
        gate_proj_weights: torch.Tensor,
        up_proj_weights: torch.Tensor,
        down_proj_weights: torch.Tensor,
        *,
        move_to_device: bool = True,
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
        mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=(mesh_rows, mesh_cols), dims=(0, 1))
        device_for_torch = device if move_to_device else None

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
                device=device_for_torch,
                memory_config=mem_config,
                mesh_mapper=mesh_mapper,
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
