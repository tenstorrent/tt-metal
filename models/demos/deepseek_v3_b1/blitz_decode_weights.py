# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Blitz decode weight overlap specs and fusion helpers.

"Overlapping" means fusing multiple weight tensors into a single
L1 buffer so they share the same base address on each core.
For each core the individual shards are concatenated into one contiguous
buffer, zero-padded to a common maximum byte size.  Kernels locate
each sub-weight at a known byte offset within the fused shard.

This module provides:

- ``*_SingleDeviceOverlapSpec`` dataclasses with OverlappedTensorSpec
  fields and shuffle/preprocessing methods.
- ``FusionGroupSpec`` constants (``Q_AB_KV_A_SPEC``, ``KV_B12_SPEC``,
  ``O_PROJ_GATE_MM_NORMS_SPEC``, ``GATE_UP_SPEC``).
- Shared preprocessing functions (``preprocess_q_ab_kv_a``,
  ``preprocess_kv_b12``, ``preprocess_gate_up``) — single source of
  truth for the shuffle/TP-concat/mesh-reshape orchestration.
- Standalone fusion functions (``fuse_q_ab_kv_a``, ``fuse_kv_b12``,
  ``fuse_o_proj_gate_mm_norms``, ``fuse_gate_up``,
  ``create_moe_routed_expert_tensors``).
- Standalone non-fusion utilities (``shuffle_dram_tiles``,
  ``shared_down_torch_for_cache``, ``moe_routed_expert_torch_for_cache``,
  ``mlp_routed_dense_stacked_torch_for_cache``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlapEntry, OverlappedTensorSpec, overlap_tensors
from models.demos.deepseek_v3_b1.tensor_cache.types import (
    FusionGroupSpec,
    MeshMapperConfig,
    RegionSpec,
    ReplicateMeshMapper,
    Shard2dMeshMapper,
)

OverlappedTensor = ttnn.OverlappedTensor


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

    The three sub-tensor shard specs use :class:`OverlappedTensorSpec`
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
    q_a_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7))}),
            raw_tensor_shape=(3584, 3072),
            dtype=ttnn.DataType.BFLOAT8_B,
        )
    )
    q_b_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7))}),
            raw_tensor_shape=(1536, 12288),
            dtype=ttnn.DataType.BFLOAT8_B,
            tp_dim=(None, 1),
        )
    )
    kv_a_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
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

    * **o_proj** — BFP8, base logical shape (8192, 7168), TP along mesh
      columns (tp_dim=(None, 0)), WIDTH_SHARDED on 112 cores.
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

    o_proj: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7)),  # 96 cores
                    ttnn.CoreRange(ttnn.CoreCoord(1, 8), ttnn.CoreCoord(8, 9)),  # 16 cores
                }
            ),
            raw_tensor_shape=(8192, 7168),
            dtype=ttnn.DataType.BFLOAT8_B,
            tp_dim=(None, 0),
        )
    )
    gate_mm: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(12, 0), ttnn.CoreCoord(12, 7))}),
            raw_tensor_shape=(7168, 256),
            dtype=ttnn.DataType.BFLOAT16,
        )
    )
    attn_norm: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=_GAMMA_CORE,
            raw_tensor_shape=(1, 7168),
            dtype=ttnn.DataType.BFLOAT16,
            tile_h=1,
            tile_w=32,
        )
    )
    q_norm: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=_GAMMA_CORE,
            raw_tensor_shape=(1, 1536),
            dtype=ttnn.DataType.BFLOAT16,
            tile_h=1,
            tile_w=32,
        )
    )
    kv_norm: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=_KV_NORM_CORE,
            raw_tensor_shape=(1, 512),
            dtype=ttnn.DataType.BFLOAT16,
            tile_h=1,
            tile_w=32,
        )
    )
    ffn_norm: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
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

    kv_b1_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
            raw_tensor_shape=(8192, 512),
            dtype=ttnn.DataType.BFLOAT8_B,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            tp_dim=(None, 0),
        )
    )
    kv_b2_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(12, 7)),  # 5×8 = 40 cores
                    ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(11, 9)),  # 12×2 = 24 cores
                }
            ),
            raw_tensor_shape=(512, 8192),
            dtype=ttnn.DataType.BFLOAT8_B,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            tp_dim=(None, 0),
        )
    )

    @property
    def kv_b1_core_range_set(self) -> ttnn.CoreRangeSet:
        return self.kv_b1_shard_spec.core_range_set

    @property
    def kv_b2_core_range_set(self) -> ttnn.CoreRangeSet:
        return self.kv_b2_shard_spec.core_range_set

    @property
    def kv_b1_proj_shape(self) -> tuple[int, int]:
        return self.kv_b1_shard_spec.raw_tensor_shape

    @property
    def kv_b2_proj_shape(self) -> tuple[int, int]:
        return self.kv_b2_shard_spec.raw_tensor_shape

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

    gate_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 9)),  # 30 cores  cols 0-2
                    ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 3)),  #  4 cores  col 3 top
                    ttnn.CoreRange(ttnn.CoreCoord(7, 0), ttnn.CoreCoord(9, 9)),  # 30 cores  cols 7-9
                }
            ),
            raw_tensor_shape=(7168, 256),
            dtype=ttnn.DataType.BFLOAT4_B,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            tp_dim=(0, 1),
        )
    )
    up_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(3, 4), ttnn.CoreCoord(3, 9)),  #  6 cores  col 3 bottom
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(6, 9)),  # 30 cores  cols 4-6
                    ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(11, 9)),  # 20 cores  cols 10-11
                    ttnn.CoreRange(ttnn.CoreCoord(12, 0), ttnn.CoreCoord(12, 7)),  #  8 cores  col 12
                }
            ),
            raw_tensor_shape=(7168, 256),
            dtype=ttnn.DataType.BFLOAT4_B,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            tp_dim=(0, 1),
        )
    )

    @property
    def gate_core_range_set(self) -> ttnn.CoreRangeSet:
        return self.gate_shard_spec.core_range_set

    @property
    def up_core_range_set(self) -> ttnn.CoreRangeSet:
        return self.up_shard_spec.core_range_set

    @property
    def gate_proj_shape(self) -> tuple[int, int]:
        return self.gate_shard_spec.raw_tensor_shape

    @property
    def up_proj_shape(self) -> tuple[int, int]:
        return self.up_shard_spec.raw_tensor_shape

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


def _infer_mesh_mapper(lanes: list[list[tuple[str, OverlappedTensorSpec]]]) -> MeshMapperConfig:
    """Derive the mesh mapper config from ``tp_dim`` values across all specs.

    For each mesh dimension, collects the unique non-None tensor dimension
    mapping.  All specs that shard on a given mesh dimension must agree on
    which tensor dimension they shard.
    """
    dims: list[int | None] = [None, None]
    for lane in lanes:
        for spec_name, spec in lane:
            for mesh_dim in range(2):
                td = spec.tp_dim[mesh_dim]
                if td is not None:
                    if dims[mesh_dim] is not None and dims[mesh_dim] != td:
                        raise ValueError(
                            f"Conflicting tp_dim[{mesh_dim}] in {spec_name!r}: "
                            f"previously saw {dims[mesh_dim]}, now {td}"
                        )
                    dims[mesh_dim] = td
    if dims[0] is None and dims[1] is None:
        return ReplicateMeshMapper()
    return Shard2dMeshMapper(dims=(dims[0], dims[1]))


def _build_fusion_group_spec(
    name: str,
    lanes: list[list[tuple[str, OverlappedTensorSpec]]],
    sharding_strategy: ttnn.TensorMemoryLayout,
    mesh_mapper_config: MeshMapperConfig | None = None,
) -> FusionGroupSpec:
    """Derive a :class:`FusionGroupSpec` from named :class:`OverlappedTensorSpec` fields.

    Each *lane* is a list of ``(name, spec)`` pairs sharing the same core
    range.  The helper stamps ``name`` onto each spec via ``replace()``,
    groups them into :class:`RegionSpec` instances (core range taken from
    the first spec in the lane), and assembles the final
    :class:`FusionGroupSpec`.

    When ``mesh_mapper_config`` is omitted the mapper is inferred from
    the ``tp_dim`` values on the specs.
    """
    if mesh_mapper_config is None:
        mesh_mapper_config = _infer_mesh_mapper(lanes)
    regions: list[RegionSpec] = []
    for lane in lanes:
        subtensors = tuple(
            replace(spec, name=n, logical_tensor_shape=spec.logical_tensor_shape or spec.raw_tensor_shape)
            for n, spec in lane
        )
        regions.append(
            RegionSpec(
                core_range_set=lane[0][1].core_range_set,
                subtensors=subtensors,
            )
        )
    return FusionGroupSpec(
        name=name,
        regions=tuple(regions),
        sharding_strategy=sharding_strategy,
        mesh_mapper_config=mesh_mapper_config,
    )


_QAB_SPEC = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
Q_AB_KV_A_SPEC = _build_fusion_group_spec(
    "q_ab_kv_a",
    [
        [("q_a_proj", _QAB_SPEC.q_a_shard_spec), ("q_b_proj", _QAB_SPEC.q_b_shard_spec)],
        [("kv_a_proj", _QAB_SPEC.kv_a_shard_spec)],
    ],
    sharding_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
)

_OV_SPEC = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC
# Explicit mesh_mapper_config to preserve fingerprint hash compatibility.
# o_proj.tp_dim=(None,0) would auto-derive dims=(None,0), but the
# original spec used dims=(None,1); changing it would invalidate caches.
O_PROJ_GATE_MM_NORMS_SPEC = _build_fusion_group_spec(
    "o_proj_gate_mm_norms",
    [
        [("o_proj", _OV_SPEC.o_proj)],
        [("gate_mm", _OV_SPEC.gate_mm)],
        [("attn_norm", _OV_SPEC.attn_norm), ("q_norm", _OV_SPEC.q_norm), ("ffn_norm", _OV_SPEC.ffn_norm)],
        [("kv_norm", _OV_SPEC.kv_norm)],
    ],
    sharding_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    mesh_mapper_config=Shard2dMeshMapper(dims=(None, 1)),
)

_KVB_SPEC = KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
KV_B12_SPEC = _build_fusion_group_spec(
    "kv_b12",
    [
        [("kv_b1_proj", _KVB_SPEC.kv_b1_shard_spec)],
        [("kv_b2_proj", _KVB_SPEC.kv_b2_shard_spec)],
    ],
    sharding_strategy=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
)

_GU_SPEC = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
GATE_UP_SPEC = _build_fusion_group_spec(
    "gate_up",
    [
        [("shared_gate_proj", _GU_SPEC.gate_shard_spec)],
        [("shared_up_proj", _GU_SPEC.up_shard_spec)],
    ],
    sharding_strategy=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
)


def _tp_factors(device) -> tuple[int, int]:
    """Returns (mla_tp, moe_tp) derived from device topology."""
    if device.get_num_devices() == 1:
        return 1, 1
    return 2, 8


def shuffle_dram_tiles(tensor: torch.Tensor, tile_size: int, num_banks: int) -> torch.Tensor:
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


def shared_down_torch_for_cache(
    down_proj_weights: torch.Tensor, moe_tp: int, mesh_shape: tuple[int, int]
) -> torch.Tensor:
    """Produce the torch tensor layout for shared expert down_proj cache storage.

    Matches the ``dp_combined`` tensor that ``from_torch`` receives in the
    shared expert down path.

    Args:
        down_proj_weights: Shape ``(K_down_per_device * moe_tp, N_down)``.
        moe_tp: Number of MoE tensor parallel slices.
        mesh_shape: ``(mesh_rows, mesh_cols)`` of the device mesh.
    """
    dp_spec = DOWN_PROJ_SINGLE_DEVICE_SPEC
    K_down_per_device = 256
    N_per_core = 64
    N_down = N_per_core * dp_spec.NUM_MATMUL_CORES
    expected_down_shape = (K_down_per_device * moe_tp, N_down)
    assert (
        tuple(down_proj_weights.shape) == expected_down_shape
    ), f"down_proj_weights must be {expected_down_shape}, got {tuple(down_proj_weights.shape)}"
    if moe_tp == 1:
        return down_proj_weights.contiguous()
    mesh_rows, mesh_cols = mesh_shape
    return (
        down_proj_weights.reshape(mesh_rows, mesh_cols, K_down_per_device, N_down)
        .permute(0, 2, 1, 3)
        .reshape(mesh_rows * K_down_per_device, mesh_cols * N_down)
    ).contiguous()


def moe_routed_expert_torch_for_cache(w: torch.Tensor, num_banks: int) -> torch.Tensor:
    """Match the tensor passed to ``from_torch`` for one MoE routed expert (one projection).

    Returns shape ``(1, 1, K, N_padded)``.

    Args:
        w: Single expert weight ``(K, N)``.
        num_banks: Number of DRAM banks on the device.
    """
    tile_w = 32
    K, N = w.shape
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    if N_padded != N:
        w = torch.nn.functional.pad(w, (0, N_padded - N))
    w_shuffled = shuffle_dram_tiles(w.unsqueeze(0), tile_w, num_banks)
    return w_shuffled.reshape(1, 1, K, N_padded).contiguous()


def mlp_routed_dense_stacked_torch_for_cache(
    experts: torch.Tensor, num_banks: int, mesh_shape: tuple[int, int]
) -> torch.Tensor:
    """Stacked torch before ``from_torch`` in dense MLP routed ``upload`` (all experts on mesh).

    Args:
        experts: ``(n_exp, K, N)`` tensor of expert weights.
        num_banks: Number of DRAM banks on the device.
        mesh_shape: ``(mesh_rows, mesh_cols)``.
    """
    tile_w = 32
    mesh_rows, mesh_cols = mesh_shape
    n_exp, K, N = experts.shape
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    processed = []
    for i in range(n_exp):
        w = experts[i]
        if N_padded != N:
            w = torch.nn.functional.pad(w, (0, N_padded - N))
        w_shuffled = shuffle_dram_tiles(w.unsqueeze(0), tile_w, num_banks)
        processed.append(w_shuffled.reshape(K, N_padded))
    stacked = torch.stack(processed).reshape(mesh_rows, mesh_cols, K, N_padded)
    return stacked.contiguous()


# ---------------------------------------------------------------------------
# Shared preprocessing functions (Phase B)
#
# These are the single source of truth for the shuffle/TP-concat/mesh-reshape
# orchestration that converts logical torch tensors into fusion-ready form.
# Both prepare_weights._preprocess_* and the fuse_* helpers below delegate
# to these.
# ---------------------------------------------------------------------------


def preprocess_q_ab_kv_a(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    kv_a: torch.Tensor,
    mesh_shape: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Shuffle and TP-concat q_a/q_b/kv_a into fusion-ready tensors.

    Args:
        q_a: Transposed q_a_proj weight ``(K, N)``.
        q_b: Deinterleaved q_b_proj weight ``(K, N)`` (full or TP1-trimmed).
        kv_a: Transposed kv_a_proj weight ``(K, N)``.
        mesh_shape: ``(rows, cols)`` of the device mesh, ``(1, 1)`` for single device.

    Returns:
        Dict with keys ``q_a_proj``, ``q_b_proj``, ``kv_a_proj`` — shuffled,
        TP-concatenated torch tensors ready for tilization.
    """
    cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    q_a_packed = cfg.shuffle_q_a(q_a)
    q_b_tp = cfg.q_b_shard_spec.tp(mesh_shape)
    q_b_slices = [cfg.shuffle_q_b(cfg.get_q_b_slice(q_b, i, mesh_shape)) for i in range(q_b_tp)]
    q_b_pre = torch.cat(q_b_slices, dim=1) if q_b_tp > 1 else q_b_slices[0]
    kv_reordered = cfg.shuffle_kv_a(kv_a)
    return {"q_a_proj": q_a_packed, "q_b_proj": q_b_pre, "kv_a_proj": kv_reordered}


def preprocess_kv_b12(
    kv_b1: torch.Tensor,
    kv_b2: torch.Tensor,
    mla_tp: int,
) -> dict[str, torch.Tensor]:
    """Shuffle and TP-concat kv_b1/kv_b2 into fusion-ready tensors.

    Args:
        kv_b1: kv_b1 projection weight ``(H, W)`` (already split from kv_b_proj).
        kv_b2: kv_b2 projection weight ``(H, W)`` (already split, full TP width).
        mla_tp: MLA tensor-parallel factor (1 for single device, 2 for 4x2 mesh).

    Returns:
        Dict with keys ``kv_b1_proj``, ``kv_b2_proj`` — kv_b2 is shuffled
        and TP-concatenated along dim 0.
    """
    cfg = KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    per_device_b2_w = cfg.kv_b2_proj_shape[1]
    b2_shuffled = [cfg.shuffle_kv_b2(kv_b2[:, i * per_device_b2_w : (i + 1) * per_device_b2_w]) for i in range(mla_tp)]
    kv_b2_pre = torch.cat(b2_shuffled, dim=0) if mla_tp > 1 else b2_shuffled[0]
    return {"kv_b1_proj": kv_b1, "kv_b2_proj": kv_b2_pre}


def preprocess_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    moe_tp: int,
    mesh_rows: int,
    mesh_cols: int,
) -> dict[str, torch.Tensor]:
    """Reshuffle and TP-stack gate/up into fusion-ready tensors.

    Args:
        gate: Shared gate projection ``(K, N)`` (transposed, TP-trimmed if needed).
        up: Shared up projection ``(K, N)`` (transposed, TP-trimmed if needed).
        moe_tp: MoE tensor-parallel factor (1 for single device, 8 for 4x2 mesh).
        mesh_rows: Number of mesh rows (1 for single device).
        mesh_cols: Number of mesh columns (1 for single device).

    Returns:
        Dict with keys ``shared_gate_proj``, ``shared_up_proj`` — reshuffled
        from block to height-sharded layout, with multi-device stack/permute
        applied when ``moe_tp > 1``.
    """
    cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    per_device_n = cfg.gate_proj_shape[1]
    stacked_h, stacked_w = cfg.stacked_shape
    gate_list, up_list = [], []
    for tp_idx in range(moe_tp):
        gate_list.append(
            cfg.reshuffle_block_to_height_sharded(
                gate[:, tp_idx * per_device_n : (tp_idx + 1) * per_device_n], cfg.gate_core_range_set
            )
        )
        up_list.append(
            cfg.reshuffle_block_to_height_sharded(
                up[:, tp_idx * per_device_n : (tp_idx + 1) * per_device_n], cfg.up_core_range_set
            )
        )

    def _stack(lst):
        if moe_tp == 1:
            return lst[0]
        return (
            torch.stack(lst)
            .reshape(mesh_rows, mesh_cols, stacked_h, stacked_w)
            .permute(0, 2, 1, 3)
            .reshape(mesh_rows * stacked_h, mesh_cols * stacked_w)
            .contiguous()
        )

    return {"shared_gate_proj": _stack(gate_list), "shared_up_proj": _stack(up_list)}


def fuse_q_ab_kv_a(
    q_a_proj_weights: torch.Tensor,
    q_b_proj_weights: torch.Tensor,
    kv_a_proj_weights: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse q_a, q_b, and kv_a projection weights into one overlapped buffer."""
    cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    mesh_shape = (device.shape[0], device.shape[1]) if device.get_num_devices() > 1 else (1, 1)
    preprocessed = preprocess_q_ab_kv_a(q_a_proj_weights, q_b_proj_weights, kv_a_proj_weights, mesh_shape)
    q_a_packed = preprocessed["q_a_proj"]
    q_b_pre = preprocessed["q_b_proj"]
    kv_reordered = preprocessed["kv_a_proj"]

    return overlap_tensors(
        [
            [
                OverlapEntry(
                    "q_a_proj",
                    q_a_packed,
                    replace(cfg.q_a_shard_spec, raw_tensor_shape=tuple(q_a_packed.shape), dtype=dtype),
                ),
                OverlapEntry(
                    "q_b_proj",
                    q_b_pre,
                    replace(cfg.q_b_shard_spec, raw_tensor_shape=tuple(q_b_pre.shape), dtype=dtype),
                ),
            ],
            [
                OverlapEntry(
                    "kv_a_proj",
                    kv_reordered,
                    replace(cfg.kv_a_shard_spec, raw_tensor_shape=tuple(kv_reordered.shape), dtype=dtype),
                ),
            ],
        ],
        device=device,
        move_to_device=move_to_device,
    )


def fuse_o_proj_gate_mm_norms(
    o_proj_weights: torch.Tensor,
    gate_mm_weights: torch.Tensor,
    attn_norm: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    ffn_norm: torch.Tensor,
    device,
    *,
    o_proj_dtype: ttnn.DataType = ttnn.bfloat8_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse o_proj, gate_mm, and RMSNorm weights into one overlapped buffer."""
    cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC

    return overlap_tensors(
        [
            [
                OverlapEntry(
                    "o_proj",
                    o_proj_weights,
                    replace(cfg.o_proj, raw_tensor_shape=tuple(o_proj_weights.shape), dtype=o_proj_dtype),
                )
            ],
            [OverlapEntry("gate_mm", gate_mm_weights, cfg.gate_mm)],
            [
                OverlapEntry("attn_norm", attn_norm, cfg.attn_norm),
                OverlapEntry("q_norm", q_norm, cfg.q_norm),
                OverlapEntry("ffn_norm", ffn_norm, cfg.ffn_norm),
            ],
            [OverlapEntry("kv_norm", kv_norm, cfg.kv_norm)],
        ],
        device=device,
        move_to_device=move_to_device,
    )


def fuse_kv_b12(
    kv_b1_proj_weights: torch.Tensor,
    kv_b2_proj_weights: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse kv_b1 and kv_b2 projection weights into one overlapped buffer."""
    cfg = KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    mla_tp, _ = _tp_factors(device)
    preprocessed = preprocess_kv_b12(kv_b1_proj_weights, kv_b2_proj_weights, mla_tp)
    kv_b1 = preprocessed["kv_b1_proj"]
    kv_b2_pre = preprocessed["kv_b2_proj"]

    return overlap_tensors(
        [
            [
                OverlapEntry(
                    "kv_b1_proj",
                    kv_b1,
                    replace(cfg.kv_b1_shard_spec, raw_tensor_shape=tuple(kv_b1.shape), dtype=dtype),
                ),
            ],
            [
                OverlapEntry(
                    "kv_b2_proj",
                    kv_b2_pre,
                    replace(
                        cfg.kv_b2_shard_spec,
                        raw_tensor_shape=tuple(kv_b2_pre.shape),
                        dtype=dtype,
                        logical_tensor_shape=cfg.kv_b2_proj_shape,
                    ),
                ),
            ],
        ],
        device=device,
        move_to_device=move_to_device,
    )


def fuse_gate_up(
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat4_b,
    move_to_device: bool = True,
) -> tuple[OverlappedTensor, OverlappedTensor, ttnn.Tensor]:
    """Fuse shared-expert gate/up projections and prepare down projection."""
    _, moe_tp = _tp_factors(device)
    cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC

    mesh_rows = device.shape[0] if device.get_num_devices() > 1 else 1
    mesh_cols = device.shape[1] if device.get_num_devices() > 1 else 1

    preprocessed = preprocess_gate_up(gate_proj_weights, up_proj_weights, moe_tp, mesh_rows, mesh_cols)
    gate_preprocessed = preprocessed["shared_gate_proj"]
    up_preprocessed = preprocessed["shared_up_proj"]

    gate_up_dict = overlap_tensors(
        [
            [
                OverlapEntry(
                    "gate_proj",
                    gate_preprocessed,
                    replace(
                        cfg.gate_shard_spec,
                        raw_tensor_shape=tuple(gate_preprocessed.shape),
                        dtype=dtype,
                        logical_tensor_shape=cfg.gate_proj_shape,
                    ),
                ),
            ],
            [
                OverlapEntry(
                    "up_proj",
                    up_preprocessed,
                    replace(
                        cfg.up_shard_spec,
                        raw_tensor_shape=tuple(up_preprocessed.shape),
                        dtype=dtype,
                        logical_tensor_shape=cfg.up_proj_shape,
                    ),
                ),
            ],
        ],
        device=device,
        move_to_device=move_to_device,
    )
    gate_ov = gate_up_dict["gate_proj"]
    up_ov = gate_up_dict["up_proj"]

    dp_spec = DOWN_PROJ_SINGLE_DEVICE_SPEC
    K_down_per_device = 256
    N_per_core = 64
    N_down = N_per_core * dp_spec.NUM_MATMUL_CORES
    matmul_core_grid = dp_spec.build_matmul_core_grid()

    if moe_tp == 1:
        dp_combined = down_proj_weights
        dp_mapper = ttnn.ReplicateTensorToMesh(device)
    else:
        dp_combined = (
            down_proj_weights.reshape(mesh_rows, mesh_cols, K_down_per_device, N_down)
            .permute(0, 2, 1, 3)
            .reshape(mesh_rows * K_down_per_device, mesh_cols * N_down)
        )
        dp_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=(mesh_rows, mesh_cols), dims=(0, 1))
    device_dp = device if move_to_device else None

    dp_shard_spec = ttnn.ShardSpec(matmul_core_grid, (K_down_per_device, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    dp_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, dp_shard_spec)

    down_tensor = ttnn.from_torch(
        dp_combined,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device_dp,
        memory_config=dp_mem,
        tile=ttnn.Tile([32, 32]),
        mesh_mapper=dp_mapper,
    )

    return gate_ov, up_ov, down_tensor


def create_moe_routed_expert_tensors(
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat4_b,
    move_to_device: bool = True,
) -> tuple[list[ttnn.Tensor], list[ttnn.Tensor], list[ttnn.Tensor]]:
    """Upload routed MoE expert gate/up/down weights as per-expert DRAM-sharded tensors."""
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

            w_shuffled = shuffle_dram_tiles(w.unsqueeze(0), tile_w, num_banks)
            w_shuffled = w_shuffled.reshape(1, 1, K, N_padded)

            tensors.append(
                ttnn.from_torch(
                    w_shuffled.contiguous(),
                    dtype=dtype,
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
