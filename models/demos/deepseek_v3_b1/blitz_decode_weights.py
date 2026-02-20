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
class QAB_KVA_PROJ_OverlapConfig:
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

    def shuffle_kv_a(self, weights: torch.Tensor) -> torch.Tensor:
        """Reorder kv_a_proj shards for the KV cache branch core layout."""
        kv_h, kv_w = weights.shape
        kv_num_cores = self.kv_core_range_set.num_cores()
        shards = weights.reshape(kv_h, kv_num_cores, kv_w // kv_num_cores)
        return shards[:, list(self.kv_a_proj_shard_order), :].reshape(kv_h, kv_w)


QAB_KVA_PROJ_OVERLAP_CFG = QAB_KVA_PROJ_OverlapConfig()


@dataclass(frozen=True)
class O_PROJ_GATE_MM_RMSNORM_GAMMA_OverlapConfig:
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


O_PROJ_GATE_MM_RMSNORM_GAMMA_OVERLAP_CFG = O_PROJ_GATE_MM_RMSNORM_GAMMA_OverlapConfig()


@dataclass(frozen=True)
class KVB12_PROJ_OverlapConfig:
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


KVB12_PROJ_OVERLAP_CFG = KVB12_PROJ_OverlapConfig()


@dataclass(frozen=True)
class GATE_UP_PROJ_OverlapConfig:
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

    # Core range sets — gate (A) and up (B) grids
    gate_core_range_set: ttnn.CoreRangeSet = field(
        default_factory=lambda: ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3)),  # 16 cores
                ttnn.CoreRange(ttnn.CoreCoord(7, 0), ttnn.CoreCoord(9, 3)),  # 12 cores
                ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(2, 9)),  # 18 cores
                ttnn.CoreRange(ttnn.CoreCoord(7, 4), ttnn.CoreCoord(9, 9)),  # 18 cores
            }
        )
    )
    up_core_range_set: ttnn.CoreRangeSet = field(
        default_factory=lambda: ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(6, 3)),  # 12 cores
                ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(12, 3)),  # 12 cores
                ttnn.CoreRange(ttnn.CoreCoord(3, 4), ttnn.CoreCoord(6, 9)),  # 24 cores
                ttnn.CoreRange(ttnn.CoreCoord(10, 4), ttnn.CoreCoord(12, 7)),  # 12 cores
                ttnn.CoreRange(ttnn.CoreCoord(10, 8), ttnn.CoreCoord(11, 9)),  # 4 cores
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

    def reshuffle_block_to_height_sharded(self, weights: torch.Tensor) -> torch.Tensor:
        """Reorder a (K, N) weight matrix into stacked HEIGHT_SHARDED form.

        ``(K, N) -> (k_parallel, sh, n_parallel, sw)
        -> permute (k_parallel, n_parallel, sh, sw) -> reshape (-1, sw)``

        Returns:
            Tensor of shape :attr:`stacked_shape`.
        """
        sh, sw = self.shard_shape
        return (
            weights.reshape(self.k_parallel, sh, self.n_parallel, sw).permute(0, 2, 1, 3).reshape(-1, sw).contiguous()
        )


GATE_UP_PROJ_OVERLAP_CFG = GATE_UP_PROJ_OverlapConfig()


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tt_q_ab_proj_and_kv_a_proj_weights(
        self,
        q_a_proj_weights: torch.Tensor,
        q_b_proj_weights: torch.Tensor,
        kv_a_proj_weights: torch.Tensor,
        cfg: QAB_KVA_PROJ_OverlapConfig | None = None,
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
        TP devices (width ``tp * per_device_width``).  Per-TP slices are
        shuffled independently and stitched with the (replicated)
        q_a_proj into separate per-TP fused tensors, which are
        concatenated along width and distributed via
        ``ShardTensor2dMesh`` across mesh columns.  q_a_proj and
        kv_a_proj are replicated on every device.

        TP is inferred from the device topology: single-device -> TP=1,
        4x2 mesh -> TP=2.

        Args:
            q_a_proj_weights: Raw q_a_proj tensor, shape ``(7168, 1536)``.
            q_b_proj_weights: Raw (unshuffled) q_b_proj tensor, shape
                ``(1536, 12288 * tp)``.
            kv_a_proj_weights: Raw kv_a_proj tensor, shape ``(7168, 576)``.
            cfg: Overlap configuration.  Defaults to the module-level
                ``QAB_KVA_PROJ_OVERLAP_CFG`` singleton.

        Returns:
            A list of three :class:`OverlappedTensor` views
            ``[q_a_proj, q_b_proj, kv_a_proj]`` that share the same
            underlying fused device buffer.
        """
        if cfg is None:
            cfg = QAB_KVA_PROJ_OVERLAP_CFG

        # -- Infer TP from device topology --------------------------------
        num_devices = self._device.get_num_devices()
        if num_devices == 1:
            tp = 1
        else:
            mesh_shape = (self._device.shape[0], self._device.shape[1])
            assert mesh_shape == (
                4,
                2,
            ), f"Only single-device or 4x2 mesh supported, got {mesh_shape[0]}x{mesh_shape[1]}"
            tp = mesh_shape[1]

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
        expected_q_b_shape = (cfg.q_b_proj_shape[0], cfg.q_b_proj_shape[1] * tp)
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
        per_device_q_b_w = cfg.q_b_proj_shape[1]
        per_tp_combined = []
        for tp_idx in range(tp):
            q_b_slice = q_b_proj_weights[:, tp_idx * per_device_q_b_w : (tp_idx + 1) * per_device_q_b_w]
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

        combined = torch.cat(per_tp_combined, dim=1) if tp > 1 else per_tp_combined[0]

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

        if tp == 1:
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
            o_proj_weights:     Raw o_proj tensor, shape (8192, 7168).
            gate_mm_weights:    Raw gate_mm tensor, shape (7168, 256).
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
        cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_OVERLAP_CFG

        # -- Validate shapes --------------------------------------------
        assert (
            o_proj_weights.shape == cfg.o_proj_shape
        ), f"o_proj must be {cfg.o_proj_shape}, got {tuple(o_proj_weights.shape)}"
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

        # -- Pack shards ------------------------------------------------
        packed = bytearray()

        # o_proj shards (BFP8_b) — 112 cores
        for i in range(o_num_cores):
            shard_data = o_proj_weights[:, i * o_shard_w : (i + 1) * o_shard_w].contiguous()
            shard_raw = BlitzDecodeWeights._tilize_and_pack_bfp8(shard_data, cfg.tile_h, cfg.tile_w)
            assert len(shard_raw) == cfg.o_proj_shard_bytes
            packed.extend(shard_raw)
            packed.extend(b"\x00" * (max_shard_bytes - cfg.o_proj_shard_bytes))

        # gate_mm shards (bfloat16) — 8 cores
        for i in range(g_num_cores):
            shard_data = gate_mm_weights[:, i * g_shard_w : (i + 1) * g_shard_w].contiguous()
            shard_raw = BlitzDecodeWeights._tilize_and_pack_bfloat16(shard_data, cfg.tile_h, cfg.tile_w)
            assert len(shard_raw) == cfg.gate_mm_shard_bytes
            packed.extend(shard_raw)
            packed.extend(b"\x00" * (max_shard_bytes - cfg.gate_mm_shard_bytes))

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
        packed.extend(gamma_shard)

        # kv_norm core (0, 8) — dedicated core
        kv_norm_shard = bytearray(max_shard_bytes)
        kv_norm_raw = BlitzDecodeWeights._pack_bfloat16_1x32(kv_norm)
        assert len(kv_norm_raw) == cfg.kv_norm_bytes
        kv_norm_shard[: len(kv_norm_raw)] = kv_norm_raw
        packed.extend(kv_norm_shard)

        # -- Build UINT32 tensor on device ------------------------------
        total_cores = o_num_cores + g_num_cores + 2  # +1 gamma core, +1 kv_norm core
        uint32_per_shard = max_shard_bytes // 4

        raw_data = torch.frombuffer(bytes(packed), dtype=torch.int32).clone()

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

        fused = ttnn.from_torch(
            raw_data.reshape(1, uint32_per_shard * total_cores),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self._device,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._device),
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
            kv_b1_proj_weights: shape (8192, 512).
            kv_b2_proj_weights: shape (512, 8192).

        Returns:
            ``[kv_b1_proj, kv_b2_proj]`` as :class:`OverlappedTensor`
            views sharing the same fused device buffer.
        """
        cfg = KVB12_PROJ_OVERLAP_CFG

        assert (
            tuple(kv_b1_proj_weights.shape) == cfg.kv_b1_proj_shape
        ), f"kv_b1 expected {cfg.kv_b1_proj_shape}, got {tuple(kv_b1_proj_weights.shape)}"
        assert (
            tuple(kv_b2_proj_weights.shape) == cfg.kv_b2_proj_shape
        ), f"kv_b2 expected {cfg.kv_b2_proj_shape}, got {tuple(kv_b2_proj_weights.shape)}"

        kv_b2_physical = cfg.shuffle_kv_b2(kv_b2_proj_weights)
        combined = torch.cat([kv_b1_proj_weights, kv_b2_physical], dim=0)

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

        fused = ttnn.from_torch(
            combined,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._device),
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

    def get_tt_gate_up_proj_weights(
        self,
        gate_proj_weights: torch.Tensor,
        up_proj_weights: torch.Tensor,
    ) -> list[OverlappedTensor]:
        """Fuse gate_proj and up_proj into one HEIGHT_SHARDED tensor.

        Both tensors are block-sharded: the K dimension is split among
        ``k_parallel`` partitions and the N dimension among ``n_parallel``
        partitions, yielding 64 shards of ``(896, 32)`` per weight.

        Gate weights live on the A compute cores, up weights on the B
        compute cores of the shared-expert dual-matmul layout.  Both are
        BFP4 with identical shard shapes, so ``ttnn.from_torch`` handles
        the conversion directly (no raw byte packing needed).

        Layout::

            -- gate region: 64 A cores (non-rectangular) --
            gate_proj (7168, 256) as bfloat4_b, block-sharded
              stacked (57344, 32), shard (896, 32)

            -- up region: 64 B cores (non-rectangular) --
            up_proj (7168, 256) as bfloat4_b, block-sharded
              stacked (57344, 32), shard (896, 32)

            combined: 128 cores, HEIGHT_SHARDED (114688, 32)

        Args:
            gate_proj_weights: Raw gate tensor, shape (7168, 256).
            up_proj_weights:   Raw up tensor, shape (7168, 256).

        Returns:
            A list of two :class:`OverlappedTensor` views
            ``[gate_proj, up_proj]`` that share the same underlying fused
            device buffer.
        """
        cfg = GATE_UP_PROJ_OVERLAP_CFG

        # -- Validate shapes --------------------------------------------
        assert (
            gate_proj_weights.shape == cfg.gate_proj_shape
        ), f"gate_proj must be {cfg.gate_proj_shape}, got {tuple(gate_proj_weights.shape)}"
        assert (
            up_proj_weights.shape == cfg.up_proj_shape
        ), f"up_proj must be {cfg.up_proj_shape}, got {tuple(up_proj_weights.shape)}"

        # -- Stack block-shards and concatenate -------------------------
        gate_stacked = cfg.reshuffle_block_to_height_sharded(gate_proj_weights)
        up_stacked = cfg.reshuffle_block_to_height_sharded(up_proj_weights)
        combined = torch.cat([gate_stacked, up_stacked], dim=0)

        # -- Place on device as HEIGHT_SHARDED --------------------------
        combined_crs = ttnn.CoreRangeSet(list(cfg.gate_core_range_set.ranges()) + list(cfg.up_core_range_set.ranges()))
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

        fused = ttnn.from_torch(
            combined,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._device),
        )

        # -- Build OverlappedTensor views --------------------------------
        tile = fused.get_tile()
        ts = tuple(tile.tile_shape)
        stacked = cfg.stacked_shape

        return [
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=stacked,
                shard_shape=cfg.shard_shape,
                core_range_set=cfg.gate_core_range_set,
                dtype=ttnn.bfloat4_b,
                tile_shape=ts,
                byte_offset=0,
            ),
            OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=stacked,
                shard_shape=cfg.shard_shape,
                core_range_set=cfg.up_core_range_set,
                dtype=ttnn.bfloat4_b,
                tile_shape=ts,
                byte_offset=0,
            ),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
