# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V3 per-device overlap spec dataclasses and singleton instances.

Each ``*_SingleDeviceOverlapSpec`` bundles the :class:`OverlappedTensorSpec`
fields, weight-shuffle methods, and a :meth:`fusion_group_spec` factory for
one fusion group.  The singleton instances (e.g.
``QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC``) are the single source of truth
consumed by the transform functions (:mod:`~weights.transforms`) and the
cache fingerprinting / packing system.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

import ttnn
from models.demos.deepseek_v3_b1.weights.cache.types import (
    FusionGroupSpec,
    MeshMapperConfig,
    RegionSpec,
    ReplicateMeshMapper,
    Shard2dMeshMapper,
)
from models.demos.deepseek_v3_b1.weights.overlap.spec import OverlappedTensorSpec


def _infer_mesh_mapper(
    lanes: list[list[tuple[str, OverlappedTensorSpec]]],
) -> MeshMapperConfig:
    """Derive the mesh mapper config from ``tp_dim`` values across all specs."""
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
    transform_version: int = 0,
) -> FusionGroupSpec:
    """Derive a :class:`FusionGroupSpec` from named :class:`OverlappedTensorSpec` fields."""
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
        transform_version=transform_version,
    )


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
    ``prepare.deinterleave_q_b_proj`` to convert HF weights first.
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

    transform_version: int = 1  # bump when shuffle/preprocess logic in this class changes

    num_qnope_heads: int = 64
    num_qrope_heads: int = 64
    qnope_head_dim: int = 128
    qrope_head_dim: int = 64
    heads_per_row: int = 8
    q_a_proj_shape: tuple[int, int] = (7168, 1536)
    q_b_proj_shape: tuple[int, int] = (1536, 12288)
    kv_a_proj_shape: tuple[int, int] = (7168, 576)

    q_a_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7))}),
            raw_tensor_shape=(3584, 3072),
            dtype=ttnn.DataType.BFLOAT4_B,
        )
    )
    q_b_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7))}),
            raw_tensor_shape=(1536, 12288),
            dtype=ttnn.DataType.BFLOAT4_B,
            tp_dim=(None, 1),
        )
    )
    kv_a_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(8, 9))}),
            raw_tensor_shape=(7168, 576),
            dtype=ttnn.DataType.BFLOAT4_B,
        )
    )

    kv_a_proj_shard_order: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 16, 8, 9, 10, 11, 12, 13, 14, 15, 17)

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
        """Extract the per-device q_b_proj slice for a given TP index."""
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

    def fusion_group_spec(self) -> FusionGroupSpec:
        """Build the ``q_ab_kv_a`` :class:`FusionGroupSpec` from this config."""
        return _build_fusion_group_spec(
            "q_ab_kv_a",
            [
                [("q_a_proj", self.q_a_shard_spec), ("q_b_proj", self.q_b_shard_spec)],
                [("kv_a_proj", self.kv_a_shard_spec)],
            ],
            sharding_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            transform_version=self.transform_version,
        )


QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC = QAB_KVA_PROJ_SingleDeviceOverlapSpec()


_GAMMA_CORE = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(12, 9), ttnn.CoreCoord(12, 9))])
_KV_NORM_CORE = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(0, 8))])


@dataclass(frozen=True)
class O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec:
    """Configuration for the o_proj / gate_mm / rmsnorm-gamma weight overlap.

    Fuses the following into a single WIDTH_SHARDED raw buffer:

    * **o_proj** — BFP4, base logical shape (8192, 7168), TP along mesh
      columns (tp_dim=(None, 0)), WIDTH_SHARDED on 112 cores.
    * **gate_mm** — BFP16, (7168, 256), WIDTH_SHARDED on 8 cores.
    * **attn_norm** — BFP16, (1, 7168), on core (12, 9).
    * **q_norm** — BFP16, (1, 1536), on core (12, 9).
    * **kv_norm** — BFP16, (1, 512), on core (0, 8).
    * **ffn_norm** — BFP16, (1, 7168), on core (12, 9).
    """

    transform_version: int = 1  # bump when shuffle/preprocess logic in this class changes

    o_proj: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 8), ttnn.CoreCoord(8, 9)),
                }
            ),
            raw_tensor_shape=(8192, 7168),
            dtype=ttnn.DataType.BFLOAT4_B,
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

    def fusion_group_spec(self) -> FusionGroupSpec:
        """Build the ``o_proj_gate_mm_norms`` :class:`FusionGroupSpec` from this config."""
        return _build_fusion_group_spec(
            "o_proj_gate_mm_norms",
            [
                [("o_proj", self.o_proj)],
                [("gate_mm", self.gate_mm)],
                [("attn_norm", self.attn_norm), ("q_norm", self.q_norm), ("ffn_norm", self.ffn_norm)],
                [("kv_norm", self.kv_norm)],
            ],
            sharding_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            mesh_mapper_config=Shard2dMeshMapper(dims=(None, 1)),
            transform_version=self.transform_version,
        )


O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC = O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec()


@dataclass(frozen=True)
class KVB12_PROJ_SingleDeviceOverlapSpec:
    """Configuration for the kv_b1 / kv_b2 weight overlap."""

    transform_version: int = 1  # bump when shuffle/preprocess logic in this class changes

    kv_b1_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
            raw_tensor_shape=(8192, 512),
            dtype=ttnn.DataType.BFLOAT4_B,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            tp_dim=(None, 0),
        )
    )
    kv_b2_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(12, 7)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(11, 9)),
                }
            ),
            raw_tensor_shape=(512, 8192),
            dtype=ttnn.DataType.BFLOAT4_B,
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

    def shuffle_kv_b2(self, weights: torch.Tensor) -> torch.Tensor:
        """Tile-level rearrange (512, 8192) into (8192, 512) for HEIGHT_SHARDED."""
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

    def fusion_group_spec(self) -> FusionGroupSpec:
        """Build the ``kv_b12`` :class:`FusionGroupSpec` from this config."""
        return _build_fusion_group_spec(
            "kv_b12",
            [
                [("kv_b1_proj", self.kv_b1_shard_spec)],
                [("kv_b2_proj", self.kv_b2_shard_spec)],
            ],
            sharding_strategy=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            transform_version=self.transform_version,
        )


KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC = KVB12_PROJ_SingleDeviceOverlapSpec()


@dataclass(frozen=True)
class GATE_UP_PROJ_SingleDeviceOverlapSpec:
    """Configuration for the gate / up projection weight overlap."""

    transform_version: int = 1  # bump when shuffle/preprocess logic in this class changes

    gate_shard_spec: OverlappedTensorSpec = field(
        default_factory=lambda: OverlappedTensorSpec(
            core_range_set=ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 3)),
                    ttnn.CoreRange(ttnn.CoreCoord(7, 0), ttnn.CoreCoord(9, 9)),
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
                    ttnn.CoreRange(ttnn.CoreCoord(3, 4), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(6, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(11, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(12, 0), ttnn.CoreCoord(12, 7)),
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

    k_parallel: int = 8
    n_parallel: int = 8

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
        """Compute shard permutation from CoreRangeSet enumeration to row-major order."""
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
        """Reorder a (K, N) weight matrix into stacked HEIGHT_SHARDED form."""
        sh, sw = self.shard_shape
        num_shards = self.k_parallel * self.n_parallel
        block_shards = (
            weights.reshape(self.k_parallel, sh, self.n_parallel, sw).permute(0, 2, 1, 3).reshape(num_shards, sh, sw)
        )
        perm = self._crs_shard_permutation(core_range_set)
        return block_shards[list(perm)].reshape(-1, sw).contiguous()

    def fusion_group_spec(self) -> FusionGroupSpec:
        """Build the ``gate_up`` :class:`FusionGroupSpec` from this config."""
        return _build_fusion_group_spec(
            "gate_up",
            [
                [("shared_gate_proj", self.gate_shard_spec)],
                [("shared_up_proj", self.up_shard_spec)],
            ],
            sharding_strategy=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            transform_version=self.transform_version,
        )


GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC = GATE_UP_PROJ_SingleDeviceOverlapSpec()


@dataclass(frozen=True)
class DOWN_PROJ_SingleDeviceSpec:
    """Configuration for the down projection weight layout."""

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
