# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""L1-resident weight placement for two decoder layers per chip.

The single source of truth for WEIGHT_PLACEMENT.md: which contiguous core range
("zone") every non-expert decode weight lives in, how it is sharded there, and the
resulting per-core L1 budget. Both resident layers stack their copy of each weight
on the same zone with the same shard shape, so every layer shares one set of
program / memory configs, exactly like ``DECODE_LAYOUTS`` does for the streamed
path today.

Pure host-side data plus arithmetic -- importable without a device or ttnn. The
ttnn conversions (``core_range_set`` / ``memory_config``) import ttnn lazily so a
future loader can call them directly::

    from .l1_placement import placement_for, core_range_set, memory_config

    spec = placement_for("q_b_proj")
    weight = ttnn.as_tensor(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=memory_config(spec))

Run as a script to print the budget table::

    python -m models.experimental.deepseek_v4_flash.tt.l1_placement
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------- #
# Chip geometry and dtype constants
# --------------------------------------------------------------------------- #

GRID_W = 12
GRID_H = 10
NUM_CORES = GRID_W * GRID_H  # 120
L1_PER_CORE = 1536 * 1024  # 1.5 MB

TILE = 32
BF4_TILE_BYTES = 576  # 512 B of 4-bit mantissas + 64 B shared exponents
BF16_TILE_BYTES = TILE * TILE * 2

# Reserved for the bf16 small tensors of ONE layer (norms, sinks, position bias,
# router bias), held on Z0's cores alongside the activations. 16 KB/core covers
# the ~1 MB they tile-pad to per layer.
NORMS_MISC_BYTES_PER_CORE_PER_LAYER = 16 * 1024

# Number of decoder layers resident on one chip. The zone loads below are sized
# for exactly this; a third layer does not fit (3 x ~76 MB > 180 MB).
RESIDENT_LAYERS = 2

# --------------------------------------------------------------------------- #
# Zones: contiguous runs of row-major core index. 120 = 64 + 32 + 16 + 8, the
# only way to cut the chip so every (power-of-two-sized) weight gets a
# contiguous range. Z0 is the single 64-wide zone the chip admits: any two
# 64-core windows in 120 cores overlap, so there can be only one.
# --------------------------------------------------------------------------- #

ZONES: dict[str, tuple[int, int]] = {
    # name: (first row-major core index, number of cores)
    "Z0": (0, 64),
    "Z1": (64, 32),
    "Z2": (96, 16),
    "Z3": (112, 8),
}


def zone_cores(zone: str) -> list[tuple[int, int]]:
    """The zone's cores as (x, y) grid coordinates, in row-major (slab) order."""
    start, count = ZONES[zone]
    return [(c % GRID_W, c // GRID_W) for c in range(start, start + count)]


def zone_core_ranges(zone: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """The zone as maximal rectangular ranges ``((x0, y0), (x1, y1))``.

    A contiguous row-major run decomposes into at most three rectangles: a
    partial first row, a block of full rows, and a partial last row.
    """
    start, count = ZONES[zone]
    end = start + count - 1  # inclusive
    ranges = []
    c = start
    while c <= end:
        x, y = c % GRID_W, c // GRID_W
        row_end = min(end, y * GRID_W + GRID_W - 1)
        if x == 0 and row_end == y * GRID_W + GRID_W - 1:
            # Full rows from here: emit them as one rectangle.
            last_full_y = (end + 1) // GRID_W - 1
            ranges.append(((0, y), (GRID_W - 1, last_full_y)))
            c = (last_full_y + 1) * GRID_W
        else:
            ranges.append(((x, y), (row_end % GRID_W, y)))
            c = row_end + 1
    return ranges


# --------------------------------------------------------------------------- #
# Placements
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WeightPlacement:
    """Where one weight lives and how it is cut, in decode_weight_layout terms.

    Exactly one of the three layouts applies:
    * full width-sharded: ``n_blocks`` set, ``k_blocks``/``batch`` unset -- shard
      ``[K, N / n_blocks]`` on ``n_blocks`` cores;
    * partial width-sharded: ``k_blocks`` and ``n_blocks`` set -- shard
      ``[K / k_blocks, N / n_blocks]`` on ``k_blocks * n_blocks`` cores, K-partials
      reduced onto the first ``n_blocks`` cores;
    * batched (BatchedLinearDecode): ``batch`` and ``b_blocks`` set -- shard
      ``[(batch / b_blocks) * K, N / n_blocks]`` on ``b_blocks * n_blocks`` cores.
    """

    name: str
    zone: str
    K: int
    N: int
    n_blocks: int
    k_blocks: Optional[int] = None
    batch: Optional[int] = None
    b_blocks: Optional[int] = None
    dtype: str = "bfloat4_b"

    @property
    def num_cores(self) -> int:
        if self.batch is not None:
            return self.b_blocks * self.n_blocks
        if self.k_blocks is not None:
            return self.k_blocks * self.n_blocks
        return self.n_blocks

    @property
    def shard_shape(self) -> tuple[int, int]:
        if self.batch is not None:
            return (self.batch // self.b_blocks) * self.K, self.N // self.n_blocks
        if self.k_blocks is not None:
            return self.K // self.k_blocks, self.N // self.n_blocks
        return self.K, self.N // self.n_blocks

    @property
    def per_core_bytes(self) -> int:
        rows, cols = self.shard_shape
        assert rows % TILE == 0 and cols % TILE == 0, f"{self.name}: shard {self.shard_shape} not tile-aligned"
        tile_bytes = BF4_TILE_BYTES if self.dtype == "bfloat4_b" else BF16_TILE_BYTES
        return (rows // TILE) * (cols // TILE) * tile_bytes

    @property
    def total_bytes(self) -> int:
        return self.per_core_bytes * self.num_cores

    def __post_init__(self):
        zone_count = ZONES[self.zone][1]
        if self.num_cores != zone_count:
            raise ValueError(f"{self.name}: {self.num_cores} cores does not fill zone {self.zone} ({zone_count})")


def _p(*args, **kwargs) -> WeightPlacement:
    return WeightPlacement(*args, **kwargs)


# Weights common to every layer type. Keys follow DECODE_LAYOUTS / the checkpoint
# names so a loader can look them up the same way it does today.
_COMMON: dict[str, WeightPlacement] = {
    p.name: p
    for p in [
        # -- Z0: the 64-wide zone. Two of the three 18 MB projections live here
        #    (all three would be 3 x 2 x 288 KB = 1728 KB/core with two layers).
        _p("q_b_proj", "Z0", K=1024, N=32768, n_blocks=64),
        _p("o_b_proj", "Z0", K=8192, N=4096, n_blocks=64),
        _p("q_a_proj", "Z0", K=4096, N=1024, k_blocks=2, n_blocks=32),
        # -- Z1: the demoted big pair (batched grouped output projection) runs
        #    32-wide for both layers.
        _p("o_a_proj", "Z1", K=4096, N=1024, batch=8, b_blocks=8, n_blocks=4),
        _p("attn_hc.fn", "Z1", K=16384, N=32, k_blocks=32, n_blocks=1),
        _p("ffn_hc.fn", "Z1", K=16384, N=32, k_blocks=32, n_blocks=1),
        # -- Z2
        _p("shared_gate_proj", "Z2", K=4096, N=2048, n_blocks=16),
        _p("shared_up_proj", "Z2", K=4096, N=2048, n_blocks=16),
        _p("kv_proj", "Z2", K=4096, N=512, n_blocks=16),
        # -- Z3
        _p("shared_down_proj", "Z3", K=2048, N=4096, n_blocks=8),
        _p("router_gate", "Z3", K=4096, N=256, n_blocks=8),
    ]
}

# Compressor projections differ by layer type (HCA projects to Dh, CSA to 2*Dh),
# mirroring the layer-type-keyed entries in DECODE_LAYOUTS.
_COMPRESSOR: dict[str, dict[str, WeightPlacement]] = {
    "compressed_sparse_attention": {
        "compressor.gate_proj": _p("compressor.gate_proj", "Z0", K=4096, N=1024, k_blocks=2, n_blocks=32),
        "compressor.kv_proj": _p("compressor.kv_proj", "Z1", K=4096, N=1024, n_blocks=32),
    },
    "heavily_compressed_attention": {
        "compressor.gate_proj": _p("compressor.gate_proj", "Z0", K=4096, N=512, k_blocks=4, n_blocks=16),
        "compressor.kv_proj": _p("compressor.kv_proj", "Z1", K=4096, N=512, k_blocks=2, n_blocks=16),
    },
    "sliding_attention": {},
}


def placements_for_layer(layer_type: str = "compressed_sparse_attention") -> dict[str, WeightPlacement]:
    """Every resident weight of one layer of ``layer_type``, keyed by name."""
    return {**_COMMON, **_COMPRESSOR[layer_type]}


def placement_for(name: str, layer_type: str = "compressed_sparse_attention") -> WeightPlacement:
    return placements_for_layer(layer_type)[name]


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


def budget_report(layer_types: tuple[str, ...] = ("heavily_compressed_attention", "compressed_sparse_attention")):
    """Per-zone L1 load for the given resident layers.

    Returns ``{zone: {"cores", "bytes", "per_core_bytes", "free_bytes"}}`` and
    raises if any core would exceed its 1.5 MB.
    """
    if len(layer_types) != RESIDENT_LAYERS:
        raise ValueError(f"this placement is sized for {RESIDENT_LAYERS} resident layers, got {len(layer_types)}")
    load = {zone: 0 for zone in ZONES}
    for lt in layer_types:
        for spec in placements_for_layer(lt).values():
            load[spec.zone] += spec.per_core_bytes
        load["Z0"] += NORMS_MISC_BYTES_PER_CORE_PER_LAYER
    report = {}
    for zone, (start, cores) in ZONES.items():
        per_core = load[zone]
        if per_core > L1_PER_CORE:
            raise ValueError(f"zone {zone} needs {per_core} B/core, over the {L1_PER_CORE} B L1")
        report[zone] = {
            "cores": cores,
            "bytes": per_core * cores,
            "per_core_bytes": per_core,
            "free_bytes": L1_PER_CORE - per_core,
        }
    return report


# --------------------------------------------------------------------------- #
# ttnn conversions (lazy import so the module stays host-only importable)
# --------------------------------------------------------------------------- #


def core_range_set(zone: str):
    """The zone as a ``ttnn.CoreRangeSet`` (<= 3 rectangles)."""
    import ttnn

    return ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))
            for (x0, y0), (x1, y1) in zone_core_ranges(zone)
        ]
    )


def memory_config(spec: WeightPlacement):
    """Width-sharded L1 ``ttnn.MemoryConfig`` placing ``spec`` in its zone."""
    import ttnn

    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set(spec.zone), list(spec.shard_shape), ttnn.ShardOrientation.ROW_MAJOR),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    pairs = {
        "HCA + CSA (typical)": ("heavily_compressed_attention", "compressed_sparse_attention"),
        "CSA + CSA (worst)": ("compressed_sparse_attention", "compressed_sparse_attention"),
        "sliding + CSA": ("sliding_attention", "compressed_sparse_attention"),
    }
    for label, kinds in pairs.items():
        rep = budget_report(kinds)
        total = sum(z["bytes"] for z in rep.values())
        print(f"\n== {label}: {total / 2**20:.2f} MB of {NUM_CORES * L1_PER_CORE / 2**20:.0f} MB")
        for zone, z in rep.items():
            ranges = ", ".join(f"({x0},{y0})-({x1},{y1})" for (x0, y0), (x1, y1) in zone_core_ranges(zone))
            print(
                f"  {zone}: {z['cores']:>3} cores  {z['per_core_bytes'] / 1024:>7.1f} KB/core  "
                f"free {z['free_bytes'] / 1024:>6.1f} KB   [{ranges}]"
            )

    print("\n== placements (CSA layer)")
    for name, spec in sorted(placements_for_layer().items(), key=lambda kv: (kv[1].zone, -kv[1].per_core_bytes)):
        print(
            f"  {spec.zone}  {name:<22} [{spec.K}, {spec.N}]"
            f"{' x' + str(spec.batch) if spec.batch else '':<4} -> shard {list(spec.shard_shape)}"
            f" on {spec.num_cores} cores, {spec.per_core_bytes / 1024:.0f} KB/core"
        )
