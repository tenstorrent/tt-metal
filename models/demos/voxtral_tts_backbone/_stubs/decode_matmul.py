# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-grid program configs for the decode-shape projections.

WHY
    A decode projection is `[1, K] x [K, N]`: one row of activation against the
    whole weight, so its cost is the weight read, not the FLOPs. Left to pick
    its own routing, `ttnn.linear` puts these on a PARTIAL grid -- e.g. the
    9216->3072 down-projection ran on 96 cores at 263 GB/s, roughly half the
    device's DRAM bandwidth.

    Occupying the grid here means a 1D multicast along N with the activation
    L1 WIDTH-SHARDED across the same cores (`mcast_in0=True`): each core owns
    `N_tiles / cores` output columns and multicasts its slice of the activation.
    Measured on the down-projection: 0.215 -> 0.157 ms/call (263 -> 361 GB/s).

    This is a COORDINATED change -- the activation's `memory_config` and the
    program config have to agree. A program config alone on a DRAM-interleaved
    activation is inert.

WHERE IT PAYS (measured, per shape)
    Only where the auto router is STARVED -- by a small N (k/v 3072->1024, 32
    output tiles: -28.6%) or a long K (down 9216->3072: -27.1%). On a wide-N
    projection it is a LOSS: gate/up 3072->9216 has 288 output tiles, already
    routes at 369 GB/s, and lost at all seven core counts swept (best +0.9%),
    because the two reshard ops this adds cost more than the routing it fixes.
    So it is applied per shape, not blanket.

WHAT THIS DELIBERATELY DOES NOT DO
    It leaves the WEIGHT interleaved in DRAM. The DRAM-width-sharded variant
    was measured too and is worse here (down_proj -6.4% vs this -27.1%, and
    gate/up +34.6%), and it would additionally force a second copy of every
    weight because that kernel only accepts a one-tile M -- prefill could not
    share it.
"""
from __future__ import annotations

import ttnn

TILE = 32


def _core_split(device, k_tiles: int, n_tiles: int, max_cores: int = 0):
    """Most cores that evenly divide BOTH tile counts and factor into the grid.

    Both must divide evenly: `in0_block_w` slices K per core and `per_core_N`
    slices N, and a remainder on either is a mis-sized shard, not just an
    imbalance. More cores is not automatically better -- see the module note --
    so `max_cores` caps the search for shapes that measured better narrower.
    """
    grid = device.compute_with_storage_grid_size()
    ceiling = min(max_cores, grid.x * grid.y) if max_cores else grid.x * grid.y
    for cores in range(ceiling, 0, -1):
        if k_tiles % cores or n_tiles % cores:
            continue
        for rows in range(1, grid.y + 1):
            if cores % rows == 0 and cores // rows <= grid.x:
                return cores, rows, cores // rows
    return None


class DecodeMatmulPlan:
    """Sharded activation/output memory configs + full-grid program config."""

    def __init__(self, device, k: int, n: int, max_cores: int = 0):
        self.k, self.n = int(k), int(n)
        split = _core_split(device, self.k // TILE, self.n // TILE, max_cores)
        if split is None:
            raise ValueError("no even core split for k=%d n=%d" % (k, n))
        cores, rows, cols = split
        self.cores, self.core_grid = cores, ttnn.CoreGrid(y=rows, x=cols)
        per_core_n = (self.n // TILE) // cores
        # out_subblock_w must DIVIDE per_core_N (and h*w <= 8 with fp32_dest_acc
        # off), so take the largest such divisor -- min(4, per_core_N) is wrong
        # the moment per_core_N is 6.
        subblock_w = next(s for s in (4, 3, 2, 1) if per_core_n % s == 0)

        self.input_memory_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, TILE, self.k), core_grid=self.core_grid, strategy=ttnn.ShardStrategy.WIDTH
        )
        self.output_memory_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, TILE, self.n), core_grid=self.core_grid, strategy=ttnn.ShardStrategy.WIDTH
        )
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(cols, rows),
            in0_block_w=(self.k // TILE) // cores,
            out_subblock_h=1,
            out_subblock_w=subblock_w,
            per_core_M=1,  # the decode row -- one tile
            per_core_N=per_core_n,
            fuse_batch=True,
            mcast_in0=True,
            fused_activation=None,
        )

    def matches(self, x) -> bool:
        """True only for the ONE-tile-row decode activation of exactly this K.

        Prefill has many rows and is compute-bound, so it keeps the default
        routing; `per_core_M=1` above is only valid for the decode row anyway.
        """
        shape = list(x.padded_shape)
        if int(shape[-1]) != self.k or int(shape[-2]) != TILE:
            return False
        batch = 1
        for dim in shape[:-2]:
            batch *= int(dim)
        return batch == 1

    def __call__(self, x, weight, compute_kernel_config=None):
        """Run the projection, returning an INTERLEAVED result.

        The caller's next op consumes interleaved, so the shard is opened and
        closed around this matmul; keeping it open across ops is a separate
        (layout-chaining) change.
        """
        kwargs = {"compute_kernel_config": compute_kernel_config} if compute_kernel_config else {}
        out = ttnn.linear(
            ttnn.to_memory_config(x, self.input_memory_config),
            weight,
            program_config=self.program_config,
            memory_config=self.output_memory_config,
            **kwargs,
        )
        return ttnn.sharded_to_interleaved(out)


def build_plan(device, k: int, n: int, max_cores: int = 0):
    """A plan, or None if this shape has no even full-grid split."""
    try:
        return DecodeMatmulPlan(device, k, n, max_cores)
    except Exception:  # noqa: BLE001 - fall back to plain ttnn.linear
        return None
