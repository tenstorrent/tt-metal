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
    imbalance.

    MORE CORES IS NOT BETTER, and for a K-heavy shape it is much worse: widening
    the shard shortens `in0_block_w` and `per_core_N` together, until each core
    is walking a long chain of tiny k-blocks to produce a single output tile with
    nothing to overlap the reduction against. down_proj (9216->3072) measured
    0.1589 ms at 96 cores and 0.0934 ms at 24. So `max_cores` is not a
    workaround -- callers are expected to SWEEP it per shape and pin what won.
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

    @staticmethod
    def is_decode_row(x) -> bool:
        """One tile row, single batch -- the decode step's activation shape.

        Says nothing about the width, so a caller can ask it about a tensor
        UPSTREAM of this matmul (e.g. the block input that will become its
        operand) to decide whether the decode path applies.
        """
        shape = list(x.padded_shape)
        if int(shape[-2]) != TILE:
            return False
        batch = 1
        for dim in shape[:-2]:
            batch *= int(dim)
        return batch == 1

    def matches(self, x) -> bool:
        """True only for the ONE-tile-row decode activation of exactly this K.

        Prefill has many rows and is compute-bound, so it keeps the default
        routing; `per_core_M=1` above is only valid for the decode row anyway.
        """
        return int(list(x.padded_shape)[-1]) == self.k and self.is_decode_row(x)

    def shares_input_with(self, other) -> bool:
        """True when `other` consumes the SAME sharded activation as this plan.

        Several projections read one activation (q/k/v all read the block's
        normalized hidden state). When their plans resolve to the same K and the
        same core grid their input shards are identical, so the caller can
        reshard ONCE and hand the result to all of them instead of paying an
        interleaved->sharded conversion per projection.
        """
        return (
            other is not None
            and other.k == self.k
            and other.core_grid.x == self.core_grid.x
            and other.core_grid.y == self.core_grid.y
        )

    def shard_input(self, x):
        """Open the shard once, for a caller that will run several projections."""
        return ttnn.to_memory_config(x, self.input_memory_config)

    def run_presharded_raw(self, x_sharded, weight, compute_kernel_config=None):
        """The matmul, leaving its result IN the plan's output shard.

        For a consumer that reads a width-sharded operand directly: it then
        reads the result out of the L1 the matmul just wrote it to, and the
        interleaved round trip below never happens.
        """
        kwargs = {"compute_kernel_config": compute_kernel_config} if compute_kernel_config else {}
        return ttnn.linear(
            x_sharded,
            weight,
            program_config=self.program_config,
            memory_config=self.output_memory_config,
            **kwargs,
        )

    def run_presharded(self, x_sharded, weight, compute_kernel_config=None):
        """The matmul itself, on an activation the caller already sharded."""
        return ttnn.sharded_to_interleaved(self.run_presharded_raw(x_sharded, weight, compute_kernel_config))

    def __call__(self, x, weight, compute_kernel_config=None):
        """Run the projection, returning an INTERLEAVED result.

        The caller's next op consumes interleaved, so the shard is opened and
        closed around this matmul; keeping it open across ops is a separate
        (layout-chaining) change.
        """
        return self.run_presharded(self.shard_input(x), weight, compute_kernel_config)


def build_plan(device, k: int, n: int, max_cores: int = 0):
    """A plan, or None if this shape has no even full-grid split."""
    try:
        return DecodeMatmulPlan(device, k, n, max_cores)
    except Exception:  # noqa: BLE001 - fall back to plain ttnn.linear
        return None
