# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN composite for Kimi K3 attention residuals (AttnRes).

Mirrors `torch_functional/attn_res.py`; `API_SPEC.md` holds the contract. Two
divergences from the torch API, both forced by ttnn:

  * `block_residual` is 4D `[1, S, N, d]` with candidates on **dim 1**, not the
    last dim. `S+1 <= 9` would tile-pad 9 -> 32 on a last dim, and padding zeros
    would enter a last-dim softmax as `exp(0) = 1`.
  * "no sealed snapshots" is `block_residual=None`, not a zero-width tensor —
    ttnn has no zero-extent dimension. The read is the identity there anyway.

Both forms of the read live here. `forward` is direct. `inter_block` + `merge`
split the mixture so the sealed half amortizes across a whole 12-layer block: the
reciprocal-RMS pass and the dots each collapse to a single pass over the sealed
set for all 24 read sites, which leaves the weighted sum as the only per-site
traffic over it. That one does not batch in composed form — it contracts over the
candidate axis, and reaching it with a matmul means making that axis a tile axis,
whose two permutes over the sealed set cost what the matmul saves
(`bringup_log.md` P8).

Distribution follows `DISTRIBUTION.md`: the stream stays sharded `[1, 1, T/R,
d/C]` exactly as the analog leaves it, the sequence axis communicates nothing,
and each read all-reduces `2(S+1)` scalars per token across the TP axis. Nothing
of width `d` ever crosses a rank boundary.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

HIDDEN_SIZE = 7168
BLOCK_SIZE = 12
EPS = 1e-5

# Both one-pass `d`-reductions are inadmissible without this. At default fidelity
# the bf16 mantissa is truncated going into the multiply, which costs the RMSNorm
# statistics kernel an order of magnitude — 4.8e-2 relative error against the
# 2.4e-3 that `mul` + `sum` achieves — and the matvec 4x. HiFi4 with fp32
# accumulation restores both (2.5e-3 and 3.2e-3) and is free on device, because
# these reductions are bandwidth-bound and the extra math passes hide under the
# reads: 232 µs against 230 at default fidelity. `WormholeComputeKernelConfig` is
# the name ttnn gives this on every non-Grayskull arch, Blackhole included.
STATS_FIDELITY = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)

# `rms_norm_pre_all_gather` keeps a whole row of `v` in one core's L1: its program
# factory sizes the circular buffers at `4 * Wt` tiles — the input double-buffered
# plus a double-buffered `x**2` intermediate — on top of ~113 KiB of scaler and
# output buffers. Measured on Blackhole's 1 572 864 B of L1, it asks for 1 590 144 B
# at a 5 760-wide row, 1 688 448 at 6 144 and 1 950 592 at 7 168, and the steps
# between those are exactly 8 192 B per tile of width, so the ceiling is 177 tiles.
# Past it the op throws at program build rather than degrading, and
# `use_2d_core_grid=True` is not the escape hatch — it splits tokens, not the row,
# and asks for more (1 971 072 B at 7 168). Any `tp_factor >= 2` on a 7 168-wide
# model fits; `tp_factor == 1` does not, so the read falls back there instead of
# making the caller carry the constraint. Wormhole's smaller L1 moves this bound,
# which is why the fallback exists at all rather than an assert.
ONE_PASS_SQUARES_MAX_WIDTH = 177 * ttnn.TILE_SIZE


class TtAttnRes(LightweightModule):
    """The AttnRes read, composed from ttnn primitives.

    Args:
        mesh_device: TTNN mesh device.
        hidden_size: **global** `d`, not the per-device shard width. Every
            reduction runs over this axis, and `mean(v**2)` divides by the global
            `d` even when each device holds `d / tp_factor` of it — that is the
            one place a sharded AttnRes would return quietly wrong numbers
            instead of failing.
        eps: `rms_norm_eps`.
        torch_queries: optional sequence of `[d]` folded queries (see
            `torch_functional.attn_res.fold_query`). Exposed as `self.queries`.
        dtype: device dtype for queries and intermediates.
        sp_axis, tp_axis: mesh axes the sequence and the hidden dim are sharded
            across. `sp_axis` is placement only — the op never communicates on it,
            which is the whole reason the sequence split is free. Both live here
            rather than in the caller so the layout has one definition; `forward`
            checks its input against it.
        num_links: fabric links for the statistics all-reduce.
        topology: one `ttnn.Topology` **per mesh axis**, not a scalar. Galaxy
            prefill is `[LINE, RING]`; applying a scalar `Ring` to a linear axis
            points a collective at a wrap link with no physical fabric edge.
        stats_dtype: dtype the statistics all-reduce runs in. `ttnn.all_reduce`
            reduces in bf16 unless the input is fp32 (`all_reduce_nanobind.cpp:48`).
            The difference is small and it is free: measured over 186 chained reads
            at `d = 7168` on a 2x4 mesh, fp32 stats give PCC 0.9999500 against bf16
            stats' 0.9999401, for ~1.5 MB of extra traffic per read against ~900 MB
            of DRAM traffic. bf16 stats land on the single-device number (0.9999408),
            which is the tell: fp32 here buys back the rounding the single-device
            path also takes, not something sharding introduced.
        fold_stats: fold the candidate axis into the last dim before the
            statistics all-reduce. A `[1, C, N, 1]` tensor tile-pads its 1-wide
            last dim to 32 and the collective bills padded bytes at the payload
            rate, so `[1, 1, N, C]` puts up to 32 candidates inside the column
            the padding already paid for. Phase 9 P4: 348 us -> 47 us on `(2,4)`
            at `C = 18`, and the cost stops scaling with the candidate count.
            Costs two `ttnn.permute` calls, which is why it is off untraced —
            there two extra launches cancel the saving exactly.
        one_pass_stats: take both `d`-reductions in a single pass over `v`, with
            `rms_norm_pre_all_gather` for the sum of squares and a matmul for the
            dot. Written as `mul` then `sum`, each reduction materializes a second
            copy of `[1, C, N, d/tp]` in DRAM and reads it back — three passes to
            produce 0.6% of the op's bytes. Phase 9 P7 on `(2,4)` at `C = 9`:
            782 -> 232 us for the squares, 793 -> 450 for the dot, ~892 us of a
            3 136 us read. Requires `STATS_FIDELITY`; see the note there. The
            squares half is width-gated and falls back on its own, so this stays
            a single knob for the caller — see `ONE_PASS_SQUARES_MAX_WIDTH`.
        fused_mix: take the mixture with `ttnn.experimental.fast_weighted_reduce_nc`
            instead of `mul` then `sum`. The composed form writes a full second
            `[1, C, N, d/tp]` to DRAM and reads it back, so it moves 3x the bytes
            of the reduction it is performing; the op MACs the weight into the
            accumulator during the one pass that has to happen anyway. Phase 10 on
            `(2,4)` at `C = 9`: 687 us -> 257 us, 2.67x, against a 228 us floor
            measured as unweighted `fast_reduce_nc` over the same tensor. The 29 us
            over that floor is what the weighting itself costs. Off is the composed
            form, kept because it is the reference the op is gated against.
    """

    def __init__(
        self,
        mesh_device,
        hidden_size=HIDDEN_SIZE,
        eps=EPS,
        torch_queries=None,
        dtype=ttnn.bfloat16,
        sp_axis=0,
        tp_axis=1,
        num_links=1,
        topology=None,
        stats_dtype=ttnn.float32,
        fold_stats=True,
        one_pass_stats=True,
        fused_mix=True,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self.num_links = num_links
        self.stats_dtype = stats_dtype
        self.fold_stats = fold_stats
        self.one_pass_stats = one_pass_stats
        self.fused_mix = fused_mix

        mesh_shape = tuple(mesh_device.shape)
        self.tp_factor = mesh_shape[tp_axis]
        self.sp_factor = mesh_shape[sp_axis]
        self.topology = topology if topology is not None else [ttnn.Topology.Linear] * len(mesh_shape)
        assert len(self.topology) == len(mesh_shape), (
            f"topology has {len(self.topology)} entries for a {len(mesh_shape)}-axis mesh; "
            "pass one Topology per axis (Galaxy prefill is [LINE, RING])"
        )

        assert (
            hidden_size % self.tp_factor == 0
        ), f"hidden_size {hidden_size} not divisible by TP factor {self.tp_factor}"
        self.shard_width = hidden_size // self.tp_factor
        # A shard narrower than a tile would put the reduction's tail in padding.
        # Every real config divides cleanly; asserting removes the question.
        assert (
            self.shard_width % ttnn.TILE_SIZE == 0
        ), f"shard width {self.shard_width} is not a multiple of {ttnn.TILE_SIZE}"
        # The matvec has no width limit; the RMSNorm statistics kernel does.
        self.one_pass_squares = one_pass_stats and self.shard_width <= ONE_PASS_SQUARES_MAX_WIDTH

        # Both mappers are None at a single device, so placement there is exactly
        # what it was before distribution existed and every prior measurement
        # stays comparable.
        self.stream_mapper, self.vector_mapper = None, None
        if self.tp_factor > 1 or self.sp_factor > 1:
            stream_dims, vector_dims = [None] * len(mesh_shape), [None] * len(mesh_shape)
            if self.sp_factor > 1:
                stream_dims[sp_axis] = 2  # tokens split across the sequence axis
            if self.tp_factor > 1:
                stream_dims[tp_axis] = 3  # hidden split across the tensor axis
                vector_dims[tp_axis] = 3
            self.stream_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=stream_dims, mesh_shape=mesh_device.shape)
            if self.tp_factor > 1:
                self.vector_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=vector_dims, mesh_shape=mesh_device.shape)

        # The inverse of `stream_mapper`, for tests and for the pipeline boundary.
        # `ConcatMesh2dToTensor` needs a real dim on both axes, so a replicated
        # axis composes as a concat of identical copies — the caller slices.
        self.stream_composer = None
        if self.stream_mapper is not None:
            compose_dims = [0, 0]
            compose_dims[sp_axis], compose_dims[tp_axis] = 2, 3
            self.stream_composer = ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=tuple(compose_dims), mesh_shape=mesh_device.shape
            )

        self.queries = [self.to_query(q) for q in torch_queries] if torch_queries is not None else []

    def to_query(self, torch_query):
        """Place one folded `[d]` query as `[1, 1, 1, d/tp_factor]` on device.

        The query is folded from two `[d]` weights (`res_norm.weight` times
        `res_proj.weight`), so it shards on `d` exactly like the stream it is
        dotted against."""
        assert (
            torch_query.numel() == self.hidden_size
        ), f"query has {torch_query.numel()} elements, expected {self.hidden_size}"
        return ttnn.from_torch(
            torch_query.reshape(1, 1, 1, self.hidden_size),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=self.vector_mapper,
        )

    def _assert_shard_width(self, stream):
        assert stream.shape[-1] == self.shard_width, (
            f"stream last dim is {stream.shape[-1]}, expected {self.shard_width} "
            f"(hidden_size {self.hidden_size} over TP factor {self.tp_factor}); "
            "AttnRes reduces over the full hidden dim and cannot infer the sharding"
        )

    def _reduce_stats(self, stats):
        """Sum a `[1, C, N, k]` statistics tensor across the TP axis.

        Takes ownership. At `tp_factor == 1` this is the identity and adds no op
        to the trace, so every single-device measurement stays comparable."""
        if self.tp_factor == 1:
            return stats

        wide = ttnn.typecast(stats, self.stats_dtype) if stats.dtype != self.stats_dtype else stats
        reduced = self._collective(wide)
        if wide is not stats:
            ttnn.deallocate(wide)
        ttnn.deallocate(stats)

        if reduced.dtype == self.dtype:
            return reduced
        narrow = ttnn.typecast(reduced, self.dtype)
        ttnn.deallocate(reduced)
        return narrow

    def _collective(self, wide):
        """One all-reduce over the TP axis. Does not consume `wide`.

        With `fold_stats`, the candidate axis rides in the last dim for the
        crossing: `[1, C, N, 1]` -> `[1, 1, N, C]` -> reduce -> back. The
        collective charges for tile padding at the payload rate, so a 1-wide
        last dim pays for 32 columns and uses one; folding fills them.

        The fold is not bit-neutral. On reduce-scatter it reassociates the partial
        sums — measured against a replicated 4x reference it doubles the error,
        7.8e-3 -> 1.6e-2 at `C = 18` — and at `N` of one tile row it can switch
        `all_reduce` to the composite algorithm outright, because the candidate
        axis was the only dim that could qualify for reduce-scatter and the folded
        shape does not have it (`ROOFLINE.md` §5). Neither layout is exact to
        begin with and both stay ~50x inside one bf16 ULP, so the gate is the
        186-read depth PCC in `test_tp_depth_walk`, not exactness: measured there,
        the two differ by <=5e-6 in *either* direction, which is reassociation
        noise rather than a precision cost."""
        if not (self.fold_stats and wide.shape[-1] == 1 and wide.shape[1] <= ttnn.TILE_SIZE):
            return ttnn.all_reduce(
                wide,
                cluster_axis=self.tp_axis,
                num_links=self.num_links,
                topology=self.topology[self.tp_axis],
            )

        folded = ttnn.permute(wide, [0, 3, 2, 1])
        crossed = ttnn.all_reduce(
            folded,
            cluster_axis=self.tp_axis,
            num_links=self.num_links,
            topology=self.topology[self.tp_axis],
        )
        ttnn.deallocate(folded)
        unfolded = ttnn.permute(crossed, [0, 3, 2, 1])
        ttnn.deallocate(crossed)
        return unfolded

    def _reduce_stats_pair(self, first, second):
        """Reduce two `[1, C, N, 1]` statistics in **one** collective.

        Packed on dim 1 rather than the last dim: the halves come back out with
        `ttnn.slice` on a tile-plane boundary instead of a sub-tile read of a
        2-wide last dim. Both forms tile-pad the last dim 1 -> 32 anyway, so
        stacking on dim 1 costs 2x a payload that is already 0.2% of the op's
        traffic and buys a slice that cannot land mid-tile."""
        if self.tp_factor == 1:
            return first, second

        candidates = first.shape[1]
        packed = ttnn.concat([first, second], dim=1)
        ttnn.deallocate(first)
        ttnn.deallocate(second)

        reduced = self._reduce_stats(packed)
        shape = list(reduced.shape)
        head = ttnn.slice(reduced, [0, 0, 0, 0], [shape[0], candidates, shape[2], shape[3]])
        tail = ttnn.slice(reduced, [0, candidates, 0, 0], shape)
        ttnn.deallocate(reduced)
        return head, tail

    def _local_sum_squares(self, v):
        """[1, C, N, d/tp] -> [1, C, N, 1]. Not yet summed across ranks.

        `mul` then `sum` is three DRAM passes over the largest tensor in the op to
        produce one scalar per (token, candidate): read `v`, write a second copy of
        it, read that back. `rms_norm_pre_all_gather` is the distributed-RMSNorm
        statistics kernel, which squares inside the reduce and does it in one —
        782 -> 232 µs traced at `C = 9`, against a 229 µs one-pass floor
        (`bringup_log.md` P7). Its 32-wide output carries the sum in column 0.

        Only where the row fits in L1 — see `ONE_PASS_SQUARES_MAX_WIDTH`.
        """
        if not self.one_pass_squares:
            squares = ttnn.mul(v, v)
            sum_squares = ttnn.sum(squares, dim=3, keepdim=True)
            ttnn.deallocate(squares)
            return sum_squares

        wide = ttnn.rms_norm_pre_all_gather(v, dtype=v.dtype, compute_kernel_config=STATS_FIDELITY)
        shape = list(wide.shape)
        column = ttnn.slice(wide, [0, 0, 0, 0], [shape[0], shape[1], shape[2], 1])
        ttnn.deallocate(wide)
        return column

    def _local_dots(self, v, q):
        """[1, C, N, d/tp] . [1, 1, 1, d/tp] -> [1, C, N, 1]. Rank-local.

        The same three-pass shape as the sum of squares, and this one is a matvec,
        so a matmul against `q` as a column does it without the intermediate —
        793 -> 450 µs traced. Still 1.97x the one-pass floor: `N = 1` wastes 31 of
        32 output columns and gives the matmul no reuse to exploit.
        """
        if not self.one_pass_stats:
            projected = ttnn.mul(v, q)
            dots = ttnn.sum(projected, dim=3, keepdim=True)
            ttnn.deallocate(projected)
            return dots

        column = ttnn.permute(q, [0, 1, 3, 2])
        dots = ttnn.matmul(v, column, compute_kernel_config=STATS_FIDELITY)
        ttnn.deallocate(column)
        return dots

    def _local_dots_by_site(self, v, queries):
        """[1, C, N, d/tp] against R folded queries -> [1, C, N, R]. Rank-local.

        R matvecs are R passes over `v`, which is the only large tensor in the op.
        Stacking the queries as columns makes them one matmul over one pass: at a
        12-layer block's 24 read sites that is 42x the one-pass floor down to 1.8x
        (`bringup_log.md` P8). The 24-wide output also idles 8 of 32 tile columns
        instead of the lone matvec's 31.

        Nothing has to be transposed to get here — the dots contract over `d`,
        which is already the last axis. The mixture contracts over candidates and
        is not so lucky, which is why only this half batches.
        """
        if not self.one_pass_stats:
            per_site = [self._local_dots(v, q) for q in queries]
            return self._concat_sites(per_site)

        columns = [ttnn.permute(q, [0, 1, 3, 2]) for q in queries]
        stacked = self._concat_sites(columns)
        dots = ttnn.matmul(v, stacked, compute_kernel_config=STATS_FIDELITY)
        ttnn.deallocate(stacked)
        return dots

    def _dots_by_site(self, v, queries):
        """[1, C, N, d/tp] against R queries -> [1, C, N, R], globally summed.

        One collective for the whole block. The site axis lands in the last dim,
        which the collective pads to a tile either way, so R <= 32 sites cross on
        the payload a single site would have cost unfolded. `fold_stats` therefore
        does not apply here and is not missed — it exists to fill that padding,
        and the sites have already filled it."""
        return self._reduce_stats(self._local_dots_by_site(v, queries))

    @staticmethod
    def _concat_sites(per_site):
        """Stack per-site tensors along the last dim, consuming them. R == 1 is
        the identity — `ttnn.concat` of one tensor would be a copy."""
        if len(per_site) == 1:
            return per_site[0]
        stacked = ttnn.concat(per_site, dim=3)
        for tensor in per_site:
            ttnn.deallocate(tensor)
        return stacked

    @staticmethod
    def _site_major(stacked):
        """`[1, ..., R]` -> `[R, ..., 1]`, consuming its input. Identity at `R == 1`.

        The batch is built with the sites in the last dim because that is where
        the matmul wants them, but the mixture loop takes them back out one at a
        time and a 1-wide last-dim slice lands mid-tile: ttnn untilizes the whole
        batch, cuts the column, and re-tilizes. On dim 0 the same column is a
        whole tile plane and the slice stays in tile layout — 26.8 -> 2.07 µs per
        extraction, against 128.6 µs for the one permute that buys it."""
        if stacked.shape[-1] == 1:
            return stacked
        moved = ttnn.permute(stacked, [3, 1, 2, 0])
        ttnn.deallocate(stacked)
        return moved

    @staticmethod
    def _site_column(stacked, site):
        """One read site's plane out of a `[R, ...]` batch, owned by the caller.

        A `ttnn.slice` that spans its input hands back a fresh handle onto the
        *same* device buffer, so at `R == 1` the plane would die with the batch.
        Copy in that one case; a narrower slice already writes its own."""
        shape = list(stacked.shape)
        if shape[0] == 1:
            return ttnn.clone(stacked)
        return ttnn.slice(stacked, [site, 0, 0, 0], [site + 1] + shape[1:])

    def _to_reciprocal_rms(self, sum_squares):
        """Globally summed squares -> `rsqrt(mean + eps)`. Consumes its input.

        Divides by the **global** `d`; `sum_squares` has already crossed the TP
        axis by the time it gets here."""
        mean_squares = ttnn.mul(sum_squares, 1.0 / self.hidden_size)
        ttnn.deallocate(sum_squares)
        reciprocal_rms = ttnn.rsqrt(ttnn.add(mean_squares, self.eps))
        ttnn.deallocate(mean_squares)
        return reciprocal_rms

    def _reciprocal_rms(self, v):
        """[1, C, N, d/tp] -> [1, C, N, 1]. RMS is a per-(token, candidate) scalar.

        One collective. `inter_block` calls this once per 12-layer block, so the
        sealed set's share of the communication amortizes with its arithmetic."""
        return self._to_reciprocal_rms(self._reduce_stats(self._local_sum_squares(v)))

    def _scores(self, v, q):
        """[1, C, N, d/tp] -> [1, C, N, 1]. Scores the normalized key against `q`
        without ever materializing the normalized tensor.

        Both `d`-reductions ride one collective — this is the per-read cost on a
        TP mesh, and it is `2(S+1)` scalars per token."""
        sum_squares, dots = self._reduce_stats_pair(self._local_sum_squares(v), self._local_dots(v, q))
        reciprocal_rms = self._to_reciprocal_rms(sum_squares)
        scores = ttnn.mul(dots, reciprocal_rms)
        ttnn.deallocate(dots)
        ttnn.deallocate(reciprocal_rms)
        return scores

    def _mix(self, v, weights):
        """Weighted sum over the candidate axis. Values enter raw — the mixture
        is over `v`, not over the normalized key.

        Rank-local at every sharding: the weight is a per-(token, candidate)
        scalar and the sum is over candidates, so no `d`-wide tensor moves.

        The fused op takes the fp32 `weights` directly — its weight operand
        accepts fp32 precisely so this call site does not have to downcast the
        score chain's output to hand it over. See `fused_mix`."""
        if self.fused_mix:
            return ttnn.experimental.fast_weighted_reduce_nc(v, weights, dim=1)
        weighted = ttnn.mul(v, weights)
        mixed = ttnn.sum(weighted, dim=1, keepdim=True)
        ttnn.deallocate(weighted)
        return mixed

    def _softmax_over_candidates(self, scores):
        """[1, C, N, 1] -> [1, C, N, 1], hand-rolled rather than `ttnn.softmax`.

        `ttnn.softmax` only reaches its attention-optimized kernel when reducing
        the last dim; the dim-1 fallback loses ~4% of the softmax mass even in
        fp32 (measured rel err 1.4e-2, row sums to 0.962), and neither
        `numeric_stable` nor `fp32_dest_acc_en` moves it. This chain measures
        15-27x closer. `ttnn.exp` itself is exact to 6e-8, so the deficit is the
        fallback reduction, not the exponential.

        Rank-local: `scores` is already identical on every TP rank."""
        shift = ttnn.max(scores, dim=1, keepdim=True)
        exponentials = ttnn.exp(ttnn.sub(scores, shift))
        ttnn.deallocate(shift)
        total = ttnn.sum(exponentials, dim=1, keepdim=True)
        weights = ttnn.div(exponentials, total)
        ttnn.deallocate(exponentials)
        ttnn.deallocate(total)
        return weights

    def forward(self, prefix_sum, block_residual, q):
        """The AttnRes read.

        Args:
            prefix_sum: `[1, 1, N, d/tp_factor]` live residual stream.
            block_residual: `[1, S, N, d/tp_factor]` sealed snapshots, or None
                for `S == 0`.
            q: `[1, 1, 1, d/tp_factor]` folded query.

        Returns:
            `[1, 1, N, d/tp_factor]`. A fresh tensor even at `S == 0`, so the
            caller's deallocation is uniform across the two paths.
        """
        self._assert_shard_width(prefix_sum)
        if block_residual is None:
            return ttnn.clone(prefix_sum)

        v = ttnn.concat([block_residual, prefix_sum], dim=1)
        scores = self._scores(v, q)
        weights = self._softmax_over_candidates(scores)
        ttnn.deallocate(scores)
        mixed = self._mix(v, weights)
        ttnn.deallocate(weights)
        ttnn.deallocate(v)
        return mixed

    def inter_block(self, block_residual, queries):
        """Sealed-snapshot half of the mixture, for every read site in a block.

        Everything that does not depend on the read site runs once for the whole
        block. The reciprocal-RMS pass is loop-invariant because a sealed snapshot
        is write-once, and the scores batch because the site axis fits in the last
        dim: one matmul, one collective, and one elementwise chain cover all of
        them. Only the mixture stays per site.

        Batching is free in the temporaries as well as the arithmetic. Every
        intermediate here carries one scalar per (token, candidate, site), so a
        1-wide last dim tile-pads to 32 and R <= 32 read sites occupy exactly what
        one site already paid for.

        Args:
            block_residual: `[1, S, N, d/tp_factor]`. Not None — `S == 0` has no
                sealed half, and `forward`'s identity path covers it.
            queries: sequence of `[1, 1, 1, d/tp_factor]` folded queries.

        Returns:
            Three lists, one entry per query: partials `[1, 1, N, d/tp_factor]`
            holding `sum_i e_i v_i`, shifts `[1, 1, N, 1]`, masses `[1, 1, N, 1]`,
            in the online-softmax convention `e_i = exp(s_i - m)`.
        """
        reciprocal_rms = self._reciprocal_rms(block_residual)
        dots = self._dots_by_site(block_residual, queries)
        scores = ttnn.mul(dots, reciprocal_rms)
        ttnn.deallocate(dots)
        ttnn.deallocate(reciprocal_rms)

        site_shifts = ttnn.max(scores, dim=1, keepdim=True)
        centered = ttnn.sub(scores, site_shifts)
        ttnn.deallocate(scores)
        exponentials = ttnn.exp(centered)
        ttnn.deallocate(centered)
        site_masses = ttnn.sum(exponentials, dim=1, keepdim=True)

        # Both reductions above contract dim 1, so the sites cannot move until here.
        exponentials = self._site_major(exponentials)
        site_shifts = self._site_major(site_shifts)
        site_masses = self._site_major(site_masses)

        partials, shifts, masses = [], [], []
        for site in range(len(queries)):
            weights = self._site_column(exponentials, site)
            partials.append(self._mix(block_residual, weights))
            ttnn.deallocate(weights)
            shifts.append(self._site_column(site_shifts, site))
            masses.append(self._site_column(site_masses, site))

        ttnn.deallocate(exponentials)
        ttnn.deallocate(site_shifts)
        ttnn.deallocate(site_masses)
        return partials, shifts, masses

    def merge(self, partial, shift, mass, prefix_sum, q):
        """Fold the live stream into a precomputed sealed-snapshot partial.

        Args:
            partial, shift, mass: this read site's entries from `inter_block`.
            prefix_sum: `[1, 1, N, d/tp_factor]` live residual stream.
            q: `[1, 1, 1, d/tp_factor]` folded query, the one that built `partial`.

        Returns:
            `[1, 1, N, d/tp_factor]`, equal to `forward` up to rounding.
        """
        self._assert_shard_width(prefix_sum)
        live_scores = self._scores(prefix_sum, q)

        merged_shift = ttnn.maximum(shift, live_scores)
        rescale = ttnn.exp(ttnn.sub(shift, merged_shift))
        live_weight = ttnn.exp(ttnn.sub(live_scores, merged_shift))
        ttnn.deallocate(live_scores)
        ttnn.deallocate(merged_shift)

        numerator = ttnn.add(ttnn.mul(partial, rescale), ttnn.mul(prefix_sum, live_weight))
        denominator = ttnn.add(ttnn.mul(mass, rescale), live_weight)
        ttnn.deallocate(rescale)
        ttnn.deallocate(live_weight)

        merged = ttnn.div(numerator, denominator)
        ttnn.deallocate(numerator)
        ttnn.deallocate(denominator)
        return merged
