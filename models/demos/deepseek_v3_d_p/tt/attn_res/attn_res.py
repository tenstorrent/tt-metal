# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN composite for Kimi K3 attention residuals (AttnRes).

Mirrors `reference/kimi_k3/attn_res/attn_res.py`. Two divergences from the torch API, both
forced by ttnn:

  * `block_residual` is 4D `[1, S, N, d]` with candidates on **dim 1**, not the
    last dim. `S+1 <= 9` would tile-pad 9 -> 32 on a last dim, and padding zeros
    would enter a last-dim softmax as `exp(0) = 1`.
  * "no sealed snapshots" is `block_residual=None`, not a zero-width tensor —
    ttnn has no zero-extent dimension. The read is the identity there anyway.

The read is split: `inter_block` + `merge`. The sealed half amortizes across a whole
12-layer block — the reciprocal-RMS pass and the dots each collapse to a single pass
over the sealed set for all 24 read sites, which leaves the weighted sum as the only
per-site traffic over it. That one does not batch in composed form: it contracts over
the candidate axis, and reaching it with a matmul means making that axis a tile axis,
whose two permutes over the sealed set cost what the matmul saves.

`merge` is one dispatch. `ttnn.experimental.deepseek_prefill.attn_res_gather_softmax` takes the live
stream's statistics, crosses the tensor-parallel axis, and folds the sealed partial in,
which is why nothing between the statistics and the result exists as a tensor.

Distribution: the stream stays sharded `[1, 1, T/R, d/C]` exactly as the analog
leaves it, the sequence axis communicates nothing, and each read all-reduces
`2(S+1)` scalars per token across the TP axis. Nothing of width `d` ever crosses
a rank boundary.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

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
# and asks for more (1 971 072 B at 7 168). Every shard width the read admits on
# Blackhole fits; Wormhole's smaller L1 moves the bound, which is why this is a
# fallback rather than an assert.
ONE_PASS_SQUARES_MAX_WIDTH = 177 * ttnn.TILE_SIZE

# Folding the candidate axis into the last dim trades the collective's padded
# payload against two permutes, so it only pays once the axis is wide enough to
# have been paying for padding. Traced on `(2,4)` at 640 rows per chip, one
# `all_reduce` folded against unfolded: C=2 costs 6.0 us, C=4 buys 2.7, C=8 buys
# 12.5, C=16 buys 50.5, C=32 buys 96.0. The sealed set starts at one snapshot and
# grows a snapshot a block, so the first blocks of a walk cross below the crossover
# and an ungated fold would be a straight loss on them.
FOLD_MIN_CANDIDATES = 4


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
        weights: optional `tt.attn_res.weights.AttnResWeights` — the whole stack's
            queries already on device, from a checkpoint or a `.tensorbin` cache.
            A caller without one places its own with `to_query`.
        dtype: device dtype for queries and intermediates.
        sp_axis, tp_axis: mesh axes the sequence and the hidden dim are sharded
            across. `sp_axis` is placement only — the op never communicates on it,
            which is the whole reason the sequence split is free. Both live here
            rather than in the caller so the layout has one definition; `merge`
            checks its input against it. The tensor-parallel axis must be wider than
            one chip: the read's exchange is what its one dispatch is built around,
            and there is no exchange to absorb at `tp_factor == 1`.
        num_links: fabric links for the statistics all-reduce.
        topology: one `ttnn.Topology` **per mesh axis**, not a scalar. Defaults to
            what the opened fabric actually wraps, which is the only source that
            cannot disagree with it: naming `Ring` on an axis the fabric does not
            wrap points the collective at a link nothing services, and it waits
            there for an arrival that is never routed.
        stats_dtype: dtype the statistics all-reduce runs in. The collective
            reduces in bf16 unless the input is fp32, which takes a dedicated path.
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
            Costs two `ttnn.permute` calls, and pays for them even untraced,
            where each launch is billed to the host: 111.3 ms against 120.3 ms
            over the 186-read walk on `(2,4)`. Width-gated even when on, because
            below a few candidates the padding is the payload; see
            `FOLD_MIN_CANDIDATES`.
        one_pass_stats: take both `d`-reductions in a single pass over `v`, with
            `rms_norm_pre_all_gather` for the sum of squares and a matmul for the
            dot. Written as `mul` then `sum`, each reduction materializes a second
            copy of `[1, C, N, d/tp]` in DRAM and reads it back — three passes to
            produce 0.6% of the op's bytes. Phase 9 P7 on `(2,4)` at `C = 9`:
            782 -> 232 us for the squares, 793 -> 450 for the dot, ~892 us of a
            3 136 us read. Requires `STATS_FIDELITY`; see the note there. The
            squares half is width-gated and falls back on its own, so this stays
            a single knob for the caller — see `ONE_PASS_SQUARES_MAX_WIDTH`.
        tt_ccl: the model's CCL semaphore pools, when the caller has them. The
            statistics all-reduce needs a persistent set either way and makes its
            own when this is `None`; sharing the model's costs nothing and gets
            the cycling `_tp_semaphores` explains.
    """

    def __init__(
        self,
        mesh_device,
        hidden_size=HIDDEN_SIZE,
        eps=EPS,
        weights=None,
        dtype=ttnn.bfloat16,
        sp_axis=0,
        tp_axis=1,
        num_links=1,
        topology=None,
        stats_dtype=ttnn.float32,
        fold_stats=True,
        one_pass_stats=True,
        tt_ccl=None,
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
        self.tt_ccl = tt_ccl
        # Sized by the token count, which reaches the op and not the constructor.
        self._exchange_scratch = {}
        self._exchange_sem = None
        self._tp_sems = None

        mesh_shape = tuple(mesh_device.shape)
        self.tp_factor = mesh_shape[tp_axis]
        self.sp_factor = mesh_shape[sp_axis]
        self.topology = list(per_axis_topology()) if topology is None else topology
        assert len(self.topology) == len(mesh_shape), (
            f"topology has {len(self.topology)} entries for a {len(mesh_shape)}-axis mesh; "
            "pass one Topology per axis, or none and take the opened fabric's"
        )

        # The read is one program because the exchange runs inside it. Without a
        # tensor-parallel axis there is no exchange, and no read either.
        assert self.tp_factor > 1, (
            f"AttnRes needs a tensor-parallel axis wider than one chip, got tp_factor {self.tp_factor} "
            f"on mesh axis {tp_axis} of a {mesh_shape} mesh"
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

        stream_dims, vector_dims = [None] * len(mesh_shape), [None] * len(mesh_shape)
        stream_dims[tp_axis] = 3  # hidden split across the tensor axis
        vector_dims[tp_axis] = 3
        # A mesh may be one chip deep on the sequence axis; the hidden split never is.
        if self.sp_factor > 1:
            stream_dims[sp_axis] = 2  # tokens split across the sequence axis
        self.stream_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=stream_dims, mesh_shape=mesh_device.shape)
        self.vector_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=vector_dims, mesh_shape=mesh_device.shape)

        # The inverse of `stream_mapper`, for tests and for the pipeline boundary.
        # `ConcatMesh2dToTensor` needs a real dim on both axes, so a replicated
        # axis composes as a concat of identical copies — the caller slices.
        compose_dims = [0, 0]
        compose_dims[sp_axis], compose_dims[tp_axis] = 2, 3
        self.stream_composer = ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=tuple(compose_dims), mesh_shape=mesh_device.shape
        )

        # Query layouts the op owns: one column per query, one stack per block.
        # Both key on identity because `ttnn.Tensor.__eq__` is elementwise, which
        # makes a tensor hashable but not usable as a dict key; each value holds
        # its queries so an address cannot be recycled under the key.
        self._column_by_query_id = {}
        self._stack_by_query_ids = {}
        self.weights = weights
        if weights is not None:
            assert weights.tensor_parallel_axis == tp_axis, (
                f"weights are sharded on mesh axis {weights.tensor_parallel_axis}, "
                f"this op reduces over axis {tp_axis}"
            )
            # Cached queries skip `to_query`, so their columns are transposed here for the
            # same reason it transposes there: after capture starts there is no free point.
            for query in weights.walk_order():
                self._query_column(query)

    def to_query(self, torch_query):
        """Place one folded `[d]` query as `[1, 1, 1, d/tp_factor]` on device.

        The query is folded from two `[d]` weights (`res_norm.weight` times
        `res_proj.weight`), so it shards on `d` exactly like the stream it is
        dotted against.

        The transposed column the dot matmul contracts against is placed here too.
        It is a function of the weight alone, so preparing it on first use instead
        would land a permute per read site wherever the caller happens to be — and
        a caller that captures a trace on its first read would capture those
        permutes and replay them for nothing. Weight layout belongs at weight
        placement, which is the one point that is unambiguously outside a trace.
        """
        assert (
            torch_query.numel() == self.hidden_size
        ), f"query has {torch_query.numel()} elements, expected {self.hidden_size}"
        row = ttnn.from_torch(
            torch_query.reshape(1, 1, 1, self.hidden_size),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=self.vector_mapper,
        )
        self._query_column(row)
        return row

    def _assert_shard_width(self, stream):
        assert stream.shape[-1] == self.shard_width, (
            f"stream last dim is {stream.shape[-1]}, expected {self.shard_width} "
            f"(hidden_size {self.hidden_size} over TP factor {self.tp_factor}); "
            "AttnRes reduces over the full hidden dim and cannot infer the sharding"
        )

    def _to_stats_dtype(self, tensor):
        """Widen to `stats_dtype`, consuming `tensor`. The identity if it already is."""
        if tensor.dtype == self.stats_dtype:
            return tensor
        wide = ttnn.typecast(tensor, self.stats_dtype)
        ttnn.deallocate(tensor)
        return wide

    def _reduce_stats(self, stats):
        """Sum a `[1, C, N, k]` statistics tensor across the TP axis.

        Takes ownership."""
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

    def _tp_semaphores(self):
        """Persistent CCL semaphores for the statistics all-reduce.

        Handing the collective its semaphores is what keeps it off `ttnn.all_reduce`'s
        fallback, which passes none and so has each leg allocate its own inside program
        construction. That fallback costs twice. The set it takes is held for as long as
        the program cache is, and the statistics shape carries the sealed set's width, so
        a deeper seal hashes a new program and takes another set with it: the pool a walk
        needs grows with seal depth instead of staying flat, and there is no depth at
        which sizing it is safe. Worse, allocating one resets it through a mesh write the
        allocator blocks on, so that the reset lands everywhere before the next program
        runs; each new depth therefore stalls the walk on a full queue drain. One set for
        the whole walk keeps the pool flat and keeps every allocation out of the walk.

        A `tt_ccl` is the better source when the caller has one. Its scatter and gather
        pools are double-buffered and handed out in turn, and that matters as soon as a
        second component has a collective in flight on the same axis: two collectives
        sharing one set of semaphores read each other's counts. A walk on its own is
        serialized and cannot race itself, so the fallback below is a fixed set.

        Sizes are the op's contract, not a choice — two barriers, index 0 for the
        reduce-scatter and 1 for the all-gather, then the scatter's three and the
        gather's two. The barrier pair is `tt_ccl`'s double-buffer slots used as that
        pair, which is what the rest of the model passes.
        """
        if self.tt_ccl is not None:
            return (
                self.tt_ccl.barrier_semaphore_handles[self.tp_axis],
                self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
            )
        if self._tp_sems is None:
            grid = self.mesh_device.compute_with_storage_grid_size()
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
            new_set = lambda count: [ttnn.create_global_semaphore(self.mesh_device, cores, 0) for _ in range(count)]
            self._tp_sems = (new_set(2), new_set(3), new_set(2))
        return self._tp_sems

    def _all_reduce(self, tensor):
        """Sum `tensor` across the TP axis. Does not consume it."""
        barriers, scatter, gather = self._tp_semaphores()
        return ttnn.experimental.all_reduce_async(
            tensor,
            cluster_axis=self.tp_axis,
            mesh_device=self.mesh_device,
            barrier_semaphores=barriers,
            rs_global_semaphores=scatter,
            ag_global_semaphores=gather,
            math_op=ttnn.ReduceType.Sum,
            num_links=self.num_links,
            topology=self.topology[self.tp_axis],
        )

    def _collective(self, wide):
        """One all-reduce over the TP axis. Does not consume `wide`.

        With `fold_stats` and an axis of at least `FOLD_MIN_CANDIDATES`, the
        candidate axis rides in the last dim for the crossing: `[1, C, N, 1]` ->
        `[1, 1, N, C]` -> reduce -> back. The collective charges for tile padding
        at the payload rate, so a 1-wide last dim pays for 32 columns and uses
        one; folding fills them.

        The fold is not bit-neutral. On reduce-scatter it reassociates the partial
        sums — measured against a replicated 4x reference it doubles the error,
        7.8e-3 -> 1.6e-2 at `C = 18` — and at `N` of one tile row it can switch
        the collective to its composite algorithm outright, because the candidate
        axis was the only dim that could qualify for reduce-scatter and the folded
        shape does not have it. Composite allocates its own semaphores whatever it
        is handed, so only shapes above one tile row keep what `_tp_semaphores` buys;
        the model's 640 rows per chip are well above it.

        Neither layout is exact to begin with and both stay ~50x inside one bf16 ULP,
        so the gate is the 186-read depth PCC in
        `tests/attn_res/model/test_forward_loop.py`, not exactness: measured there,
        the two differ by <=5e-6 in *either* direction, which is reassociation noise
        rather than a precision cost."""
        if not (self.fold_stats and wide.shape[-1] == 1 and FOLD_MIN_CANDIDATES <= wide.shape[1] <= ttnn.TILE_SIZE):
            return self._all_reduce(wide)

        folded = ttnn.permute(wide, [0, 3, 2, 1])
        crossed = self._all_reduce(folded)
        ttnn.deallocate(folded)
        unfolded = ttnn.permute(crossed, [0, 3, 2, 1])
        ttnn.deallocate(crossed)
        return unfolded

    def _local_sum_squares(self, v):
        """[1, C, N, d/tp] -> [1, C, N, 1]. Not yet summed across ranks.

        `mul` then `sum` is three DRAM passes over the largest tensor in the op to
        produce one scalar per (token, candidate): read `v`, write a second copy of
        it, read that back. `rms_norm_pre_all_gather` is the distributed-RMSNorm
        statistics kernel, which squares inside the reduce and does it in one —
        782 -> 232 µs traced at `C = 9`, against a 229 µs one-pass floor. Its
        32-wide output carries the sum in column 0.

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

        return ttnn.matmul(v, self._query_columns([q]), compute_kernel_config=STATS_FIDELITY)

    def _local_dots_by_site(self, v, queries):
        """[1, C, N, d/tp] against R folded queries -> [1, C, N, R]. Rank-local.

        R matvecs are R passes over `v`, which is the only large tensor in the op.
        Stacking the queries as columns makes them one matmul over one pass: at a
        12-layer block's 24 read sites that is 42x the one-pass floor down to 1.8x.
        The 24-wide output also idles 8 of 32 tile columns instead of the lone
        matvec's 31.

        Nothing has to be transposed to get here — the dots contract over `d`,
        which is already the last axis. The mixture contracts over candidates and
        is not so lucky, which is why only this half batches.
        """
        if not self.one_pass_stats:
            per_site = [self._local_dots(v, q) for q in queries]
            return self._concat_sites(per_site)

        return ttnn.matmul(v, self._query_columns(queries), compute_kernel_config=STATS_FIDELITY)

    def _query_column(self, q):
        """The `[1, 1, d/tp, 1]` transpose of one folded query, held by the op."""
        if id(q) not in self._column_by_query_id:
            self._column_by_query_id[id(q)] = (q, ttnn.permute(q, [0, 1, 3, 2]))
        return self._column_by_query_id[id(q)][1]

    def _query_columns(self, queries):
        """R folded queries `[1, 1, 1, d/tp]` -> one `[1, 1, d/tp, R]` matmul operand.

        The op owns what this returns; callers must not deallocate it. Queries are
        folded weights, so both the transpose and the stack are functions of the
        weights alone and neither belongs in a per-read-site cost.

        Holding the stack means the first call for a block builds it, and only a
        caller that captures a trace has to care where that lands. Capture already
        requires a compile pass over the same call sequence — a program built
        during capture is not the one replayed — so the stack is built in that
        pass alongside the programs, and the concat never enters a trace. The
        columns themselves are placed by `to_query` and do not even need that.
        """
        key = tuple(id(q) for q in queries)
        if key not in self._stack_by_query_ids:
            columns = [self._query_column(q) for q in queries]
            stack = columns[0] if len(columns) == 1 else ttnn.concat(columns, dim=3)
            self._stack_by_query_ids[key] = (list(queries), stack)
        return self._stack_by_query_ids[key][1]

    def _dots_by_site(self, v, queries):
        """[1, C, N, d/tp] against R queries -> [1, C, N, R], globally summed.

        One collective for the whole block. The site axis lands in the last dim,
        which the collective pads to a tile either way, so R <= 32 sites cross on
        the payload a single site would have cost unfolded. `fold_stats` therefore
        does not apply here and is not missed — it exists to fill that padding,
        and the sites have already filled it."""
        return self._reduce_stats(self._local_dots_by_site(v, queries))

    @staticmethod
    def _concat_sites(per_site, dim=3):
        """Stack per-site tensors along `dim`, consuming them. R == 1 is the
        identity — `ttnn.concat` of one tensor would be a copy."""
        if len(per_site) == 1:
            return per_site[0]
        stacked = ttnn.concat(per_site, dim=dim)
        for tensor in per_site:
            ttnn.deallocate(tensor)
        return stacked

    @staticmethod
    def _site_major(stacked):
        """`[1, ..., R]` -> `[R, ..., 1]`, consuming its input. Identity at `R == 1`.

        The batch is built with the sites in the last dim because that is where the
        matmul wants them, but every consumer below addresses one site at a time. On
        the last dim a site is a 1-wide cut mid-tile, which costs an untilize of the
        whole batch and a re-tilize; on dim 0 it is a whole tile plane and so a page
        offset — 26.8 -> 2.07 µs per site, against 128.6 µs for the one permute."""
        if stacked.shape[-1] == 1:
            return stacked
        moved = ttnn.permute(stacked, [3, 1, 2, 0])
        ttnn.deallocate(stacked)
        return moved

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

    def _exchange_semaphore(self):
        """Arrival semaphore for the fused read's exchange.

        Global rather than program-local: a program-local semaphore is re-initialized
        at every launch, which races a peer already a read ahead. One serves the whole
        walk — the op resets it in kernel after waiting on it.
        """
        if self._exchange_sem is None:
            grid = self.mesh_device.compute_with_storage_grid_size()
            self._exchange_sem = ttnn.create_global_semaphore(
                self.mesh_device,
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]),
                0,
            )
        return self._exchange_sem

    def _exchange_stats(self, num_tokens):
        """Scratch for one read's statistics, this rank's plane and its peers'.

        Written and read back inside the one pass, so it carries nothing between reads
        and a single buffer per token count serves the walk. Replicated rather than
        mapped: a peer's plane is written by page address, so the buffer has to sit at
        the same address on every chip of the tensor-parallel axis.
        """
        stats = self._exchange_scratch.get(num_tokens)
        if stats is None:
            stats = ttnn.zeros(
                [1, 2 * self.tp_factor, num_tokens, 1],
                dtype=self.stats_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
            )
            self._exchange_scratch[num_tokens] = stats
        return stats

    def _mix(self, v, weights):
        """Weighted sum over the candidate axis, one per read site batched on
        `weights`' dim 0: `[1, C, N, d/tp]` and `[R, C, N, 1]` -> `[R, 1, N, d/tp]`.

        Values enter raw — the mixture is over `v`, not over the normalized key.

        Rank-local at every sharding: the weight is a per-(token, candidate)
        scalar and the sum is over candidates, so no `d`-wide tensor moves.

        Batching the sites into the op is what keeps `v` out of the loop. `v` is
        the largest tensor in the module and every site weights the same one, so
        a site-at-a-time mixture streams it R times; the op reads it once per group
        of sites instead. It also takes the fp32 `weights` directly — its weight
        operand accepts fp32 precisely so this call site does not have to downcast
        the score chain's output to hand it over."""
        return ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(v, weights, dim=1)

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
            block_residual: `[1, S, N, d/tp_factor]`. Not None — the walk places no
                read site before the first seal, so every executed read has one.
            queries: sequence of `[1, 1, 1, d/tp_factor]` folded queries.

        Returns:
            The partials `[R, 1, N, d/tp_factor]`, each site's holding `sum_i e_i v_i`,
            and the shifts and masses `[R, 1, N, 1]`, all batched over read sites in
            the online-softmax convention `e_i = exp(s_i - m)`. `merge` takes the
            batches whole and names the site; keep them alive until the block's last
            read.
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
        # Site-major is what makes a site addressable without an op: on dim 0 a site
        # is a whole tile plane, so every consumer below reaches it as a page offset.
        # It is also the axis the mixture batches over, which is what lets the whole
        # block's sealed half be one dispatch.
        exponentials = self._site_major(exponentials)
        site_shifts = self._site_major(site_shifts)
        site_masses = self._site_major(site_masses)

        # `merge` reads these alongside the live stream's statistics through one
        # circular buffer, which is one unpack configuration and so one dtype.
        # Widening here is exact and costs two dispatches a block against the
        # scoring program it lets every one of the block's reads drop.
        site_shifts = self._to_stats_dtype(site_shifts)
        site_masses = self._to_stats_dtype(site_masses)

        partials = self._mix(block_residual, exponentials)

        ttnn.deallocate(exponentials)
        return partials, site_shifts, site_masses

    def merge(self, partial, shift, mass, running_sum, q, site=0, pending=None):
        """Fold the live stream into a precomputed sealed-snapshot partial.

        Args:
            partial, shift, mass: `inter_block`'s batches over read sites, passed
                whole. The fused op reads site `site` out of them itself, so the
                batches survive the read and the caller frees them once per block.
            running_sum: `[1, 1, N, d/tp_factor]` live residual stream.
            q: `[1, 1, 1, d/tp_factor]` folded query, the one that built site
                `site`'s partial.
            site: which read site of the batches this is.
            pending: a write into the stream the caller has not settled yet. Scored
                and folded as part of `running_sum`, and handed back summed. Borrowed.

        Returns:
            `[1, 1, N, d/tp_factor]` — or, where `pending` was given, that and the
            settled stream behind it.
        """
        self._assert_shard_width(running_sum)

        # The live stream's statistics are taken inside this program, which is why the
        # exchange they need is here rather than around it and why the read is one
        # dispatch. Nothing between the statistics and the result exists as a tensor.
        outputs = ttnn.experimental.deepseek_prefill.attn_res_gather_softmax(
            partial,
            running_sum,
            shift,
            mass,
            q,
            self._exchange_stats(running_sum.shape[-2]),
            self._exchange_semaphore(),
            site=site,
            cluster_axis=self.tp_axis,
            # Omitting this leaves the op on the process-wide fabric topology, which the
            # sealed half's all_reduce does not use: one read would route its two halves
            # by different rules wherever the two disagree.
            topology=self.topology[self.tp_axis],
            inv_hidden_size=1.0 / self.hidden_size,
            eps=self.eps,
            pending=pending,
        )
        return tuple(outputs) if pending is not None else outputs[0]
