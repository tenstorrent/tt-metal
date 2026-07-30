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
split the mixture so the sealed half amortizes across a whole 12-layer block; in
this composed form only the reciprocal-RMS pass actually amortizes, which is one
of the two passes over the sealed set. Batching the weighted sum across read
sites needs a token-batched matmul over the candidate axis — see the Phase-5
learnings in `bringup_log.md`.

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
        reduced = ttnn.all_reduce(
            wide,
            cluster_axis=self.tp_axis,
            num_links=self.num_links,
            topology=self.topology[self.tp_axis],
        )
        if wide is not stats:
            ttnn.deallocate(wide)
        ttnn.deallocate(stats)

        if reduced.dtype == self.dtype:
            return reduced
        narrow = ttnn.typecast(reduced, self.dtype)
        ttnn.deallocate(reduced)
        return narrow

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
        """[1, C, N, d/tp] -> [1, C, N, 1]. Not yet summed across ranks."""
        squares = ttnn.mul(v, v)
        sum_squares = ttnn.sum(squares, dim=3, keepdim=True)
        ttnn.deallocate(squares)
        return sum_squares

    def _local_dots(self, v, q):
        """[1, C, N, d/tp] . [1, 1, 1, d/tp] -> [1, C, N, 1]. Rank-local."""
        projected = ttnn.mul(v, q)
        dots = ttnn.sum(projected, dim=3, keepdim=True)
        ttnn.deallocate(projected)
        return dots

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

    def _dots(self, v, q):
        """[1, C, N, d/tp] . [1, 1, 1, d/tp] -> [1, C, N, 1], globally summed."""
        return self._reduce_stats(self._local_dots(v, q))

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
        scalar and the sum is over candidates, so no `d`-wide tensor moves."""
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

        The reciprocal-RMS pass runs once and is reused across read sites: a
        sealed snapshot is write-once, so its RMS is loop-invariant. On a TP mesh
        that amortizes a collective too — one per block instead of one per read.

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
        partials, shifts, masses = [], [], []

        for q in queries:
            dots = self._dots(block_residual, q)
            scores = ttnn.mul(dots, reciprocal_rms)
            ttnn.deallocate(dots)

            shift = ttnn.max(scores, dim=1, keepdim=True)
            exponentials = ttnn.exp(ttnn.sub(scores, shift))
            ttnn.deallocate(scores)

            masses.append(ttnn.sum(exponentials, dim=1, keepdim=True))
            partials.append(self._mix(block_residual, exponentials))
            shifts.append(shift)
            ttnn.deallocate(exponentials)

        ttnn.deallocate(reciprocal_rms)
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
