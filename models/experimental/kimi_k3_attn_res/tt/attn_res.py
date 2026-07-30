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
        hidden_size: `d`. Reductions run over this axis, so a stream sharded on
            it needs the statistics all-reduce that Phase 8 adds; until then
            `forward` rejects a sharded stream rather than reducing locally and
            returning quietly wrong numbers.
        eps: `rms_norm_eps`.
        torch_queries: optional sequence of `[d]` folded queries (see
            `torch_functional.attn_res.fold_query`). Exposed as `self.queries`.
        dtype: device dtype for queries and intermediates.
        cluster_axis, num_links, topology: distribution knobs, tracked for
            Phase 8. `topology` becomes a per-axis tuple there — Galaxy prefill
            is `[LINE, RING]` and a scalar `Ring` deadlocks a TP all-gather on a
            column wrap link with no physical fabric edge.
    """

    def __init__(
        self,
        mesh_device,
        hidden_size=HIDDEN_SIZE,
        eps=EPS,
        torch_queries=None,
        dtype=ttnn.bfloat16,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology
        self.queries = [self.to_query(q) for q in torch_queries] if torch_queries is not None else []

    def to_query(self, torch_query):
        """Place one folded `[d]` query as `[1, 1, 1, d]` on device."""
        assert (
            torch_query.numel() == self.hidden_size
        ), f"query has {torch_query.numel()} elements, expected {self.hidden_size}"
        return ttnn.from_torch(
            torch_query.reshape(1, 1, 1, self.hidden_size),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )

    def _reject_sharded(self, stream):
        assert stream.shape[-1] == self.hidden_size, (
            f"stream is sharded on the reduction axis ({stream.shape[-1]} of {self.hidden_size}); "
            "AttnRes needs the statistics all-reduce that Phase 8 adds"
        )

    def _reciprocal_rms(self, v):
        """[1, C, N, d] -> [1, C, N, 1]. RMS is a per-(token, candidate) scalar."""
        squares = ttnn.mul(v, v)
        mean_squares = ttnn.mul(ttnn.sum(squares, dim=3, keepdim=True), 1.0 / self.hidden_size)
        ttnn.deallocate(squares)
        reciprocal_rms = ttnn.rsqrt(ttnn.add(mean_squares, self.eps))
        ttnn.deallocate(mean_squares)
        return reciprocal_rms

    def _dots(self, v, q):
        """[1, C, N, d] . [1, 1, 1, d] -> [1, C, N, 1]."""
        projected = ttnn.mul(v, q)
        dots = ttnn.sum(projected, dim=3, keepdim=True)
        ttnn.deallocate(projected)
        return dots

    def _scores(self, v, q):
        """[1, C, N, d] -> [1, C, N, 1]. Scores the normalized key against `q`
        without ever materializing the normalized tensor."""
        reciprocal_rms = self._reciprocal_rms(v)
        dots = self._dots(v, q)
        scores = ttnn.mul(dots, reciprocal_rms)
        ttnn.deallocate(dots)
        ttnn.deallocate(reciprocal_rms)
        return scores

    def _mix(self, v, weights):
        """Weighted sum over the candidate axis. Values enter raw — the mixture
        is over `v`, not over the normalized key."""
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
        fallback reduction, not the exponential."""
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
            prefix_sum: `[1, 1, N, d]` live residual stream.
            block_residual: `[1, S, N, d]` sealed snapshots, or None for `S == 0`.
            q: `[1, 1, 1, d]` folded query.

        Returns:
            `[1, 1, N, d]`. A fresh tensor even at `S == 0`, so the caller's
            deallocation is uniform across the two paths.
        """
        self._reject_sharded(prefix_sum)
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
        sealed snapshot is write-once, so its RMS is loop-invariant.

        Args:
            block_residual: `[1, S, N, d]`. Not None — `S == 0` has no sealed
                half, and `forward`'s identity path covers it.
            queries: sequence of `[1, 1, 1, d]` folded queries.

        Returns:
            Three lists, one entry per query: partials `[1, 1, N, d]` holding
            `sum_i e_i v_i`, shifts `[1, 1, N, 1]`, masses `[1, 1, N, 1]`, in the
            online-softmax convention `e_i = exp(s_i - m)`.
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
            prefix_sum: `[1, 1, N, d]` live residual stream.
            q: `[1, 1, 1, d]` folded query, the same one that built `partial`.

        Returns:
            `[1, 1, N, d]`, equal to `forward` up to rounding.
        """
        self._reject_sharded(prefix_sum)
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
