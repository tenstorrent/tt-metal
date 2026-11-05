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
12-layer block, as two dispatches for all 24 read sites.
`ttnn.experimental.deepseek_prefill.attn_res_block_stats` takes every site's dot and the
sum of squares in one pass over the sealed set and gathers them across the
tensor-parallel axis inside the same program; `attn_res_block_mix` sums the ranks'
partials, scores, and mixes in a second. Each pass reads every sealed tile once.

`merge` is one dispatch. `ttnn.experimental.deepseek_prefill.attn_res_gather_softmax` takes the live
stream's statistics, crosses the tensor-parallel axis, and folds the sealed partial in,
which is why nothing between the statistics and the result exists as a tensor.

Distribution: the stream stays sharded `[1, 1, T/R, d/C]` exactly as the analog
leaves it, and the sequence axis communicates nothing. The TP axis carries statistics
only: once a block, each rank's per-(token, candidate) partials, gathered as one tile
column per rank; once a read, the live stream's two scalars per token. Nothing of width
`d` ever crosses a rank boundary.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_num_links, per_axis_topology

HIDDEN_SIZE = 7168
BLOCK_SIZE = 12
EPS = 1e-5

# Both one-pass `d`-reductions — the sum of squares and the dots, which
# `attn_res_block_stats` takes as matmuls in one pass — are inadmissible without this.
# At default fidelity the bf16 mantissa is truncated going into the multiply, which
# costs the sum of squares an order of magnitude — 4.8e-2 relative error against the
# 2.4e-3 that `mul` + `sum` achieves — and the dots 4x. HiFi4 with fp32
# accumulation restores both (2.5e-3 and 3.2e-3) and is free on device, because
# these reductions are bandwidth-bound and the extra math passes hide under the
# reads: 232 µs against 230 at default fidelity. `WormholeComputeKernelConfig` is
# the name ttnn gives this on every non-Grayskull arch, Blackhole included.
STATS_FIDELITY = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


def as_sealed_list(sealed):
    """A sealed set as the list of snapshot tensors it was sealed as: None is the empty set,
    and a single `[1, S, N, d]` tensor is a one-element list."""
    if sealed is None:
        return []
    return list(sealed) if isinstance(sealed, (list, tuple)) else [sealed]


class TtAttnRes(LightweightModule):
    """The AttnRes read, on the fused `deepseek_prefill` ops.

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
        num_links: fabric links for the block statistics' gather. Defaults to every link
            the box has on the tensor-parallel axis: the block's statistics are the largest
            payload AttnRes moves.
        topology: one `ttnn.Topology` **per mesh axis**, not a scalar. Defaults to
            what the opened fabric actually wraps, which is the only source that
            cannot disagree with it: naming `Ring` on an axis the fabric does not
            wrap points the gather at a link nothing services, and it waits there
            for an arrival that is never routed.
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
        num_links=None,
        topology=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self.num_links = get_num_links(mesh_device, tp_axis) if num_links is None else num_links
        # Sized by the token count, which reaches the op and not the constructor.
        self._exchange_scratch = {}
        self._gathered_stats_by_shape = {}
        self._exchange_sem = None
        # `attn_res_block_stats` counts its peers' arrivals here, on its gather cores: the
        # last `num_links` cores of the grid's last row. One serves every call, whatever the
        # sealed depth, since the op consumes it by subtraction. Creating a global semaphore
        # resets it through a blocking mesh write, and a write inside a trace capture region
        # is fatal, so it is made here rather than on first use.
        grid = mesh_device.compute_with_storage_grid_size()
        self._block_stats_sem = ttnn.create_global_semaphore(
            mesh_device,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(grid.x - self.num_links, grid.y - 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1)
                    )
                ]
            ),
            0,
        )

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
        self._score_stack_by_query_ids = {}
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

    def _query_column(self, q):
        """The `[1, 1, d/tp, 1]` transpose of one folded query, held by the op."""
        if id(q) not in self._column_by_query_id:
            self._column_by_query_id[id(q)] = (q, ttnn.permute(q, [0, 1, 3, 2]))
        return self._column_by_query_id[id(q)][1]

    def _score_columns(self, queries):
        """R folded queries -> one `[1, 1, d/tp, 32]` operand whose column 1 + r is query r.

        Column 0 and the columns past the last site are zero, so `attn_res_block_stats`
        lands each site's dot in the column `attn_res_block_mix` reads it from and leaves
        column 0 free for the sum of squares it accumulates there: both statistics share a
        single tile column and one gather carries them.

        The op owns what this returns; callers must not deallocate it. The stack is a
        function of the weights alone, so it is built on first use and held rather than
        rebuilt per block. Only a caller that captures a trace has to care where that first
        use lands, and capture already requires a compile pass over the same call sequence
        — a program built during capture is not the one replayed — so the stack is built in
        that pass alongside the programs and the concat never enters a trace.
        """
        key = tuple(id(q) for q in queries)
        if key not in self._score_stack_by_query_ids:
            columns = [self._query_column(q) for q in queries]
            # Explicit `ttnn.zeros` and one concat: `zeros_like` and `ttnn.pad` leave
            # garbage in the stack on a mesh.
            zeros = lambda width: ttnn.zeros(
                [1, 1, self.shard_width, width],
                dtype=columns[0].dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
            )
            blocks = [zeros(1), *columns]
            if len(blocks) < ttnn.TILE_SIZE:
                blocks.append(zeros(ttnn.TILE_SIZE - len(blocks)))
            stack = ttnn.concat(blocks, dim=3)
            ttnn.deallocate(blocks[0])
            if len(blocks) > 1 + len(columns):
                ttnn.deallocate(blocks[-1])
            self._score_stack_by_query_ids[key] = (list(queries), stack)
        return self._score_stack_by_query_ids[key][1]

    def _inter_block_fused(self, sealed, queries, num_closed):
        """`inter_block` as two dispatches: both statistics in one pass over the sealed set,
        gathered across the tensor-parallel axis in the same program, then the scoring and
        the mixture in a second pass.

        Each pass reads every sealed tile once, for all the block's read sites, and the
        scores stay in fp32 throughout.
        """
        # Gathered, not reduced: the mixture sums the ranks' partials in fp32 dest, so the
        # exchange only moves them, overlapped with the reduction of the next candidate.
        stats = ttnn.experimental.deepseek_prefill.attn_res_block_stats(
            sealed,
            self._score_columns(queries),
            gathered=self._gathered_stats(sealed, sealed[0].shape[-2]),
            cluster_axis=self.tp_axis,
            semaphore=self._block_stats_sem,
            topology=self.topology[self.tp_axis],
            num_links=self.num_links,
            compute_kernel_config=STATS_FIDELITY,
        )

        return ttnn.experimental.deepseek_prefill.attn_res_block_mix(
            sealed,
            stats,
            num_sites=len(queries),
            eps=self.eps,
            num_closed_sites=num_closed,
        )

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
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
            )
            self._exchange_scratch[num_tokens] = stats
        return stats

    def _gathered_stats(self, sealed, num_tokens):
        """Where a block's statistics are gathered: `[1, C, N, 32 * tp]` float32, rank p's in
        tile column p.

        Kept for the walk, one per sealed-set depth, for the same reason as the read's
        exchange buffer: peers write into it by address, so it has to sit at the same
        address on every chip and must not be freed and reallocated while a chip that has
        not caught up could still be using that memory.

        Reuse is handshaked inside the op: a peer writes here only once this chip's
        `attn_res_block_stats` has started, and a chip runs its programs in order, so the
        previous call's `attn_res_block_mix` is done reading by then. Any order of calls is
        safe, back-to-back `inter_block`s of one shape included.
        """
        num_candidates = sum(snapshot.shape[1] for snapshot in sealed)
        key = (num_candidates, num_tokens)
        if key not in self._gathered_stats_by_shape:
            self._gathered_stats_by_shape[key] = ttnn.zeros(
                [1, num_candidates, num_tokens, ttnn.TILE_SIZE * self.tp_factor],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
            )
        return self._gathered_stats_by_shape[key]

    def inter_block(self, sealed, queries, num_closed=0):
        """Sealed-snapshot half of the mixture, for every read site in a block.

        Everything that does not depend on the read site runs once for the whole
        block: a sealed snapshot is write-once, so its sum of squares is loop-invariant,
        and every site's dot against it comes out of the same pass. The sites batch for
        free because each one's statistic is a single scalar per (token, candidate): the
        whole block's fit in one fp32 tile column beside the sum of squares, which is
        also why a batch holds at most 31 sites.

        Args:
            sealed: the sealed set, one `[1, S, N, d/tp_factor]` tensor or the list of
                them it was sealed as, whose planes in order are the candidates. Not
                empty — the walk places no read site before the first seal, so every
                executed read has one.
            queries: sequence of `[1, 1, 1, d/tp_factor]` folded queries.
            num_closed: how many leading sites have no live stream to fold in. Their read
                is the sealed set's mixture alone, `sum_i e_i v_i / sum_i e_i`, and comes
                back whole rather than as a batch entry.

        Returns:
            `(partials, shifts, masses, closed)`. The partials `[R, 1, N, d/tp_factor]`,
            each site's holding `sum_i e_i v_i`, and the shifts and masses `[R, 1, N, 1]`,
            all batched over the open read sites in the online-softmax convention
            `e_i = exp(s_i - m)`. `merge` takes the batches whole and names the site; keep
            them alive until the block's last read. `closed` is the closed sites' reads,
            `[num_closed, 1, N, d/tp_factor]`, with the batches starting at the first open
            site; `None` without `num_closed`.
        """
        sealed = as_sealed_list(sealed)
        assert sealed, "inter_block needs a sealed snapshot to score against"
        assert len(queries) < ttnn.TILE_SIZE, (
            f"inter_block scores at most {ttnn.TILE_SIZE - 1} read sites at once, got {len(queries)}: "
            "`attn_res_block_stats` lands every site's dot in one fp32 tile column, beside the sum "
            "of squares in column 0"
        )
        return self._inter_block_fused(sealed, queries, num_closed)

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
            # sealed half's gather does not use: one read would route its two halves
            # by different rules wherever the two disagree.
            topology=self.topology[self.tp_axis],
            inv_hidden_size=1.0 / self.hidden_size,
            eps=self.eps,
            pending=pending,
        )
        return tuple(outputs) if pending is not None else outputs[0]
