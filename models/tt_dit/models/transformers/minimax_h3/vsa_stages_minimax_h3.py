# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-side VSA coarse stage for MiniMax-H3 (R3 of VSA_SCOPE.md), unfused ttnn ops.

Per attention call, given per-device Q/K/V heads (post QK-norm + RoPE, tiled order):

(a) pool local Q/K/V per 64-token tile via a matmul with the host-built block-diagonal
    averaging matrix (entries 1/valid_count, zero rows for pad tiles);
(b) all-gather pooled K_c/V_c on the SP axis (persistent-buffer all-gather);
(c) scores = Q_c @ K_c^T / sqrt(head_dim); O_c = softmax(scores) @ V_c broadcast tile -> 64 tokens;
(d) selection: top-k over candidate columns only, every row's index list =
    [all exempt tile ids] + [its top-k candidate ids] (exempt-query rows list all real tile ids),
    emitted as the uint32 sentinel-tailed index tensor vsa_sdpa consumes.

The pooling matmuls run transposed (X^T @ A^T) so the averaging matrix broadcasts across heads as
a batch-1 weight; the pooled tensors are tiny ([heads, head_dim, n_tiles]), so the extra
transposes are cheap. Everything here is static per (geometry, sparsity): no host readback, all
shapes fixed -- trace-compatible.

Exempt-query rows are blended in via a host-built row mask (int32 select on device), because the
per-shard count of exempt rows differs between shards and SPMD mesh ops need uniform shapes.
"""

from __future__ import annotations

import math

import torch
from loguru import logger

import ttnn

from ....pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, MiniMaxH3VSAGeometry
from ....utils.tensor import from_torch
from ....utils.tracing import StateTensor

VSA_INDEX_SENTINEL = 0xFFFFFFFF
_TOPK_K_MULTIPLE = 16  # ttnn.experimental.topk_large_indices wants k % 16 == 0, k in [16, 2048]
# vsa_sdpa's raw-selection prefix is a fixed 32-entry array in the reader kernel
# (vsa_sdpa_device_operation.cpp asserts it); above that the exempt prefix must be assembled on host.
_MAX_RAW_EXEMPT_IDS = 32


from dataclasses import dataclass  # noqa: E402

DEFAULT_VSA_PLACEMENT = "interleaved"


@dataclass(frozen=True)
class MiniMaxH3VSAConfig:
    """Host-side VSA knobs (R5): the attention path is selected by passing this to the model."""

    sparsity: float
    # tile placement across SP shards: "identity" | "striped" | "interleaved". Interleaved (default)
    # spreads the exempt dense-list tiles across shards AND evenly within each shard, so no device
    # (and no worker/pass inside vsa_sdpa) carries a disproportionate share of the dense rows.
    placement: str = DEFAULT_VSA_PLACEMENT
    k_chunk_blocks: int = 1  # vsa_sdpa's m: listed blocks gathered per L1 chunk (v1 kernel only)
    streaming: bool = True  # vsa_sdpa kernel: streaming leader/worker (default) or the v1 per-row gather
    # pooled K^T/V gathers tile-aligned via a padded per-shard coarse numbering (needs streaming);
    # default on: -1.1 ms per block at 15 s, validated by the stage oracle, sparsity-0 and block tests
    padded_pooling: bool = True
    # vsa_sdpa distributed-window kernel (v18): every core of a head group fetches a slice of each KV
    # window and rows gather from the peers' L1 (larger visits); rows are dealt by static cost so the
    # dense rows do not pile onto one core. Needs streaming.
    distributed: bool = False
    # vsa_sdpa KV streaming order: "bstrideR.S" (default: runs of R consecutive slots interleaved across S
    # spatial segments, so one streaming window carries work for several cores -- the leader/consumer
    # convoy, VSA_STREAM_DESIGN.md 11; measured 19.5 -> 17.7 ms on the real 15 s shard) | "identity" |
    # "canonical" | "zorder" (see MiniMaxH3VSAGeometry.stream_order). Deterministic for a given order.
    stream_order: str = "identity"  # bstride4.16 is -9.5% standalone but needs the leader protocol fix (log)
    # Build the VSA modules and load their gates, then run the DENSE attention path anyway. An A/B
    # lever, not a serving knob: a VSA student's `to_gate_compress` has no module to bind to when the
    # attention is built without a config, so this is the only way to hold the adapter weights fixed
    # while changing the attention.
    bypass: bool = False


def compute_topk(sparsity: float, num_candidates: int) -> int:
    """Candidate tiles to keep; FastVideo's compute_topk, clamped to [1, n]."""
    return max(1, min(math.ceil((1 - sparsity) * num_candidates), num_candidates))


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


class MiniMaxH3VSACoarseStage:
    """Coarse VSA stage for one ladder rung.

    Rung-constant tensors (broadcast, pad/exempt/blend-keep structure) upload in `__init__`. The
    per-request content -- averaging matrices, candidate mask, block counts, dense blend -- lives in
    `StateTensor`s that `bind_request` refreshes: the rung's trace bakes the addresses `__call__`
    reads, so a new request copies into the same buffers rather than allocating fresh ones. `k` and
    every device shape derive from the geometry's rung structure (fixed across requests), so the
    stage is built once per bucket and reused for every request that lands on it.
    """

    def __init__(
        self,
        geometry: MiniMaxH3VSAGeometry,
        *,
        sparsity: float,
        padded_pooling: bool = False,
        head_dim: int,
        num_heads: int,
        mesh_device: ttnn.MeshDevice,
        sp_axis: int,
        ccl_manager=None,
    ) -> None:
        self.geometry = geometry
        self.sparsity = sparsity
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.ccl_manager = ccl_manager

        n_tiles = geometry.n_tiles
        tiles_per_shard = geometry.tiles_per_shard
        rows_per_shard = tiles_per_shard  # one selection row per 64-token q tile
        exempt_ids = torch.nonzero(geometry.is_exempt, as_tuple=False).reshape(-1)
        self.n_exempt = int(exempt_ids.numel())
        self.n_candidates = geometry.candidate_capacity  # rung-constant top-k denominator
        self.k = compute_topk(sparsity, self.n_candidates)
        self.k_pad = max(_TOPK_K_MULTIPLE, _round_up(self.k, _TOPK_K_MULTIPLE))
        # Index width: global tile count, padded so W*4 bytes meets DRAM row alignment and the
        # static row layout [exempt | top-k | sentinel tail] fits.
        self.index_width = _round_up(max(n_tiles, self.n_exempt + self.k), _TOPK_K_MULTIPLE)

        # Padded pooling pads each shard's tile axis to a power of two >= 32 (zero columns) so the
        # pooled K^T / V gathers are tile-aligned (the CCL otherwise falls back to a broadcast+concat
        # composite, ~1.6 ms per block at 15 s). Scores/top-k then run in the padded per-shard
        # numbering (shard j, slot s -> j * slots + s); vsa_sdpa maps ids back (coarse_slots_shift).
        # An exempt-tile count past the kernel's raw prefix assembles indices on host; padded pooling
        # rides on raw selection (the kernel maps padded ids back), so it stands down together with it.
        self.raw_selection_ok = self.n_exempt <= _MAX_RAW_EXEMPT_IDS
        if padded_pooling and not self.raw_selection_ok:
            logger.info(
                f"VSA: {self.n_exempt} exempt tiles exceeds the kernel's {_MAX_RAW_EXEMPT_IDS}-id raw prefix; "
                "falling back to host index assembly and unpadded pooling"
            )
            padded_pooling = False
        self.padded_pooling = padded_pooling
        self.slots_per_shard = tiles_per_shard
        if padded_pooling:
            self.slots_per_shard = 32
            while self.slots_per_shard < tiles_per_shard:
                self.slots_per_shard *= 2
        self.coarse_slots_shift = self.slots_per_shard.bit_length() - 1 if padded_pooling else 0
        self.n_coarse_cols = geometry.sp_factor * self.slots_per_shard if padded_pooling else n_tiles

        # tile -> token broadcast for the coarse output, B = kron(I_tiles_local, ones(1, 64)) as
        # [tiles_local, S_local] (replicated: every shard has the same local structure). A 0/1
        # matmul copies each tile's row to its 64 tokens exactly; used in place of
        # repeat_interleave, whose permute/concat/tilize chain cost ~3 ms at 15 s. Stored transposed
        # as [S_local, slots]: o_c = B^T @ o_c_tiles[slots, H*d] lands directly in the gate's layout.
        bcast = torch.kron(torch.eye(tiles_per_shard), torch.ones(1, VSA_TILE_TOKENS))
        if padded_pooling:  # pooled rows are padded to slots_per_shard; the extra rows broadcast nothing
            bcast = torch.nn.functional.pad(bcast, (0, 0, 0, self.slots_per_shard - tiles_per_shard))
        self.bcast_flat = from_torch(
            bcast.T.contiguous().reshape(1, 1, tiles_per_shard * VSA_TILE_TOKENS, self.slots_per_shard),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            mesh_axes=None,
        )

        # padded pooling adds (slots_per_shard - tiles_per_shard) zero-key slots per shard: the top-k masks
        # them via cand_mask, but the coarse o_c softmax must not give them exp(0) weight either (it did:
        # attention-level oracle 92.7% -> fixed). Real pad TILES stay in, as in the unpadded path.
        self.pool_pad_mask = None
        if padded_pooling:
            pad_mask = torch.zeros(geometry.sp_factor, self.slots_per_shard)
            pad_mask[:, tiles_per_shard:] = -float("inf")
            self.pool_pad_mask = from_torch(
                pad_mask.reshape(1, 1, 1, self.n_coarse_cols), device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=None
            )

        # Rung-constant selection structure: exempt-slot ids and per-shard exempt-row layout are fixed
        # by the placement, so the exempt prefix, sentinel tail and blend-keep mask upload once here;
        # only the dense-row content (which real tiles exist) is per-request.
        self._host_exempt_ids = exempt_ids
        self._host_row_exempt = geometry.is_exempt.reshape(geometry.sp_factor, rows_per_shard)
        self._upload_row_structure(num_heads)
        if self.raw_selection_ok:  # rung-constant; the raw kernel reads it inside the trace
            self.dense_row_mask_tensor()

        # Per-request content, refreshed in place by `bind_request`.
        self.a_t_kv = StateTensor()
        self.a_t_q = StateTensor()
        self.a_kv_flat = StateTensor()
        self.cand_mask = StateTensor()
        self.blend_dense = StateTensor()
        self._block_counts = StateTensor()
        self._bound_signature = None
        # The tile <-> packed-row gathers run outside the traced block stack, but trace replay
        # overwrites device memory allocated after capture, so these buffers must be as
        # address-stable as the in-trace content: StateTensors, refreshed in place.
        self.pack_idx = StateTensor()
        self.unpack_idx = StateTensor()

    def _upload_row_structure(self, num_heads: int) -> None:
        """Head-expanded, rung-constant selection tensors: exempt prefix, sentinel tail, blend keep."""
        geometry = self.geometry
        rows = geometry.tiles_per_shard
        w = self.index_width
        mesh_axes = [self.sp_axis, None, None, None]

        def upload(x: torch.Tensor, dtype, layout=ttnn.Layout.TILE) -> ttnn.Tensor:
            return from_torch(x.contiguous(), device=self.mesh_device, dtype=dtype, layout=layout, mesh_axes=mesh_axes)

        prefix = (
            self._host_exempt_ids.to(torch.int32).reshape(1, 1, 1, -1).expand(geometry.sp_factor, num_heads, rows, -1)
        )
        self.exempt_prefix = upload(prefix, ttnn.uint32, ttnn.Layout.ROW_MAJOR)

        tail = w - self.n_exempt - self.k
        self.sentinel_tail = None
        if tail > 0:
            sentinel = torch.full((geometry.sp_factor, num_heads, rows, tail), -1, dtype=torch.int32)
            self.sentinel_tail = upload(sentinel, ttnn.uint32, ttnn.Layout.ROW_MAJOR)

        row_exempt = self._host_row_exempt.to(torch.int32)  # [sp, rows]
        keep = (1 - row_exempt).reshape(geometry.sp_factor, 1, rows, 1).expand(-1, num_heads, -1, w)
        self.blend_keep = upload(keep, ttnn.int32)

    def _averaging_torch(self, geometry: MiniMaxH3VSAGeometry):
        """(a_t_kv, a_t_q, a_kv_flat) host tensors: shard s owns the diagonal block of A^T."""
        matrix = geometry.averaging_matrix()  # [n_tiles, padded_len] fp64->fp32
        tps = geometry.tiles_per_shard
        rows_local = geometry.padded_len // geometry.sp_factor
        a_t = torch.cat(
            [
                matrix[s * tps : (s + 1) * tps, s * rows_local : (s + 1) * rows_local].T
                for s in range(geometry.sp_factor)
            ],
            dim=0,
        )  # [padded_len, tiles_per_shard]
        if self.padded_pooling:
            a_t = torch.nn.functional.pad(a_t, (0, self.slots_per_shard - tps))
        pl, sl = geometry.padded_len, self.slots_per_shard
        return (
            a_t.reshape(1, 1, pl, sl),
            (a_t / math.sqrt(self.head_dim)).reshape(1, 1, pl, sl),  # scores scale baked into Q's matrix
            a_t.T.contiguous().reshape(1, 1, sl, pl),  # left-multiply form for the transpose-free V pool
        )

    def _cand_mask_torch(self, geometry: MiniMaxH3VSAGeometry) -> torch.Tensor:
        """Additive score-column mask: 0 for real candidates, -inf elsewhere (empty slots included)."""
        cand = torch.where(geometry.is_candidate, 0.0, -float("inf")).to(torch.float32)
        if self.padded_pooling:  # shard-major padded columns; pad slots are never candidates
            tps = geometry.tiles_per_shard
            padded = torch.full((geometry.sp_factor, self.slots_per_shard), -float("inf"))
            padded[:, :tps] = cand.reshape(geometry.sp_factor, tps)
            cand = padded.reshape(-1)
        return cand.reshape(1, 1, 1, self.n_coarse_cols)

    def _block_counts_torch(self, geometry: MiniMaxH3VSAGeometry) -> torch.Tensor:
        counts = torch.zeros(self.index_width, dtype=torch.int32)
        counts[: geometry.n_tiles] = geometry.valid_counts.to(torch.int32)
        return counts.reshape(1, 1, 1, -1)

    def _blend_dense_torch(self, geometry: MiniMaxH3VSAGeometry) -> torch.Tensor:
        """[sp, H, rows, W] dense-list masked by exempt rows: final = sparse * keep + this."""
        w = self.index_width
        real_ids = torch.nonzero(geometry.valid_counts > 0, as_tuple=False).reshape(-1)
        dense_row = torch.full((w,), VSA_INDEX_SENTINEL, dtype=torch.int64)
        dense_row[: real_ids.numel()] = real_ids
        row_exempt = self._host_row_exempt.to(torch.int32)
        dense = dense_row.to(torch.int32).reshape(1, 1, 1, w) * row_exempt.reshape(geometry.sp_factor, 1, -1, 1)
        return dense.expand(-1, self.num_heads, -1, -1)

    def bind_request(self, geometry: MiniMaxH3VSAGeometry, *, traced: bool) -> None:
        """Refresh the per-request content buffers for `geometry`; a no-op when the request repeats.

        Must run before the rung's trace captures (the pipeline binds on the untraced pass); a traced
        call copies into the buffers the capture baked. `gather_index` fully determines the content,
        so an unchanged one leaves the buffers as-is -- steady-state serving of one shape uploads nothing.
        """
        signature = hash(geometry.gather_index.numpy().tobytes())
        if signature == self._bound_signature and self.a_t_kv.value is not None:
            return
        self.geometry = geometry
        md, sp = self.mesh_device, self.sp_axis
        a_t_kv, a_t_q, a_kv_flat = self._averaging_torch(geometry)
        self.a_t_kv.update(from_torch(a_t_kv, device=md, dtype=ttnn.bfloat16, mesh_axes=[..., sp, None]), traced=traced)
        self.a_t_q.update(from_torch(a_t_q, device=md, dtype=ttnn.bfloat16, mesh_axes=[..., sp, None]), traced=traced)
        self.a_kv_flat.update(
            from_torch(a_kv_flat, device=md, dtype=ttnn.bfloat16, mesh_axes=[..., None, sp]), traced=traced
        )
        self.cand_mask.update(
            from_torch(self._cand_mask_torch(geometry), device=md, dtype=ttnn.bfloat16, mesh_axes=None), traced=traced
        )
        self._block_counts.update(
            from_torch(
                self._block_counts_torch(geometry),
                device=md,
                dtype=ttnn.uint32,
                layout=ttnn.Layout.ROW_MAJOR,
                mesh_axes=None,
            ),
            traced=traced,
        )
        if not self.raw_selection_ok:
            self.blend_dense.update(
                from_torch(
                    self._blend_dense_torch(geometry), device=md, dtype=ttnn.int32, mesh_axes=[sp, None, None, None]
                ),
                traced=traced,
            )
        self.pack_idx.update(
            from_torch(
                geometry.row_source.to(torch.int32).reshape(1, -1),
                device=md,
                dtype=ttnn.uint32,
                layout=ttnn.Layout.ROW_MAJOR,
                mesh_axes=None,
            ),
            traced=traced,
        )
        # untile_index is [seq_len] (this request's packed length); pad to the rung-constant row
        # count so the in-place refresh keeps one buffer shape. The pad rows land past seq_len in
        # the unpacked tensor, beyond every output-index selection, so their content is a don't-care.
        rung_rows = geometry.n_tiles * VSA_TILE_TOKENS
        unpack = torch.zeros(rung_rows, dtype=torch.int32)
        unpack[: geometry.untile_index.numel()] = geometry.untile_index.to(torch.int32)
        self.unpack_idx.update(
            from_torch(
                unpack.reshape(1, -1),
                device=md,
                dtype=ttnn.uint32,
                layout=ttnn.Layout.ROW_MAJOR,
                mesh_axes=None,
            ),
            traced=traced,
        )
        self._bound_signature = signature

    def _all_gather(self, x: ttnn.Tensor, dim: int) -> ttnn.Tensor:
        if self.geometry.sp_factor == 1:
            return x
        return self.ccl_manager.all_gather_persistent_buffer(x, dim=dim, mesh_axis=self.sp_axis)

    def _pool_program_config(self, m: int, k: int, n: int | None = None):
        """Full-grid multicast config for the pooling products ([H*d, S_local] @ [S_local, slots], or
        [slots, S_local] @ [S_local, H*d]): the default picked 80 cores and 0.48 ms at 15 s; the full grid
        with in0_block_w=4 measured 0.34 ms (0.22 ms for the flat form). Cached per (m, k, n)."""
        n = self.slots_per_shard if n is None else n
        key = (m, k, n)
        cache = self.__dict__.setdefault("_pool_cfg", {})
        if key not in cache:
            grid = self.mesh_device.compute_with_storage_grid_size()
            m_tiles, n_tiles, k_tiles = m // 32, n // 32, k // 32
            if min(m_tiles, n_tiles, k_tiles) == 0:  # sub-tile shapes (tiny tests): default matmul config
                cache[key] = None
                return None
            in0_block_w = next((w for w in (4, 2, 1) if k_tiles % w == 0), 1)
            cache[key] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=-(-m_tiles // grid.y),
                per_core_N=-(-n_tiles // grid.x),
                transpose_mcast=False,
                fused_activation=None,
            )
        return cache[key]

    def pool_t(self, x_bhnd: ttnn.Tensor, *, scaled: bool) -> ttnn.Tensor:
        """Head-split [1, H, S_local, d] -> pooled and TRANSPOSED [1, H, d, slots] (fp math in bf16, matches
        the oracle to bf16). One large transpose of the input, none of the output: K is consumed in this
        orientation (k_c^T for the scores) and Q's transpose back is on the small pooled tensor."""
        _, num_heads, s_local, d = x_bhnd.shape
        x_t = ttnn.transpose(x_bhnd, 2, 3)  # [1, H, d, S_local]
        # Fold heads into M: per head the product has only (d/32) x (slots/32) output tiles, which is all
        # the parallelism a batched matmul gets (the batch-broadcast form measured 3.4 ms). As one
        # [H*d, S_local] @ [S_local, slots] product the same math spans the whole grid. The merge of
        # adjacent tile-aligned dims is a view, no data movement.
        x_t = ttnn.reshape(x_t, [1, 1, num_heads * d, s_local])
        pooled_t = ttnn.matmul(
            x_t,
            self.a_t_q.value if scaled else self.a_t_kv.value,
            program_config=self._pool_program_config(num_heads * d, s_local),
        )  # [1, 1, H*d, slots]
        ttnn.deallocate(x_t)
        return ttnn.reshape(pooled_t, [1, num_heads, d, pooled_t.shape[-1]])

    def pool(self, x_bhnd: ttnn.Tensor, *, scaled: bool) -> ttnn.Tensor:
        """[1, H, S_local, d] -> pooled [1, H, slots, d]."""
        pooled_t = self.pool_t(x_bhnd, scaled=scaled)
        pooled = ttnn.transpose(pooled_t, 2, 3)  # small: [1, H, d, slots] -> [1, H, slots, d]
        ttnn.deallocate(pooled_t)
        return pooled

    def pool_flat(self, x_1bnf: ttnn.Tensor, num_heads: int) -> ttnn.Tensor:
        """Un-split [1, 1, S_local, H*d] -> pooled head-split [1, H, slots, d] with no large transpose:
        A[slots, S_local] @ x, then the head split on the small pooled tensor (26 us at 15 s)."""
        _, _, s_local, hd = x_1bnf.shape
        pooled = ttnn.matmul(
            self.a_kv_flat.value, x_1bnf, program_config=self._pool_program_config(self.slots_per_shard, s_local, hd)
        )  # [1, 1, slots, H*d]
        heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            pooled, num_heads=num_heads, num_kv_heads=0, transpose_k_heads=False
        )
        ttnn.deallocate(pooled)
        return heads  # [1, H, slots, d]

    def __call__(
        self,
        q_bhnd: ttnn.Tensor,
        k_bhnd: ttnn.Tensor,
        v_bhnd: ttnn.Tensor | None,
        *,
        v_1bnf: ttnn.Tensor | None = None,
        compute_o_c: bool = True,
        raw_selection: bool = False,
    ) -> tuple[ttnn.Tensor | None, ttnn.Tensor]:
        """Run the coarse stage. Returns (o_c [1,1,S_local,H*d] bf16 or None, indices [1,H,rows,W] uint32 ROW_MAJOR).
        o_c comes in the head-concatenated layout: apply the gate after ``concatenate_heads`` on the fine output.
        Pass V un-split as ``v_1bnf`` ([1,1,S_local,H*d], as it is before the head split) when available: it
        pools with no transpose; ``v_bhnd`` is the head-split fallback (tests).

        ``compute_o_c=False`` skips the coarse-output branch (an all-zero gate weight contributes
        nothing, so skipping it gives identical output); selection always runs.

        ``raw_selection=True`` returns the top-k rows as-is ([1,H,rows,k_pad]); the exempt prefix,
        sentinel tail and dense-row blend move into the streaming kernel via
        ``vsa_sdpa(..., list_len=self.k, exempt_ids=self.exempt_ids, dense_row_mask=self.dense_row_mask_tensor())``.
        Saves ~10 layout ops (~2.5 ms per block at 15 s / 768p).
        """
        num_heads = q_bhnd.shape[1]
        assert self.a_t_kv.value is not None, "call bind_request before the coarse stage runs"

        q_c = self.pool(q_bhnd, scaled=True)  # scores scale baked into the Q averaging matrix
        k_c_t = self.pool_t(k_bhnd, scaled=False)  # [1, H, d, slots]: the scores consume K^T directly

        k_c_t_g = self._all_gather(k_c_t, dim=3)  # [1, H, d, n_tiles]

        scores = ttnn.matmul(q_c, k_c_t_g)  # [1, H, tiles_local, n_tiles]
        ttnn.deallocate(q_c)

        # (c) coarse output, broadcast tile -> 64 tokens
        o_c = None
        if compute_o_c:
            if v_1bnf is not None:
                v_c = self.pool_flat(v_1bnf, num_heads)
            else:
                assert v_bhnd is not None, "the coarse output needs V (head-split or un-split)"
                v_c = self.pool(v_bhnd, scaled=False)
            v_c_g = self._all_gather(v_c, dim=2)  # [1, H, n_tiles, d]
            if self.pool_pad_mask is not None:
                scores_m = ttnn.add(scores, self.pool_pad_mask)
                ttnn.deallocate(scores)
                scores = scores_m
            probs = ttnn.softmax(scores, dim=-1)
            # batched [H, slots, cols] @ [H, cols, d]: the default config ran on 32 cores (0.62 ms at
            # 15 s); splitting batch x M over the grid with 2 M-tiles per core measured 0.13 ms
            grid = self.mesh_device.compute_with_storage_grid_size()
            # Tile counts must be the PADDED counts (ceil), not floor: the matmul validates
            # per_core_M / per_core_N against the tile-padded dims. On the unpadded coarse path
            # tiles_per_shard need not be tile-even (e.g. 88 -> 3 tiles), so a floor //32 picked an
            # even m_tiles that does not divide the real 3-tile M -- a TT_FATAL in the o_c matmul.
            m_tiles = -(-probs.shape[2] // 32)
            n_tiles = -(-v_c_g.shape[3] // 32)
            k_tiles = -(-probs.shape[3] // 32)
            o_c_cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=next((w for w in (4, 2, 1) if k_tiles % w == 0), 1),
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=2 if m_tiles % 2 == 0 else 1,
                per_core_N=n_tiles,
            )
            o_c_tiles = ttnn.matmul(probs, v_c_g, program_config=o_c_cfg)  # [1, H, slots, d]
            ttnn.deallocate(probs)
            # broadcast tile -> 64 tokens as a 0/1 matmul in the head-concatenated layout (see bcast_flat):
            # [S_local, T] @ [T, H*d] -> o_c [1, 1, S_local, H*d]; the head concat runs on the small pooled
            # tensor. Replaces two large transposes around the [H*d, T] @ [T, S_local] form.
            o_flat = ttnn.transformer.concatenate_heads(o_c_tiles)  # [1, slots, H*d]
            ttnn.deallocate(o_c_tiles)
            o_flat = ttnn.reshape(o_flat, [1, 1, o_flat.shape[-2], o_flat.shape[-1]])
            o_c = ttnn.matmul(self.bcast_flat, o_flat)  # [1, 1, S_local, H*d]
            ttnn.deallocate(o_flat)

        # (e) selection: top-k over candidate columns only
        masked = ttnn.add(scores, self.cand_mask.value)  # -inf on non-candidate columns
        ttnn.deallocate(scores)
        masked_rm = ttnn.to_layout(masked, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(masked)
        topk_ids = ttnn.experimental.topk_large_indices(masked_rm, k=self.k_pad)  # [1,H,rows,k_pad] uint32
        ttnn.deallocate(masked_rm)
        if self.padded_pooling:  # drop the pad q-tile rows (pooled Q was padded to slots_per_shard rows)
            topk_ids = ttnn.slice(topk_ids, [0, 0, 0, 0], [1, num_heads, self.geometry.tiles_per_shard, self.k_pad])
        if raw_selection:
            return o_c, topk_ids
        assert not self.padded_pooling, "padded_pooling needs raw_selection (the kernel maps the padded ids)"
        if self.k != self.k_pad:
            topk_ids = ttnn.slice(topk_ids, [0, 0, 0, 0], [1, num_heads, self.geometry.tiles_per_shard, self.k])

        parts = [self.exempt_prefix, topk_ids] + ([self.sentinel_tail] if self.sentinel_tail is not None else [])
        sparse_rows = ttnn.concat(parts, dim=-1)  # [1, H, rows, W]
        ttnn.deallocate(topk_ids)

        # exempt-query rows take the dense list: final = sparse * keep + dense_masked (int32 select;
        # the sentinel 0xFFFFFFFF is -1 in int32 and survives the arithmetic)
        sparse_i32 = ttnn.typecast(ttnn.to_layout(sparse_rows, ttnn.TILE_LAYOUT), ttnn.int32)
        ttnn.deallocate(sparse_rows)
        blended = ttnn.add(ttnn.multiply(sparse_i32, self.blend_keep), self.blend_dense.value)
        ttnn.deallocate(sparse_i32)
        indices = ttnn.to_layout(ttnn.typecast(blended, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(blended)

        return o_c, indices

    @property
    def exempt_ids(self) -> list[int]:
        """Global ids of the exempt (dense-list) blocks every row attends (raw-selection mode)."""
        return [int(b) for b in self._host_exempt_ids.tolist()]

    @property
    def dense_row_hint(self) -> list[int]:
        """q-tile rows that take the dense list on SOME shard (the union over shards): the distributed
        kernel weights them as dense when dealing rows to cores (an op attribute, so mesh-uniform)."""
        return sorted(set(torch.nonzero(self._host_row_exempt.any(dim=0)).reshape(-1).tolist()))

    def stream_order_tensor(self, kind: str) -> ttnn.Tensor:
        """[1,1,1,n_tiles] uint32 permutation for vsa_sdpa(stream_order=...), replicated. Cached per kind."""
        cache = self.__dict__.setdefault("_stream_order_cache", {})
        if kind not in cache:
            perm = self.geometry.stream_order(kind).to(torch.int32).reshape(1, 1, 1, -1)
            cache[kind] = from_torch(
                perm, device=self.mesh_device, dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, mesh_axes=[None] * 4
            )
        return cache[kind]

    def dense_row_mask_tensor(self) -> ttnn.Tensor:
        """[1,1,1,words] uint32 ROW_MAJOR per device: bit q_tile set -> that row takes the dense list
        (raw-selection mode). Sharded over the SP axis like the selection constants. Cached."""
        if not hasattr(self, "_dense_row_mask"):
            rows = self.geometry.tiles_per_shard
            words = _round_up(max(8, (rows + 31) // 32), 8)  # words*4 bytes must be a multiple of 32
            mask = torch.zeros(self.geometry.sp_factor, words, dtype=torch.int64)
            row_exempt = self._host_row_exempt  # [sp, rows] bool
            for s in range(self.geometry.sp_factor):
                for r in torch.nonzero(row_exempt[s]).reshape(-1).tolist():
                    mask[s, r // 32] |= 1 << (r % 32)
            self._dense_row_mask = from_torch(
                mask.to(torch.int32).reshape(self.geometry.sp_factor, 1, 1, words),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.Layout.ROW_MAJOR,
                mesh_axes=[self.sp_axis, None, None, None],
            )
        return self._dense_row_mask

    def block_counts_tensor(self) -> ttnn.Tensor:
        """[1,1,1,W] uint32 valid tokens per block (vsa_sdpa's block_counts input); per-request."""
        return self._block_counts.value
