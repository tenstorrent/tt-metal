# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer of the full-indexer layers.

Adapted from models/demos/deepseek_v3_d_p/tt/mla/indexer.py (TtIndexer), without its q-LoRA input, Hadamard basis and
full-mesh striped cache. Per chip (SP row r of 8, TP column t of 4), for one 5120-token chunk at ``start``:

    x        [1, 1, 640, 1536] attn_norm -> all_gather(axis 1) -> x_full [1, 1, 640, 6144]
    keys     k = LayerNorm(x_full @ wk) (eps 1e-6, bias), RoPE on dims 0-63 (rope layers) -> bfp8 ->
             update_padded_kv_cache into the served index cache (SP-sharded, TP-replicated, block-cyclic)
    queries  this column's 160 of the row's 640 rows (mesh_partition axis 1): q = LayerNorm(x) @ wq_b (all 32 heads,
             replicated) -> heads [1, 32, 160, 128], RoPE on dims 0-63
    weights  w = x @ weights_proj * 32^-0.5 * 128^-0.5 (folded at load) [1, 1, 160, 32], times the softmax scaling
             f[p, h] = s_h log(p + 1) + b_h. The reference scales q by f; f > 0 on every position (checked at load),
             so relu(f q.k) = f relu(q.k) and f moves onto the head weight exactly. f is a per-layer table applied
             with rotary_embedding_indexed (cos = f, sin = 0), so any 32-aligned start needs no host slice.
    score    INDEXER_SCORE "composed" (default): high_bw_all_gather over SP of this slot's index keys (rank-major) ->
             natural order by a reshape / permute of the block-cyclic blocks -> per head relu(q_h k^T) (linear, fp32)
             times the head weight, summed over the 32 heads in fp32; the chunk's own key columns get a
             constant per-chip causal mask (0 / -inf, built at load). "ring": ring_indexer_score_dsa (same gather and
             causal rule in one op, but it sums the heads in bf16 DEST).
             Then topk_large_indices k=2048 over [0, end) -> uint32 [1, 1, 160, 2048], 0xFFFFFFFF pads
             (a row that sees fewer than 2048 keys), sorted by score, not by position.

The output stays split over TP and SP: chip (r, t) holds query rows 640 r + 160 t + [0, 160) of the chunk.
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import mm_configs
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.attention import _stamp_sp_topology, _write_cache_meta
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import SP_AXIS, TP_AXIS
from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import seqbuf
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.rope import RowTables, log_rows, log_split, rope_tables, scale_tables, select_split, split_tables
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

NORM_EPS = 1e-6


def hadamard_matrix(n: int) -> torch.Tensor:
    """The Sylvester Walsh-Hadamard matrix H_n / sqrt(n) (fp32; symmetric, orthonormal); must match the rotation the
    decode indexer applies to the keys it caches."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert h.shape[0] == n, f"{n} is not a power of two"
    return h / n**0.5


def _compute_config(mesh, fidelity: str, fp32: bool):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=False,
    )


def _topology():
    return getattr(ttnn.Topology, settings.get("INDEXER_TOPOLOGY"))


class TtIndexer:
    # read at call time, not build time: settings may change between runs of one loaded model
    @property
    def mm_cfg(self):
        return _compute_config(self.mesh, settings.get("INDEXER_MATH_FIDELITY"), settings.get("INDEXER_FP32_ACC"))

    @property
    def score_cfg(self):
        return _compute_config(self.mesh, settings.get("INDEXER_SCORE_FIDELITY"), False)

    @property
    def own_bf16(self):
        return self.composed and settings.get("INDEXER_OWN_KEYS") == "bfloat16"

    @property
    def shift(self):
        return self.composed and settings.get("INDEXER_SCORE_SHIFT")

    # the score path, LayerNorm and query dtype are read at call time (both paths' tables are built)
    @property
    def composed(self):
        return settings.get("INDEXER_SCORE") == "composed"

    @property
    def norm_composed(self):
        return settings.get("INDEXER_NORM") == "composed"

    @property
    def q_fp32(self):
        return settings.get("INDEXER_Q_DTYPE") == "float32"

    @property
    def q_norm_w(self):
        return self._norm["composed" if self.norm_composed else "fused"][0]

    @property
    def q_norm_b(self):
        return self._norm["composed" if self.norm_composed else "fused"][1]

    @property
    def k_norm_w(self):
        return self._norm["composed" if self.norm_composed else "fused"][2]

    @property
    def k_norm_b(self):
        return self._norm["composed" if self.norm_composed else "fused"][3]

    def __init__(
        self,
        mesh,
        w: dict,
        *,
        inv_freq: torch.Tensor | None,
        n_heads: int,
        head_dim: int,
        rope_dim: int,
        topk: int,
        max_seq: int,
        chunk: int,
        num_users: int = 2,
        cache=None,
        cache_layer: int = 0,
        cache_layers: int = 1,
    ):
        """w: idx_wq_b [Hi*D, H], idx_wk [D, H], idx_weights_proj [Hi, H], idx_q_norm_w / _b [H], idx_k_norm_w / _b
        [D], idx_ssmax_s / _b [Hi] (host, any float dtype). inv_freq: the RoPE inverse frequencies (None on NoPE
        layers). cache: the served index cache [num_users * cache_layers, 1, max_seq / 8, D] (allocated here when
        None); this layer owns its slot ``cache_layer``."""
        self.mesh, self.n_heads, self.head_dim, self.rope_dim, self.topk = mesh, n_heads, head_dim, rope_dim, topk
        self.sp, self.tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
        self.max_seq, self.chunk = max_seq, chunk
        self.rows = chunk // self.sp  # this SP row's query / key rows per chunk
        self.q_rows = self.rows // self.tp  # this chip's query rows
        assert self.q_rows % ttnn.TILE_SIZE == 0, f"{self.q_rows} query rows per chip are not whole tiles"
        self.cache_layer, self.cache_layers = cache_layer, cache_layers
        wdt = ttnn.bfloat8_b

        def repl(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.float().contiguous(),
                device=mesh,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def norm_w(t, composed):  # [n], TILE (as TtIndexer's k_norm); fp32 [1, n] for the composed LayerNorm
            return repl(t.float().reshape(1, -1), ttnn.float32) if composed else repl(t.float())

        self.wq_b = repl(w["idx_wq_b"].float().t(), wdt)  # [H, Hi*D]
        self.wk = repl(w["idx_wk"].float().t(), getattr(ttnn, settings.get("INDEXER_WK_DTYPE")))  # [H, D]
        scale = n_heads**-0.5 * head_dim**-0.5
        self.wproj = repl(w["idx_weights_proj"].float().t() * scale)  # [H, Hi], bf16 (the gate is precision-sensitive)
        self._norm = {
            mode: tuple(
                norm_w(w[f"idx_{n}"], mode == "composed") for n in ("q_norm_w", "q_norm_b", "k_norm_w", "k_norm_b")
            )
            for mode in ("composed", "fused")
        }

        # the host inputs of the max_seq-sized tables, kept for resize; the metadata path's fp32 tables are built
        # when first used
        self._inv_freq = inv_freq
        self._ssmax_sb = (w["idx_ssmax_s"].float().clone(), w["idx_ssmax_b"].float().clone())
        self._split = None
        self._repl = repl
        self.w_dtype = getattr(ttnn, settings.get("INDEXER_HEAD_WEIGHT_DTYPE"))
        self._rot = None
        if rope_dim:
            rot = torch.zeros(rope_dim, rope_dim)
            rot[torch.arange(0, rope_dim, 2), torch.arange(1, rope_dim, 2)] = 1  # (x[2i], x[2i+1]) -> (-x[2i+1], x[2i])
            rot[torch.arange(1, rope_dim, 2), torch.arange(0, rope_dim, 2)] = -1
            self._rot = repl(rot, ttnn.float32)
        # INDEXER_HADAMARD: H128 / sqrt(128) (symmetric and orthonormal) in each dtype keys and queries take
        self.hadamard = None
        if settings.get("INDEXER_HADAMARD"):
            h = hadamard_matrix(head_dim)
            self.hadamard = {ttnn.float32: repl(h, ttnn.float32), ttnn.bfloat16: repl(h, ttnn.bfloat16)}
        self.cols = repl(torch.arange(chunk).float().reshape(1, 1, 1, chunk), ttnn.float32)

        self.num_users, self._own_cache = num_users, cache is None
        self.cache = cache
        self.rope = self.ssmax = self.ssmax_sb = self.k_full = None
        self._eager = None
        self.build_seq(max_seq, cache)
        # causal mask of this chip's query rows over the chunk's own key columns: chip (r, t) holds chunk rows
        # q = rows r + q_rows t + i, which see chunk column j iff j <= q (the prefix is always visible)
        q = torch.arange(self.sp * self.tp * self.q_rows).view(self.sp, self.tp, self.q_rows, 1)
        mask = torch.where(torch.arange(chunk)[None, None, None, :] > q, float("-inf"), 0.0)
        dims = [None, None]
        dims[SP_AXIS], dims[TP_AXIS] = 0, 1
        self.chunk_mask = ttnn.from_torch(
            mask,
            device=mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )
        grid = mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.sems = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)] for _ in range(2)]
        self.sem_i = 0

    # ---- the max_seq-sized tensors (rebuilt by resize, shared over layers with SEQ_BUFFERS=shared)
    def build_seq(self, max_seq: int, cache=None) -> None:
        """RoPE and softmax-scaling tables, the gather scratch and (allocated here when None) the cache, for max_seq.
        The eager path's fp32 RowTables (q / k RoPE, f, query positions) are built on first eager use (_eager_tables):
        the metadata path selects from SplitTables instead, so a traced model never allocates them."""
        mesh, chunk, repl = self.mesh, self.chunk, self._repl
        self.max_seq = max_seq
        # softmax scaling f[p, h] = s_h log(p + 1) + b_h, fp32 on the host, as the reference (in the query dtype)
        s, b = self._ssmax_sb
        f = s[None, :] * torch.arange(1, max_seq + 1).float().log()[:, None] + b[None, :]
        assert bool((f > 0).all()), "softmax scaling is not positive everywhere: it cannot move onto the head weight"
        self.rope = rope_tables(mesh, self._inv_freq, max_seq, chunk) if self.rope_dim else None
        self.ssmax = self.ssmax_sb = None
        if self.w_dtype == ttnn.float32 and settings.get("SSMAX_TABLE") == "shared_log":
            # f = s * log(p + 1) + b in fp32 on the device from one log table per mesh; s_h, b_h as rows
            self.ssmax_sb = (repl(s.reshape(1, 1, 1, -1), ttnn.float32), repl(b.reshape(1, 1, 1, -1), ttnn.float32))
        elif self.w_dtype != ttnn.float32:
            self.ssmax = scale_tables(mesh, f, chunk)
        del f
        self._eager = None
        self._split = None
        self.cache = cache if cache is not None else self.cache
        if self.cache is None:
            self.cache = init_kvpe_cache(
                self.head_dim,
                mesh,
                max_seq,
                tuple(mesh.shape),
                SP_AXIS,
                self.cache_layers,
                num_users=self.num_users,
                dtype=ttnn.bfloat8_b,
            )
        # the score's persistent gather output (this column's whole key cache, one slot, rank-major)
        dt = self.cache.dtype
        self.k_full = seqbuf.get(
            mesh, ("k_full", max_seq, self.head_dim, str(dt)), lambda: repl(torch.zeros(1, 1, max_seq, self.head_dim), dt)
        )
        # the composed score's SP topology and mask, built for either INDEXER_SCORE path (read at run time)
        _stamp_sp_topology(self.cache, mesh)

    def _eager_tables(self) -> dict:
        """The eager path's fp32 RowTables, built on first eager use: q_rope / k_rope (cos, sin, rot) for fp32 RoPE,
        f_rows (this layer's f, SSMAX_TABLE=per_layer; or the shared log rows) and q_pos. Outside a trace capture."""
        if self._eager is None:
            mesh, chunk, max_seq, rope_dim = self.mesh, self.chunk, self.max_seq, self.rope_dim
            pos = torch.arange(max_seq)[:, None].float()
            out = {"q_rope": None, "k_rope": None, "f_rows": None, "log_rows": None}
            if self.w_dtype == ttnn.float32 and self.ssmax_sb is not None:
                out["log_rows"] = log_rows(mesh, max_seq, chunk, self.q_rows, self.q_rows)
            elif self.w_dtype == ttnn.float32:
                # rotary_embedding_indexed takes no fp32 input: the fp32 f of this chip's query rows, sliced per call
                s, b = self._ssmax_sb
                out["f_rows"] = RowTables(mesh, s[None, :] * (pos + 1).log() + b[None, :], chunk, self.q_rows, self.q_rows)
            if rope_dim:
                inv_freq = self._inv_freq

                def fp32_rope():
                    # composed fp32 RoPE: x * cos + (x @ rot) * sin, cos / sin of this chip's query rows (q_rope) or of
                    # its SP row's key rows (k_rope: fp32 keys, which rotary_embedding_indexed does not take)
                    freqs = pos * inv_freq[None, :].float()
                    cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)
                    sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)
                    return (
                        (RowTables(mesh, cos, chunk, self.q_rows, self.q_rows), RowTables(mesh, sin, chunk, self.q_rows, self.q_rows)),
                        (RowTables(mesh, cos, chunk, self.rows), RowTables(mesh, sin, chunk, self.rows)),
                    )

                key = ("idx_rope", max_seq, chunk, self.q_rows, self.rows, seqbuf.tensor_key(inv_freq))
                out["qk_rope"] = seqbuf.get(mesh, key, fp32_rope)
                out["q_rope"] = (*out["qk_rope"][0], self._rot)
                out["k_rope"] = (*out["qk_rope"][1], self._rot)
            # a chunk whose start is not chunk-aligned (a follow-up turn) is rotated over the SP rows (RowTables): its
            # causal mask comes from the positions of this chip's query rows, compared with the chunk's key columns
            out["q_pos"] = seqbuf.get(
                mesh, ("q_pos", max_seq, chunk, self.q_rows), lambda: RowTables(mesh, pos, chunk, self.q_rows, self.q_rows)
            )
            self._eager = out
        return self._eager

    @property
    def q_rope(self):
        return self._eager_tables()["q_rope"]

    @property
    def k_rope(self):
        return self._eager_tables()["k_rope"]

    @property
    def f_rows(self):
        return self._eager_tables()["f_rows"]

    @property
    def q_pos(self):
        return self._eager_tables()["q_pos"]

    def release_seq(self) -> None:
        """Frees this layer's max_seq-sized tensors (a shared one stays: seqbuf.release; the split tables:
        rope.release_split); its own cache too."""
        e = self._eager or {}
        for obj in (self.rope, self.ssmax, self.ssmax_sb, self.k_full, e.get("f_rows"), e.get("log_rows"), e.get("qk_rope"), e.get("q_pos")):
            seqbuf.free(obj)
        self.rope = self.ssmax = self.ssmax_sb = self.k_full = self._eager = self._split = None
        if self._own_cache and self.cache is not None:
            ttnn.deallocate(self.cache)
            self.cache = None

    # ---- pieces
    def _layer_norm(self, x, w, b):
        """LayerNorm over the last dim (eps NORM_EPS). INDEXER_NORM "composed": mean, centre, variance, rsqrt, scale and
        shift as separate fp32 ops."""
        if not self.norm_composed:
            return ttnn.layer_norm(
                x,
                epsilon=NORM_EPS,
                weight=w,
                bias=b,
                compute_kernel_config=self.mm_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        mem = ttnn.DRAM_MEMORY_CONFIG
        x32 = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
        mu = ttnn.mean(x32, dim=-1, keepdim=True, compute_kernel_config=self.mm_cfg)
        xc = ttnn.subtract(x32, mu, memory_config=mem)
        ttnn.deallocate(mu)
        if x32 is not x:
            ttnn.deallocate(x32)
        sq = ttnn.multiply(xc, xc, memory_config=mem)
        var = ttnn.mean(sq, dim=-1, keepdim=True, compute_kernel_config=self.mm_cfg)
        ttnn.deallocate(sq)
        r = ttnn.rsqrt(ttnn.add(var, NORM_EPS), fast_and_approximate_mode=False)
        ttnn.deallocate(var)
        xn = ttnn.multiply(xc, r, memory_config=mem)
        ttnn.deallocate(xc)
        ttnn.deallocate(r)
        xw = ttnn.multiply(xn, w, memory_config=mem)
        ttnn.deallocate(xn)
        out = ttnn.add(xw, b, memory_config=mem)
        ttnn.deallocate(xw)
        return out

    def _rope(self, x, start: int, subshard: bool, meta=None):
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            x,
            self.rope.cos,
            self.rope.sin,
            self.rope.trans,
            start if meta is None else meta.start,
            cluster_axis=SP_AXIS,
            seq_subshard_axis=TP_AXIS if subshard else None,
            rotary_dim=self.rope_dim,
            compute_kernel_config=self.mm_cfg,
        )

    def _split_tables(self) -> dict:
        """The metadata path's exact fp32 tables (rope.SplitTables, selected at the device-held start; the trace-safe
        form of the RowTables slices): cos and sin (shared by every layer with the same frequencies) and this layer's
        softmax scaling f, built from the same host fp32 values as the RowTables. Built on first use, which must be
        outside a trace capture (ChunkMeta's eager warm-up chunk)."""
        if self._split is None:
            pos = torch.arange(self.max_seq)[:, None].float()
            out = {}
            if self.rope_dim:
                freqs = pos * self._inv_freq[None, :].float()
                out["cos"] = split_tables(self.mesh, torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2), self.chunk)
                out["sin"] = split_tables(self.mesh, torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2), self.chunk)
            if self.ssmax_sb is None:
                s, b = self._ssmax_sb
                out["f"] = split_tables(self.mesh, s[None, :] * (pos + 1).log() + b[None, :], self.chunk)
            else:
                out["log"] = log_split(self.mesh, self.max_seq, self.chunk)  # SSMAX_TABLE=shared_log: one per mesh
            for t in out.values():
                t.prepare(self.rows)
                t.prepare(self.q_rows)
            self._split = out
        return self._split

    def _selected(self, meta, name: str, rows: int, subshard: bool):
        """fp32 [1, 1, rows, W] per chip of split table ``name`` at the chunk start meta holds, computed once per chunk
        and shared by the layers with the same table (meta.derived; not deallocated by the caller)."""
        return select_split(meta, self._split_tables()[name], rows, subshard)

    def write_k(self, x_full, start: int, slot: int, end: int, meta=None):
        """Index keys of this chunk -> the cache rows [start, end) of (slot, cache_layer). Returns the chunk's keys of
        every SP row in x's dtype [1, 1, chunk, D] (all_gather over SP, natural order) when INDEXER_OWN_KEYS is bfloat16,
        else None."""
        k = ttnn.linear(
            x_full,
            self.wk,
            program_config=mm_configs.get_indexer(x_full, self.wk),
            dtype=ttnn.float32 if self.norm_composed else None,  # the composed LayerNorm runs in fp32
            compute_kernel_config=self.mm_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kn = self._layer_norm(k, self.k_norm_w, self.k_norm_b)
        ttnn.deallocate(k)
        if self.rope is not None and kn.dtype == ttnn.float32:
            cs = None
            if meta is not None:
                cs = (self._selected(meta, "cos", self.rows, False), self._selected(meta, "sin", self.rows, False))
            kn = self._rope_fp32(kn, self.k_rope if cs is None else (None, None, self._rot), start, self.rows, cs=cs)
        elif self.rope is not None:
            kr = self._rope(kn, start, subshard=False, meta=meta)
            ttnn.deallocate(kn)
            kn = kr
        kn = self._rotate(kn)
        own = None
        if self.own_bf16:
            own = self._natural(
                ttnn.all_gather(kn, dim=2, cluster_axis=SP_AXIS, num_links=settings.get("NUM_LINKS"), topology=_topology()), start
            )
        kc = ttnn.typecast(kn, self.cache.dtype)
        ttnn.deallocate(kn)
        if meta is not None:
            _write_cache_meta(self.cache, kc, meta, self.cache_layer, self.cache_layers, self.chunk)
            ttnn.deallocate(kc)
            return own
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            self.cache,
            kc,
            slot_idx=slot,
            layer_idx=self.cache_layer,
            num_layers=self.cache_layers,
            kv_actual_global=start,
            cluster_axis=SP_AXIS,
            valid_global=end,
        )
        ttnn.deallocate(kc)
        if end < start + self.chunk and end % ttnn.TILE_SIZE:  # the last record's pad rows, zero before the ack
            ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
                self.cache,
                slot,
                self.cache_layer,
                self.cache_layers,
                end,
                self.chunk,
                SP_AXIS,
                pad_align=ttnn.TILE_SIZE,
            )
        return own

    def _rotate(self, x):
        """INDEXER_HADAMARD: x [.., D] @ H128 / sqrt(128) in x's dtype (deallocates x); x itself when off. Keys and queries
        both turn, so every score q.k is unchanged."""
        if self.hadamard is None:
            return x
        out = ttnn.linear(
            x, self.hadamard[x.dtype], dtype=x.dtype, compute_kernel_config=self.mm_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(x)
        return out

    def _queries_fp32(self, qn, start: int, meta=None):
        """LayerNorm'd query rows [1, 1, q_rows, H] -> fp32 heads [1, Hi, q_rows, D], RoPE'd in fp32 (q_rope)."""
        q = ttnn.linear(
            qn, self.wq_b, dtype=ttnn.float32, compute_kernel_config=self.mm_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qn)
        qh = ttnn.permute(ttnn.reshape(q, [1, self.q_rows, self.n_heads, self.head_dim]), (0, 2, 1, 3))
        ttnn.deallocate(q)
        if not self.rope_dim:
            return qh
        cs = None
        if meta is not None:
            cs = (self._selected(meta, "cos", self.q_rows, True), self._selected(meta, "sin", self.q_rows, True))
        return self._rope_fp32(qh, self.q_rope if cs is None else (None, None, self._rot), start, self.q_rows, cs=cs)

    def _rope_fp32(self, x, tables, start: int, rows: int, cs=None):
        """fp32 [1, n, rows, D] -> the same, dims [0, rope_dim) RoPE'd by composition with fp32 row tables (cos, sin,
        rot) sliced at ``start``; deallocates x. cs: (cos, sin) rows already selected (the metadata path's; kept)."""
        cos_t, sin_t, rot = tables
        r, d, n = self.rope_dim, self.head_dim, x.shape[1]
        cos, sin = (cos_t.rows(start), sin_t.rows(start)) if cs is None else cs  # [1, 1, rows, r]
        xr = ttnn.slice(x, [0, 0, 0, 0], [1, n, rows, r])
        sw = ttnn.linear(xr, rot, dtype=ttnn.float32, compute_kernel_config=self.mm_cfg)
        a = ttnn.multiply(xr, cos, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # cos rows broadcast over heads
        bb = ttnn.multiply(sw, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(sw)
        if cs is None and start % self.chunk:  # selections of their own (a chunk-aligned start's are table slices)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        rotated = ttnn.add(a, bb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a)
        ttnn.deallocate(bb)
        if r == d:
            ttnn.deallocate(x)
            return rotated
        out = ttnn.concat([rotated, ttnn.slice(x, [0, 0, 0, r], [1, n, rows, d])], dim=3)
        ttnn.deallocate(rotated)
        ttnn.deallocate(x)
        return out

    def _gather_keys(self, slot: int):
        """This slot's index keys of every SP row in natural order: bf16 [1, 1, max_seq, D] (rows past the valid
        end are stale; the caller slices [0, end))."""
        gathered = ttnn.experimental.high_bw_all_gather(
            self.cache,
            dim=2,
            output_tensor=self.k_full,
            cluster_axis=SP_AXIS,
            num_links=settings.get("NUM_LINKS"),
            input_batch_index=slot * self.cache_layers + self.cache_layer,
        )  # rank-major: rank r holds blocks r, r + sp, ... of ``rows`` positions
        k16 = ttnn.typecast(gathered, ttnn.bfloat16)  # exact: every bfp8 value is a bf16 value
        blocks = self.max_seq // self.chunk
        kn = ttnn.permute(ttnn.reshape(k16, [self.sp, blocks, self.rows, self.head_dim]), (1, 0, 2, 3))
        ttnn.deallocate(k16)  # kn [blocks, sp, rows, D]: natural block b * sp + r
        return ttnn.reshape(kn, [1, 1, self.max_seq, self.head_dim])

    def _head_sum(self, qh, w32, k):
        """sum_h w[:, h] * relu(q_h . k^T), fp32 [1, 1, q_rows, len(k)]."""
        if settings.get("INDEXER_HEAD_SUM") == "batched":
            return self._head_sum_batched(qh, w32, k)
        acc = None
        for h in range(self.n_heads):
            q_h = ttnn.slice(qh, [0, h, 0, 0], [1, h + 1, self.q_rows, self.head_dim])
            s_h = ttnn.linear(
                q_h,
                k,
                transpose_b=True,
                activation="relu",
                dtype=ttnn.float32,
                compute_kernel_config=self.mm_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            w_h = ttnn.slice(w32, [0, 0, 0, h], [1, 1, self.q_rows, h + 1])
            t = ttnn.multiply(s_h, w_h, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # w column broadcast over keys
            ttnn.deallocate(s_h)
            if acc is None:
                acc = t
            else:
                nxt = ttnn.add(acc, t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(acc)
                ttnn.deallocate(t)
                acc = nxt
        return acc

    def _head_sum_batched(self, qh, w32, k):
        """_head_sum with every head in one matmul per block of keys: relu(q . k^T) [1, H, q_rows, block] fp32, times
        the head weights [1, H, q_rows, 1], summed over the heads. The same fp32 products and weights as the per-head
        loop (the heads are summed in a different order); 4 ops per block of INDEXER_HEAD_SUM_BLOCK keys instead of 5
        per head."""
        wt = ttnn.permute(w32, (0, 3, 2, 1))  # [1, H, q_rows, 1]
        n, blk = k.shape[-2], settings.get("INDEXER_HEAD_SUM_BLOCK")
        blk = blk or n
        parts = []
        for a in range(0, n, blk):
            z = min(a + blk, n)
            kb = k if (a == 0 and z == n) else ttnn.slice(k, [0, 0, a, 0], [1, 1, z, self.head_dim])
            s = ttnn.linear(
                qh,
                kb,
                transpose_b=True,
                activation="relu",
                dtype=ttnn.float32,
                compute_kernel_config=self.mm_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            t = ttnn.multiply(s, wt, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # head weight column over the keys
            ttnn.deallocate(s)
            parts.append(ttnn.sum(t, dim=1, keepdim=True, compute_kernel_config=self.mm_cfg))
            ttnn.deallocate(t)
        ttnn.deallocate(wt)
        if len(parts) == 1:
            return parts[0]
        out = ttnn.concat(parts, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in parts:
            ttnn.deallocate(t)
        return out

    def _score_composed(self, qh, wf, start: int, slot: int, end: int, own_k=None):
        """[1, 1, q_rows, end] bf16 ROW_MAJOR scores of keys [0, end), -inf on future columns. own_k: the chunk's
        keys in bf16 (write_k), scored instead of their bfp8 cache rows. Slices and reshapes may be views of their
        input, so only tensors that own their buffer are deallocated, after their last use."""
        k = self._gather_keys(slot) if start or own_k is None else None
        w32 = wf if wf.dtype == ttnn.float32 else ttnn.typecast(wf, ttnn.float32)
        parts = []
        if start:
            parts.append(self._head_sum(qh, w32, ttnn.slice(k, [0, 0, 0, 0], [1, 1, start, self.head_dim])))
        # the chunk's own key columns are scored whole: on a short last chunk the columns past ``end`` are pad keys,
        # which a real query row never sees (causal: they are after it) and topk_large_indices' valid_length drops
        if own_k is not None:
            own = self._head_sum(qh, w32, own_k)
        else:
            own = self._head_sum(qh, w32, ttnn.slice(k, [0, 0, start, 0], [1, 1, start + self.chunk, self.head_dim]))
        if k is not None:
            ttnn.deallocate(k)
        if w32 is not wf:
            ttnn.deallocate(w32)
        rotated = start % self.chunk != 0
        mask = self._rotated_mask(start) if rotated else self.chunk_mask
        parts.append(ttnn.add(own, mask, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        ttnn.deallocate(own)
        if rotated:
            ttnn.deallocate(mask)
        sc = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sc16 = ttnn.typecast(sc, ttnn.bfloat16)
        if self.shift:
            sc16 = self._shift_to_cut(sc, sc16)
        for p in parts:
            ttnn.deallocate(p)
        if sc is not parts[0]:
            ttnn.deallocate(sc)
        logits = ttnn.to_layout(sc16, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(sc16)
        return logits

    def _rotated_mask(self, start: int):
        """The causal mask of a chunk whose start is not chunk-aligned: 0 where chunk column j (natural position
        start + j) <= this chip's query row's position, -inf after (log of the 0 / 1 comparison, as
        deepseek_v3_d_p's sliding-window mask)."""
        pos = self.q_pos.rows(start)  # [1, 1, q_rows, 1] absolute positions
        rel = ttnn.subtract(pos, float(start))
        ttnn.deallocate(pos)
        seen = ttnn.le(self.cols, rel)  # [1, 1, q_rows, chunk]
        ttnn.deallocate(rel)
        mask = ttnn.log(seen)
        ttnn.deallocate(seen)
        return mask

    def _natural(self, own, start: int):
        """The chunk's keys gathered over SP ([1, 1, chunk, D], SP row-major) -> natural order. A chunk at ``start``
        is rotated over the SP rows (the served layout, update_padded_kv_cache's offsets): the row b holding ``start``
        has its first rows - off positions, rows b + 1, ..., b - 1 the next whole blocks, and row b's last ``off`` rows
        are the chunk's last positions. Deallocates ``own`` when it reorders."""
        w = self.rows
        b, off = (start // w) % self.sp, start % w
        if not b and not off:
            return own
        d = self.head_dim
        cut = lambda a, z: ttnn.slice(own, [0, 0, a, 0], [1, 1, z, d])  # noqa: E731
        pieces = [cut(b * w, b * w + w - off)] + [
            cut(((b + k) % self.sp) * w, ((b + k) % self.sp + 1) * w) for k in range(1, self.sp)
        ]
        if off:
            pieces.append(cut(b * w + w - off, (b + 1) * w))
        out = ttnn.concat(pieces, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(own)
        return out

    def _shift_to_cut(self, sc, sc16):
        """bf16 (sc - c): c is each row's top-k-th bf16 score (ttnn.topk values), so the scores that decide the cut
        sit near 0, where bf16 resolves them finely; fp32 subtraction keeps the order. A row with fewer than k
        visible keys (c = -inf) keeps every one of them and is shifted by 0."""
        vals, ind = ttnn.topk(sc16, k=self.topk, dim=-1)
        ttnn.deallocate(ind)
        ttnn.deallocate(sc16)
        c = ttnn.typecast(ttnn.slice(vals, [0, 0, 0, self.topk - 1], [1, 1, self.q_rows, self.topk]), ttnn.float32)
        ttnn.deallocate(vals)
        c0 = ttnn.where(ttnn.isfinite(c), c, 0.0)
        ttnn.deallocate(c)
        shifted = ttnn.subtract(sc, c0, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # c column broadcast over keys
        ttnn.deallocate(c0)
        out = ttnn.typecast(shifted, ttnn.bfloat16)
        ttnn.deallocate(shifted)
        return out

    def _score_ring(self, qh, wf, start: int, slot: int, end: int, meta=None):
        k_chunk = settings.get("INDEXER_K_CHUNK")
        cfg = ttnn.IndexerScoreProgramConfig(
            q_chunk_size=64 if self.q_rows % 64 == 0 else 32, k_chunk_size=k_chunk, head_group_size=0
        )
        sems = self.sems[self.sem_i]
        self.sem_i = 1 - self.sem_i
        if meta is None:
            where = dict(
                chunk_start_idx=start,
                # the op needs a tile-aligned length; the extra columns are zeroed pad rows, and topk_large_indices'
                # valid_length=end drops them
                kv_len=-(-end // ttnn.TILE_SIZE) * ttnn.TILE_SIZE,
                cache_batch_idx=slot * self.cache_layers + self.cache_layer,
            )
        else:
            # trace-safe: the kernel derives kv_len = min(start + chunk, ceil32(end)), the eager value
            where = dict(chunk_start_idx_tensor=meta.start, valid_end_tensor=meta.end, cache_batch_idx_tensor=meta.slot)
        logits = ttnn.experimental.ring_indexer_score_dsa(
            qh,
            self.k_full,
            wf,
            self.cache,
            sems,
            cluster_axis=SP_AXIS,
            topology=_topology(),
            num_links=settings.get("NUM_LINKS"),
            program_config=cfg,
            compute_kernel_config=self.score_cfg,
            **where,
            index_cache_num_layers=self.cache_layers,
            index_cache_layer_idx=self.cache_layer,
            seq_subshard_axis=TP_AXIS,
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=self.rows,
        )  # [1, 1, 160, max_seq] bf16 ROW_MAJOR, -inf on future columns, stale past end
        return logits

    def __call__(self, x: ttnn.Tensor, start: int, slot: int = 0, end: int | None = None, meta=None) -> ttnn.Tensor:
        """x: attn_norm [1, 1, S/8, H/4] per chip (ATTN_NORM_DTYPE: fp32 or bf16), the chunk at absolute positions [start, start + S).
        Returns the top-k key positions uint32 [1, 1, S/32, 2048] per chip (module docstring). meta (chunk_meta.py):
        the chunk's slot, start and end read on the device (the trace-safe path; ring score only)."""
        end = start + self.chunk if end is None else end
        assert meta is None or not self.composed, "the metadata path scores with the ring op (INDEXER_SCORE=ring)"
        if x.shape[-1] == self.wk.shape[-2]:  # NORM_GATHER=input: attn_norm handed on the full row
            x_full = x
        else:
            x_full = ttnn.all_gather(x, dim=3, cluster_axis=TP_AXIS, num_links=settings.get("NUM_LINKS"), topology=_topology())
        own_k = self.write_k(x_full, start, slot, end, meta=meta)

        x_q = ttnn.mesh_partition(x_full, dim=2, cluster_axis=TP_AXIS)  # [1, 1, 160, H]
        if x_full is not x:
            ttnn.deallocate(x_full)
        qn = self._layer_norm(x_q, self.q_norm_w, self.q_norm_b)
        if self.q_fp32:
            qh = self._queries_fp32(qn, start, meta=meta)
        else:
            q = ttnn.linear(
                qn,
                self.wq_b,
                program_config=mm_configs.get_indexer(qn, self.wq_b),
                dtype=ttnn.bfloat16,  # nlp_create_qkv_heads / rotary_embedding_indexed are bf16 (qn may be fp32)
                compute_kernel_config=self.mm_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qn)
            qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                q,
                num_heads=self.n_heads,
                num_kv_heads=0,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1, Hi, 160, D]
            ttnn.deallocate(q)
            if self.rope is not None:
                qr = self._rope(qh, start, subshard=True, meta=meta)
                ttnn.deallocate(qh)
                qh = qr
        qh = self._rotate(qh)
        wts = ttnn.linear(
            x_q,
            self.wproj,
            program_config=mm_configs.get_indexer(x_q, self.wproj),
            dtype=self.w_dtype,
            compute_kernel_config=self.mm_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_q)
        if self.ssmax_sb is not None:
            # SSMAX_TABLE=shared_log: f = s * log(p + 1) + b in fp32 from the shared log table's rows
            if meta is None:
                lg = self._eager_tables()["log_rows"].rows(start)
            else:
                lg = self._selected(meta, "log", self.q_rows, True)
            fs = ttnn.multiply(lg, self.ssmax_sb[0], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if meta is None and start % self.chunk:
                ttnn.deallocate(lg)
            f = ttnn.add(fs, self.ssmax_sb[1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(fs)
            wf = ttnn.multiply(wts, f, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(f)
        elif self.w_dtype == ttnn.float32:
            f = self.f_rows.rows(start) if meta is None else self._selected(meta, "f", self.q_rows, True)
            wf = ttnn.multiply(wts, f, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if meta is None and start % self.chunk:
                ttnn.deallocate(f)  # a selection of its own (a chunk-aligned start's is a slice of the table)
        else:
            wf = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                wts,
                self.ssmax.cos,
                self.ssmax.sin,
                self.ssmax.trans,
                start if meta is None else meta.start,
                cluster_axis=SP_AXIS,
                seq_subshard_axis=TP_AXIS,
                compute_kernel_config=self.mm_cfg,
            )  # w * f[p, h]: cos = f, sin = 0
        ttnn.deallocate(wts)

        if not self.composed and wf.dtype != ttnn.bfloat16:
            w16 = ttnn.typecast(wf, ttnn.bfloat16)  # the ring op takes bf16 head weights and queries
            ttnn.deallocate(wf)
            wf = w16
        if not self.composed and qh.dtype != ttnn.bfloat16:
            q16 = ttnn.typecast(qh, ttnn.bfloat16)
            ttnn.deallocate(qh)
            qh = q16
        if self.composed:
            logits = self._score_composed(qh, wf, start, slot, end, own_k)
            if own_k is not None:
                ttnn.deallocate(own_k)
        else:
            logits = self._score_ring(qh, wf, start, slot, end, meta=meta)
        ttnn.deallocate(qh)
        ttnn.deallocate(wf)
        if meta is None:
            idx = ttnn.experimental.topk_large_indices(logits, k=self.topk, valid_length=end)
        else:
            # min(start + chunk, ceil32(end)): end on a 32-aligned end; otherwise the extra columns are the pad keys
            # [end, ceil32(end)), -inf (future) to every real query row, so only pad rows' top-k can differ
            idx = ttnn.experimental.topk_large_indices(
                logits, k=self.topk, valid_length_tensor=meta.start, valid_length_offset=self.chunk, valid_end_tensor=meta.end
            )
        ttnn.deallocate(logits)
        return idx
