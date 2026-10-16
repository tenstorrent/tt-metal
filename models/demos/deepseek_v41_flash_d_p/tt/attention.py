# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 window attention (reference/ds41_ref.py:_attention for a ratio-0 layer; inference Attention) on the
SP x TP mesh, and the served window-KV cache it reads and writes (serving_contract.md "KV cache", "Attention and
cache writes"; plan.md "Attention and cache writes").

    attn_out [S, H] = fp8_linear(grouped_wo_a(fp8_qdq(unrope(softmax_sink(q . window) window))), wo_b)
    q = rope(fp8_linear(q_resid, wq_b)) [S, 64 heads x 512]; window of position p = KV rows [p - 127, p]

Layout per chip (r, c) (mesh axis 0 = SP rows, axis 1 = TP columns, both read from mesh.shape): the inputs are
row-split, TP-replicated (q_resid [1, 1, S/SP, 1280] fp32, kv [1, 1, S/SP, 512] bf16; row r = positions
[start + r S/SP, ...)); the output is the hidden split [1, 1, S/SP, H/TP] fp32 (tt/layout.py).

Chain (one path on every box; nothing on the host per chunk except picking load-time tables by integers):
    window KV      all_gather over axis 0 -> [S, 512] bf16 (every chip, natural order); rows at or past actual_end
                   zeroed (iota < end - start, load-time iota); ROW_MAJOR
    cache write    each SP row's rows in the server's order (ttnn.embedding with the load-time index row of
                   (start mod S) / 32), then ttnn.experimental.deepseek_prefill.update_padded_kv_cache(kv_actual_global=
                   start, valid_global=end, tp_axis=1): positions [start, end) land at the owner chip, TP-dedup, the
                   last record's pad rows written as zeros
    window carry   positions [start - 128, start) read back from the served cache: the same local rows sliced on every
                   chip, gathered over the mesh, the owner's block kept (one or two pieces; zeros before position 0)
    table          [carry 128 | chunk S] rows x 512, ROW_MAJOR, replicated; the RoPE tail de-interleaved ([even |
                   odd], as q's)
    q              tt/quant.fp8_qdq -> ttnn.linear wq_b (the chip's heads, bf16 weights with each head's rope rows
                   de-interleaved, HiFi4 + fp32 acc, fp32 out) -> heads [1, H/TP, S/SP, 512] -> RoPE (fp32 SFPU,
                   tt/rope.rope_heads) -> x 512^-0.5 (the softmax scale folded into q: the op's scale is then 1, exact;
                   K-27) -> bf16 (stays fp32 with WINDOW_SDPA_SCORES=fp32, DS41-31) -> all_to_all head -> seq over
                   axis 1 (sparse_sdpa needs H % 32 == 0: 16 heads per chip on 4x4; done whenever TP > 1, one path
                   everywhere) -> [1, 64, S/(SP TP), 512]
    attention      settings WINDOW_SDPA = "high_precision" (default, owner 2026-10-15): the sparse_sdpa path below.
                   "exact" (comparison only; DS41-19): tt/window_exact.ExactWindow on the chip's own heads
                   and rows (no reshard): q [1, H/TP, S/SP, 512] fp32 -> blocks of 32 queries x their 160 table rows,
                   q and the probabilities as bf16 pieces, lane-masked keys / values (exact products), fp32 SFPU
                   softmax with the fp32 sink, fp32 output. "high_precision" / "source": the q -> all_to_all ->
                   ttnn.bringup / ttnn.transformer sparse_sdpa(attention_sink = sink / 1, scale 1) path: per query the
                   128 window rows (load-time index list: row d + 1 + j of the table, a sentinel tail for positions
                   < 0) -> all_to_all back -> fp32 (high_precision: the op's own fp32 output, WINDOW_SDPA_OUT)
    output         inverse RoPE + re-interleave -> fp8_qdq -> heads concatenated per group ->
                   grouped wo_a (the chip's whole groups, one batched matmul) -> fp8_qdq -> wo_b (K-split: the chip's
                   groups) -> exact sum over TP (all_gather + SFPU add, K-30) -> the chip's hidden slice

The served cache (TtWindowCache): position g on mesh row (g / W) % SP, column ((g mod W) / (W / TP)), local row
(g / S) (W / TP) + g mod (W / TP) (W = S / SP; update_padded_kv_cache's sp x tp linearized block-cyclic order),
bf16 ROW_MAJOR, one ND shard [1, 1, 32, 512] per 32-token record round-robin over the DRAM banks (init_kvpe_cache).
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM, SP_AXIS, TP_AXIS, compute_config, tp_sum
from .quant import fp8_qdq, round_tf32
from .rope import RopeTable, deinterleave_perm, rope_heads
from .settings import settings
from .window_exact import ExactWindow

WINDOW = 128  # sliding window (cfg.window); the table's carry region
SENTINEL = -1  # uploaded as int32 -1, read through uint32: 0xFFFFFFFF, sparse_sdpa's "no key"
SDPA_K_CHUNK = 128


def _mapper(mesh, dims):
    return ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)


def _upload(mesh, t, dtype, layout, dims=None):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if dims is None else _mapper(mesh, dims),
    )


class WindowGeometry:
    """Chunk and mesh numbers the window path depends on (from mesh.shape and the serving chunk)."""

    def __init__(self, mesh, chunk: int):
        self.sp, self.tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
        self.n = self.sp * self.tp
        self.chunk = chunk
        assert chunk % (32 * self.n) == 0, f"chunk {chunk} does not split into tiles over {self.n} chips"
        self.w = chunk // self.sp  # tokens per SP row per chunk (the server's W)
        self.stripe = self.w // self.tp  # tokens per chip per W block (the cache's TP-dedup stripe)
        self.q_rows = self.stripe  # query rows per chip after the head -> seq reshard (== W when TP == 1)

    def cache_rows(self, max_seq: int) -> int:
        """Local rows per chip for positions [0, max_seq) (whole chunk periods)."""
        return -(-max_seq // self.chunk) * self.stripe

    def server_rows(self, start: int) -> torch.Tensor:
        """[SP, W] natural chunk offsets d (position start + d) of each SP row's rows in the server's order
        (serving_contract.md "Input": row (g / W) % SP, positions ascending)."""
        s, w, sp = self.chunk, self.w, self.sp
        intra, c_start = start % w, (start // w) % sp
        j = torch.arange(w)
        out = []
        for r in range(sp):
            k = (r - c_start) % sp
            out.append(torch.where(j < w - intra, j, s - w + j) if k == 0 else k * w - intra + j)
        return torch.stack(out)

    def query_offsets(self) -> torch.Tensor:
        """[SP, TP, q_rows] natural chunk offsets of the queries each chip holds after the reshard."""
        d = torch.arange(self.chunk).view(self.sp, self.w)
        return d.view(self.sp, self.tp, self.q_rows)

    @staticmethod
    def start_class(start: int) -> int:
        """The index table a chunk uses: the carry is whole from position 128 on."""
        return min(start, WINDOW)


class WindowTables:
    """Load-time tables shared by every window layer of a model (they depend only on the chunk and the mesh):
    the server-order row index per (start mod S) / 32, the attention index lists per start class (0, 32, 64, 96,
    >= 128), the chunk-row iota for the pad mask, and a zero carry."""

    def __init__(self, mesh, chunk: int):
        g = self.geo = WindowGeometry(mesh, chunk)
        self.mesh = mesh
        # server order: [SP, 1, S / 32, W] -> chip row r holds its [S / 32, W] table (replicated over TP)
        srv = torch.stack([g.server_rows(32 * k) for k in range(chunk // 32)], dim=1)  # [SP, S/32, W]
        self.srv = _upload(mesh, srv.unsqueeze(1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, None))
        # attention index lists: query d reads table rows d + 1 + j (positions start + d - 127 + j); rows of
        # positions < 0 (start < 128) dropped, the valid ones first, a sentinel tail
        d = g.query_offsets().unsqueeze(-1)  # [SP, TP, q, 1]
        j = torch.arange(WINDOW)
        self.index = {}
        for cls in range(0, WINDOW + 1, 32):
            lo = torch.clamp(d + 1, min=WINDOW - cls)
            n = d + WINDOW - lo + 1
            ids = torch.where(j < n, lo + j, torch.full_like(lo + j, SENTINEL))
            self.index[cls] = _upload(mesh, ids.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, 1))
        self.iota = _upload(
            mesh, torch.arange(chunk, dtype=torch.float32).view(1, 1, chunk, 1), ttnn.float32, ttnn.TILE_LAYOUT
        )
        self.zero_carry = _upload(mesh, torch.zeros(1, 1, WINDOW, 512), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self._exact = {}

    def exact(self, heads_per_chip: int, width: int, pieces: int, compute_config) -> ExactWindow:
        """The exact window attention's load-time tables (tt/window_exact.py), one per head count / pieces, shared
        by every window layer."""
        key = (heads_per_chip, width, pieces)
        if key not in self._exact:
            self._exact[key] = ExactWindow(
                self.mesh, self.geo.chunk, heads_per_chip, width, WINDOW, pieces, compute_config
            )
        return self._exact[key]


class TtWindowCache:
    """The served 0_window_kv cache of one layer and slot: [1, 1, rows, 512] bf16 ROW_MAJOR per chip, ND-sharded one
    32-token record per shard (init_kvpe_cache), TP-dedup (update_padded_kv_cache's sp x tp order, module docstring).
    ``write`` and ``carry`` run per chunk on the device; the ``*_host`` methods are the harness / serving boundary."""

    WIDTH = 512

    def __init__(self, mesh, tables: WindowTables, max_seq: int):
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

        self.mesh, self.tables, self.geo = mesh, tables, tables.geo
        g = self.geo
        self.rows = g.cache_rows(max_seq)
        self.max_seq = self.rows * g.n
        self.tp_axis = TP_AXIS if g.tp > 1 else None
        self.tensor = init_kvpe_cache(
            kvpe_cache_head_dim=self.WIDTH,
            mesh_device=mesh,
            seq_len=self.max_seq,
            mesh_shape=list(mesh.shape),
            sp_axis=SP_AXIS,
            num_kvpe_cache_layers=1,
            num_users=1,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            tp_axis=self.tp_axis,
        )

    # ------------------------------------------------------------------ per chunk (device)
    def write(self, kv_rm: ttnn.Tensor, start: int, end: int) -> None:
        """kv_rm [1, 1, S, 512] bf16 ROW_MAJOR, natural chunk order, replicated, rows at or past end zero: positions
        [start, end) into the cache (the last record's pad rows as zeros)."""
        g = self.geo
        assert start % 32 == 0 and start < end <= start + g.chunk and end <= self.max_seq, (start, end, self.max_seq)
        k = (start % g.chunk) // 32
        idx = ttnn.slice(self.tables.srv, [0, 0, k, 0], [1, 1, k + 1, g.w], memory_config=DRAM)
        rows = ttnn.embedding(
            ttnn.reshape(idx, [1, g.w]),
            ttnn.reshape(kv_rm, [g.chunk, self.WIDTH]),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=DRAM,
        )
        ttnn.deallocate(idx)
        slab = ttnn.reshape(rows, [1, 1, g.w, self.WIDTH])
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            self.tensor,
            slab,
            slot_idx=0,
            layer_idx=0,
            num_layers=1,
            kv_actual_global=start,
            cluster_axis=SP_AXIS,
            valid_global=end,
            tp_axis=self.tp_axis,
        )
        ttnn.deallocate(rows)

    def _gather_all(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, n, 512] TILE per chip -> [1, 1, N n, 512] on every chip, chip-linear order r TP + c."""
        out = t
        if self.geo.tp > 1:
            out = ttnn.all_gather(out, dim=2, cluster_axis=TP_AXIS, memory_config=DRAM)
        if self.geo.sp > 1:
            nxt = ttnn.all_gather(out, dim=2, cluster_axis=SP_AXIS, memory_config=DRAM)
            if out is not t:
                ttnn.deallocate(out)
            out = nxt
        return out

    def carry(self, start: int) -> ttnn.Tensor:
        """The window rows of positions [start - 128, start) ([1, 1, 128, 512] bf16 ROW_MAJOR, replicated; rows of
        positions < 0 are zero) read from the cache: per owner stripe (one or two when the stripe is >= 128 tokens,
        more on a narrower one), the same local rows sliced on every chip, gathered over the mesh, the owner's block
        kept."""
        g = self.geo
        n = min(WINDOW, start)
        if n == 0:
            return self.tables.zero_carry
        pieces = [] if n == WINDOW else [ttnn.slice(self.tables.zero_carry, [0, 0, 0, 0], [1, 1, WINDOW - n, 512])]
        p = start - n
        while p < start:
            s_idx, u = divmod(p, g.stripe)
            take = min(g.stripe - u, start - p)
            q, b = s_idx % g.n, s_idx // g.n
            loc = b * g.stripe + u
            sl = ttnn.slice(self.tensor, [0, 0, loc, 0], [1, 1, loc + take, self.WIDTH], memory_config=DRAM)
            tl = ttnn.to_layout(sl, ttnn.TILE_LAYOUT, memory_config=DRAM)
            ttnn.deallocate(sl)
            gat = self._gather_all(tl)
            if gat is not tl:
                ttnn.deallocate(tl)
            mine = ttnn.slice(gat, [0, 0, q * take, 0], [1, 1, (q + 1) * take, self.WIDTH], memory_config=DRAM)
            ttnn.deallocate(gat)
            pieces.append(ttnn.to_layout(mine, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM))
            ttnn.deallocate(mine)
            p += take
        if len(pieces) == 1:
            return pieces[0]
        out = ttnn.concat(pieces, dim=2, memory_config=DRAM)
        for t in pieces:
            if t is not self.tables.zero_carry:
                ttnn.deallocate(t)
        return out

    # ------------------------------------------------------------------ harness / serving boundary (host)
    def load_host(self, rows: torch.Tensor, p0: int) -> None:
        """Write host rows [n, 512] at positions [p0, p0 + n) (p0 % 32 == 0) through the device write path."""
        g = self.geo
        assert p0 % 32 == 0, p0
        n = rows.shape[0]
        for a in range(0, n, g.chunk):
            m = min(g.chunk, n - a)
            nat = torch.zeros(g.chunk, self.WIDTH)
            nat[:m] = rows[a : a + m].float()
            srv = nat[g.server_rows(p0 + a)].reshape(g.sp, 1, g.w, self.WIDTH)
            slab = _upload(self.mesh, srv, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (0, None))
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                self.tensor,
                slab,
                slot_idx=0,
                layer_idx=0,
                num_layers=1,
                kv_actual_global=p0 + a,
                cluster_axis=SP_AXIS,
                valid_global=p0 + a + m,
                tp_axis=self.tp_axis,
            )
            ttnn.deallocate(slab)

    def to_host(self, length: int) -> torch.Tensor:
        """Natural-order rows of positions [0, ceil32(length)) as host bf16 [ceil32(length), 512]."""
        g = self.geo
        shards = [ttnn.to_torch(t).reshape(self.rows, self.WIDTH) for t in ttnn.get_device_tensors(self.tensor)]
        blocks = self.rows // g.stripe
        stacked = torch.stack([s.view(blocks, g.stripe, self.WIDTH) for s in shards], dim=1)  # [blocks, N, stripe]
        nat = stacked.reshape(blocks * g.n * g.stripe, self.WIDTH)
        return nat[: -(-length // 32) * 32].to(torch.bfloat16)

    def ring_host(self, length: int, window: int = WINDOW) -> torch.Tensor:
        """The reference's window_kv state: the ring [window, 512], slot p % window = position p of the last window
        positions before ``length`` (zeros elsewhere)."""
        ring = torch.zeros(window, self.WIDTH)
        if length:
            nat = self.to_host(length).float()
            pos = torch.arange(max(0, length - window), length)
            ring[pos % window] = nat[pos]
        return ring

    def load_ring(self, ring: torch.Tensor, length: int, window: int = WINDOW) -> None:
        """Load the reference's ring view of positions [length - window, length) into the cache."""
        n = min(window, length)
        if n == 0:
            return
        p0 = length - n
        assert p0 % 32 == 0, f"prefix length {length}: the ring's first position {p0} is not 32-aligned"
        pos = torch.arange(p0, length)
        self.load_host(ring.float()[pos % window], p0)

    def free(self) -> None:
        ttnn.deallocate(self.tensor)


class TtWindowAttention:
    """attention of one window-only layer. ``__call__(q_resid, kv, start, end, cache)``: q_resid [1, 1, S/SP, 1280]
    fp32 and kv [1, 1, S/SP, 512] bf16 (row-split, TP-replicated, natural blocks of positions [start, start + S)),
    end = actual_end -> attn_out [1, 1, S/SP, H/TP] fp32 (hidden split); writes the chunk's window rows into ``cache``.
    Does not consume its inputs."""

    def __init__(self, mesh, cfg, w: dict, rope: RopeTable, tables: WindowTables):
        self.mesh, self.cfg, self.rope, self.tables, self.geo = mesh, cfg, rope, tables, tables.geo
        g = self.geo
        self.heads, self.d, self.rd = cfg.n_heads, cfg.head_dim, cfg.rope_dim
        self.groups, self.rank = cfg.o_groups, cfg.o_lora_rank
        assert cfg.window == WINDOW and self.d == TtWindowCache.WIDTH, (cfg.window, self.d)
        assert self.heads % g.tp == 0 and self.groups % g.tp == 0, (self.heads, self.groups, g.tp)
        self.h_l, self.g_l = self.heads // g.tp, self.groups // g.tp
        self.hpg = self.heads // self.groups  # heads per wo_a group
        self.ckc = compute_config(mesh)
        self.scale = self.d**-0.5
        self.sdpa = settings.get("WINDOW_SDPA")
        self.high_precision = self.sdpa == "high_precision"
        self.fp32_out = self.high_precision and settings.get("WINDOW_SDPA_OUT") == "fp32"
        # fp32 q and Float32 scores / probabilities in the op (DS41-31); needs the fp32 output
        self.fp32_scores = self.fp32_out and settings.get("WINDOW_SDPA_SCORES") == "fp32"
        if self.sdpa != "exact":
            assert (
                self.heads % 32 == 0
            ), f"sparse_sdpa needs a multiple of 32 heads per chip ({self.heads} after reshard)"
        self.o_sum = settings.get("ATTN_O_SUM")
        perm = deinterleave_perm(self.d, self.rd)
        # wq_b [heads * 512, q_lora] -> each head's rope rows de-interleaved -> [q_lora, heads * 512], columns split
        # over TP (chip c: heads [c H/TP, (c+1) H/TP)); bf16 exact (e4m3 x 2^k)
        wq_b = w["wq_b"].float().view(self.heads, self.d, -1)[:, perm].reshape(self.heads * self.d, -1)
        assert torch.equal(wq_b.bfloat16().float(), wq_b), "wq_b must be bf16-exact"
        self.wq_b = _upload(
            mesh, wq_b.t().reshape(1, 1, -1, self.heads * self.d), ttnn.bfloat16, ttnn.TILE_LAYOUT, (None, 3)
        )
        # wo_a [groups * rank, hpg * 512] -> [1, groups, hpg * 512, rank], groups split over TP; bf16-exact
        wo_a = w["wo_a"].float().view(self.groups, self.rank, -1)
        assert torch.equal(wo_a.bfloat16().float(), wo_a), "wo_a must be bf16-exact"
        self.wo_a = _upload(mesh, wo_a.transpose(1, 2).unsqueeze(0), ttnn.bfloat16, ttnn.TILE_LAYOUT, (None, 1))
        # wo_b [H, groups * rank] -> [groups * rank, H], K (the chip's groups) split over TP
        wo_b = w["wo_b"].float()
        assert torch.equal(wo_b.bfloat16().float(), wo_b), "wo_b must be bf16-exact"
        self.wo_b = _upload(mesh, wo_b.t().reshape(1, 1, -1, wo_b.shape[0]), ttnn.bfloat16, ttnn.TILE_LAYOUT, (None, 2))
        self.hidden = wo_b.shape[0]
        # the sink in sparse_sdpa's units (it multiplies by the op's scale, 1 here): every head on every chip
        sink = w["attn_sink"].float().reshape(1, 1, 1, self.heads)
        if self.sdpa == "exact":
            # fp32 sink per (head, block row) [1, H/TP, 32, 1], heads split over TP like wq_b's columns
            col = w["attn_sink"].float().view(1, self.heads, 1, 1).expand(1, self.heads, 32, 1)
            self.sink_col = ttnn.from_torch(
                col.contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=_mapper(mesh, (None, 1)),
            )
            self.window = tables.exact(self.h_l, self.d, int(settings.get("WINDOW_PIECES")), self.ckc)
        else:
            self.sink = _upload(mesh, sink, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)

    # ------------------------------------------------------------------ parts
    def _reshard(self, t, in_dim, out_dim):
        if self.geo.tp == 1:
            return t
        return ttnn.experimental.all_to_all_async_generic(
            t, in_dim=in_dim, out_dim=out_dim, num_links=2, memory_config=DRAM, cluster_axis=TP_AXIS
        )

    def _table(self, kv: ttnn.Tensor, start: int, end: int, cache: TtWindowCache) -> ttnn.Tensor:
        """Writes the chunk's rows into the cache; returns the sparse table [1, 1, 128 + S, 512] bf16 ROW_MAJOR
        (carry | chunk), rope tail de-interleaved."""
        g = self.geo
        full = kv
        if g.sp > 1:
            full = ttnn.all_gather(kv, dim=2, cluster_axis=SP_AXIS, memory_config=DRAM)  # [1, 1, S, 512] bf16
        keep = ttnn.lt(self.tables.iota, float(end - start), dtype=ttnn.bfloat16, memory_config=DRAM)
        masked = ttnn.multiply(full, keep, dtype=ttnn.bfloat16, memory_config=DRAM)  # pad rows -> 0 (exact)
        ttnn.deallocate(keep)
        if full is not kv:
            ttnn.deallocate(full)
        kv_rm = ttnn.to_layout(masked, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(masked)
        cache.write(kv_rm, start, end)
        carry = cache.carry(start)
        table = ttnn.concat([carry, kv_rm], dim=2, memory_config=DRAM)
        if carry is not self.tables.zero_carry:
            ttnn.deallocate(carry)
        ttnn.deallocate(kv_rm)
        # the rope tail [interleaved pairs] -> [even | odd] (q's order): ROW_MAJOR transpose, exact
        rows, half, r0 = table.shape[2], self.rd // 2, self.d - self.rd
        tail = ttnn.slice(table, [0, 0, 0, r0], [1, 1, rows, self.d], memory_config=DRAM)
        tt = ttnn.transpose(ttnn.reshape(tail, [1, rows, half, 2]), -2, -1, memory_config=DRAM)
        ttnn.deallocate(tail)
        nope = ttnn.slice(table, [0, 0, 0, 0], [1, 1, rows, r0], memory_config=DRAM)
        ttnn.deallocate(table)
        out = ttnn.concat([nope, ttnn.reshape(tt, [1, 1, rows, self.rd])], dim=-1, memory_config=DRAM)
        ttnn.deallocate(nope)
        ttnn.deallocate(tt)
        return out

    def _q_heads(self, q_resid: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """q_resid -> scaled, roped q of the chip's heads [1, H/TP, S/SP, 512] fp32 (rope tail [even | odd])."""
        rows = q_resid.shape[-2]
        xq = fp8_qdq(q_resid, dtype=ttnn.bfloat16)
        qf = ttnn.linear(xq, self.wq_b, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(xq)
        q4 = ttnn.transpose(ttnn.reshape(qf, [1, rows, self.h_l, self.d]), 1, 2, memory_config=DRAM)
        ttnn.deallocate(qf)
        qr = rope_heads(q4, cos, sin)  # tail stays [even | odd], the table's order
        ttnn.deallocate(q4)
        qs = ttnn.multiply(qr, self.scale, dtype=ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(qr)
        return qs

    def _q(self, q_resid: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """q_resid -> scaled, roped q heads [1, H, S/(SP TP), 512] ROW_MAJOR (after the reshard): fp32 with
        WINDOW_SDPA_SCORES=fp32, else bf16."""
        qs = self._q_heads(q_resid, cos, sin)
        if self.fp32_scores:
            qb = round_tf32(qs)  # the op's FPU reads q at TF32: rounded to nearest here, not truncated there
            ttnn.deallocate(qs)
        else:
            qb = ttnn.typecast(qs, ttnn.bfloat16, memory_config=DRAM)
            ttnn.deallocate(qs)
        qa = self._reshard(qb, in_dim=1, out_dim=2)
        if qa is not qb:
            ttnn.deallocate(qb)
        q_rm = ttnn.to_layout(qa, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(qa)
        return q_rm

    def _sdpa(self, q_rm, table, index):
        if self.high_precision:
            return ttnn.bringup.sparse_sdpa(
                q_rm,
                table,
                index,
                self.d,
                kv_format=ttnn.bringup.SparseKVFormat.BF16,
                scale=1.0,
                k_chunk_size=SDPA_K_CHUNK,
                compute_kernel_config=self.ckc,
                attention_sink=self.sink,
                high_precision=True,
                fp32_output=self.fp32_out,
                fp32_scores=self.fp32_scores,
            )
        return ttnn.transformer.sparse_sdpa(
            q_rm,
            table,
            index,
            self.d,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=1.0,
            k_chunk_size=SDPA_K_CHUNK,
            compute_kernel_config=self.ckc,
            attention_sink=self.sink,
        )

    def _out(self, o_rm: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """sparse_sdpa output [1, H, S/(SP TP), 512] ROW_MAJOR (fp32, or bf16 with WINDOW_SDPA_OUT=bf16 / the source
        op) -> attn_out [1, 1, S/SP, H/TP] fp32."""
        ot = ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT, memory_config=DRAM)
        oh = self._reshard(ot, in_dim=2, out_dim=1)  # [1, H/TP, S/SP, 512]
        if oh is not ot:
            ttnn.deallocate(ot)
        if oh.dtype == ttnn.float32:
            return self._out_heads(oh, cos, sin)
        of = ttnn.typecast(oh, ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(oh)
        return self._out_heads(of, cos, sin)

    def _out_heads(self, of: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """attention output of the chip's heads [1, H/TP, S/SP, 512] fp32 (rope tail [even | odd]; consumed) ->
        attn_out [1, 1, S/SP, H/TP] fp32."""
        rows = of.shape[2]
        ou = rope_heads(of, cos, sin, inverse=True, interleave=True)  # the reference's order again
        ttnn.deallocate(of)
        oq = fp8_qdq(ou, dtype=ttnn.bfloat16)  # groups of 32 inside each head (512 % 32 == 0)
        ttnn.deallocate(ou)
        # heads of one wo_a group side by side: [G/TP, hpg, S/SP, 512] -> [1, G/TP, S/SP, hpg * 512]
        og = ttnn.experimental.nlp_concat_heads(
            ttnn.reshape(oq, [self.g_l, self.hpg, rows, self.d]), memory_config=DRAM
        )
        ttnn.deallocate(oq)
        og = ttnn.reshape(og, [1, self.g_l, rows, self.hpg * self.d])
        y = ttnn.matmul(og, self.wo_a, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(og)
        # [1, G/TP, S/SP, rank] -> [1, 1, S/SP, G/TP * rank] (the chip's K range of wo_b, group-major)
        parts = [ttnn.slice(y, [0, k, 0, 0], [1, k + 1, rows, self.rank], memory_config=DRAM) for k in range(self.g_l)]
        ttnn.deallocate(y)
        yt = ttnn.concat(parts, dim=-1, memory_config=DRAM) if len(parts) > 1 else parts[0]
        if len(parts) > 1:
            for t in parts:
                ttnn.deallocate(t)
        yq = fp8_qdq(yt, dtype=ttnn.bfloat16)
        ttnn.deallocate(yt)
        z = ttnn.linear(yq, self.wo_b, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(yq)
        if self.geo.tp == 1:
            return z
        if self.o_sum == "reduce_scatter":
            out = ttnn.reduce_scatter(z, dim=3, cluster_axis=TP_AXIS, memory_config=DRAM)
        else:
            full = tp_sum(self.mesh, z, how="gather_add")
            out = ttnn.mesh_partition(full, dim=3, cluster_axis=TP_AXIS, memory_config=DRAM)
            ttnn.deallocate(full)
        ttnn.deallocate(z)
        return out

    def __call__(self, q_resid, kv, start: int, end: int | None, cache: TtWindowCache) -> ttnn.Tensor:
        g = self.geo
        s = q_resid.shape[-2] * g.sp
        assert s == g.chunk, f"chunk of {s} rows; the window tables were built for {g.chunk}"
        end = start + s if end is None else int(end)
        cos, sin = self.rope.rows(start, s)
        table = self._table(kv, start, end, cache)
        if self.sdpa == "exact":
            qs = self._q_heads(q_resid, cos, sin)
            o = self.window(qs, table, start, self.sink_col)
            ttnn.deallocate(qs)
            ttnn.deallocate(table)
            out = self._out_heads(o, cos, sin)
            for t in (cos, sin):
                ttnn.deallocate(t)
            return out
        q_rm = self._q(q_resid, cos, sin)
        o = self._sdpa(q_rm, table, self.tables.index[g.start_class(start)])
        ttnn.deallocate(q_rm)
        ttnn.deallocate(table)
        out = self._out(o, cos, sin)
        for t in (o, cos, sin):
            ttnn.deallocate(t)
        return out


def attention_weights(ck, layer: int) -> dict:
    p = f"layers.{layer}.attn."
    return {
        "wq_b": ck.weight(p + "wq_b.weight", torch.float32),
        "wo_a": ck.weight(p + "wo_a.weight", torch.float32),
        "wo_b": ck.weight(p + "wo_b.weight", torch.float32),
        "attn_sink": ck.weight(p + "attn_sink", torch.float32),
    }


def build_attention(mesh, ck, cfg, layer: int, max_seq: int, chunk: int, rope=None, tables=None):
    """The window attention of a ratio-0 layer (its own rope table and window tables unless shared ones are given)."""
    from .kv import rope_freqs

    assert not cfg.ratio(layer), f"layer {layer} has compress ratio {cfg.ratio(layer)}: not a window-only layer"
    rope = rope or RopeTable(mesh, rope_freqs(cfg, layer), max_seq)
    tables = tables or WindowTables(mesh, chunk)
    return TtWindowAttention(mesh, cfg, attention_weights(ck, layer), rope, tables)
