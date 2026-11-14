# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact HOST reference of the fused greedy accept + drafter-input build (stage S1 of work/notes/45, section 2).

The device code will compute, from the verify trace's argmax ids, what today's host does in three places:
  1. the plugin's greedy walk (vllm_tt_plugin.spec_accept._greedy_accept + _counts_from_accepted, padding rows count 1),
  2. MTPSpecEngine.propose_draft_tokens -> accepted_plan / decline logic (spec_engine.py:2067-2135, 2035-2060),
  3. the drafter input build _dr_pack_host (SE:2343) and the reseed page table _dr_rs_table (SE:2391).
This file holds pure-torch functions that give those values bit for bit (QWEN36_SPEC_HOSTSET=1 layout only), so the
device ops have a single target. No ttnn import. tests/test_spec_fused_host.py proves them against the existing host path.

Rows are ENGINE rows (after the slot remap), r = u*T + j, T = K + 1, R = B*T.

Fused aux buffer (what the host stages BEFORE the step; only values known before the walk). uint32, int32 torch
[FUSED_AUX_ROWS, W], row i padded to W = 32-word multiple of max(B, 5) like tp_common.pack_u32_rows:
  row 0  num_valid[u]   verify num_valid_drafts after sanitize (0 for inactive rows), words 0..B-1
  row 1  active[u]      1 if the row is in the step (start_pos[u,0] >= 0)
  row 2  declined[u]    1 if the propose declines the row whatever m is: inactive | no-draft (skip) | fits_none
  row 3  keep_kv[u]     1 if skip & active (declined row still writes MTP KV slot p: m_kv = m + 1)
  row 4  p0[u]          position of column 0 (start0 = max(start_pos[u,0], 0))
  row 5  consts         [0] scratch block id, [1] common rope_delta as two's complement uint32 (kept for the audit and
                        old readers; the device bodies no longer read it), [2] rope table rows, [3] B, [4] K
  row 6  row_delta[u]   each ENGINE row's own M-RoPE delta, two's complement uint32, words 0..B-1 (inactive rows too:
                        the row's own value, like model._vfy_aux_host(rope_delta=None)); the bodies add it per row
A row whose fit result depends on m ('mixed', see fused_fit_mode) cannot run fused unless it is declined anyway
(inactive or skip); fused_accept_aux_host raises ValueError then (the step falls back to the host path).
Host-only decisions that stay outside the fused step: all rows declined (no drafter replay, SE:2043/2060), B == 1 skip_all
(SE:2128), the eager drafter (B == 1 and p > eager_reseed_len, SE:2036), the page tables themselves.
"""

from typing import NamedTuple

import torch

from models.demos.blackhole.qwen36.tt.tp_common import (
    HOSTSET_PACK_ALIGN,
    pack_u32_rows,
    rope_row_index,
    spec_accept_kernel_cores,
    spec_accept_kernel_enabled,
)

FUSED_AUX_ROWS = 7
AUX_NUM_VALID, AUX_ACTIVE, AUX_DECLINED, AUX_KEEP, AUX_P0, AUX_CONSTS, AUX_DELTA = range(7)
CONST_SCRATCH, CONST_ROPE_DELTA, CONST_ROPE_ROWS, CONST_B, CONST_K = range(5)


def signed_rope_delta(word):
    """The rope delta back from its fused-aux uint32 word (fused_accept_aux_host stores ``int(delta) & 0xFFFFFFFF``)."""
    w = int(word) & 0xFFFFFFFF
    return w - 2**32 if w >= 2**31 else w


def row_delta_list(rope_delta, B):
    """Per-row rope deltas as B python ints. ``rope_delta``: an int (all rows the same) or a list / tensor of B ints."""
    if isinstance(rope_delta, (list, tuple, torch.Tensor)):
        out = [int(v) for v in torch.as_tensor(rope_delta).reshape(-1).tolist()]
        if len(out) != B:
            raise ValueError(f"row_delta_list: {len(out)} deltas for {B} rows")
        return out
    return [int(rope_delta)] * B


def aux_row_deltas(aux, B):
    """The B signed per-row rope deltas of a fused aux (row AUX_DELTA)."""
    return [signed_rope_delta(w) for w in torch.as_tensor(aux)[AUX_DELTA, :B].tolist()]


def ridx_deltas(deltas, T, K):
    """Per-position deltas for the drafter pack's ridx rows: the R reseed rows (user-major, T per user) then the
    K x B step rows (step-major), the order of torch.cat([rpos, steps.reshape(-1)])."""
    d = [int(x) for x in deltas]
    return [x for x in d for _ in range(T)] + d * K


def fused_fit_mode(start0, capacity, active, K):
    """Per row 'fits_all' | 'fits_none' | 'mixed': draft_rows_fit (SE:431, p + 1 + K < capacity) for the anchor
    p = start0 + m over every m in 0..K. An inactive row is reported 'fits_none' (it declines whatever m is)."""
    out = []
    for s, a in zip(torch.as_tensor(start0).reshape(-1).tolist(), torch.as_tensor(active).reshape(-1).tolist()):
        if not a:
            out.append("fits_none")
            continue
        fits = [int(s) + m + 1 + K < int(capacity) for m in range(K + 1)]
        out.append("fits_all" if all(fits) else "fits_none" if not any(fits) else "mixed")
    return out


def fused_accept_aux_host(
    num_valid, start0, active, skip, capacity, K, scratch_block, rope_delta, rope_rows, common_delta=None
):
    """The fused aux buffer int32 [FUSED_AUX_ROWS, W] (layout in the module doc). ``skip`` [B] bool = rows in the
    engine's _no_draft set (propose_skip_plan marked); ``capacity`` = table_capacity(cap_blocks, block_size).
    ``rope_delta``: an int (every row the same) or B per-row ints (row AUX_DELTA). ``common_delta`` (optional) is the
    value for row 5 col 1; default: the shared value when all rows agree, else 0."""
    nv = torch.as_tensor(num_valid).reshape(-1).to(torch.int64).tolist()
    B = len(nv)
    st = torch.as_tensor(start0).reshape(-1).to(torch.int64).tolist()
    act = [bool(a) for a in torch.as_tensor(active).reshape(-1).tolist()]
    sk = [bool(s) for s in torch.as_tensor(skip).reshape(-1).tolist()]
    modes = fused_fit_mode(st, capacity, act, K)
    declined, keep = [], []
    for u in range(B):
        if modes[u] == "mixed" and not sk[u]:
            raise ValueError(f"row {u}: draft_rows_fit depends on m (mixed); the step cannot run fused")
        declined.append(int((not act[u]) or sk[u] or modes[u] == "fits_none"))
        keep.append(int(sk[u] and act[u]))
    consts = [0] * 5
    row_d = row_delta_list(rope_delta, B)
    common = common_delta if common_delta is not None else (row_d[0] if len(set(row_d)) == 1 else 0)
    consts[CONST_SCRATCH], consts[CONST_ROPE_DELTA] = int(scratch_block), common & 0xFFFFFFFF
    consts[CONST_ROPE_ROWS], consts[CONST_B], consts[CONST_K] = int(rope_rows), B, int(K)
    rows = [nv, [int(a) for a in act], declined, keep, st, consts, [d & 0xFFFFFFFF for d in row_d]]
    width = -(-max(len(r) for r in rows) // HOSTSET_PACK_ALIGN) * HOSTSET_PACK_ALIGN
    out = torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.int64)
    if int(out.min()) < 0:
        raise ValueError("fused aux: negative word")
    return ((out + 2**31) % 2**32 - 2**31).to(torch.int32)  # uint32 bit pattern in an int32 tensor


def fused_greedy_walk(ids, toks, nv, K):
    """The host greedy walk alone: ids / toks int [B, T] (verify argmax ids; the verify token block, cols 1..K drafts),
    nv [B] valid drafts per row. Returns (committed [B, T] int64, -1 past the count; counts [B] int64 in [1, T]).
    A row with no valid draft (and every padding row) commits column 0's argmax."""
    ids = torch.as_tensor(ids).to(torch.int64)
    toks = torch.as_tensor(toks).to(torch.int64)
    nv = torch.as_tensor(nv).to(torch.int64).reshape(-1)
    T = K + 1
    jj = torch.arange(K)[None, :]
    valid = jj < nv[:, None]
    rejected = valid & (ids[:, :K] != toks[:, 1:])
    first_rej = rejected.to(torch.int8).argmax(dim=1)
    counts = torch.where(rejected.any(dim=1), first_rej + 1, nv + 1)  # [B] in [1, T]
    committed = torch.where(torch.arange(T)[None, :] < counts[:, None], ids, torch.full_like(ids, -1))
    return committed, counts


class FusedGreedyOut(NamedTuple):
    committed: torch.Tensor  # [B, T] int32, -1 past count
    counts: torch.Tensor  # [B] int32 in [1, T]
    pack: torch.Tensor  # [2K+5, W] int32 (uint32 words) = _dr_pack_host's pack
    rs_pt: torch.Tensor  # [R, nb] int32 = _dr_rs_table's full content
    mtp_pt: torch.Tensor  # [B, nb] int32: pt, declined rows -> scratch block
    declined: torch.Tensor  # [B] bool, final decline (aux declined | non-fit(m))
    num_valid_out: torch.Tensor  # [B] int32: K for a drafting row, 0 for a declined one
    drafter_runs: bool  # False when every row declines (the host skips the drafter replay)


def fused_greedy_reference(ids, tok_buf, aux, pt, *, B, K):
    """Fused greedy accept + drafter inputs. ``ids``: verify argmax ids, uint32/int [R] or [1, R] (row u*T+j);
    ``tok_buf``: the verify token buffer (row u*T+j = column j token; cols 1..K drafts, padded tail = col 0);
    ``aux``: fused_accept_aux_host output; ``pt`` [B, nb] int32 page table (engine rows, as passed to propose)."""
    T, R = K + 1, B * (K + 1)
    aux = torch.as_tensor(aux).to(torch.int64) & 0xFFFFFFFF
    nv, act, dec0, keep, p0 = (aux[i, :B] for i in range(5))
    scratch, rope_rows = (int(aux[AUX_CONSTS, i]) for i in (CONST_SCRATCH, CONST_ROPE_ROWS))
    delta = ridx_deltas(aux_row_deltas(aux, B), T, K)
    ids = torch.as_tensor(ids).reshape(-1)[:R].to(torch.int64).reshape(B, T)
    toks = torch.as_tensor(tok_buf).reshape(-1)[:R].to(torch.int64).reshape(B, T)
    # 1. greedy walk (spec_accept._greedy_accept + _counts_from_accepted); a row with no drafts commits 1
    committed, counts = fused_greedy_walk(ids, toks, nv, K)
    # 2. propose: accepted_plan (SE:155) + decline (SE:2105-2125) + m_kv (SE:2035)
    m = counts - 1
    anchor_pos = p0 + m  # committed_positions[u, m] - 1 with cpos[u, j] = p0 + 1 + j
    declined = dec0.bool()
    m_kv = m + (declined & keep.bool()).to(torch.int64)
    # 3. _dr_pack_host (SE:2343), with the decline overrides of SE:2043-2053 (p, pending -> 0)
    pending = torch.where(declined, torch.zeros_like(m), committed.gather(1, m[:, None]).reshape(-1))
    p = torch.where(declined, torch.zeros_like(p0), anchor_pos)
    real = torch.arange(T)[None, :] < m_kv[:, None]  # [B, T]
    zero = torch.zeros((), dtype=torch.int64)
    rtok = torch.where(real, committed, zero).reshape(-1)
    rpos = torch.where(real, (p0[:, None] + torch.arange(T)[None, :]), zero).reshape(-1)  # cpos - 1 = p0 + j
    steps = p[None, :] + torch.arange(K)[:, None]  # [K, B]
    ridx = rope_row_index(torch.cat([rpos, steps.reshape(-1)]), delta, rope_rows)
    anchor_rows = torch.arange(B) * T + m
    pack = pack_u32_rows([pending, *steps, rtok, rpos, anchor_rows, ridx[:R], *ridx[R:].reshape(K, B).to(torch.int64)])
    # 4. page tables: _dr_rs_table (SE:2391) content and the declined-row mtp table (SE:2046)
    pt = torch.as_tensor(pt).to(torch.int32)
    rs_pt = torch.where(
        real.reshape(-1, 1), pt.repeat_interleave(T, dim=0), torch.full_like(pt.repeat_interleave(T, dim=0), scratch)
    )
    mtp_pt = torch.where(declined[:, None], torch.full_like(pt, scratch), pt)
    num_valid_out = torch.where(declined, 0, K).to(torch.int32)
    return FusedGreedyOut(
        committed.to(torch.int32),
        counts.to(torch.int32),
        pack,
        rs_pt.to(torch.int32),
        mtp_pt.to(torch.int32),
        declined,
        num_valid_out,
        not bool(declined.all()),
    )


class FusedGreedyDevice:
    """Device twin of fused_greedy_reference (stage S1 on device). ``allocate`` creates the persistent buffers (same
    shape / dtype / layout as the model's; replicated over the mesh), the host stages the four inputs, ``body`` is a
    trace-capturable op sequence that writes pack_out / rs_pt_out / result_out IN PLACE (their addresses never change).

    Inputs : ids_in uint32 RM [1,1,1,32] (verify argmax ids, word r = u*T+j), tok_in uint32 RM [1,R],
             aux_in int32 RM [7,W] (fused_accept_aux_host output; the rope delta is a true signed int32),
             pt_rows_in int32 RM [R,nb] (page table row pt[r // T]).
    Outputs: pack_out uint32 RM [2K+5,W] (= fused_greedy_reference .pack), rs_pt_out int32 RM [R,nb] (= .rs_pt),
             result_out int32 RM [2,W]: row 0 = committed flattened r = u*T+j (-1 past the count), row 1 = counts[B];
             both zero past their length.
    All values are integers < 2^24, so the math runs in fp32 TILE and is exact; -1 is made in int32.
    QWEN36_SPEC_ACCEPT_KERNEL=1: ``body`` is ONE kernel on the same buffers (spec_fused_kernels.AcceptKernel); the op graph
    below stays the reference (and the fallback when the buffers do not fit the kernel)."""

    _kernel = None  # QWEN36_SPEC_ACCEPT_KERNEL: the AcceptKernel of this device (None = the op graph)

    def __init__(self):
        self.bufs = {}

    @staticmethod
    def _dev(mesh, t, dtype, layout):
        import ttnn

        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @staticmethod
    def _check_bound(name, t, shape, dtype, layout):
        got = (tuple(int(d) for d in t.shape), t.dtype, t.layout)
        assert got == (tuple(shape), dtype, layout), f"bound {name}: {got} != {(tuple(shape), dtype, layout)}"

    @classmethod
    def allocate(cls, mesh, B, K, nb, W, ids_in=None, tok_in=None, pack_out=None, rs_pt_out=None):
        """Persistent buffers. ``ids_in`` / ``tok_in`` / ``pack_out`` / ``rs_pt_out`` bind the LIVE buffers of the engine
        (verify ids output, verify token buffer, drafter pack, reseed page table) instead of allocating standalone ones;
        a bound tensor must have exactly the shape / dtype / layout allocated here. Default (None): standalone."""
        import ttnn

        self = cls()
        self.mesh, self.B, self.K, self.T, self.R, self.nb, self.W = mesh, B, K, K + 1, B * (K + 1), nb, W
        T, R = self.T, self.R
        assert R <= W and B <= W and R <= 32
        RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        u32, i32, f32 = ttnn.uint32, ttnn.int32, ttnn.float32
        z = lambda *s, dt=torch.int32: torch.zeros(*s, dtype=dt)
        self.ids_in = self.ids_standin = None
        if ids_in is not None:
            self.bind_ids(ids_in)
        else:
            self.ids_in = self.ids_standin = self._dev(mesh, z(1, 1, 1, 32), u32, RM)
        self.tok_in, self.pack_out, self.rs_pt_out = tok_in, pack_out, rs_pt_out
        for name, t, shape, dt in (
            ("tok_in", tok_in, (1, R), u32),
            ("pack_out", pack_out, (2 * K + 5, W), u32),
            ("rs_pt_out", rs_pt_out, (R, nb), i32),
        ):
            if t is not None:
                cls._check_bound(name, t, shape, dt, RM)
        if tok_in is None:
            self.tok_in = self._dev(mesh, z(1, R), u32, RM)
        self.aux_in = self._dev(mesh, z(FUSED_AUX_ROWS, W), i32, RM)
        self.pt_rows_in = self._dev(mesh, z(R, nb), i32, RM)
        if pack_out is None:
            self.pack_out = self._dev(mesh, z(2 * K + 5, W), u32, RM)
        if rs_pt_out is None:
            self.rs_pt_out = self._dev(mesh, z(R, nb), i32, RM)
        self.result_out = self._dev(mesh, z(2, W), i32, RM)
        f = lambda t: self._dev(mesh, t.to(torch.float32), f32, TL)
        self.c_iota_t = f(torch.arange(T).reshape(1, T))  # j along columns
        self.c_iota_k = f(torch.arange(K).reshape(1, K))
        self.c_ut = f((torch.arange(B) * T).reshape(B, 1))  # u * T
        self.c_one = f(torch.ones(1, 1))
        if spec_accept_kernel_enabled() and cls is FusedGreedyDevice:  # the sampled subclass keeps its own body
            from models.demos.blackhole.qwen36.tt.spec_fused_kernels import attach_accept_kernel

            self._kernel = attach_accept_kernel(self, spec_accept_kernel_cores())
        return self

    def bind_ids(self, ids):
        """Point ``ids_in`` at the verify trace's ids output (a trace output: it exists only after the verify capture, so
        the engine warms ``body`` on a standalone ids buffer and binds the live one just before the capture)."""
        import ttnn

        self._check_bound("ids_in", ids, (1, 1, 1, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        if self._kernel is not None:  # the kernel was compiled on the stand-in: the live ids must share its layout
            self._kernel.check_rebind(ids)
        self.ids_in = ids

    def _stage_one(self, dst, t, dt):
        import ttnn

        host = ttnn.from_torch(
            torch.as_tensor(t).to(torch.int32),
            dtype=dt,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(host, dst)

    def stage_aux(self, aux):
        """Host -> device copy of the aux words (the only per-step input besides the page-table rows)."""
        import ttnn

        assert tuple(torch.as_tensor(aux).shape) == (FUSED_AUX_ROWS, self.W), tuple(torch.as_tensor(aux).shape)
        self._stage_one(self.aux_in, aux, ttnn.int32)

    def stage_pt_rows(self, pt_rows):
        """Copy of the page-table rows [R, nb] (row r = pt[r // T]); the engine does it only when the table changed."""
        import ttnn

        self._stage_one(self.pt_rows_in, pt_rows, ttnn.int32)

    def stage(self, ids_words, tok_words, aux, pt_rows):
        """Standalone use: host -> device copies of all four inputs into the persistent buffers (replay-safe)."""
        import ttnn

        R = self.R
        ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        ids[0, 0, 0, :R] = torch.as_tensor(ids_words).reshape(-1)[:R].to(torch.int32)
        self._stage_one(self.ids_in, ids, ttnn.uint32)
        self._stage_one(self.tok_in, torch.as_tensor(tok_words).reshape(1, R), ttnn.uint32)
        self.stage_aux(aux)
        self.stage_pt_rows(pt_rows)

    def body(self):
        """The trace-capturable sequence. No host round trip, no allocation-dependent output address."""
        import ttnn

        if self._kernel is not None:  # QWEN36_SPEC_ACCEPT_KERNEL: one program instead of the op graph
            self._kernel.run()
            return
        B, K, T, R, W, nb = self.B, self.K, self.T, self.R, self.W, self.nb
        TL, RM, f32 = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, ttnn.float32
        tl = lambda x: ttnn.to_layout(x, TL)
        fl = lambda x: ttnn.typecast(x, f32)
        cat = lambda xs, d: ttnn.concat(xs, dim=d)
        # ---- inputs -> fp32 TILE
        ids = fl(tl(ttnn.reshape(ttnn.slice(self.ids_in, [0, 0, 0, 0], [1, 1, 1, R]), (B, T))))  # [B,T]
        tok = fl(tl(ttnn.reshape(self.tok_in, (B, T))))  # [B,T]
        aux = fl(tl(self.aux_in))  # [7,W]
        auxt = ttnn.transpose(aux, 0, 1)  # [W,7]
        col = lambda i: ttnn.slice(auxt, [0, i], [B, i + 1])  # [B,1]
        nv, dec, keep, p0 = col(AUX_NUM_VALID), col(AUX_DECLINED), col(AUX_KEEP), col(AUX_P0)
        c = lambda i: ttnn.slice(aux, [AUX_CONSTS, i], [AUX_CONSTS + 1, i + 1])  # [1,1]
        scratch, rows = c(CONST_SCRATCH), c(CONST_ROPE_ROWS)
        delta = col(AUX_DELTA)  # [B,1] per-row rope delta (broadcasts over axis 1 of [B,T] / [B,K])
        rows_m1 = ttnn.subtract(rows, self.c_one)
        # ---- greedy walk: m = length of the valid, matching draft prefix
        eq = fl(ttnn.eq(ttnn.slice(ids, [0, 0], [B, K]), ttnn.slice(tok, [0, 1], [B, T])))  # [B,K]
        acc = ttnn.multiply(eq, fl(ttnn.lt(self.c_iota_k, nv)))
        m = ttnn.sum(ttnn.cumprod(acc, dim=1), dim=1, keepdim=True)  # [B,1]
        cnt = ttnn.add(m, 1.0)
        one_hot = fl(ttnn.eq(self.c_iota_t, m))  # [B,T]
        committed = ttnn.subtract(ttnn.multiply(ttnn.add(ids, 1.0), fl(ttnn.le(self.c_iota_t, m))), 1.0)  # [B,T]
        ndec = ttnn.rsub(dec, 1.0)
        pending = ttnn.multiply(ttnn.sum(ttnn.multiply(one_hot, ids), dim=1, keepdim=True), ndec)
        p = ttnn.multiply(ttnn.add(p0, m), ndec)
        anchor = ttnn.add(self.c_ut, m)
        real = fl(ttnn.lt(self.c_iota_t, ttnn.add(m, ttnn.multiply(dec, keep))))  # [B,T]
        rtok = ttnn.multiply(committed, real)
        rpos = ttnn.multiply(ttnn.add(p0, self.c_iota_t), real)
        steps = ttnn.add(p, self.c_iota_k)  # [B,K]
        ridx = lambda x: ttnn.minimum(ttnn.relu(ttnn.add(x, delta)), rows_m1)
        # ---- assemble pack [2K+5, W]
        padw = lambda x, n: ttnn.pad(x, [(0, 0), (0, W - n)], value=0.0)
        g1 = padw(ttnn.transpose(cat([pending, steps], 1), 0, 1), B)  # [K+1,W]
        g2 = padw(ttnn.reshape(cat([rtok, rpos], 0), (2, R)), R)
        g3 = padw(ttnn.transpose(anchor, 0, 1), B)
        g4 = padw(ttnn.reshape(ridx(rpos), (1, R)), R)
        g5 = padw(ttnn.transpose(ridx(steps), 0, 1), B)  # [K,W]
        pack = cat([g1, g2, g3, g4, g5], 0)
        ttnn.copy(ttnn.to_layout(ttnn.typecast(pack, ttnn.uint32), RM), self.pack_out)
        # ---- result [2,W]
        res = cat([padw(ttnn.reshape(committed, (1, R)), R), padw(ttnn.transpose(cnt, 0, 1), B)], 0)
        ttnn.copy(ttnn.to_layout(ttnn.typecast(res, ttnn.int32), RM), self.result_out)
        # ---- reseed page table: real rows keep their pt row, the others get the scratch block
        pt = fl(tl(self.pt_rows_in))  # [R,nb]
        real_col = ttnn.reshape(real, (R, 1))
        rs = ttnn.add(ttnn.multiply(ttnn.subtract(pt, scratch), real_col), scratch)
        ttnn.copy(ttnn.to_layout(ttnn.typecast(rs, ttnn.int32), RM), self.rs_pt_out)
