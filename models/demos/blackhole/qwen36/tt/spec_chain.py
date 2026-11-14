# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact HOST reference of the S3 "chain trace" CH (work/notes/63_m1_s3_s4_design.md, section 1): the device builds the
NEXT verify's inputs from the previous fused step's result words, the drafter ids and one host overlay. Pure torch
(no ttnn call). tests/test_spec_chain_host.py proves it equals the composition of today's host builders
(sanitize_verify_block, Qwen36Model._vfy_aux_host, spec_ctrl_page, rope_row_index, fused_accept_aux_host).

Rows are ENGINE rows, r = u*T + j, T = K + 1, R = B*T. Every tensor is int32 holding a uint32 bit pattern unless said.

RES_C int32 [4, W], W = ceil32(max(R, B)) (shared result buffer, one D2H for the host):
  row RES_COMMITTED  committed[R]   walk result, row u*T+j, -1 past counts[u]  (the ONLY signed row)
  row RES_COUNTS     counts[B]      in [1, T]; m = counts - 1
  row RES_P0         p0_prev[B]     position of column 0 of the step that made it (aux row AUX_P0)
  row RES_OVERFLOW   overflow[B]    sampled top-k tie overflow flag (poison in chain mode; not read by CH)

OVR int32 [11, W_o], W_o = ceil32(max(R, B, 5)) (host overlay for the NEXT step, ONE H2D copy before CH):
  rows 0..5          the fused aux layout of the next step (spec_fused.AUX_*): nv'[B], active'[B], declined'[B],
                     keep'[B], p0_host[B] (used only where mask=1), consts[5] (scratch, rope_delta two's complement,
                     rope rows, B, K). All m-independent, so the host knows them at enqueue time.
  row OVR_MASK       mask[B]        1 = host owns the row this step (finish, padding, join): device values are dropped
  row OVR_TOKENS     host_tokens[R] sanitized block (pad tail = column 0, inactive row = 0), used where mask=1
  row OVR_MI         mi_host[B]     used where mask=1 (0 for an inactive row)
  row OVR_WORDS      word 0 = GDN conv-window parity of the next verify (0/1); other words 0
  row OVR_DELTA      row_delta[B]   each engine row's own rope delta, two's complement (the chain's ridx; copied into the
                     next fused aux row AUX_DELTA)
"""

import torch

from models.demos.blackhole.qwen36.tt.spec_fused import (
    AUX_ACTIVE,
    AUX_CONSTS,
    AUX_DECLINED,
    AUX_DELTA,
    AUX_KEEP,
    AUX_NUM_VALID,
    AUX_P0,
    CONST_B,
    CONST_K,
    CONST_ROPE_DELTA,
    CONST_ROPE_ROWS,
    CONST_SCRATCH,
    FUSED_AUX_ROWS,
    row_delta_list,
    signed_rope_delta,
)
from models.demos.blackhole.qwen36.tt.tp_common import (
    HOSTSET_PACK_ALIGN,
    pack_u32_rows,
    rope_row_index,
    spec_chain_kernel_enabled,
)

RES_ROWS = 4
RES_COMMITTED, RES_COUNTS, RES_P0, RES_OVERFLOW = range(4)
OVR_ROWS = 11
OVR_MASK, OVR_TOKENS, OVR_MI, OVR_WORDS, OVR_DELTA = 6, 7, 8, 9, 10


def _align(n):
    return -(-int(n) // HOSTSET_PACK_ALIGN) * HOSTSET_PACK_ALIGN


def _i64(x):
    return torch.as_tensor(x).reshape(-1).to(torch.int64)


def _wrap32(t):
    """int64 -> int32 holding the uint32 bit pattern."""
    return ((t + 2**31) % 2**32 - 2**31).to(torch.int32)


def res_c_reference(committed, counts, p0, overflow):
    """RES_C int32 [4, W] from committed [B, T] (-1 past count) / counts [B] / p0_prev [B] / overflow [B]."""
    committed = torch.as_tensor(committed).to(torch.int64)
    B, T = committed.shape
    rows = [committed.reshape(-1), _i64(counts), _i64(p0), _i64(overflow)]
    W = _align(max(B * T, B))
    out = torch.zeros(RES_ROWS, W, dtype=torch.int64)
    for i, r in enumerate(rows):
        out[i, : r.numel()] = r
    return out.to(torch.int32)


def pack_ovr_host(
    nv,
    active,
    declined,
    keep,
    p0_host,
    mask,
    host_tokens,
    mi_host,
    parity,
    *,
    K,
    scratch_block,
    rope_delta,
    rope_rows,
    common_delta=None,
):
    """OVR int32 [11, W_o] (layout in the module doc). All per-row args have B entries; host_tokens is [B, T] or
    [R] (zeros are fine for rows with mask=0); parity is 0 or 1."""
    nv, act, dec, keep, p0h, mask, mih = (_i64(x) for x in (nv, active, declined, keep, p0_host, mask, mi_host))
    B, T = nv.numel(), K + 1
    tok = _i64(host_tokens) if host_tokens is not None else torch.zeros(B * T, dtype=torch.int64)
    assert tok.numel() == B * T and parity in (0, 1)
    consts = [0] * 5
    row_d = row_delta_list(rope_delta, B)
    common = common_delta if common_delta is not None else (row_d[0] if len(set(row_d)) == 1 else 0)
    consts[CONST_SCRATCH], consts[CONST_ROPE_DELTA] = int(scratch_block), int(common) & 0xFFFFFFFF
    consts[CONST_ROPE_ROWS], consts[CONST_B], consts[CONST_K] = int(rope_rows), B, int(K)
    dwords = torch.tensor([d & 0xFFFFFFFF for d in row_d], dtype=torch.int64)
    rows = [nv, act, dec, keep, p0h, torch.tensor(consts), mask, tok, mih, torch.tensor([int(parity)]), dwords]
    out = torch.zeros(OVR_ROWS, _align(max(B * T, B, 5)), dtype=torch.int64)
    for i, r in enumerate(rows):
        out[i, : r.numel()] = r
    if int(out.min()) < 0:
        raise ValueError("pack_ovr_host: negative word")
    return _wrap32(out)


def chain_reference(
    res_c, drafts, fused_aux_prev, ovr, *, B, K, Nv, rope_rows, ctrl_words, rope_delta=None, ring_idx=None
):
    """Pure-torch twin of CH. Returns (token_buf int32 [R] (uint32 words), vfy_aux int32 [3, W] (uint32 words),
    next_fused_aux int32 [7, W_aux]).

      m = counts - 1                                     (design note 63 section 1.3, step 2-3)
      p0' = ((p0_prev + m + 1) * (1-mask) + p0_host * mask) * active'      next start = cpos[u, m] (SE:3093)
      mi' = (m * (1-mask) + mi_host * mask) * active'
      tok0 = committed[u, m]; cols j=0..K-1: draft d_j if j < nv' else tok0 (sanitize_verify_block pad rule, SE:324-350)
      tokens = ((tok0, cols) * (1-mask) + host_tokens * mask) * active'
      row0 = p0' + j (model._vfy_row_positions, model.py:2309); row1 = rope_row_index (tp_common.py:156)
      row2 = spec_ctrl_page (gdn/tp.py:301): [parity, mi'[B], ((mi'*B+u)*Nv+h)[B*Nv]] zero padded to ctrl_words
      ``ring_idx`` (QWEN36_MAMBA_SPEC: one int [B, T*Nv] ring_page_idx per GDN group): rows 2 .. 1 + G instead, the ctrl
      page of group g = [parity, mi'[B], ring_idx[g][u, mi'*Nv + h][B*Nv]] (the start state in window column mi')
      next aux = ovr rows 0..3,5 and OVR_DELTA (as aux row 6) with row 4 = p0' (spec_fused.fused_accept_aux_host layout)
    ``fused_aux_prev`` (the step that made res_c; may be None) is only checked: its p0 row must equal RES_P0."""
    T, R = K + 1, B * (K + 1)
    res = torch.as_tensor(res_c).to(torch.int64)
    ovr = torch.as_tensor(ovr).to(torch.int64)
    committed = res[RES_COMMITTED, :R].reshape(B, T)
    counts, p0_prev = res[RES_COUNTS, :B], res[RES_P0, :B]
    if fused_aux_prev is not None:
        prev = torch.as_tensor(fused_aux_prev).to(torch.int64)
        if not torch.equal(prev[AUX_P0, :B], p0_prev):
            raise ValueError("chain_reference: res_c p0_prev differs from the previous fused aux p0")
    drafts = torch.as_tensor(drafts).to(torch.int64).reshape(B, K)
    nv, act, dec, keep, p0_host = (ovr[i, :B] for i in (AUX_NUM_VALID, AUX_ACTIVE, AUX_DECLINED, AUX_KEEP, AUX_P0))
    mask, mi_host, parity = ovr[OVR_MASK, :B], ovr[OVR_MI, :B], int(ovr[OVR_WORDS, 0])
    host_tok = ovr[OVR_TOKENS, :R].reshape(B, T)
    nm = 1 - mask
    m = counts - 1
    p0n = (nm * (p0_prev + m + 1) + mask * p0_host) * act
    min_ = (nm * m + mask * mi_host) * act
    iota_t = torch.arange(T)[None, :]
    tok0 = (committed * (iota_t == m[:, None])).sum(dim=1)  # not zeroed for declined rows
    cols = torch.where(torch.arange(K)[None, :] < nv[:, None], drafts, tok0[:, None])
    tok_dev = torch.cat([tok0[:, None], cols], dim=1)
    tokens = (nm[:, None] * tok_dev + mask[:, None] * host_tok) * act[:, None]
    pos = (p0n[:, None] + iota_t).reshape(-1)
    # per-row deltas: ovr row OVR_DELTA (signed words); ``rope_delta`` (int | B ints) overrides it when given
    row_d = (
        row_delta_list(rope_delta, B)
        if rope_delta is not None
        else [signed_rope_delta(w) for w in ovr[OVR_DELTA, :B].tolist()]
    )
    ridx = rope_row_index(pos, [d for d in row_d for _ in range(T)], rope_rows).to(torch.int64)
    ctrls = chain_ctrl_rows(min_, parity, B=B, K=K, Nv=Nv, ctrl_words=ctrl_words, ring_idx=ring_idx)
    token_buf = _wrap32(tokens.reshape(-1))
    vfy_aux = pack_u32_rows([pos, ridx, *ctrls])
    w_aux = _align(max(B, 5))
    aux = torch.cat([ovr[:AUX_DELTA, :w_aux], ovr[OVR_DELTA : OVR_DELTA + 1, :w_aux]], 0).clone()
    aux[AUX_P0, :B] = p0n
    return token_buf, vfy_aux, _wrap32(aux)


def chain_ctrl_rows(min_, parity, *, B, K, Nv, ctrl_words, ring_idx=None):
    """The GDN ctrl page(s) of chain_reference: [parity, mi'[B], start block or page per (u, h)] zero padded to
    ctrl_words, int64. Without ``ring_idx`` one page of ring blocks (mi'*B+u)*Nv+h; with it one page per GDN group whose
    start pages are ring_idx[g][u, mi'*Nv + h]."""
    min_ = torch.as_tensor(min_).to(torch.int64).reshape(B)
    if ring_idx is None:
        blks = [((min_ * B + torch.arange(B))[:, None] * Nv + torch.arange(Nv)[None, :]).reshape(-1)]
    else:
        cols = (min_[:, None] * Nv + torch.arange(Nv)[None, :]).to(torch.int64)
        blks = [
            torch.gather(torch.as_tensor(r).to(torch.int64).reshape(B, (K + 1) * Nv), 1, cols).reshape(-1)
            for r in ring_idx
        ]
    ctrls = []
    for blk in blks:
        ctrl = torch.zeros(ctrl_words, dtype=torch.int64)
        ctrl[0] = int(parity)
        ctrl[1 : 1 + B] = min_
        ctrl[1 + B : 1 + B + B * Nv] = blk
        ctrls.append(ctrl)
    return ctrls


def fused_fit_mode_range(p0_prev, capacity, active, K):
    """Fit mode of the NEXT fused step's row when the step before it is still in flight (two-step window).
    p0_prev = start0 of the PREVIOUS step. Its commit index m in 0..K makes the next start0 = p0_prev + m + 1, and
    the next step's draft fit (draft_rows_fit, SE:431; fused_fit_mode, spec_fused.py:40) tests anchor
    p = start0 + m' for m' in 0..K: p0_prev + m + m' + 2 + K < capacity. 'fits_all' iff true for every (m, m'),
    'fits_none' iff false for every (m, m'), else 'mixed' (not chainable). Closed form over the extremes
    (m + m' in [0, 2K]): all <=> p0_prev + 3K + 2 < capacity; none <=> p0_prev + K + 2 >= capacity.
    An inactive row is 'fits_none' (it declines whatever m is)."""
    out = []
    for s, a in zip(torch.as_tensor(p0_prev).reshape(-1).tolist(), torch.as_tensor(active).reshape(-1).tolist()):
        if not a:
            out.append("fits_none")
        elif int(s) + 3 * K + 2 < int(capacity):
            out.append("fits_all")
        elif int(s) + K + 2 >= int(capacity):
            out.append("fits_none")
        else:
            out.append("mixed")
    return out


def fused_fit_mode_bounds(start_lo, start_hi, capacity, active, K):
    """Fit mode of a fused step's row when its own start0 is only known to lie in [start_lo, start_hi] (a chained step enqueued
    before the step it follows was read; deferred serving, work/notes/63 S4b). The step's draft fit tests the anchor p = start0 + m'
    for m' in 0..K (draft_rows_fit: p + 1 + K < capacity): 'fits_all' iff start_hi + 2K + 1 < capacity, 'fits_none' iff
    start_lo + K + 1 >= capacity, else 'mixed' (not chainable). start_lo == start_hi is spec_fused.fused_fit_mode;
    start0 in [p0_prev + 1, p0_prev + K + 1] is fused_fit_mode_range. An inactive row is 'fits_none'."""
    out = []
    lo, hi = (torch.as_tensor(x).reshape(-1).tolist() for x in (start_lo, start_hi))
    for a, b, act in zip(lo, hi, torch.as_tensor(active).reshape(-1).tolist()):
        if not act:
            out.append("fits_none")
        elif int(b) + 2 * K + 1 < int(capacity):
            out.append("fits_all")
        elif int(a) + K + 1 >= int(capacity):
            out.append("fits_none")
        else:
            out.append("mixed")
    return out


class ChainDevice:
    """Device twin of chain_reference (design note 63 section 1.3): ONE captured body that builds the next verify's
    inputs from the previous fused step's result words, the K drafter id tensors and one host overlay.

    Inputs : result_in  int32 RM [2 or 3, W] (S1 result_out [2,W] or S2 [3,W]: rows 0-1 are read: committed, counts)
             aux_prev_in int32 RM [7, W_a] (the fused aux_in of the step that made result_in; row AUX_P0 is read)
             draft_ins  K uint32 RM tensors [1,1,B,1] (the drafter's draft_argmax outputs, B valid ids each)
             ovr_in     int32 RM [11, W_o] (pack_ovr_host layout, staged by the host: ``stage_ovr``)
    Outputs, written IN PLACE with ttnn.copy (addresses never change):
             tok_out    uint32 RM [1, R]   (= model._vfy_token_buf)
             vaux_out   uint32 RM [3, W_v] (= model._vfy_aux_buf; W_v = align(max(R, spec_ctrl_words(B, Nv))));
                        [2 + G, W_v] with ``ring_pages`` (QWEN36_MAMBA_SPEC: the fp32 [B, T*G*Nv] start-page table of
                        the G GDN groups, SpecBlockBook.pages_cat): one ctrl page per group, start pages gathered at
                        column mi' (chain_reference with ring_idx)
             ring_windows  with ``ring_windows`` (chained state moves): each group's ring_page_idx (uint32 RM
                        [B, T*Nv]) = the fp32 window-page table [B, T*G*T*Nv] (SpecBlockBook.windows_cat) at accepted
                        count mi' + 1, so the verify writes the window of the count the pending verify accepted
             next_aux_out = aux_prev_in itself (int32 [7, W_a]): every value is read before the final copy.
    All values are integers < 2^24, so fp32 TILE math is exact.
    QWEN36_SPEC_CHAIN_KERNEL=1: ``body`` is ONE single-core kernel on the same buffers (spec_fused_kernels.ChainKernel); the op
    graph below stays the reference (and the fallback when the buffers do not fit the kernel)."""

    _kernel = None  # QWEN36_SPEC_CHAIN_KERNEL: the ChainKernel of this device (None = the op graph)

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
    def _check_bound(name, t, shape, dtype):
        import ttnn

        got = (tuple(int(d) for d in t.shape), t.dtype, t.layout)
        want = (tuple(shape), dtype, ttnn.ROW_MAJOR_LAYOUT)
        assert got == want, f"bound {name}: {got} != {want}"

    @classmethod
    def allocate(
        cls,
        mesh,
        B,
        K,
        Nv,
        result_rows=3,
        result_in=None,
        aux_prev_in=None,
        draft_ins=None,
        tok_out=None,
        vaux_out=None,
        ring_pages=None,
        ring_windows=None,
        ring_pages_rm=None,
        ring_windows_rm=None,
    ):
        """Persistent buffers; every ``*_in`` / ``*_out`` argument binds a LIVE engine tensor (shape / dtype / layout
        checked), None allocates a standalone one. ``ring_pages`` = (fp32 TILE [B, T * G * Nv] table, G);
        ``ring_windows`` = (fp32 TILE [B, T * G * T * Nv] table, [G] uint32 RM [B, T * Nv] ring_page_idx outputs), needs
        ``ring_pages``: see the class docstring. ``ring_pages_rm``: the same start pages as ``ring_pages`` in uint32 RM
        [B, T * G * Nv], which QWEN36_SPEC_CHAIN_KERNEL reads; ``ring_windows_rm``: likewise the window table in uint32
        RM [B, T * G * T * Nv]."""
        import ttnn
        from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_words

        self = cls()
        self.mesh, self.B, self.K, self.Nv, self.T, self.R = mesh, B, K, Nv, K + 1, B * (K + 1)
        self.ring_pages, self.G = (None, 1) if ring_pages is None else ring_pages
        n_ctrl = self.G
        if self.ring_pages is not None:
            got = (tuple(int(d) for d in self.ring_pages.shape), self.ring_pages.dtype, self.ring_pages.layout)
            want = ((B, (K + 1) * self.G * Nv), ttnn.float32, ttnn.TILE_LAYOUT)
            assert got == want, f"bound ring_pages: {got} != {want}"
        self.ring_windows, self.ring_idx_outs = (None, []) if ring_windows is None else ring_windows
        self.ring_pages_rm, self.ring_windows_rm = ring_pages_rm, ring_windows_rm
        if self.ring_windows is not None:
            T = K + 1
            assert self.ring_pages is not None, "ring_windows needs ring_pages"
            got = (tuple(int(d) for d in self.ring_windows.shape), self.ring_windows.dtype, self.ring_windows.layout)
            want = ((B, T * self.G * T * Nv), ttnn.float32, ttnn.TILE_LAYOUT)
            assert got == want, f"bound ring_windows: {got} != {want}"
            assert len(self.ring_idx_outs) == self.G, (len(self.ring_idx_outs), self.G)
            for g, t in enumerate(self.ring_idx_outs):
                cls._check_bound(f"ring_idx_outs[{g}]", t, (B, T * Nv), ttnn.uint32)
            if ring_windows_rm is not None:
                cls._check_bound("ring_windows_rm", ring_windows_rm, (B, T * self.G * T * Nv), ttnn.uint32)
        R = self.R
        self.W_res, self.W_a = _align(max(R, B)), _align(max(B, 5))
        self.W_o, self.ctrl_words = _align(max(R, B, 5)), spec_ctrl_words(B, Nv)
        self.W_v = _align(max(R, self.ctrl_words))
        assert 1 + B + B * Nv <= self.W_v and R <= self.W_res and self.W_a <= self.W_o
        RM, TL, u32, i32, f32 = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, ttnn.uint32, ttnn.int32, ttnn.float32
        z = lambda *s: torch.zeros(*s, dtype=torch.int32)
        if result_in is not None:
            assert int(result_in.shape[0]) in (2, 3), tuple(result_in.shape)
            self._check_bound("result_in", result_in, (int(result_in.shape[0]), self.W_res), i32)
        self.result_in = result_in if result_in is not None else cls._dev(mesh, z(result_rows, self.W_res), i32, RM)
        for name, t, shape, dt in (
            ("aux_prev_in", aux_prev_in, (FUSED_AUX_ROWS, self.W_a), i32),
            ("tok_out", tok_out, (1, R), u32),
            ("vaux_out", vaux_out, (2 + n_ctrl, self.W_v), u32),
        ):
            if t is not None:
                cls._check_bound(name, t, shape, dt)
        self.aux_prev_in = (
            aux_prev_in if aux_prev_in is not None else cls._dev(mesh, z(FUSED_AUX_ROWS, self.W_a), i32, RM)
        )
        self.tok_out = tok_out if tok_out is not None else cls._dev(mesh, z(1, R), u32, RM)
        self.vaux_out = vaux_out if vaux_out is not None else cls._dev(mesh, z(2 + n_ctrl, self.W_v), u32, RM)
        if draft_ins is None:
            draft_ins = [cls._dev(mesh, z(1, 1, B, 1), u32, RM) for _ in range(K)]
        assert len(draft_ins) == K
        for i, t in enumerate(draft_ins):
            cls._check_bound(f"draft_ins[{i}]", t, (1, 1, B, 1), u32)
        self.draft_ins = list(draft_ins)
        self.ovr_in = cls._dev(mesh, z(OVR_ROWS, self.W_o), i32, RM)
        f = lambda t: cls._dev(mesh, t.to(torch.float32), f32, TL)
        self.c_iota_t = f(torch.arange(self.T).reshape(1, self.T))
        self.c_iota_k = f(torch.arange(K).reshape(1, K))
        self.c_iota_h = f(torch.arange(Nv).reshape(1, Nv))
        self.c_u = f(torch.arange(B).reshape(B, 1))
        if spec_chain_kernel_enabled():
            from models.demos.blackhole.qwen36.tt.spec_fused_kernels import attach_chain_kernel

            self._kernel = attach_chain_kernel(self)
        return self

    def bind_draft_ins(self, draft_ins):
        """Point the K draft inputs at the drafter trace's LIVE id outputs (trace outputs exist only after the drafter capture; the
        engine warms ``body`` on the stand-ins from ``allocate`` and binds the live ones just before the chain capture).
        """
        import ttnn

        assert len(draft_ins) == self.K, (len(draft_ins), self.K)
        for i, t in enumerate(draft_ins):
            self._check_bound(f"draft_ins[{i}]", t, (1, 1, self.B, 1), ttnn.uint32)
        if self._kernel is not None:  # the kernel was compiled on the stand-ins: the live ids must share their layout
            self._kernel.check_rebind(draft_ins)
        self.draft_ins = list(draft_ins)

    def stage_ovr(self, ovr):
        """Host -> device copy of the overlay (pack_ovr_host output, int32 [11, W_o]); the one per-step H2D input."""
        import ttnn

        ovr = torch.as_tensor(ovr).to(torch.int32)
        assert tuple(ovr.shape) == (OVR_ROWS, self.W_o), tuple(ovr.shape)
        host = ttnn.from_torch(
            ovr, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(host, self.ovr_in)

    def body(self):
        """The trace-capturable sequence (no host round trip, no allocation-dependent output address)."""
        import ttnn

        if self._kernel is not None:  # QWEN36_SPEC_CHAIN_KERNEL: one program instead of the op graph
            self._kernel.run()
            return
        B, K, T, R, Nv = self.B, self.K, self.T, self.R, self.Nv
        W_a, W_v = self.W_a, self.W_v
        TL, RM, f32 = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, ttnn.float32
        fl = lambda x: ttnn.typecast(ttnn.to_layout(x, TL), f32)
        cat = lambda xs, d: ttnn.concat(xs, dim=d)
        padw = lambda x, n: ttnn.pad(x, [(0, 0), (0, W_v - n)], value=0.0)
        # ---- inputs -> fp32 TILE (result rows 0-1 only; aux_prev p0 row only)
        res = fl(ttnn.slice(self.result_in, [0, 0], [2, self.W_res]))  # [2,W]
        committed = ttnn.reshape(ttnn.slice(res, [0, 0], [1, R]), (B, T))
        counts = ttnn.slice(ttnn.transpose(res, 0, 1), [0, RES_COUNTS], [B, RES_COUNTS + 1])  # [B,1]
        auxp_t = ttnn.transpose(fl(self.aux_prev_in), 0, 1)  # [W_a,7]
        p0_prev = ttnn.slice(auxp_t, [0, AUX_P0], [B, AUX_P0 + 1])
        ovr = fl(self.ovr_in)  # [11,W_o]
        ovr_t = ttnn.transpose(ovr, 0, 1)
        col = lambda i: ttnn.slice(ovr_t, [0, i], [B, i + 1])  # [B,1]
        nv, act, p0_host = col(AUX_NUM_VALID), col(AUX_ACTIVE), col(AUX_P0)
        mask, mi_host = col(OVR_MASK), col(OVR_MI)
        host_tok = ttnn.reshape(ttnn.slice(ovr, [OVR_TOKENS, 0], [OVR_TOKENS + 1, R]), (B, T))
        parity = ttnn.slice(ovr, [OVR_WORDS, 0], [OVR_WORDS + 1, 1])  # [1,1]
        c = lambda i: ttnn.slice(ovr, [AUX_CONSTS, i], [AUX_CONSTS + 1, i + 1])
        rows = c(CONST_ROPE_ROWS)
        delta = col(OVR_DELTA)  # [B,1] per-row rope delta
        rows_m1 = ttnn.subtract(rows, 1.0)
        nm = ttnn.rsub(mask, 1.0)
        # ---- m, p0', mi'
        m = ttnn.subtract(counts, 1.0)
        p0n = ttnn.multiply(ttnn.add(ttnn.multiply(ttnn.add(p0_prev, counts), nm), ttnn.multiply(p0_host, mask)), act)
        min_ = ttnn.multiply(ttnn.add(ttnn.multiply(m, nm), ttnn.multiply(mi_host, mask)), act)  # [B,1]
        # ---- tokens: tok0 = committed[u, m]; cols = draft j if j < nv' else tok0
        tok0 = ttnn.sum(ttnn.multiply(fl(ttnn.eq(self.c_iota_t, m)), committed), dim=1, keepdim=True)  # [B,1]
        drafts = cat([ttnn.reshape(fl(d), (B, 1)) for d in self.draft_ins], 1)  # [B,K]
        lt = fl(ttnn.lt(self.c_iota_k, nv))
        cols = ttnn.add(ttnn.multiply(drafts, lt), ttnn.multiply(tok0, ttnn.rsub(lt, 1.0)))
        tok_dev = cat([tok0, cols], 1)  # [B,T]
        tokens = ttnn.multiply(ttnn.add(ttnn.multiply(tok_dev, nm), ttnn.multiply(host_tok, mask)), act)
        ttnn.copy(ttnn.to_layout(ttnn.typecast(ttnn.reshape(tokens, (1, R)), ttnn.uint32), RM), self.tok_out)
        # ---- verify aux [3, W_v]: kv positions, rope rows, GDN ctrl page
        pos = ttnn.add(p0n, self.c_iota_t)  # [B,T]
        ridx = ttnn.minimum(ttnn.relu(ttnn.add(pos, delta)), rows_m1)
        if self.ring_pages is None:
            blks = [
                ttnn.add(ttnn.multiply(ttnn.add(ttnn.multiply(min_, float(B)), self.c_u), float(Nv)), self.c_iota_h)
            ]
        else:  # QWEN36_MAMBA_SPEC: start page of (g, u, h) = the table at column mi'*G*Nv + g*Nv + h (one-hot over T)
            GN = self.G * Nv
            sel = fl(ttnn.eq(self.c_iota_t, min_))  # [B,T]
            acc = None
            for t in range(T):
                term = ttnn.multiply(
                    ttnn.slice(self.ring_pages, [0, t * GN], [B, (t + 1) * GN]), ttnn.slice(sel, [0, t], [B, t + 1])
                )
                acc = term if acc is None else ttnn.add(acc, term)  # [B, G*Nv]
            blks = [ttnn.slice(acc, [0, g * Nv], [B, (g + 1) * Nv]) for g in range(self.G)]
            if self.ring_windows is not None:  # chained state moves: the window of count mi' + 1, per group
                GTN = self.G * T * Nv
                win = None
                for t in range(T):
                    term = ttnn.multiply(
                        ttnn.slice(self.ring_windows, [0, t * GTN], [B, (t + 1) * GTN]),
                        ttnn.slice(sel, [0, t], [B, t + 1]),
                    )
                    win = term if win is None else ttnn.add(win, term)  # [B, G*T*Nv]
                for g, out in enumerate(self.ring_idx_outs):
                    w = ttnn.slice(win, [0, g * T * Nv], [B, (g + 1) * T * Nv])
                    ttnn.copy(ttnn.to_layout(ttnn.typecast(w, ttnn.uint32), RM), out)
        ctrls = [
            padw(cat([parity, ttnn.transpose(min_, 0, 1), ttnn.reshape(blk, (1, B * Nv))], 1), 1 + B + B * Nv)
            for blk in blks
        ]  # [1, 1+B+B*Nv] each
        vaux = cat([padw(ttnn.reshape(pos, (1, R)), R), padw(ttnn.reshape(ridx, (1, R)), R), *ctrls], 0)
        ttnn.copy(ttnn.to_layout(ttnn.typecast(vaux, ttnn.uint32), RM), self.vaux_out)
        # ---- next fused aux = ovr rows 0..3, 5, OVR_DELTA with row 4 = p0' (written over aux_prev_in: all reads are above)
        p0row = ttnn.pad(ttnn.transpose(p0n, 0, 1), [(0, 0), (0, W_a - B)], value=0.0)
        nxt = cat(
            [
                ttnn.slice(ovr, [0, 0], [AUX_P0, W_a]),
                p0row,
                ttnn.slice(ovr, [AUX_P0 + 1, 0], [AUX_DELTA, W_a]),
                ttnn.slice(ovr, [OVR_DELTA, 0], [OVR_DELTA + 1, W_a]),
            ],
            0,
        )
        ttnn.copy(ttnn.to_layout(ttnn.typecast(nxt, ttnn.int32), RM), self.aux_prev_in)
