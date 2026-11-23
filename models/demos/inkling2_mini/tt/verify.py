# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""R-token target step for greedy speculative decoding (verify pass), exact per row.

`VerifyStep(model, R)` runs R tokens at positions p0 .. p0 + R - 1 through the 48 applications in one pass; row t
must come out bit-equal to the plain 1-row step of token t at p0 + t. Every piece runs the 1-row program's math
per row:
- residual path (attn-res, seal, row norms, loop messages, short convs): F.per_row (fused/common.py), one copy of
  the 1-row program per row on its own cores, launched as one program;
- message gates and router: their R-row forms (fused/gate.py num_tokens / num_rows, fused/router.py rows);
- attention: the pending-row path of Inkling2AttentionFast (decode_rows' pieces: rings not written during the pass,
  sdpa_decode pending_kv / rows_per_token), here also in the triple input mode (global layers, loop passes 2-3),
  with the R conv windows built in one program (fused/kernels/verify_windows.cpp) into per-token buffers, so the
  window state is only replaced by the commit, and stacked for one projection;
- experts: the multi-row bank expert step (union of the rows' experts read once, per-row chains);
- head: one unembed for all rows, argmax per row on the raw logits + vocab mask (softcap is monotone, so the
  argmax is the plain step's; shortcut: exact ties after the plain step's softcap can pick another index, upgrade:
  the R-row fused tail).
Nothing the next pass reads is written for a token that may be rejected: rings, conv rings and window states change
only in `commit(n)` (the first n rows, n = accepted + 1). The drafter context of all R rows is committed inside the
pass: rejected rows sit at positions >= the next anchor, which the next draft overwrites first.

Off by default: only `SpecGenerator.generate_rows` (tt/spec_generator.py) uses it. R <= 3 (pending-row sdpa limit).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.inkling2_mini.convert import cache as wcache
from models.demos.inkling2_mini.tt.attention import ROWS, WINDOW
from models.demos.inkling2_mini.tt.attention_fast import (
    FAR_CHUNK,
    FAR_GRP,
    GROUPS,
    V_WIDTH,
    DecodePositionsFast,
    _NoFar,
)
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.attn import _cb, _crs, _kernel, _stage
from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv
from models.demos.inkling2_mini.tt.fused.conv import commit as conv_commit
from models.demos.inkling2_mini.tt.fused.gate import FusedMessageGate, FusedQKVGate
from models.demos.inkling2_mini.tt.model import APPS, HIDDEN, LOOP_END, LOOP_START, cache_index

R_MAX = 3
WIN_ROWS = 9  # rows of a conv window the projection reads (attention_fast: w_all.bank.x_rows)
_SHIFT = {}


def _shifts(device, back=False):
    """[1, 1, 32, 32] bf16 0/1 matrices: shift t moves rows 0..8 to 9t..9t+8 (back: 9t.. to 0..)."""
    key = (id(device), back)
    if key not in _SHIFT:
        mats = []
        for t in range(R_MAX):
            m = torch.zeros(32, 32)
            for i in range(WIN_ROWS):
                if back:
                    m[i, WIN_ROWS * t + i] = 1.0
                else:
                    m[WIN_ROWS * t + i, i] = 1.0
            mats.append(ttnn.from_torch(m.view(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                        device=device))  # fmt: skip
        _SHIFT[key] = (device, mats)
    return _SHIFT[key][1]


def _gate_weights(root, L):
    """Host tensors of layer L's message gates only (the rest of the layer is not read)."""
    root = wcache._root(root)
    out = {}
    for e in wcache.manifest(root)["tensors"]:
        if e["layer"] == L and e["name"].startswith("msg_gate."):
            out[e["name"]] = ttnn.load_tensor(root / e["file"])
    return out


def windows_to(win, hists, xs):
    """All R tokens' conv windows in one program (fused/kernels/verify_windows.cpp): token t's window from token
    t-1's (hists[0] = the state), written to hists[t + 1] and to rows 9t.. of the projection input (win.act, L1)."""
    R = len(xs)
    for t in [*hists, *(x for xt in xs for x in xt)]:
        F.check_row_input(t)
        assert t.dtype == ttnn.bfloat16, t.dtype
    n = win.ntiles
    per_tok = []
    for t in range(R):
        xq, xk, xv = xs[t] if len(xs[t]) == 3 else xs[t] * 3
        per_tok += [xq.buffer_address(), xk.buffer_address(), xv.buffer_address(), hists[t + 1].buffer_address()]
    rt = [[hists[0].buffer_address(), win.act.buffer_address(), i * n] + per_tok for i in range(len(win.cores))]
    crs = _crs(win.cores)
    named = dict(ntiles=n, ngroups=1 if len(xs[0]) == 1 else 2, rows=R)
    kernels = [_kernel("verify_windows.cpp", win.cores, named, rt, ttnn.ReaderConfigDescriptor())]
    cbs = [_cb(0, ttnn.bfloat16, n, crs), _cb(1, ttnn.bfloat16, n, crs), _stage(24, crs, n * 384 + 128)]
    io = [*hists, *(x for xt in xs for x in xt), win.act]
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return win.act


def attn_rows(attn, xs, state, positions, rows, p0, hists):
    """Inkling2AttentionFast.decode_rows for both input modes: xs[t] = [x] (single) or [xq, xk, xv] (triple);
    token t's window reads hists[t] (hists[0] = state.x_hist) and writes hists[t + 1]. Returns the o-projection
    input (rows 0 .. R-1 = tokens)."""
    assert attn.FUSED and attn.window is not None and len(xs) == rows.R
    R = rows.R

    # One w_all for all tokens: token t's window (rows 0..8 used, = the projection input the plain step projects)
    # goes to rows 9t.. of one input (windows_to, one program for all tokens), w_all reads 9R rows
    # (rows ride in the tile: each output row is the 1-row projection's); token t's rows are shifted back to row 0
    # (exact 0/1 matmul, DRAM) for its prep and output gate.
    act = windows_to(attn.window, hists, xs)
    # decode staging (mk/overlap.py) swaps w_all.bank for a proxy that reads the next app's weights from L1 where
    # host programs filled them; no verify program hosts fills, so the verify pass uses the raw bank linear
    proxy = attn.w_all.bank
    bank = proxy.st.bl if hasattr(proxy, "st") else proxy
    bank.x_rows = WIN_ROWS * R if WIN_ROWS * R <= 16 else 0  # x_rows.hpp reads <= 16 rows; 0 = whole tiles (same bytes)
    attn.w_all.bank = bank
    try:
        proj_all = attn.w_all(act, ttnn.bfloat16, attn.mm_cfg)
    finally:
        bank.x_rows = WIN_ROWS
        attn.w_all.bank = proxy

    def project(t):
        """Token t's projection with its rows at 0 (prep and AttnOut read some columns from row 0 only)."""
        r = ROWS["single" if len(xs[t]) == 1 else "triple"]
        return keep(proj_all, t), {"q": 0, "k": r["k"], "v": r["v"]}

    def keep(proj, t):
        """Token t's projection rows at row 0 (DRAM) for AttnOut, which reads the gate from projection row 0."""
        if t == 0:
            return proj  # rows 0.. already; proj_all is not rewritten before this app's AttnOut calls
        return ttnn.matmul(_shifts(attn.device, back=True)[t], proj, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           compute_kernel_config=attn.hifi4)  # fmt: skip

    if attn.is_global:
        assert FAR_GRP
        shared = (
            attn.far_rows_cfg is not None
            and R > 1
            and p0 >= WINDOW
            and (p0 - WINDOW) // FAR_CHUNK == (p0 + R - 1 - WINDOW) // FAR_CHUNK
        )
        projs = []
        for t, pos in enumerate(positions):
            proj, sel = project(t)
            _, mask, sink_t = attn.prep(proj, pos, sel, state, rows=(t, rows) if shared else (t, _NoFar(rows)))
            outs = (rows.o_near[t], rows.r[t], rows.sink_far if shared else attn.near.sink, t if shared else 0)
            attn.near(attn.prep.q_grp, mask, sink_t, state.k_ring, state.v_ring, pend=(rows, positions[0].pos, t + 1),
                      outs=outs)  # fmt: skip
            if shared:
                projs.append(proj)
                continue
            far_pos, valid = pos.far_grp
            far = ttnn.transformer.flash_multi_latent_attention_decode(
                attn.prep.q_far, state.kv_grp, head_dim_v=V_WIDTH, is_causal=True, cur_pos_tensor=far_pos,
                attention_sink=attn.near.sink, scale=1.0, program_config=attn.far_grp_cfg,
                compute_kernel_config=attn.hifi4,
            )  # fmt: skip
            attn.out(far, proj, (rows.o_near[t], rows.r[t], valid), row=t)
        if shared:
            far = ttnn.transformer.flash_multi_latent_attention_decode(
                rows.q_far, state.kv_grp, head_dim_v=V_WIDTH, is_causal=True,
                cur_pos_tensor=positions[R - 1].far_grp[0], attention_sink=rows.sink_far, scale=1.0,
                program_config=attn.far_rows_cfg, compute_kernel_config=attn.hifi4,
            )  # fmt: skip
            for t, proj in enumerate(projs):
                attn.out(far, proj, (rows.o_near[t], rows.r[t], positions[t].far_grp[1]), row=t, far_row=GROUPS * t)
        return attn.out.act
    projs = []
    for t, pos in enumerate(positions):
        proj, sel = project(t)
        attn.prep(proj, pos, sel, state, rows=(t, rows))
        projs.append(proj)
    a = ttnn.transformer.scaled_dot_product_attention_decode(
        rows.q, state.k_grp, state.v_grp, is_causal=False, attn_mask=rows.mask, attention_sink=rows.sink, scale=1.0,
        program_config=attn.rows_cfg, compute_kernel_config=attn.hifi4, pending_kv=rows.pending,
        pending_pos=positions[0].pos,
    )  # fmt: skip
    for t, proj in enumerate(projs):
        attn.out(a, proj, row=t)
    return attn.out.act


class VerifyStep:
    def __init__(self, model, R, root=None):
        """model: a TappedModel (tt/spec_generator.py) with its drafter attached."""
        assert 2 <= R <= R_MAX, R
        m = self.m = model
        dev = self.device = m.device
        self.R = R
        root = root or wcache.DEFAULT_ROOT
        self.positions = [DecodePositionsFast(dev, m.max_ctx) for _ in range(R)]
        self.tokens = [
            ttnn.from_torch(
                torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
            )  # fmt: skip
            for _ in range(R)
        ]
        self.rows = {}  # app state -> RowsOut
        self.hists = {}  # app state -> [state.x_hist, window after token 0, .., after token R-1]
        for L, p in APPS:
            c = cache_index(L, p)
            st = m.states[c][0]
            self.rows[c] = m.layers[L].attn.rows_out(R, st)
            self.hists[c] = [st.x_hist] + [ttnn.clone(st.x_hist) for _ in range(R)]
        self.gates = {}  # loop layer -> (R-token QKV gate, R-row mlp gate)
        for L in range(LOOP_START, LOOP_END):
            w = _gate_weights(root, L)
            self.gates[L] = (FusedQKVGate(m.rows, w, num_tokens=R), FusedMessageGate(m.rows, w, "mlp", num_rows=R))
        self.d = m.drafter
        self.ids = ttnn.from_torch(
            torch.zeros(1, R, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
        )  # fmt: skip  positions p0 .. p0+R-1 (drafter commit RoPE rows)
        self.phase = 0
        self.p0 = None
        self.out = None
        self.collect = None  # debug: a list that receives [R clones of each app's output rows] per app

    # ------------------------------------------------------------ host inputs
    def set_inputs(self, tokens, p0):
        """tokens: R host ints (anchor + R-1 drafts) at p0 .. p0 + R - 1. A traced pass replays only at the conv
        phase (p0 % 4) and far-chunk sharing it was captured with (Generator keeps one trace per phase)."""
        self.set_positions(p0)
        self.set_tokens(tokens)

    def set_tokens(self, tokens):
        assert len(tokens) == self.R
        for t, tok in enumerate(tokens):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.tensor([[tok]], dtype=torch.int32), dtype=ttnn.uint32), self.tokens[t]
            )

    def set_positions(self, p0):
        """The position inputs of a pass at p0 (independent of the drafts: the spec loop writes them while the
        draft runs)."""
        for t, pos in enumerate(self.positions):
            pos.update(p0 + t)
        ids = torch.arange(p0, p0 + self.R, dtype=torch.int32).view(1, -1)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self.ids)
        self.phase = p0 % FusedShortConv.PHASES
        self.p0 = p0

    # ------------------------------------------------------------ the pass
    def _app(self, L, p, hs, n_sealed, msgs):
        """hs: R row tensors (layer input of each token). msgs: (qkv, mlp) R-row message tensors or None.
        Returns R output rows."""
        m, R, dev = self.m, self.R, self.device
        lay = m.layers[L]
        c = cache_index(L, p)
        attn_st, conv_a, conv_m = m.states[c]
        triple = msgs is not None or lay.is_global
        rep = 3 if (msgs is not None and not lay.is_global) else None
        nq = 3 if triple else 1

        if lay.attn_res is None:
            xs_in = hs  # layer 0: the embedding rows
        else:
            xs_in = F.per_row(dev, R, lambda r: lay.attn_res(None, hs[r], n_sealed + 1, split=False, repeat_rows=rep,
                                                             row_out=False))  # fmt: skip
        if msgs is not None:
            x_all = self.gates[L][0](xs_in[0], msgs[0])  # rows 3r + j: token r's q / k / v input
            normed = F.per_row(dev, R, lambda r: lay.norm(x_all, num_rows=3, first_row=3 * r))
        elif lay.attn_res is None:
            normed = F.per_row(dev, R, lambda r: lay.norm(xs_in[r]))
        else:
            normed = F.per_row(dev, R, lambda r: lay.norm(xs_in[r], num_rows=nq, first_row=nq * r))
        xs = [list(n) if triple else [n[0]] for n in normed]
        act = attn_rows(lay.attn, xs, attn_st, self.positions, self.rows[c], self.p0, self.hists[c])
        y = lay.attn.w_o(act, ttnn.bfloat16, lay.attn.mm_cfg)  # one o-projection, row t = token t
        pend_a = [lay.conv_attn.pending_slot(conv_a, r) for r in range(R)]
        resid = F.per_row(dev, R, lambda r: lay.conv_attn.conv_add(
            y, conv_a, (self.phase + r) % FusedShortConv.PHASES, None if lay.boundary else hs[r], ttnn.float32,
            row_out=False, x_row=r, slot=pend_a[r]))  # fmt: skip
        if lay.boundary:
            F.per_row(dev, R, lambda r: m.frows.seal(None, hs[r], n_sealed))
            n_sealed += 1

        if msgs is None:
            x = F.per_row(dev, R, lambda r: lay.mlp_res(None, resid[r], n_sealed + 1, norm=lay.moe_norm))[0]
        else:
            mm = F.per_row(dev, R, lambda r: lay.mlp_res(None, resid[r], n_sealed + 1, row_out=False)[0])[0]
            g = self.gates[L][1](mm, msgs[1])
            x = F.per_row(dev, R, lambda r: lay.moe_norm(g, num_rows=1, first_row=r)[0])[0]
        yy, _ = lay.moe(x, normed=True)  # R-row router + multi-row expert step -> [1, 1, R, C] bf16 L1
        pend_m = [lay.conv_mlp.pending_slot(conv_m, r) for r in range(R)]
        out = F.per_row(dev, R, lambda r: lay.conv_mlp.conv_add(
            yy, conv_m, (self.phase + r) % FusedShortConv.PHASES, resid[r], ttnn.bfloat16, row_out=True, x_row=r,
            slot=pend_m[r]))  # fmt: skip
        return list(out), n_sealed

    def step(self):
        """One verify pass on (tokens, positions). Returns {"token": [device uint32 [1, 1, R, 1]] (row t = token t's
        argmax), "logits": raw + vocab mask [1, 1, R, V] fp32}."""
        m, R, dev = self.m, self.R, self.device
        for pos in self.positions:
            pos.prepare()
        hs = []
        for t in range(R):
            e = ttnn.embedding(self.tokens[t], m.embed, layout=ttnn.TILE_LAYOUT)
            hs.append(m._rms(ttnn.reshape(e, [1, 1, 1, HIDDEN]), m.embed_norm))
        taps = {t: [] for t in range(R)}
        from models.demos.inkling2_mini.tt.spec_generator import TAP_LP

        n, pre, h12_in, all_msgs = 0, None, None, None
        for L, p in APPS:
            if L == LOOP_START:
                if p == 1:
                    pre = (list(hs), n)
                else:
                    keys = (h12_in, hs)
                    all_msgs = F.per_row(dev, R, lambda r: m.msg(keys[0][r], keys[1][r]))[0]
                    hs, n = list(pre[0]), pre[1]
            if L == 12:
                h12_in = list(hs)
            if (L, p) in TAP_LP:
                for t in range(R):
                    taps[t].append(ttnn.clone(hs[t]))
            msgs = all_msgs[L - LOOP_START] if (LOOP_START <= L < LOOP_END and p > 1) else None
            hs, n = self._app(L, p, hs, n, msgs)
            if self.collect is not None:
                self.collect.append([ttnn.clone(h) for h in hs])
        # final attn-res: one shared [1, 1, R, C] output (row r = token r); the final norm is row-wise
        x_all = m._rms(F.per_row(dev, R, lambda r: m.final_res(None, hs[r], n + 1)[0])[0], m.final_norm)
        fin = [ttnn.slice(x_all, [0, 0, t, 0], [1, 1, t + 1, HIDDEN]) for t in range(R)]
        logits = m.unembed(x_all)  # [1, 1, R, V] fp32 (bank linear, every row)
        logits = ttnn.add(logits, m.vocab_mask)
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        am = ttnn.argmax(rm, dim=-1, keepdim=True)  # [1, 1, R, 1]: every row in one op
        tokens = [am]
        # drafter context for every row in one pass (rejected rows land at positions the next draft overwrites first)
        feats = [ttnn.pad(ttnn.concat(taps[t] + [fin[t]], dim=2), [(0, 0), (0, 0), (0, 28), (0, 0)], 0.0)
                 for t in range(R)]  # fmt: skip
        self.d.commit_rows(feats, self.positions[0].pos, self.ids)
        self.out = {"token": tokens, "logits": logits}
        return self.out

    def commit(self, n):
        """After the accept: tokens 0 .. n-1 become the model state (rings, conv rings, window states, drafter f_rnn),
        as n plain steps would have left it."""
        m, R = self.m, self.R
        assert 1 <= n <= R
        pairs = []
        for L, p in APPS:
            c = cache_index(L, p)
            lay = m.layers[L]
            attn_st, conv_a, conv_m = m.states[c]
            lay.attn.commit_rows(attn_st, self.rows[c], self.positions[0], n)
            ttnn.copy(self.hists[c][n], attn_st.x_hist)
            for conv, ring in ((lay.conv_attn, conv_a), (lay.conv_mlp, conv_m)):
                pairs += [
                    (conv.pending_slot(ring, r), ring[(self.phase + r) % FusedShortConv.PHASES]) for r in range(n)
                ]
        conv_commit(self.device, pairs, F.pages(HIDDEN))
        ttnn.copy(ttnn.slice(self.d.f_rnn_rows, [0, 0, n - 1, 0], [1, 1, n, HIDDEN]), self.d.f_rnn)
