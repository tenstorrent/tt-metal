# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""R-token target step for greedy speculative decoding (verify pass), exact per row.

`VerifyStep(model, R)` runs R tokens at positions p0 .. p0 + R - 1 through the 48 applications in one pass; row t
must come out bit-equal to the plain 1-row step of token t at p0 + t. Every piece runs the 1-row program's math
per row:
- residual path (attn-res, seal, row norms, loop messages, short convs): F.per_row (fused/common.py), one copy of
  the 1-row program per row on its own cores, launched as one program;
- message gates and router: their R-row forms (fused/gate.py num_tokens / num_rows, fused/router.py rows);
- attention: Inkling2AttentionFast.decode_rows / commit_rows (one window + projection, prep, near and combine per
  pass; sdpa_decode pending rows; far shared by the R tokens when their bounds share a K chunk, else per token);
- experts: the multi-row bank expert step (union of the rows' experts read once, per-row chains);
- head: the fused logits tail's R-row form (FusedLogitsTail.rows: one unembed, one softcap / argmax program),
  each row's logits and token bit-equal to the plain step's.
Nothing the next pass reads is written for a token that may be rejected: rings, conv rings and window states change
only in `commit(n)` (the first n rows, n = accepted + 1). The drafter context of all R rows is committed inside the
pass: rejected rows sit at positions >= the next anchor, which the next draft overwrites first.

Off by default: only `SpecGenerator.generate_rows` (tt/spec_generator.py) uses it. R <= 3 (pending-row sdpa limit).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.inkling2_mini.convert import cache as wcache
from models.demos.inkling2_mini.tt.attention_fast import DecodePositionsFast
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv
from models.demos.inkling2_mini.tt.fused.conv import commit as conv_commit
from models.demos.inkling2_mini.tt.fused.gate import FusedMessageGate, FusedQKVGate
from models.demos.inkling2_mini.tt.model import APPS, HIDDEN, LOOP_END, LOOP_START, cache_index

R_MAX = 3


def _gate_weights(root, L):
    """Host tensors of layer L's message gates only (the rest of the layer is not read)."""
    root = wcache._root(root)
    out = {}
    for e in wcache.manifest(root)["tensors"]:
        if e["layer"] == L and e["name"].startswith("msg_gate."):
            out[e["name"]] = ttnn.load_tensor(root / e["file"])
    return out


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
        for L, p in APPS:
            c = cache_index(L, p)
            self.rows[c] = m.layers[L].attn.rows_out(R, m.states[c][0])
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
        xs = [list(n) if triple else n[0] for n in normed]
        # decode staging (mk/overlap.py) swaps w_all.bank for a proxy that reads the app's weights from L1 where host
        # programs filled them; no verify program hosts those fills, so the pass projects through the raw bank
        proxy = lay.attn.w_all.bank
        lay.attn.w_all.bank = proxy.st.bl if hasattr(proxy, "st") else proxy
        try:
            act = lay.attn.decode_rows(xs, attn_st, self.positions, self.rows[c], p0=self.p0)
        finally:
            lay.attn.w_all.bank = proxy
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
        """One verify pass on (tokens, positions). Returns {"token": [device uint32, R values] (value t = token t's
        argmax), "logits": per-row softcapped logits (fused head)}."""
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
        if m.head_tail is not None:  # the plain step's fused tail, all rows in one program: logits / token bit-equal
            lt = m.head_tail.rows(x_all)
            logits = [lg for lg, _ in lt]
            tokens = [ttnn.concat([tk for _, tk in lt], dim=-1)]  # [1, 1, 1, R]: one host read
        else:
            logits = ttnn.add(m.unembed(x_all), m.vocab_mask)  # [1, 1, R, V] fp32
            # shortcut: argmax on raw + mask, not on the softcapped row (ties after softcap may differ); only
            # reached without the fused head
            tokens = [ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)]
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
            lay.attn.commit_rows(attn_st, self.rows[c], self.positions[0], n)  # rings + window state
            for conv, ring in ((lay.conv_attn, conv_a), (lay.conv_mlp, conv_m)):
                pairs += [
                    (conv.pending_slot(ring, r), ring[(self.phase + r) % FusedShortConv.PHASES]) for r in range(n)
                ]
        conv_commit(self.device, pairs, F.pages(HIDDEN))
        ttnn.copy(ttnn.slice(self.d.f_rnn_rows, [0, 0, n - 1, 0], [1, 1, n, HIDDEN]), self.d.f_rnn)
