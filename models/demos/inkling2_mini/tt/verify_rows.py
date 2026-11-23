# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""R-row decode step for speculative verify: model.step / model._app's dataflow for R tokens at pos .. pos + R - 1,
each row bit-equal to the 1-row step at its position (fused/common.py per_row and the R-row modules).

    rs = RowsStep(model, R)
    out = rs.step(tokens, pos, attention, experts)   # [(logits, token)] per row
    rs.commit(n_accepted)                             # conv rings: the accepted rows' inputs

Attention and the routed experts are passed in (S6 / M5 / G2 own their R-row versions):
    attention(L, p, xq, xk, xv) -> a: [1, 1, R, C] fp32 attention output rows (the short conv's input; a dense twin
        of R vectors if it has one). xq / xk / xv: per row, its q / k / v input rows (bf16 [1, 1, 1, C]).
    experts(L, x, weights, ids) -> y: [1, 1, R, C] bf16 L1 (with its dense twin, as moe_bank's R-row step);
        x the R-row MoE input (bf16 L1), weights / ids the R-row router outputs.
State: the step writes no conv ring (rows read their earlier rows' inputs directly); commit(n) puts rows 0..n-1 into
the rings. The attention's own state (KV ring, window histories) is the attention's to keep per accepted row.
Every output is a module buffer, valid until the next step. First call eager (it allocates buffers).
"""

import torch

import ttnn
from models.demos.inkling2_mini.tt import model as MD
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv, commit
from models.demos.inkling2_mini.tt.residual import cache_index


class RowsStep:
    def __init__(self, model, n_rows):
        self.m, self.R = model, n_rows
        self.qkv = {L: lay.qkv_gate.with_rows(3 * n_rows) for L, lay in enumerate(model.layers) if lay.qkv_gate}
        self.mlp = {L: lay.mlp_gate.with_rows(n_rows) for L, lay in enumerate(model.layers) if lay.mlp_gate}
        self.tokens = ttnn.from_torch(
            torch.zeros(1, n_rows, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=model.device,
        )
        self.phases = [0] * n_rows

    def _rows(self, fn):
        return F.per_row(self.m.device, self.R, fn)

    def set_inputs(self, tokens, pos):
        """Host writes before a step: the R tokens, positions pos .. pos + R - 1."""
        assert len(tokens) == self.R
        self.pos = pos
        self.phases = [(pos + r) % FusedShortConv.PHASES for r in range(self.R)]
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([tokens], dtype=torch.int32), dtype=ttnn.uint32), self.tokens
        )

    def _pending(self, conv, ring, r):
        return conv.pending_slot(ring, r)

    def app(self, L, p, h, n_sealed, msgs, attention, experts):
        """model._app for R rows. h: R-row layer input (one [1, 1, R, C] tensor or Rows of 1-row tensors)."""
        m, R, lay = self.m, self.R, self.m.layers[L]
        _, conv_a, conv_m = m.states[cache_index(L, p)]
        row = (lambda r: (h, r)) if not isinstance(h, F.Rows) else (lambda r: (h[r], 0))
        if lay.attn_res is None:
            xs = self._rows(lambda r: lay.norm(row(r)[0], num_rows=1, first_row=row(r)[1]))
        else:
            rep = 3 if (msgs is not None and not lay.is_global) else None
            x = self._rows(
                lambda r: lay.attn_res(None, h[r], n_sealed + 1, split=False, repeat_rows=rep, row_out=False)
            )[0]
            if msgs is not None:
                x = self.qkv[L](x, msgs[0])
            n_in = x.shape[-2] // R
            xs = self._rows(lambda r: lay.norm(x, num_rows=n_in, first_row=n_in * r))
        qkv = [t if len(t) == 3 else t * 3 for t in xs]
        a = attention(L, p, *[[t[i] for t in qkv] for i in range(3)])
        resid = self._rows(
            lambda r: lay.conv_attn.conv_add(
                a, conv_a, self.phases[r], None if lay.boundary else row(r)[0], ttnn.float32, row_out=False,
                x_row=r, slot=self._pending(lay.conv_attn, conv_a, r),
            )  # fmt: skip
        )
        if lay.boundary:
            self._rows(lambda r: m.frows.seal(None, row(r)[0], n_sealed, h_row=row(r)[1]))
            n_sealed += 1
        if msgs is None:
            xm = self._rows(lambda r: lay.mlp_res(None, resid[r], n_sealed + 1, norm=lay.moe_norm))[0]
        else:
            mm = self._rows(lambda r: lay.mlp_res(None, resid[r], n_sealed + 1, row_out=False)[0])[0]
            g = self.mlp[L](mm, msgs[1])
            xm = self._rows(lambda r: lay.moe_norm(g, num_rows=1, first_row=r))[0][0]
        weights, ids = lay.moe.router(xm)
        y = experts(L, xm, weights, ids)
        out = self._rows(
            lambda r: lay.conv_mlp.conv_add(
                y, conv_m, self.phases[r], resid[r], ttnn.bfloat16, row_out=False, x_row=r,
                slot=self._pending(lay.conv_mlp, conv_m, r),
            )  # fmt: skip
        )
        return out, n_sealed

    def step(self, attention, experts):
        """One R-row decode step on (set_inputs). Returns [(logits, token)] per row (module buffers)."""
        m, R = self.m, self.R
        e = ttnn.embedding(self.tokens, m.embed, layout=ttnn.TILE_LAYOUT)
        h = m._rms(ttnn.reshape(e, [1, 1, R, MD.HIDDEN]), m.embed_norm)
        n, pre, h12_in, all_msgs = 0, None, None, None
        for L, p in MD.APPS:
            if L == MD.LOOP_START:
                if p == 1:
                    pre = (h, n)
                else:
                    keys = (h12_in, h)
                    all_msgs = self._rows(lambda r: m.msg(keys[0][r], keys[1][r]))[0]
                    h, n = pre
            if L == 12:
                h12_in = h
            msgs = all_msgs[L - MD.LOOP_START] if (MD.LOOP_START <= L < MD.LOOP_END and p > 1) else None
            h, n = self.app(L, p, h, n, msgs, attention, experts)
        x = self._rows(lambda r: m.final_res(None, h[r], n + 1, split=False))[0]
        x = m._rms(x, m.final_norm)
        return m.head_tail.rows(x)

    def commit(self, n_accepted):
        """After a verify step: rows 0..n_accepted-1's conv inputs into their ring slots (all 96 convs, one program),
        as n_accepted 1-row steps leave the rings."""
        m, pairs = self.m, []
        for L, p in MD.APPS:
            lay = m.layers[L]
            _, conv_a, conv_m = m.states[cache_index(L, p)]
            for conv, ring in ((lay.conv_attn, conv_a), (lay.conv_mlp, conv_m)):
                pairs += [(conv.pending_slot(ring, r), ring[self.phases[r]]) for r in range(n_accepted)]
        commit(m.device, pairs, F.pages(MD.HIDDEN))
