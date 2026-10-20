# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of tt/spec_chain.py (work/notes/63 S3a): chain_reference equals the composition of today's host builders;
fused_fit_mode_range equals brute force. No device. Run: pytest -o addopts="" --import-mode=importlib -q
models/demos/blackhole/qwen36/tests/test_spec_chain_host.py"""

from types import SimpleNamespace

import torch

from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_words
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_chain import (
    chain_reference,
    fused_fit_mode_range,
    pack_ovr_host,
    res_c_reference,
)
from models.demos.blackhole.qwen36.tt.spec_engine import sanitize_verify_block
from models.demos.blackhole.qwen36.tt.spec_fused import fused_accept_aux_host, fused_fit_mode

K, T, SCRATCH = 3, 4, 5000


def _case(g, B, Nv, parity):
    ri = lambda lo, hi, *s: torch.randint(lo, hi, s, generator=g)
    cap = int(ri(40, 200, 1)) if torch.rand(1, generator=g) < 0.7 else 100000
    rope_rows = int(ri(30, 90, 1))  # small: exercises the clamp
    delta = int(ri(-40, 40, 1))
    counts = ri(1, T + 1, B)
    committed = ri(0, 248320, B, T)
    committed[torch.arange(T)[None, :] >= counts[:, None]] = -1
    p0_prev = ri(1, 70, B)
    drafts = ri(0, 248320, B, K)
    mask = (torch.rand(B, generator=g) < 0.3).long()
    act = (torch.rand(B, generator=g) > 0.2).long()
    skip = (torch.rand(B, generator=g) < 0.25).bool()
    nv = torch.where(act.bool(), ri(0, K + 1, B), torch.zeros(B, dtype=torch.long))
    p0_host = torch.where(act.bool(), ri(1, 70, B), torch.zeros(B, dtype=torch.long))
    mi_host = torch.where(act.bool(), ri(0, T, B), torch.zeros(B, dtype=torch.long))
    host_tok = torch.where(act.bool()[:, None], ri(0, 248320, B, T), torch.zeros(B, T, dtype=torch.long))
    for u in range(B):  # sanitized pad rule for the host row
        host_tok[u, int(nv[u]) + 1 :] = host_tok[u, 0] if act[u] else 0
    # decline known at enqueue time: range mode for device rows, exact mode for host rows
    rmode = fused_fit_mode_range(p0_prev, cap, act, K)
    hmode = fused_fit_mode(p0_host, cap, act, K)
    skip = skip & act.bool()
    dec, keep = [], []
    for u in range(B):
        mode = hmode[u] if mask[u] else rmode[u]
        if mode == "mixed" and not skip[u]:
            skip[u] = bool(act[u])  # not chainable as a drafting row; the planner would skip it
        dec.append(int((not act[u]) or skip[u] or mode == "fits_none"))
        keep.append(int(skip[u] and act[u]))
    return dict(
        cap=cap,
        rope_rows=rope_rows,
        delta=delta,
        counts=counts,
        committed=committed,
        p0_prev=p0_prev,
        drafts=drafts,
        mask=mask,
        act=act,
        nv=nv,
        p0_host=p0_host,
        mi_host=mi_host,
        host_tok=host_tok,
        dec=torch.tensor(dec),
        keep=torch.tensor(keep),
        skip=skip,
    )


def _fake_model(B, Nv, rope_rows, delta, parity):
    return SimpleNamespace(
        rope=SimpleNamespace(rope_delta=delta, max_seq_len=rope_rows),
        _vfy_fused=True,
        _vfy_gdn=[SimpleNamespace(_spec_win_par=parity) for _ in range(3)],
        _vfy_B=B,
        _vfy_Nv=Nv,
        _vfy_T=T,
    )


def test_chain_reference_equals_host_builders():
    g = torch.Generator().manual_seed(1234)
    n_checked = 0
    for it in range(400):
        B = int(torch.randint(1, 9, (1,), generator=g))
        Nv, parity = int(torch.randint(1, 5, (1,), generator=g)) * 2, it % 2
        c = _case(g, B, Nv, parity)
        R = B * T
        # previous step's result and aux (consistent p0)
        prev_aux = torch.zeros(6, 32, dtype=torch.int32)
        prev_aux[4, :B] = c["p0_prev"].to(torch.int32)
        res = res_c_reference(c["committed"], c["counts"], c["p0_prev"], torch.zeros(B))
        ovr = pack_ovr_host(
            c["nv"],
            c["act"],
            c["dec"],
            [int(k) for k in c["keep"]],
            c["p0_host"],
            c["mask"],
            c["host_tok"],
            c["mi_host"],
            parity,
            K=K,
            scratch_block=SCRATCH,
            rope_delta=c["delta"],
            rope_rows=c["rope_rows"],
        )
        ctrl_words = spec_ctrl_words(B, Nv)
        tok_buf, vfy_aux, nxt = chain_reference(
            res,
            c["drafts"],
            prev_aux,
            ovr,
            B=B,
            K=K,
            Nv=Nv,
            rope_delta=c["delta"],
            rope_rows=c["rope_rows"],
            ctrl_words=ctrl_words,
        )
        # ---- host composition: what the plugin + engine build today for the next verify ----
        m = c["counts"] - 1
        tok0 = c["committed"][torch.arange(B), m]
        tokens = torch.zeros(B, T, dtype=torch.long)
        start_pos = torch.full((B, T), -1, dtype=torch.long)
        mi = [0] * B
        for u in range(B):
            if not c["act"][u]:
                continue  # inactive row: start_pos < 0, n = 0
            if c["mask"][u]:
                tokens[u] = c["host_tok"][u]
                p0 = int(c["p0_host"][u])
                mi[u] = int(c["mi_host"][u])
            else:
                tokens[u] = torch.cat([tok0[u : u + 1], c["drafts"][u]])
                p0 = int(c["p0_prev"][u]) + int(m[u]) + 1
                mi[u] = int(m[u])
            start_pos[u] = p0 + torch.arange(T)
        clean, start0, active = sanitize_verify_block(tokens, start_pos, c["nv"], K)
        fake = _fake_model(B, Nv, c["rope_rows"], c["delta"], parity)
        rp = Qwen36Model._vfy_row_positions(fake, start0)
        host_aux = Qwen36Model._vfy_aux_host(fake, rp, mi)
        host_fa = fused_accept_aux_host(
            c["nv"], start0, active, c["skip"], c["cap"], K, SCRATCH, c["delta"], c["rope_rows"]
        )
        assert torch.equal(tok_buf.to(torch.int64), clean.reshape(-1)), (it, "tokens")
        assert torch.equal(vfy_aux, host_aux), (it, "vfy_aux")
        assert torch.equal(nxt, host_fa), (it, "fused aux")
        n_checked += 1
    assert n_checked == 400


def test_chain_reference_pad_rows_zero_and_declined_tok0():
    # a declined unmasked row (nv'=0) keeps tok0 in every column; an inactive unmasked row is all zero
    B, Nv = 2, 2
    committed = torch.tensor([[11, 12, -1, -1], [21, -1, -1, -1]])
    res = res_c_reference(committed, [2, 1], [10, 7], [0, 0])
    ovr = pack_ovr_host(
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        None,
        [0, 0],
        1,
        K=K,
        scratch_block=SCRATCH,
        rope_delta=-3,
        rope_rows=64,
    )
    drafts = torch.tensor([[7, 8, 9], [4, 5, 6]])
    tb, aux, nxt = chain_reference(
        res, drafts, None, ovr, B=B, K=K, Nv=Nv, rope_delta=-3, rope_rows=64, ctrl_words=spec_ctrl_words(B, Nv)
    )
    assert tb.tolist() == [12, 12, 12, 12, 0, 0, 0, 0]
    assert aux[0, :8].tolist() == [12, 13, 14, 15, 0, 1, 2, 3]
    assert nxt[4, :2].tolist() == [12, 0]


def test_fused_fit_mode_range_vs_brute_force():
    for K_ in (1, 2, 3, 4):
        for cap in range(1, 40):
            for p0 in range(0, 40):
                for act in (0, 1):
                    got = fused_fit_mode_range([p0], cap, [act], K_)[0]
                    if not act:
                        assert got == "fits_none"
                        continue
                    fits = [p0 + m + 1 + mm + 1 + K_ < cap for m in range(K_ + 1) for mm in range(K_ + 1)]
                    want = "fits_all" if all(fits) else "fits_none" if not any(fits) else "mixed"
                    assert got == want, (K_, cap, p0)
                    # consistency with the existing single-step rule at every next start0
                    modes = {fused_fit_mode([p0 + m + 1], cap, [1], K_)[0] for m in range(K_ + 1)}
                    if got != "mixed":
                        assert modes == {got}
