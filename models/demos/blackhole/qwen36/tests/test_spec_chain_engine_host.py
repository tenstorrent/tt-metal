# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of chain mode in the sync-serving spec engine (QWEN36_SPEC_CHAIN, stage S3c, work/notes/63 section 1): a fused step that
follows a fused step on the same live rows lets the device build the verify token / aux buffers (the chain trace), over fake
persistent buffers and a call log (the approach of test_spec_fused_engine_host.py). The fake chain trace is the host reference
(spec_chain.chain_reference) run on what the staged buffers hold. No device.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_chain_engine_host.py
"""

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import (
    DRAFT,
    FA,
    FS,
    TOPK,
    VFY,
    K,
    T,
    _block,
    _ids_for,
    _plugin_propose_args,
    _pt,
    _SampledRig,
    _sampling,
    _smooth_logits,
)
from models.demos.blackhole.qwen36.tests.test_spec_hostset_host import _hostset_vfy_model
from models.demos.blackhole.qwen36.tests.test_spec_specstep_host import _Buf, _fake_ttnn
from models.demos.blackhole.qwen36.tt import spec_engine as se_mod
from models.demos.blackhole.qwen36.tt.spec_chain import (
    OVR_MASK,
    OVR_WORDS,
    ChainDevice,
    chain_reference,
    res_c_reference,
)
from models.demos.blackhole.qwen36.tt.tp_common import (
    spec_chain_audit_enabled,
    spec_chain_audit_every,
    spec_chain_enabled,
)

CHG, CHS = 21, 22  # chain trace ids (greedy / sampled variant)


class _ChainRig(_SampledRig):
    """_SampledRig + two fake ChainDevice shells (S1 / S2 result buffer) and the fake chain trace. ``chain=False`` leaves the engine
    exactly as without QWEN36_SPEC_CHAIN (no ChainDevice, no trace id)."""

    def __init__(self, monkeypatch, B, chain=True):
        super().__init__(monkeypatch, B)
        self.corrupt_ch = None  # "token" | "aux": the fake chain trace writes a wrong word
        self.ch_runs = []
        if chain:
            e = self.e
            e._ch_g, e._ch_s = self._shell(self.fa.result_out), self._shell(self.fs.result_out)
            e._ch_trace_g, e._ch_trace_s = CHG, CHS
            e.chain_breaks = {}

    def _shell(self, result_in):
        B, m = self.B, self.m
        ch = ChainDevice()
        ch.mesh, ch.B, ch.K, ch.T, ch.R, ch.Nv = "mesh", B, K, T, B * T, m._vfy_Nv
        ch.W_o, ch.ctrl_words = 32, m._vfy_ctrl_words
        ch.ovr_in = _Buf("ovr_g" if result_in is self.fa.result_out else "ovr_s")
        ch.result_in, ch.aux_prev_in, ch.tok_out, ch.vaux_out = (
            result_in,
            self.fa.aux_in,
            m._vfy_token_buf,
            m._vfy_aux_buf,
        )
        return ch

    def _exec(self, dev, tid, cq_id=0, blocking=False):
        if tid not in (CHG, CHS):
            return super()._exec(dev, tid, cq_id, blocking)
        self.log.append(("trace", tid))
        self.ch_runs.append(tid)
        e, m, B = self.e, self.m, self.B
        ch = e._ch_g if tid == CHG else e._ch_s
        res, aux = ch.result_in.data, ch.aux_prev_in.data
        res_c = res_c_reference(res[0, : self.R].reshape(B, T), res[1, :B], aux[4, :B], torch.zeros(B))
        drafts = torch.stack([o.data[:B] for o in e._dr_outs], dim=1)
        tok, vaux, nxt = chain_reference(
            res_c,
            drafts,
            aux,
            ch.ovr_in.data,
            B=B,
            K=K,
            Nv=m._vfy_Nv,
            rope_delta=m.rope.rope_delta,
            rope_rows=m.rope.max_seq_len,
            ctrl_words=m._vfy_ctrl_words,
        )
        if self.corrupt_ch == "token":
            tok[0] ^= 1
        if self.corrupt_ch == "aux":
            vaux[0, 1] += 1
        m._vfy_token_buf.data, m._vfy_aux_buf.data, self.fa.aux_in.data = tok.reshape(1, -1), vaux, nxt


class _Flow:
    """The plugin side of a sync serving loop over a rig: it keeps the verify block, walks nothing itself (the engine's result is the
    commit), proposes through the engine and builds the next block from the commit + the DraftResult."""

    def __init__(self, r, B, nv, base=50, stride=10):
        self.r, self.B = r, B
        self.tok, self.pos, self.nvd = _block(B, nv, base=base, stride=stride)
        self.acc, self.pt, self.res, self.dr = torch.ones(B, dtype=torch.int32), _pt(B), None, None

    def step(self, accept=None, sampling=None, logits=None, pt=None, spec_mode="fused_sample"):
        r = self.r
        if logits is not None:
            r.set_logits(logits)
        elif accept is not None:
            r.set_ids(_ids_for(self.tok, self.nvd, accept))
        self.res = r.step(
            self.tok, self.pos, self.nvd, self.acc, self.pt if pt is None else pt, sampling, spec_mode=spec_mode
        )
        return self.res

    def propose(self, mutate=None):
        cp, cpos, cc = _plugin_propose_args(self.res, self.tok, self.pos)
        if mutate is not None:
            mutate(cp, cpos, cc)
        self.dr = self.r.e.propose_draft_tokens(K, cp, cpos, cc)
        return self.dr

    def next_block(self, finish=(), leave=()):
        """The block the plugin builds after the commit: tok0 = the bonus token, the drafts, positions p0 + count + j."""
        B, res, dr = self.B, self.res, self.dr
        tok, pos, nvd = self.tok.clone(), torch.full((B, T), -1, dtype=torch.int32), torch.zeros(B, dtype=torch.int32)
        for u in range(B):
            if int(self.pos[u, 0]) < 0 or u in finish or u in leave:
                continue
            c = int(res.accepted_counts[u])
            n = int(dr.num_valid[u])
            tok[u, 0] = res.accepted_token_ids[u, c - 1]
            tok[u, 1 : 1 + n] = dr.draft_token_ids[u, :n]
            p = int(self.pos[u, 0]) + c
            pos[u, : n + 1] = p + torch.arange(n + 1, dtype=torch.int32)
            nvd[u] = n
        for u in finish:
            self.r.e.release(u)
        self.tok, self.pos, self.nvd, self.acc = tok, pos, nvd, res.accepted_counts.clone()


def _run(monkeypatch, B, chain, steps, nv=(3, 2, 1), **flow_kw):
    """``steps``: list of accept vectors; one propose + next block between them. Returns (rig, flow, [result per step], [log per step])."""
    r = _ChainRig(monkeypatch, B, chain=chain)
    f = _Flow(r, B, list(nv)[:B])
    out, logs = [], []
    for i, acc in enumerate(steps):
        f.step(acc)
        out.append(f.res)
        logs.append(r.log[:])
        f.propose()
        f.next_block(**flow_kw)
    return r, f, out, logs


def _strip_kv(log):
    """The verify's own page-table copies (they follow the KV write table, which moves with the positions) sit between the chain trace
    and the verify trace; they are checked to, then dropped."""
    out, i_ch = [], None
    for i, x in enumerate(log):
        if x[0] == "h2d" and x[1] in ("kvpt", "kvpt_users", "kvpt1"):
            assert (
                i_ch is not None and ("trace", VFY) not in out
            ), "kv table copies belong between the chain trace and the verify"
            continue
        if x[0] == "trace" and x[1] in (CHG, CHS):
            i_ch = i
        out.append(x)
    return out


def _same_result(a, b):
    return (
        torch.equal(a.accepted_token_ids, b.accepted_token_ids)
        and torch.equal(a.accepted_counts, b.accepted_counts)
        and torch.equal(a.argmax_ids, b.argmax_ids)
    )


# --------------------------------------------------------------------------- #
# (i) flags
# --------------------------------------------------------------------------- #
def test_flags_defaults(monkeypatch):
    for n in ("QWEN36_SPEC_CHAIN", "QWEN36_SPEC_CHAIN_AUDIT", "QWEN36_SPEC_CHAIN_AUDIT_EVERY"):
        monkeypatch.delenv(n, raising=False)
    assert not spec_chain_enabled() and spec_chain_audit_enabled() and spec_chain_audit_every() == 1
    monkeypatch.setenv("QWEN36_SPEC_CHAIN", "1")
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT", "0")
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT_EVERY", "4")
    assert spec_chain_enabled() and not spec_chain_audit_enabled() and spec_chain_audit_every() == 4
    monkeypatch.setenv("QWEN36_SPEC_CHAIN", "0")
    assert not spec_chain_enabled()


# --------------------------------------------------------------------------- #
# (ii) the chain step's call order
# --------------------------------------------------------------------------- #
def test_chain_step_call_order_greedy(monkeypatch):
    B = 3
    r = _ChainRig(monkeypatch, B)
    f = _Flow(r, B, [3, 2, 1])
    f.step([3, 1, 0])
    log1 = r.log[:]
    assert ("trace", CHG) not in log1 and "ovr_g" not in r.names("h2d") and r.e.chain_breaks == {"first": 1}
    f.propose()
    f.next_block()
    tok_w, aux_w = r.m._vfy_token_buf.writes, r.m._vfy_aux_buf.writes
    f.step([1, 2, 0])
    log = _strip_kv(r.log)
    assert [x[:2] for x in log] == [
        ("h2d", "ovr_g"),  # the overlay replaces fa_aux; same table, same declined rows: no fa / mtp table copies
        ("trace", CHG),
        ("trace", VFY),  # no token / aux h2d between
        ("trace", FA),
        ("trace", DRAFT),
        *[("from_device", f"out{k}") for k in range(K)],
        ("sync",),
        ("to_torch", "ids"),
        ("to_torch", "fa_result"),
        *[("to_torch", f"out{k}") for k in range(K)],
        ("to_torch", "token"),  # the audit (default on), after the sync
        ("to_torch", "aux"),
        ("to_torch", "fa_aux"),
    ]
    assert [x[0] for x in r.log].count("sync") == 1
    assert (r.m._vfy_token_buf.writes, r.m._vfy_aux_buf.writes) == (tok_w, aux_w), "the host copied no token / aux word"
    assert r.m._vfy_aux_staged is None
    e = r.e
    assert e.chain_steps == 1 and e.chain_audits == 1 and e.chain_breaks == {"first": 1} and e.fused_steps == 2
    # a changed table: the table copies stay (fa rows, mtp, then the verify's own after the chain trace)
    f.propose()
    f.next_block()
    f.step([0, 0, 0], pt=_pt(B, 7))
    assert [x[:2] for x in r.log][:8] == [
        ("h2d", "ovr_g"),
        ("h2d", "fa_ptrows"),
        ("h2d", "mtp"),
        ("trace", CHG),
        ("h2d", "kvpt"),
        ("h2d", "kvpt_users"),
        ("h2d", "kvpt1"),
        ("trace", VFY),
    ]
    assert not {"token", "aux"} & set(r.names("h2d"))
    assert e.chain_steps == 2


def test_chain_step_call_order_sampled_and_variant_choice(monkeypatch):
    B = 3
    r = _ChainRig(monkeypatch, B)
    e, f = r.e, _Flow(r, B, [3, 2, 1])
    lg = _smooth_logits(B * T, 1)
    f.step(logits=lg, sampling=_sampling(B, f.nvd, [0.8, 0.0, 1.2]))  # sampled step 1 (first: host staged)
    assert ("trace", FS) in r.log and ("trace", CHG) not in r.log and ("trace", CHS) not in r.log
    f.propose()
    f.next_block()
    f.step(logits=_smooth_logits(B * T, 2), sampling=_sampling(B, f.nvd, [0.9, 0.9, 0.0]))  # sampled after sampled: CHS
    assert [x[:2] for x in _strip_kv(r.log)][:7] == [
        ("h2d", "ovr_s"),
        ("h2d", "fs_samp"),
        ("trace", CHS),
        ("trace", VFY),
        ("trace", TOPK),
        ("trace", FS),
        ("trace", DRAFT),
    ]
    assert r.ch_runs == [CHS] and e._ch_s.ovr_in.writes == 1 and e._ch_g.ovr_in.writes == 0
    f.propose()
    f.next_block()
    f.step([1, 1, 1])  # greedy after sampled: the variant of the step that RAN is CHS, the step itself runs FA
    assert r.ch_runs == [CHS, CHS] and ("trace", FA) in r.log and ("trace", FS) not in r.log
    assert [x[:2] for x in _strip_kv(r.log)][:3] == [("h2d", "ovr_s"), ("trace", CHS), ("trace", VFY)]
    f.propose()
    f.next_block()
    f.step(logits=_smooth_logits(B * T, 3), sampling=_sampling(B, f.nvd, [0.7, 0.7, 0.7]))  # sampled after greedy: CHG
    assert r.ch_runs == [CHS, CHS, CHG] and [x[:2] for x in r.log][:2] == [("h2d", "ovr_g"), ("h2d", "fs_samp")]
    assert e.chain_steps == 3 and e.chain_audits == 3 and e.chain_breaks == {"first": 1}


@pytest.mark.parametrize("B,nv", [(3, (3, 2, 1)), (4, (3, 3, 3, 3)), (2, (0, 3))])
def test_chain_results_equal_the_host_staged_path(monkeypatch, B, nv):
    """Over many steps with varied accept counts the chained engine returns what the unchained one does, step for step, and
    leaves the same words in the device buffers."""
    g = torch.Generator().manual_seed(B)
    steps = [torch.randint(0, 4, (B,), generator=g).tolist() for _ in range(9)]
    rc, fc, out_c, _ = _run(monkeypatch, B, True, steps, nv)
    rn, fn, out_n, _ = _run(monkeypatch, B, False, steps, nv)
    for i, (a, b) in enumerate(zip(out_c, out_n)):
        assert _same_result(a, b), f"step {i}"
    assert torch.equal(rc.m._vfy_token_buf.data.reshape(-1), rn.m._vfy_token_buf.data.reshape(-1).to(torch.int32))
    assert (rc.m._vfy_aux_buf.data.to(torch.int64) & 0xFFFFFFFF).equal(
        rn.m._vfy_aux_buf.data.to(torch.int64) & 0xFFFFFFFF
    )
    assert torch.equal(rc.fa.aux_in.data.to(torch.int32), rn.fa.aux_in.data.to(torch.int32))
    e = rc.e
    assert e.chain_steps == len(steps) - 1 and e.chain_audits == e.chain_steps and e.chain_breaks == {"first": 1}
    assert rc.m._vfy_gdn[0]._spec_win_par == rn.m._vfy_gdn[0]._spec_win_par == len(steps) % 2


def test_parity_word_is_the_one_the_host_would_stage_and_flips_once_per_replay(monkeypatch):
    B = 3
    r = _ChainRig(monkeypatch, B)
    f = _Flow(r, B, [3, 2, 1])
    pars = []
    for i in range(6):
        before = r.m._vfy_gdn[0]._spec_win_par
        f.step([2, 1, 3])
        pars.append((before, r.m._vfy_gdn[0]._spec_win_par))
        assert r.m._vfy_aux_buf.data[2, 0].item() == before  # the ctrl page's word 0 = the parity the replay read
        if i:
            assert int(r.e._ch_g.ovr_in.data[OVR_WORDS, 0]) == before  # the overlay carried it
        assert all(dn._spec_win_par == r.m._vfy_gdn[0]._spec_win_par for dn in r.m._vfy_gdn)
        f.propose()
        f.next_block()
    assert pars == [(i % 2, (i + 1) % 2) for i in range(6)]


# --------------------------------------------------------------------------- #
# (iii) rows that finish; break reasons
# --------------------------------------------------------------------------- #
def test_a_finished_row_is_an_overlay_row_and_the_chain_continues(monkeypatch):
    B = 3
    steps = [[3, 1, 2], [1, 1, 1], [2, 2, 2]]
    res, rigs = [], []
    for chain in (True, False):
        r = _ChainRig(monkeypatch, B, chain=chain)
        f = _Flow(r, B, [3, 2, 1])
        out = []
        for i, acc in enumerate(steps):
            f.step(acc)
            out.append(f.res)
            f.propose()
            f.next_block(finish=(1,) if i == 0 else ())
        res.append(out)
        rigs.append(r)
    for a, b in zip(*res):
        assert _same_result(a, b)
    rc = rigs[0]
    assert rc.e.chain_steps == 2 and rc.e.chain_breaks == {"first": 1}, "finishing a row did not break the chain"
    ovr = rc.e._ch_g.ovr_in.data.to(torch.int64)
    assert ovr[OVR_MASK, :B].tolist() == [0, 1, 0]  # the last chain step: row 1 is a host (padding) row


def _engine_drafts(e, B):
    """What a plugin that never called propose would still find in the drafter outputs (no DraftResult exists then)."""
    d = type("D", (), {})()
    d.num_valid = torch.full((B,), K, dtype=torch.int32)
    d.draft_token_ids = torch.stack([o.data[:B] for o in e._dr_outs], 1).to(torch.int32)
    return d


def _drive(rig, B, reason):
    """Two steps (three for 'unpark' / 'not_captured' / 'non_fused_step') of a plugin flow in which something breaks the chain at the
    last step. Returns (call log of the last step, its result)."""
    e = rig.e
    nv, base, stride = (3, 2, 1), 50, 10
    if reason == "mixed_fit":  # prev start 503: range mode 'mixed' (capacity 512); the next start 504 fits exactly
        nv, base, stride = (0, 0, 0), 503, 0
    f = _Flow(rig, B, list(nv), base=base, stride=stride)
    if reason == "row_set":
        e._active.discard(2)
        f.pos[2], f.nvd[2] = -1, 0
    f.step([0, 0, 0] if reason == "mixed_fit" else [1, 2, 0])
    if (
        reason == "cache_miss"
    ):  # the plugin capped row 0's commit at one token: its propose does not match the engine's cache
        f.res.accepted_counts[0], f.res.accepted_token_ids[0, 1:] = 1, -1
    if reason == "no_propose":
        f.dr = _engine_drafts(e, B)
    else:
        f.propose()
        if reason == "propose":
            cp, cpos, cc = _plugin_propose_args(f.res, f.tok, f.pos)
            e.propose(cp, cpos, cc, f.pt)  # a direct propose replays the drafter
    f.next_block()
    if reason == "prefill":
        e._chain_drop("prefill")
    elif reason == "slot_change":
        e.host_slotmap, e._last_hidden = False, [None] * B
        e.remap_slots([1, 0, 2])
        f.tok, f.pos, f.nvd, f.acc, f.pt = (x[[1, 0, 2]].clone() for x in (f.tok, f.pos, f.nvd, f.acc, f.pt))
    elif reason in ("park", "unpark"):
        f.pos[2], f.nvd[2] = -1, 0
    elif reason == "row_set":
        e._active.add(2)
        f.tok[2, 0], f.pos[2, 0], f.nvd[2], f.acc[2] = 5, 90, 0, 1
    elif reason == "counts_differ":
        f.acc[0] = 1 if int(f.acc[0]) > 1 else 2
    elif reason == "position_differs":
        f.pos[1] += 1
    elif reason == "tokens_differ":
        f.tok[1, 1] += 3
    elif reason == "not_captured":
        e._fa_trace = None
    if reason == "non_fused_step":
        rig.set_ids(_ids_for(f.tok, f.nvd, [0, 0, 0]))
        f.step(spec_mode="argmax_ids")
    else:
        f.step([0, 1, 0])
    if reason in ("unpark", "not_captured"):
        if reason == "not_captured":
            e._fa_trace = FA
        f.propose()
        f.next_block()
        if reason == "unpark":  # row 2 returns with a fresh one-token block
            f.tok[2, 0], f.pos[2, 0], f.nvd[2], f.acc[2] = 7, 120, 0, 1
    if reason in ("unpark", "not_captured", "non_fused_step"):
        f.step([0, 1, 0])
    return rig.log[:], f.res


_BREAK_COUNT = {
    "unpark": {"first": 1, "park": 1, "unpark": 1},
    "not_captured": {"first": 1, "not_captured": 1},
    "non_fused_step": {"first": 1, "non_fused_step": 1},
}


@pytest.mark.parametrize(
    "reason",
    [
        "no_propose",
        "cache_miss",
        "propose",
        "prefill",
        "slot_change",
        "non_fused_step",
        "not_captured",
        "park",
        "unpark",
        "row_set",
        "counts_differ",
        "position_differs",
        "tokens_differ",
        "mixed_fit",
    ],
)
def test_every_break_reason_falls_back_to_the_host_staged_step(monkeypatch, reason):
    B = 3
    r = _ChainRig(monkeypatch, B)  # one rig at a time: each one patches the module-level ttnn fakes
    log_c, res_c = _drive(r, B, reason)
    rn = _ChainRig(monkeypatch, B, chain=False)
    rn.e._prestage_next_verify = (
        lambda *a, **k: None
    )  # chain mode turns the prestage off (it needs m); compare like with like
    log_n, res_n = _drive(rn, B, reason)
    e = r.e
    assert e.chain_steps == 0 and not r.ch_runs, f"{reason}: the chain must not run"
    want = _BREAK_COUNT.get(reason, {"first": 1, reason: 1})
    assert e.chain_breaks == want, (reason, e.chain_breaks)
    assert log_c == log_n, f"{reason}: the chain-armed engine did not run the unchained engine's call sequence"
    assert _same_result(res_c, res_n)
    assert not any(x[1] in ("ovr_g", "ovr_s") for x in log_c if x[0] == "h2d")


def test_chain_resumes_after_a_break(monkeypatch):
    B = 3
    r = _ChainRig(monkeypatch, B)
    f = _Flow(r, B, [3, 2, 1])
    f.step([1, 1, 1])
    f.propose()
    f.next_block()
    f.step([1, 1, 1])  # chain
    f.propose(mutate=lambda cp, cpos, cc: None)
    f.next_block()
    r.e._chain_drop("prefill")  # something (a prefill of another slot) ran between the steps
    f.step([1, 1, 1])  # break
    f.propose()
    f.next_block()
    f.step([1, 1, 1])  # the step before ran the device accept again: chain
    assert r.e.chain_steps == 2 and r.e.chain_breaks == {"first": 1, "prefill": 1}


# --------------------------------------------------------------------------- #
# (iv) audit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "what,match",
    [
        ("token", "verify token buffer differs at engine row 0 column 0"),
        ("aux", "verify aux row 0 \\(kv position\\) differs at word 1"),
    ],
)
def test_audit_raises_on_a_corrupted_device_buffer(monkeypatch, what, match, expect_error):
    B = 3
    r = _ChainRig(monkeypatch, B)
    f = _Flow(r, B, [3, 2, 1])
    f.step([1, 1, 1])
    f.propose()
    f.next_block()
    r.corrupt_ch = what
    with expect_error(RuntimeError, match):
        f.step([1, 1, 1])


def test_audit_off_and_every_n(monkeypatch):
    B = 2
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT", "0")
    r = _ChainRig(monkeypatch, B)
    f = _Flow(r, B, [3, 2])
    f.step([1, 1])
    f.propose()
    f.next_block()
    r.corrupt_ch = "token"
    f.step([1, 1])  # not audited: no raise, no extra reads
    assert r.e.chain_audits == 0 and "token" not in r.names("to_torch") and "fa_aux" not in r.names("to_torch")
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT", "1")
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT_EVERY", "3")
    r = _ChainRig(monkeypatch, B)
    f = _Flow(r, B, [3, 2])
    audited = []
    for i in range(8):
        f.step([1, 1])
        audited.append("token" in r.names("to_torch"))
        f.propose()
        f.next_block()
    assert audited == [False, True, False, False, True, False, False, True]  # chain steps 0, 3, 6
    assert r.e.chain_steps == 7 and r.e.chain_audits == 3


# --------------------------------------------------------------------------- #
# (v) prestage, logging, flag off
# --------------------------------------------------------------------------- #
def test_prestage_next_verify_is_disabled_in_chain_mode(monkeypatch):
    B = 3
    for chain, want in ((True, []), (False, ["aux"])):
        r = _ChainRig(monkeypatch, B, chain=chain)
        r.e._row_active = None
        del r.log[:]
        cpos = torch.arange(1, T + 1)[None, :].repeat(B, 1) + 10
        r.e._prestage_next_verify(cpos, [0, 1, 2])
        assert r.names("h2d") == want, chain


def test_periodic_log_line_carries_the_chain_counters_only_in_chain_mode(monkeypatch):
    from loguru import logger

    B = 2
    for chain in (True, False):
        r = _ChainRig(monkeypatch, B, chain=chain)
        r.e._fa_trace = None  # every step a cheap fallback; the line is written at step 256
        msgs = []
        sink = logger.add(lambda m: msgs.append(str(m)), level="INFO")
        try:
            f = _Flow(r, B, [1, 1])
            for _ in range(256):
                r.set_ids(_ids_for(f.tok, f.nvd, [0, 0]))
                r.e.decode_forward(
                    f.tok,
                    f.pos,
                    num_valid_drafts=f.nvd,
                    accepted_counts=f.acc,
                    page_table=f.pt,
                    spec_mode="fused_sample",
                )
        finally:
            logger.remove(sink)
        line = [m for m in msgs if "fused steps" in m][0]
        assert ("chain steps 0, breaks" in line and "audits 0" in line) == chain, line
        assert "\n" not in line.strip()


def _flag_off_scenario(monkeypatch):
    """A multi-step plugin flow on engines WITHOUT the chain objects (greedy, sampled, a cache miss, an argmax step, a padding row)."""
    logs = []
    B = 3
    r = _SampledRig(monkeypatch, B)
    f = _Flow(r, B, [3, 2, 1])
    for i in range(4):
        f.step([1 + i % 3, 1, 0])
        logs.append(r.log[:])
        f.propose()
        f.next_block()
        logs.append(r.log[:])
    lg = _smooth_logits(B * T, 4)
    f.step(logits=lg, sampling=_sampling(B, f.nvd, [0.8, 0.0, 1.2]))
    logs.append(r.log[:])
    f.propose(mutate=lambda cp, cpos, cc: cc.__setitem__(0, 1))  # cache miss: normal propose
    f.next_block(finish=(2,))
    logs.append(r.log[:])
    f.step([1, 1, 0])
    logs.append(r.log[:])
    f.propose()
    f.next_block()
    f.step([0, 0, 0], spec_mode="argmax_ids")
    logs.append(r.log[:])
    return [
        [tuple(str(y) if not isinstance(y, (int, bool, str, tuple)) else y for y in x) for x in lg_] for lg_ in logs
    ]


def test_flag_off_call_logs_are_the_pre_chain_call_logs(monkeypatch):
    for n in ("QWEN36_SPEC_CHAIN", "QWEN36_SPEC_CHAIN_AUDIT"):
        monkeypatch.delenv(n, raising=False)
    got = _flag_off_scenario(monkeypatch)
    flat = [x for lg_ in got for x in lg_]
    assert not any(("ovr" in str(x)) or x == ("trace", CHG) for x in flat)
    import hashlib

    digest = hashlib.sha256(repr(got).encode()).hexdigest()
    assert digest == _GOLDEN_FLAG_OFF_DIGEST, digest


# sha256 of the call logs of this scenario, recorded on the engine + model WITHOUT the chain changes (the S2 tree, work/patches/m1_s2_done_*)
_GOLDEN_FLAG_OFF_DIGEST = "bc69c904b8bd430201f56e6a0ec0ca5a86656794f4e9b1febd8e96498925b121"


# --------------------------------------------------------------------------- #
# (vi) model + prepare / capture / shutdown
# --------------------------------------------------------------------------- #
def test_model_skip_flags_and_default_unchanged(monkeypatch):
    B = 4
    m, bufs = _hostset_vfy_model(B, T)
    log = []
    _fake_ttnn(monkeypatch, log, m)
    tokens = [[u * 10 + j for j in range(T)] for u in range(B)]
    pos, mi = [10, 20, 30, 40], [0, 1, 2, 3]
    m.verify_traced(tokens, pos, mi)  # default: both copies
    assert bufs["token"].writes == 1 and bufs["aux"].writes == 1 and m._vfy_aux_staged is not None
    assert m.vfy_gdn_parity() == 1
    del log[:]
    h = m.verify_stage_and_enqueue(tokens, pos, mi, skip_tokens_h2d=True, skip_aux_h2d=True)
    assert bufs["token"].writes == 1 and bufs["aux"].writes == 1 and m._vfy_aux_staged is None
    assert log == [("trace", 7)], log
    m.verify_after_external_sync(h)
    assert m.vfy_gdn_parity() == 0  # flipped once by the replay's after-sync half
    m.verify_stage_and_enqueue(tokens, pos, mi, skip_tokens_h2d=True)  # tokens skipped only: the aux copy is made
    assert bufs["token"].writes == 1 and bufs["aux"].writes == 2


def test_prepare_and_capture_chain_with_stubs(monkeypatch):
    calls = []

    class _Stub:
        def __init__(self, name, **kw):
            self.name, self.kw, self.draft_ins, self.K, self.B = name, kw, [f"stand{name}{i}" for i in range(K)], K, 3
            self.ovr_in = _Buf("ovr")

        def body(self):
            calls.append(("body", self.name, tuple(self.draft_ins)))

        def bind_draft_ins(self, outs):
            calls.append(("bind", self.name))
            self.draft_ins = list(outs)

    def alloc(mesh, B, K_, Nv, **kw):
        calls.append(("alloc", "g" if kw["result_in"] is r.fa.result_out else "s", Nv))
        return _Stub("g" if kw["result_in"] is r.fa.result_out else "s", **kw)

    r = _ChainRig(monkeypatch, 3, chain=False)
    e, m = r.e, r.m
    monkeypatch.setattr(se_mod.ChainDevice, "allocate", staticmethod(alloc))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda d: calls.append(("sync",)))
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda mesh, cq_id=0: calls.append(("begin",)) or 100 + len(calls))
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda mesh, tid, cq_id=0: calls.append(("end", tid)))
    # flag off: nothing happens
    monkeypatch.delenv("QWEN36_SPEC_CHAIN", raising=False)
    e._prepare_chain()
    assert e._ch_g is None and not calls
    # flag on, model without the packed aux buffer: chain stays off
    monkeypatch.setenv("QWEN36_SPEC_CHAIN", "1")
    aux = m._vfy_aux_buf
    m._vfy_aux_buf = None
    e._prepare_chain()
    assert e._ch_g is None and not calls
    m._vfy_aux_buf, m._vfy_aux_staged = aux, "mirror"
    e._prepare_chain()
    assert e._ch_g is not None and e._ch_s is not None and m._vfy_aux_staged is None
    assert [c[:2] for c in calls if c[0] == "alloc"] == [("alloc", "g"), ("alloc", "s")]
    assert e._ch_g.kw["tok_out"] is m._vfy_token_buf and e._ch_g.kw["vaux_out"] is m._vfy_aux_buf
    assert e._ch_g.kw["aux_prev_in"] is r.fa.aux_in and e._ch_s.kw["aux_prev_in"] is r.fa.aux_in
    assert [c[1] for c in calls if c[0] == "body"] == ["g", "s"]
    del calls[:]
    e._capture_chain()
    assert [c[:2] for c in calls if c[0] == "bind"] == [("bind", "g"), ("bind", "s")]
    assert e._ch_g.draft_ins == e._dr_outs and e._ch_trace_g is not None and e._ch_trace_s is not None
    assert [c[0] for c in calls] == ["bind", "bind", "begin", "body", "end", "begin", "body", "end", "sync"]
    # an allocation that does not fit leaves chain mode off
    e._ch_g = e._ch_s = None

    def bad(*a, **k):
        raise AssertionError("bound result_in: shape mismatch")

    monkeypatch.setattr(se_mod.ChainDevice, "allocate", staticmethod(bad))
    e._prepare_chain()
    assert e._ch_g is None
