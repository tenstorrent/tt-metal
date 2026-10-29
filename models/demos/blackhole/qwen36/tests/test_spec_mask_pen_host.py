"""Phase 3 merge (grammar masks P3a + device penalties P8): the two features in one fused sampled step, on the host (fake device).

What this file checks (no device, no server):
  * trace order of a step with a mask row and a penalty row: main, mask, pen, topk, [stats], accept (certificate variant), pen update,
    drafter; the model enqueue order (one call and the split forward_only + post) is main, mask, pen, topk, stats;
  * a greedy row that carries a mask row and a penalty is rewritten ONCE (T 1, top_k 1, top_p 0); a plugin-rewritten penalty row is
    not rewritten again; a greedy penalty row without a constrained mask row is still refused;
  * the committed ids of such a step equal the host walk over penalized-then-masked logits;
  * the fallback and the overflow redo apply both (penalties exact fp32, then the mask) and repair the device counts;
  * on the host reference the mask and the penalties commute (equal bit for bit in both orders).
"""

import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import (
    DRAFT,
    FS,
    S_V,
    TOPK,
    VFY,
    K,
    T,
    _block,
    _plateau_logits,
    _pt,
    _SampledRig,
    _smooth_logits,
    _union_from_logits,
)
from models.demos.blackhole.qwen36.tests.test_spec_logprobs_engine_host import STATS, _StatsMixin
from models.demos.blackhole.qwen36.tests.test_spec_mask_engine_host import MASKT, _mask, _MaskMixin
from models.demos.blackhole.qwen36.tests.test_spec_penalties_host import EMU, PEN, UPD, _pen_sampling
from models.demos.blackhole.qwen36.tests.test_spec_topp_cert_host import FSC, _CertMixin
from models.demos.blackhole.qwen36.tt import spec_masks as SM
from models.demos.blackhole.qwen36.tt.spec_penalties import (
    PEN_ROWS,
    PenaltyDevice,
    apply_penalties_device_model,
    apply_penalties_exact,
    column_history_counts,
    counts_from_ids,
    mask_from_ids,
)

ONES = lambda B: torch.ones(B, dtype=torch.int32)  # noqa: E731


class _PenMix:
    """The fake pen trace of the merged rig: it edits the logits the verify 'computed' AFTER the mask trace (the rig's current
    ``self.logits``, masked when a mask trace ran), refreshes the topk union, and leaves the logits trace output raw (the logits trace
    recomputes the raw logits). The fake update trace counts the committed ids."""

    def _add_pen(self):
        B = self.B
        self.m._vfy_pen_trace_id = PEN
        self.e._pen_upd_trace = UPD
        self.pen = PenaltyDevice.allocate("mesh", B, K, S_V, S_V, torch.zeros(1, B * T, dtype=torch.int32), ops=EMU)
        self.e._pen = self.pen

    def _exec(self, dev, tid, cq_id=0, blocking=False):
        super()._exec(dev, tid, cq_id, blocking)
        if tid == PEN:
            self.pen.tok_in = self.m._vfy_token_buf.data.reshape(1, -1)
            x = self.logits.reshape(1, 1, self.R, S_V).clone()
            self.pen.apply_body(x)
            self.logits = x.reshape(self.R, S_V)
            m = self.m
            vals, ids = _union_from_logits(self.logits)
            uv = torch.zeros(1, 1, 32, vals.shape[1], dtype=torch.bfloat16)
            ui = torch.zeros(1, 1, 32, vals.shape[1], dtype=torch.int32)
            uv[0, 0, : self.R], ui[0, 0, : self.R] = vals, ids
            m._vfy_topk_vals_out.data, m._vfy_topk_ids_out.data = uv, ui
        if tid == UPD:
            self.pen.bind_result(self.fs.result_out.data)
            self.pen.update_body()


class _Both(_MaskMixin, _PenMix, _CertMixin, _StatsMixin, _SampledRig):
    """_SampledRig with the stats trace, the certificate accept, the mask trace AND the penalty traces (call order = the engine's)."""

    def __init__(self, monkeypatch, B):
        super().__init__(monkeypatch, B)
        self.B = B
        self._add_stats()
        self._add_cert()
        self._add_mask(monkeypatch, True)
        self._add_pen()


def _traces(r):
    return [x[1] for x in r.log if x[0] == "trace"]


def _both_sampling(B, nvd, temps, pen_rows, mask, *, rep=1.4, freq=0.2, pres=0.1, rebuild=None):
    ss = _pen_sampling(B, nvd, temps, pen_rows, rep=rep, freq=freq, pres=pres, rebuild=rebuild)
    ss["mask"] = mask
    return ss


def _expected(raw, tok, nvd, u, allowed, pen, exact):
    """Committed ids of user u: argmax walk over penalized (device model or fp32 exact) then masked logits. ``pen`` None = no
    penalty, else (rep, freq, pres, prompt ids, output ids). ``allowed`` None = no mask, else bool [T, V]."""
    rows = raw[u * T : (u + 1) * T].float()
    if pen is not None:
        rep, freq, pres, prompt, outs = pen
        cnt = counts_from_ids(outs, S_V)
        hist = column_history_counts(cnt, tok[u, 1:], K, T, device_rule=True)
        msk = mask_from_ids(prompt, S_V).reshape(1, -1).expand(T, -1)
        a = (rows, hist, msk, torch.full((T,), rep), torch.full((T,), freq), torch.full((T,), pres))
        rows = apply_penalties_exact(*a) if exact else apply_penalties_device_model(*a).float()
    if allowed is not None:
        rows = rows.masked_fill(~allowed, float("-inf"))
    am = [int((r == r.max()).nonzero()[0]) for r in rows]
    n = 0
    while n < int(nvd[u]) and am[n] == int(tok[u, n + 1]):
        n += 1
    return am[: n + 1]


def _lists():
    outs = [[1, 2, 3], [5, 6, 7, 7], [0]]
    prompts = [[4, 5], [9, 10, 11], [3]]
    return outs, prompts


def _rebuild(rows):
    outs, prompts = _lists()
    return {u: (torch.tensor(prompts[u]), torch.tensor(outs[u])) for u in rows}


MASK_ROWS = {
    (0, 0): range(0, 64, 2),
    (0, 1): range(0, 64, 2),
    (0, 2): range(0, 64, 2),
    (1, 0): range(1, 64, 2),
    (1, 1): range(1, 64, 2),
    (1, 2): range(1, 64, 2),
}


def _case(monkeypatch, temps, pen_rows, seed=31, rows=None, nvs=(3, 2, 1)):
    B = len(temps)
    r = _Both(monkeypatch, B)
    logits = _smooth_logits(B * T, seed)
    r.set_logits(logits)
    tok, pos, nvd = _block(B, list(nvs))
    mask = _mask(B, MASK_ROWS if rows is None else rows)
    ss = _both_sampling(B, nvd, temps, pen_rows, mask, rebuild=_rebuild(pen_rows))
    return r, logits, tok, pos, nvd, ss, mask


def _allowed(mask, B):
    return SM.unpack_allowed(mask.reshape(B * T, -1), S_V).reshape(B, T, S_V)


# --------------------------------------------------------------------------- #
# trace order
# --------------------------------------------------------------------------- #
def test_a_step_with_a_mask_row_and_a_penalty_row_replays_the_traces_in_order(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, [0.0, 0.0, 0.0], pen_rows=[1])
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    tr = _traces(r)
    # main, mask, pen, topk, accept (certificate variant: a masked step), pen update, drafter
    assert tr == [VFY, MASKT, PEN, TOPK, FSC, UPD, DRAFT], tr
    assert FS not in tr and STATS not in tr
    assert r.e.mask_steps == 1 and r.e.pen_steps == 1 and not r.e.__dict__.get("fused_fallbacks")
    h2d = [x[1] for x in r.log if x[0] == "h2d"]
    assert h2d.count("mask_in") == 1
    i_mask_h2d = [i for i, x in enumerate(r.log) if x[:2] == ("h2d", "mask_in")][0]
    assert r.log.index(("trace", VFY)) < i_mask_h2d < r.log.index(("trace", MASKT)) < r.log.index(("trace", PEN))
    assert sum(1 for x in r.log if x[0] == "sync") == 1
    # every committed token is allowed by the mask
    allowed = _allowed(mask, B)
    for u in range(B):
        for j in range(int(res.accepted_counts[u])):
            assert bool(allowed[u, j, int(res.accepted_token_ids[u, j])])


def test_a_certificate_row_adds_the_stats_trace_between_topk_and_accept(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, [0.0, 0.0, 0.8], pen_rows=[1])
    ss["top_k"][2], ss["top_p"][2] = 0, 0.9  # a certificate row (top_k 0 with top_p < 1)
    r.fs.samp_in.data = torch.zeros(4 + B, 32)
    r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert _traces(r)[:6] == [VFY, MASKT, PEN, TOPK, STATS, FSC], _traces(r)
    # the fake stats are zero: the certificate fails and the engine fetches the logits (a trailing logits trace) after the step
    assert _traces(r)[6:8] == [UPD, DRAFT], _traces(r)


def test_mask_only_and_pen_only_steps_keep_their_own_trace_logs(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, [0.0, 0.0, 0.0], pen_rows=[])
    r.step(tok, pos, nvd, ONES(B), _pt(B), ss)  # penalties key present, every row neutral
    assert _traces(r) == [VFY, MASKT, TOPK, FSC, DRAFT]
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, [0.0, 0.0, 0.0], pen_rows=[1])
    del ss["mask"]
    r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert _traces(r) == [VFY, PEN, TOPK, FS, UPD, DRAFT]
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, [0.0, 0.0, 0.0], pen_rows=[1], rows={(1, 0): [3, 4]})
    r.step(tok, pos, nvd, ONES(B), _pt(B), {k: v for k, v in ss.items() if k not in ("mask", "penalties")})
    assert PEN not in _traces(r) and MASKT not in _traces(r) and UPD not in _traces(r)


def test_model_enqueue_order_one_call_and_split_post(monkeypatch, expect_error):
    B = 3
    r = _Both(monkeypatch, B)
    m = r.m
    r.set_logits(_smooth_logits(B * T, 3))
    r.fs.samp_in.data = torch.zeros(4 + B, 32)
    tokens = [[u * 10 + j for j in range(T)] for u in range(B)]
    pos = [100 + 7 * u for u in range(B)]
    buf = torch.full((8, 32, 970), -1, dtype=torch.int32)
    full = lambda **kw: m.verify_stage_and_enqueue(tokens, pos, [0] * B, **kw)  # noqa: E731
    del r.log[:]
    h = full(execute_topk=True, execute_stats=True, execute_mask=True, mask_host=buf, execute_pen=True)
    one_call = _traces(r)
    assert one_call == [VFY, MASKT, PEN, TOPK, STATS] and not h.post_pending
    del r.log[:]
    h = full(forward_only=True)
    assert _traces(r) == [VFY] and h.post_pending
    m.verify_enqueue_post(h, execute_topk=True, execute_stats=True, execute_mask=True, mask_host=buf, execute_pen=True)
    assert _traces(r) == one_call and not h.post_pending
    # the pen trace alone keeps its place before the topk trace
    del r.log[:]
    full(execute_topk=True, execute_pen=True)
    assert _traces(r) == [VFY, PEN, TOPK]
    # validation
    with expect_error(ValueError, "topk replay"):
        full(execute_pen=True)
    m._vfy_pen_trace_id = None
    with expect_error(ValueError, "captured pen trace"):
        full(execute_topk=True, execute_pen=True)
    h = full(forward_only=True)
    with expect_error(ValueError, "captured pen trace"):
        m.verify_enqueue_post(h, execute_topk=True, execute_pen=True)


# --------------------------------------------------------------------------- #
# the greedy rewrite (once)
# --------------------------------------------------------------------------- #
def test_a_greedy_row_with_a_mask_row_and_a_penalty_is_rewritten_once(monkeypatch):
    B = 3
    r = _Both(monkeypatch, B)
    tok, pos, nvd = _block(B, [3, 2, 1])
    mask = _mask(B, MASK_ROWS)
    ss = _both_sampling(B, nvd, [0.0, 0.0, 0.0], [1], mask, rebuild=_rebuild([1]))
    # row 1: the plugin's own rewrite of a greedy penalty row (sampled, T 1, top_k 1, top_p 0): not rewritten again
    # row 0: a plain greedy row with a constrained mask: rewritten by the engine, once
    # row 2: a plain greedy row with an unconstrained mask: stays greedy
    ss["greedy"][0] = True
    ss["temperature"][0], ss["top_k"][0], ss["top_p"][0] = 0.0, 1, 1.0
    # a penalty on the plain greedy row 0 as well: the mask attach rewrites it first, the penalty check then accepts it
    ss["penalties"]["repetition"][0] = 1.3
    sp = r.e._engine_sampling(ss, ONES(B))
    assert sp is not None
    for u in (0, 1):
        assert (float(sp["temperature"][u]), int(sp["top_k"][u]), float(sp["top_p"][u]), bool(sp["greedy"][u])) == (
            1.0,
            1,
            0.0,
            False,
        )
    assert bool(sp["greedy"][2]) and int(sp["top_k"][2]) == 1
    assert r.e.__dict__.get("mask_greedy_rewrites", 0) == 1, "only row 0 is rewritten by the engine"
    assert sp["pen"].any_live and bool(sp["pen"].live[0]) and bool(sp["pen"].live[1]) and not bool(sp["pen"].live[2])


def test_a_greedy_penalty_row_without_a_constrained_mask_row_is_still_refused(monkeypatch, expect_error):
    B = 3
    r = _Both(monkeypatch, B)
    tok, pos, nvd = _block(B, [3, 2, 1])
    mask = _mask(B, {(0, 0): [5, 9]})  # only row 0 is constrained
    ss = _both_sampling(B, nvd, [0.0, 0.0, 0.0], [2], mask, rebuild=_rebuild([2]))
    ss["greedy"][2] = True  # greedy penalty row 2 has no constrained mask row
    with expect_error(ValueError, "marked greedy"):
        r.e._engine_sampling(ss, ONES(B))


def test_split_step_checks_the_greedy_penalty_row_after_the_mask_arrives(monkeypatch, expect_error):
    B = 3
    r = _Both(monkeypatch, B)
    tok, pos, nvd = _block(B, [3, 2, 1])
    mask = _mask(B, MASK_ROWS)
    ss = _both_sampling(B, nvd, [0.0, 0.0, 0.0], [0], mask, rebuild=_rebuild([0]))
    ss["greedy"][0] = True
    ss["temperature"][0], ss["top_k"][0], ss["top_p"][0] = 0.0, 1, 1.0
    del ss["mask"]
    r.e._split_begin = True
    sp = r.e._engine_sampling(ss, ONES(B))  # no refusal at the begin: the mask is not known yet
    r.e._split_begin = False
    assert "pen" in sp and bool(sp["greedy"][0])
    r.e._finish_sampling_mask(sp, mask)  # the mask rewrites the greedy row: accepted
    assert not bool(sp["greedy"][0]) and int(sp["top_k"][0]) == 1
    sp2 = dict(sp, greedy=torch.ones(B, dtype=torch.bool))
    sp2.pop("mask")
    with expect_error(ValueError, "marked greedy"):
        r.e._finish_sampling_mask(sp2, None)  # no constraint arrived: refused


# --------------------------------------------------------------------------- #
# the result of a step with both
# --------------------------------------------------------------------------- #
def test_committed_ids_equal_the_host_walk_over_penalized_then_masked_logits(monkeypatch):
    B = 3
    outs, prompts = _lists()
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, [0.0, 0.0, 0.0], pen_rows=[0, 1], seed=33)
    ss["greedy"][0] = True  # plain greedy rows are rewritten by the mask; row 1 is the plugin's rewritten form
    ss["temperature"][0], ss["top_k"][0], ss["top_p"][0] = 0.0, 1, 1.0
    raw = logits.to(torch.bfloat16)
    allowed = _allowed(mask, B)
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert _traces(r)[:3] == [VFY, MASKT, PEN]
    for u, pen in ((0, True), (1, True), (2, False)):
        p = (1.4, 0.2, 0.1, prompts[u], outs[u]) if pen else None
        want = _expected(raw, tok, nvd, u, allowed[u], p, exact=False)
        got = [int(t) for t in res.accepted_token_ids[u] if int(t) >= 0]
        assert got == want[: len(got)] and len(got) == int(res.accepted_counts[u]), (u, got, want)
    # the counts hold the rebuilt output plus this step's commits
    c = r.pen.counts.reshape(PEN_ROWS, S_V)
    for u in (0, 1):
        add = counts_from_ids([int(t) for t in res.accepted_token_ids[u] if int(t) >= 0], S_V)
        assert torch.equal(c[u * T], counts_from_ids(outs[u], S_V) + add)


# --------------------------------------------------------------------------- #
# fallbacks and redo apply both
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reason", ["no_sampled_trace", "no_mask_trace"])
def test_engine_fallback_applies_penalties_then_the_mask_and_repairs_the_counts(monkeypatch, reason):
    B = 3
    outs, prompts = _lists()
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, [0.0, 0.0, 0.0], pen_rows=[1], seed=35)
    if reason == "no_sampled_trace":
        r.e._fs_trace = None
    else:
        r.m._vfy_mask_ready = False  # the mask traces were not built: the masked step walks on the host
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert r.e.fused_fallbacks, r.e.fused_fallbacks
    tr = _traces(r)
    assert PEN not in tr and UPD not in tr and MASKT not in tr
    raw = logits.to(torch.bfloat16)
    allowed = _allowed(mask, B)
    for u in range(B):
        p = (1.4, 0.2, 0.1, prompts[u], outs[u]) if u == 1 else None
        want = _expected(raw, tok, nvd, u, allowed[u], p, exact=True)
        got = [int(t) for t in res.accepted_token_ids[u] if int(t) >= 0]
        assert got == want[: len(got)] and len(got) == int(res.accepted_counts[u]), (u, got, want)
    got1 = [int(t) for t in res.accepted_token_ids[1] if int(t) >= 0]
    assert torch.equal(
        r.pen.counts.reshape(PEN_ROWS, S_V)[1 * T], counts_from_ids(outs[1], S_V) + counts_from_ids(got1, S_V)
    ), "the host-walked commit is in the counts"
    assert r.e.pen_host_walks == 1


def test_overflow_redo_applies_both_and_counts_the_redone_commit_once(monkeypatch):
    B = 2
    r = _Both(monkeypatch, B)
    logits = _plateau_logits(
        B * T, 5
    )  # two high logits then a plateau: the top_k 3 ties reach the last candidate = overflow
    r.set_logits(logits)
    tok, pos, nvd = _block(B, [2, 2])
    mask = _mask(B, {(0, 0): range(0, 64, 2), (0, 1): range(0, 64, 2), (0, 2): range(0, 64, 2)})
    rebuild = {0: (torch.tensor([1]), torch.tensor([2]))}
    ss = _both_sampling(B, nvd, [0.9, 0.9], [0], mask, rep=1.05, freq=0.0, pres=0.0, rebuild=rebuild)
    ss["uniforms"] = torch.full((B, K + 2), 0.5)
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert r.e.fused_fallbacks.get("overflow_redo") == 1, r.e.fused_fallbacks
    got = [int(t) for t in res.accepted_token_ids[0] if int(t) >= 0]
    allowed = _allowed(mask, B)
    assert all(bool(allowed[0, j, t]) for j, t in enumerate(got)), "the redo walk applied the mask"
    assert torch.equal(
        r.pen.counts.reshape(PEN_ROWS, S_V)[0], counts_from_ids([2] + got, S_V)
    ), "garbage of the overflowed device result was counted, or the redo was not"


# --------------------------------------------------------------------------- #
# mask and penalties commute on the host reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_mask_then_penalties_equals_penalties_then_mask_on_the_host_reference(seed):
    g = torch.Generator().manual_seed(seed)
    N = 6
    logits = torch.randn(N, S_V, generator=g) * 4
    logits[0, :5] = -float("inf")  # an earlier -inf stays -inf under both orders
    hist = torch.randint(0, 4, (N, S_V), generator=g).float()
    pm = (torch.rand(N, S_V, generator=g) < 0.2).float()
    rep = torch.tensor([1.0, 1.3, 0.8, 1.5, 1.1, 1.0])
    freq = torch.tensor([0.0, 0.2, 0.0, 0.5, 0.1, 0.3])
    pres = torch.tensor([0.0, 0.1, 0.4, 0.0, 0.2, 0.0])
    allowed = torch.rand(N, S_V, generator=g) < 0.4
    allowed[:, 7] = True  # at least one token allowed per row
    words = SM.pack_allowed(allowed, words=SM.MASK_WORDS)
    pen = lambda x: apply_penalties_exact(x, hist, pm, rep, freq, pres)  # noqa: E731
    msk = lambda x: SM.apply_to_logits(x, words)  # noqa: E731
    a = msk(pen(logits))
    b = pen(msk(logits))
    assert torch.equal(a, b)
    assert bool(torch.isinf(a[~allowed]).all()) and bool((a[~allowed] < 0).all())
    # the device model (bf16 rounding) commutes too
    lb = logits.to(torch.bfloat16)
    pd = lambda x: apply_penalties_device_model(x, hist, pm, rep, freq, pres)  # noqa: E731
    assert torch.equal(msk(pd(lb)).float(), pd(msk(lb)).float())
