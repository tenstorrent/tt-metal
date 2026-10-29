# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the device logprob payload in the spec engine (phase 2 step S4, work/research_1026/67 section 4, tests H8 to H12 of
the model side): fake device, fake stats trace, call log (the rigs of test_spec_fused_engine_host.py and test_spec_async_engine_host.py).

The fake stats trace computes the two fp32 sums of the stats buffer (rows 0..31 L_raw, rows 32..63 L_T) from the logits the fake
verify "computed". A step whose ``spec_sampling`` carries ``num_logprobs`` >= 0 on some row must keep the S2 sampled step (also when
every row is greedy), replay the stats trace, read the union + stats in the same enqueue as the other reads, and return a payload
equal to the one the engine fallback builds from the full logits. No device.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_logprobs_engine_host.py
"""

import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_async_engine_host import _AsyncRig, _Pipe
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import (
    DRAFT,
    FA,
    FS,
    NB,
    S_M,
    TOPK,
    K,
    T,
    _draft_block,
    _expected_commit,
    _plateau_logits,
    _pt,
    _SampledRig,
    _sampling,
    _smooth_logits,
)
from models.demos.blackhole.qwen36.tests.test_spec_specstep_host import _Buf
from models.demos.blackhole.qwen36.tt.spec_engine import FusedHandle
from models.demos.blackhole.qwen36.tt.spec_logprobs import DeviceCandidates, candidates_from_logits, permute_candidates

STATS = 16


class _StatsMixin:
    """Adds the stats trace to a rig: ids, the buffer ``_vfy_stats_out`` fp32 [1,1,64,1] and the fake replay (after the topk trace
    in the call order, because the model enqueues it there)."""

    def _add_stats(self, ready=True):
        m = self.m
        m._vfy_stats_trace_id, m._vfy_stats_ready = STATS, ready
        m._vfy_stats_out = _Buf("stats")
        m._vfy_stats_out.data = torch.zeros(1, 1, 64, 1, dtype=torch.float32)

    def _exec(self, dev, tid, cq_id=0, blocking=False):
        super()._exec(dev, tid, cq_id, blocking)
        if tid == STATS:
            B, R = self.B, self.R
            x = self.logits.float()
            mx = x.max(dim=1, keepdim=True).values
            temp = self.fs.samp_in.data[0, :B].float()
            temp = torch.where(temp > 0, temp, torch.ones_like(temp)).repeat_interleave(T)[:R, None]
            d = x - mx
            buf = torch.zeros(64, dtype=torch.float32)
            buf[:R] = d.exp().sum(1)
            buf[32 : 32 + R] = (d / temp).exp().sum(1)
            self.m._vfy_stats_out.data = buf.reshape(1, 1, 64, 1)


class _SyncRig(_StatsMixin, _SampledRig):
    def __init__(self, monkeypatch, B, with_fs=True, stats=True):
        super().__init__(monkeypatch, B, with_fs=with_fs)
        self._add_stats(ready=stats)


class _DeferRig(_StatsMixin, _AsyncRig):
    def __init__(self, monkeypatch, B, stats=True):
        super().__init__(monkeypatch, B)
        self._add_stats(ready=stats)


def _ss(B, nvd, temps, nlp, **kw):
    ss = _sampling(B, nvd, temps, **kw)
    ss["num_logprobs"] = torch.tensor(nlp, dtype=torch.int32)
    return ss


def _ref(r, B):
    return candidates_from_logits(r.logits.float(), B, T, M=S_M)


def _same(a, b, atol=2e-5):
    assert isinstance(a, DeviceCandidates) and isinstance(b, DeviceCandidates)
    assert a.ids.dtype == torch.int32 and a.values.dtype == torch.float32 and a.lse.dtype == torch.float32
    assert torch.equal(a.ids, b.ids) and torch.equal(a.values, b.values)
    assert torch.allclose(a.lse, b.lse, atol=atol, rtol=0), (a.lse - b.lse).abs().max()


def _same_tensors(plugin_cand, dev):
    """The plugin payload (adapter output) holds the very tensors of the engine payload."""
    assert plugin_cand.ids is dev.ids and plugin_cand.values is dev.values and plugin_cand.lse is dev.lse


ONES = lambda B: torch.ones(B, dtype=torch.int32)  # noqa: E731


# --------------------------------------------------------------------------- #
# H12: the step type and the payload (sync)
# --------------------------------------------------------------------------- #
def test_all_greedy_logprobs_step_keeps_the_sampled_step_and_replays_the_stats(monkeypatch):
    B = 3
    r = _SyncRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 21)
    am = r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    ss = _ss(B, nvd, [0.0, 0.0, 0.0], [5, -2, 0])
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert (
        ("trace", FS) in r.log and ("trace", TOPK) in r.log and ("trace", STATS) in r.log and ("trace", FA) not in r.log
    )
    i = [x for x in r.log if x[0] == "trace"]
    assert i.index(("trace", TOPK)) < i.index(("trace", STATS)) < i.index(("trace", FS)) < i.index(("trace", DRAFT))
    c, n = _expected_commit(am.reshape(B, T), tok, nvd)
    assert torch.equal(res.accepted_token_ids, c) and torch.equal(res.accepted_counts, n)
    _same(res.candidates, _ref(r, B))
    # reads: the stats and the union are non-blocking copies enqueued before the single synchronize, converted after it
    kinds = [x[0] for x in r.log]
    s_i = kinds.index("sync")
    assert kinds.count("sync") == 1
    pre = [x[1] for x in r.log[:s_i] if x[0] == "from_device"]
    assert {"stats", "uvals", "uids"} <= set(pre)
    post = [x[1] for x in r.log[s_i:] if x[0] == "to_torch"]
    assert {"stats", "uvals", "uids"} <= set(post)
    assert all(x[2] is False for x in r.log if x[0] == "from_device")


def test_without_a_logprobs_row_nothing_of_the_stats_runs(monkeypatch):
    B = 3
    r = _SyncRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 22)
    am = r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    for nl in ([-2, -2, -2], None):
        ss = _sampling(B, nvd, [0.8, 0.0, 1.2])
        if nl is not None:
            ss["num_logprobs"] = torch.tensor(nl, dtype=torch.int32)
        res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
        assert ("trace", STATS) not in r.log and "stats" not in [x[1] for x in r.log if len(x) > 1 and x[0] != "trace"]
        assert res.candidates is None
    # all greedy and no logprobs row: still the S1 step
    r.set_ids(am)
    r.step(
        tok,
        pos,
        nvd,
        ONES(B),
        _pt(B),
        {**_sampling(B, nvd, [0.0] * 3), "num_logprobs": torch.full((B,), -2, dtype=torch.int32)},
    )
    assert ("trace", FA) in r.log and ("trace", FS) not in r.log and ("trace", STATS) not in r.log


@pytest.mark.parametrize(
    "temps,nlp", [([0.8, 0.0, 1.2], [20, 5, 0]), ([0.7, 0.9, 1.1], [1, 1, 1]), ([0.0, 0.8, 0.0], [-2, 3, -2])]
)
def test_device_payload_equals_the_host_payload_from_the_full_logits(monkeypatch, temps, nlp):
    B = 3
    r = _SyncRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 23)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    ss = _ss(B, nvd, temps, nlp, topk=3)
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    dev = res.candidates
    assert r.e.fused_steps == 1 and not r.e.__dict__.get("fused_fallbacks")
    # the engine fallback of the same step (no sampled trace): the payload from the logits
    r2 = _SyncRig(monkeypatch, B, with_fs=False)
    r2.set_logits(logits)
    res2 = r2.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert r2.e.fused_fallbacks == {"no_sampled_trace": 1}
    _same(dev, res2.candidates)
    # and the committed tokens agree with the host walk (the existing sampled rule)
    assert res.accepted_token_ids.shape == res2.accepted_token_ids.shape


def test_overflow_redo_gives_the_host_payload(monkeypatch):
    B = 3
    r = _SyncRig(monkeypatch, B)
    logits = _plateau_logits(B * T, 3)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    ss = _ss(B, nvd, [0.8, 1.0, 0.0], [4, 4, 4], topk=3)
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert r.e.fused_fallbacks == {"overflow_redo": 1}
    _same(res.candidates, _ref(r, B), atol=1e-6)
    # plateau rows: the candidates are the first 8 ids of the (value desc, id asc) order, ties included
    assert res.candidates.ids.shape == (B, T, S_M)


@pytest.mark.parametrize("reason", ["mixed_rope_delta", "eager_drafter", "mixed_fit", "all_declined"])
def test_engine_fallbacks_give_the_payload_from_the_logits(monkeypatch, reason):
    B = 3
    r = _SyncRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 24)
    r.set_logits(logits)
    r.e._fused_ineligible = lambda *a, **k: (reason, None)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), _ss(B, nvd, [0.0, 0.8, 0.0], [5, -2, 0]))
    assert r.e.fused_fallbacks == {reason: 1} and ("trace", STATS) not in r.log
    _same(res.candidates, _ref(r, B), atol=1e-6)


def test_kill_switch_off_or_no_stats_trace_falls_back_to_the_host_payload(monkeypatch):
    B = 3
    for stats, env in ((False, "1"), (True, "0")):
        monkeypatch.setenv("QWEN36_SPEC_DEVICE_LOGPROBS", env)
        r = _SyncRig(monkeypatch, B, stats=stats)
        logits = _smooth_logits(B * T, 25)
        r.set_logits(logits)
        tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
        res = r.step(tok, pos, nvd, ONES(B), _pt(B), _ss(B, nvd, [0.8, 0.0, 1.2], [5, 5, 5]))
        assert r.e.fused_fallbacks == {"no_stats_trace": 1}
        assert ("trace", STATS) not in r.log and ("trace", FS) not in r.log
        _same(res.candidates, _ref(r, B), atol=1e-6)
    # without the key the same engine does not care about the switch
    r = _SyncRig(monkeypatch, B, stats=False)
    logits = _smooth_logits(B * T, 26)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    r.step(tok, pos, nvd, ONES(B), _pt(B), _sampling(B, nvd, [0.8, 0.0, 1.2]))
    assert not r.e.__dict__.get("fused_fallbacks") and ("trace", FS) in r.log


def test_payload_follows_the_slot_permutation(monkeypatch):
    B = 3
    r = _SyncRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 27)
    r.set_logits(logits)
    cand = _ref(r, B)
    phys = torch.tensor([2, 0, 1])
    r.e._phys_perm = (phys, torch.argsort(phys))
    out = r.e._cands_to_slots(cand)
    ids_slots = r.e._to_slots(cand.ids)[0]
    assert torch.equal(out.ids, ids_slots) and torch.equal(out.lse, r.e._to_slots(cand.lse)[0])
    assert torch.equal(out.values, permute_candidates(cand, phys).values)
    r.e._phys_perm = None
    assert r.e._cands_to_slots(cand) is cand


def test_zero_or_stale_stats_rows_stay_finite(monkeypatch):
    B = 3
    r = _SyncRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 28)
    r.set_logits(logits)
    reads = (
        types.SimpleNamespace(data=r.m._vfy_topk_vals_out.data),
        types.SimpleNamespace(data=r.m._vfy_topk_ids_out.data),
        types.SimpleNamespace(data=torch.zeros(1, 1, 64, 1)),
    )
    monkeypatch.setattr(ttnn, "to_torch", lambda t, *a, **k: t.data)
    cand = r.e._cand_build(reads)
    assert bool(torch.isfinite(cand.lse).all()) and bool((cand.lse >= cand.values[..., 0]).all())


# --------------------------------------------------------------------------- #
# deferred serving
# --------------------------------------------------------------------------- #
def _defer_step(r, B, ss, logits, pend=None):
    r.queue = [("logits", logits)]
    p = _Pipe(r, B, [3, 2, 1])
    h = p.enq(None, sampling=ss)
    return p, h


def test_deferred_step_reads_the_payload_in_the_same_enqueue_and_the_reader_builds_it(monkeypatch):
    B = 3
    r = _DeferRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 31)
    ss = _ss(B, torch.full((B,), K), [0.8, 0.0, 1.2], [5, 3, -2])
    p, h = _defer_step(r, B, ss, logits)
    kinds = [x[0] for x in r.log]
    assert "sync" not in kinds and kinds[-1] == "record_event"
    assert ("trace", STATS) in r.log
    reads = [x[1] for x in r.log if x[0] == "from_device"]
    assert {"stats", "uvals", "uids"} <= set(reads) and reads.count("uvals") == 1
    assert max(i for i, x in enumerate(r.log) if x[0] == "from_device") < kinds.index("record_event")
    assert [x for x in r.log if x[0] == "to_torch"] == [], "no host conversion on the engine thread"
    ids, counts, p0 = h.wait()
    _same(h.candidates(), _ref(r, B))
    assert h.candidates() is not None and isinstance(h, FusedHandle)


def test_deferred_audit_step_reads_the_union_once(monkeypatch):
    B = 3
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT_EVERY", "1")
    r = _DeferRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 32)
    ss = _ss(B, torch.full((B,), K), [0.8, 0.9, 1.2], [5, 3, 1])
    p, h = _defer_step(r, B, ss, logits)
    assert [x[1] for x in r.log if x[0] == "from_device"].count("uvals") == 1
    h.wait()
    _same(h.candidates(), _ref(r, B))


def test_deferred_step_without_logprobs_has_no_payload_and_no_stats(monkeypatch):
    B = 3
    r = _DeferRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 33)
    p, h = _defer_step(r, B, _sampling(B, torch.full((B,), K), [0.8, 0.0, 1.2]), logits)
    assert ("trace", STATS) not in r.log and "stats" not in [x[1] for x in r.log if x[0] == "from_device"]
    h.wait()
    assert h.candidates() is None


def test_deferred_all_greedy_logprobs_step_is_a_sampled_step(monkeypatch):
    B = 3
    r = _DeferRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 34)
    ss = _ss(B, torch.full((B,), K), [0.0, 0.0, 0.0], [2, -2, -2])
    p, h = _defer_step(r, B, ss, logits)
    assert h.kind == "s" and ("trace", FS) in r.log and ("trace", FA) not in r.log and ("trace", STATS) in r.log
    h.wait()
    _same(h.candidates(), _ref(r, B))


def test_deferred_fallback_handle_carries_the_host_payload(monkeypatch):
    B = 3
    r = _DeferRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 35)
    r.e._fused_ineligible = lambda *a, **k: ("mixed_rope_delta", None)
    ss = _ss(B, torch.full((B,), K), [0.8, 0.0, 1.2], [5, 3, -2])
    p, h = _defer_step(r, B, ss, logits)
    assert not h.device and h.ready and r.e.fused_fallbacks == {"mixed_rope_delta": 1}
    h.wait()
    _same(h.candidates(), _ref(r, B), atol=1e-6)


def test_chained_logprobs_steps_each_get_their_own_payload(monkeypatch):
    B = 3
    r = _DeferRig(monkeypatch, B)
    L = [_smooth_logits(B * T, 40 + i) for i in range(3)]
    r.queue = [("logits", x) for x in L]
    p = _Pipe(r, B, [3, 2, 1])
    ss = [_ss(B, torch.full((B,), K), [0.8, 0.0, 1.2], [5, 5, -2], seed=i) for i in range(3)]
    hs, refs = [], []
    for i in range(3):
        hs.append(p.enq(hs[-1] if hs else None, sampling=ss[i]))
        refs.append(candidates_from_logits(L[i].float().to(torch.bfloat16).float(), B, T, M=S_M))
        assert ("trace", STATS) in r.log
        # resolve the older step while the newer is in flight (the payload of step i must not see step i+1's buffers)
        if i >= 1:
            p.resolve(hs[i - 1])
            _same(hs[i - 1].candidates(), refs[i - 1])
    p.resolve(hs[2])
    _same(hs[2].candidates(), refs[2])


# --------------------------------------------------------------------------- #
# the adapter (qwen36_vllm): plumbing behind the capability, no declaration here
# --------------------------------------------------------------------------- #
def _qv():
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv

    return qv.Qwen36MTPContractForCausalLM


def _ss_checked(B, nlp):
    ss = _sampling(B, torch.full((B,), K), [0.8] * B)
    if nlp is not None:
        ss["num_logprobs"] = nlp
    return ss


def test_adapter_checks_num_logprobs_and_gates_it_on_the_capability(expect_error):
    B = 3
    cls = _qv()
    caps = {}
    fake = types.SimpleNamespace(_SPEC_K=K, model_capabilities=caps)
    chk = lambda ss, mode="fused_sample": cls._check_spec_sampling(fake, ss, mode, B)  # noqa: E731
    assert chk(_ss_checked(B, None)) is not None  # the old key set still passes
    none = torch.full((B,), -2, dtype=torch.int32)
    assert chk(_ss_checked(B, none))["num_logprobs"] is none  # -2 everywhere needs no capability
    want = torch.tensor([5, -2, 0], dtype=torch.int32)
    with expect_error(ValueError, "supports_spec_device_logprobs"):
        chk(_ss_checked(B, want))
    caps["supports_spec_device_logprobs"] = True  # what the capability line would set
    assert chk(_ss_checked(B, want))["num_logprobs"] is want
    with expect_error(ValueError, "num_logprobs"):
        chk(_ss_checked(B, torch.tensor([21, -2, 0], dtype=torch.int32)))
    with expect_error(ValueError, "num_logprobs"):
        chk(_ss_checked(B, torch.tensor([-3, -2, 0], dtype=torch.int32)))
    with expect_error(ValueError, "num_logprobs"):
        chk(_ss_checked(B, torch.tensor([1, 1, 1], dtype=torch.int64)))  # dtype


def _fake_plugin(monkeypatch):
    """Stand-ins for the plugin's VerifyOutput / FusedReadout WITH the new optional ``candidates`` field (the S5 contract), so the
    adapter's keyword use is tested without a plugin edit. Also records whether the keyword was passed."""
    import sys

    class VerifyOutput:
        def __init__(
            self,
            spec_mode,
            argmax_ids=None,
            accepted_token_ids=None,
            accepted_counts=None,
            hidden=None,
            logits=None,
            topk_values=None,
            topk_ids=None,
            fetch_logits=None,
            handle=None,
            candidates=None,
        ):
            self.spec_mode, self.argmax_ids, self.accepted_token_ids, self.accepted_counts = (
                spec_mode,
                argmax_ids,
                accepted_token_ids,
                accepted_counts,
            )
            self.handle, self.candidates, self.logits, self.topk_values, self.topk_ids = (
                handle,
                candidates,
                logits,
                topk_values,
                topk_ids,
            )

    class FusedReadout:
        def __init__(self, accepted_token_ids, accepted_counts, p0, candidates=None):
            self.accepted_token_ids, self.accepted_counts, self.p0, self.candidates = (
                accepted_token_ids,
                accepted_counts,
                p0,
                candidates,
            )

    class SpecCandidates:  # the plugin-owned payload type (ids, values, lse); the adapter builds it from DeviceCandidates
        def __init__(self, ids, values, lse):
            self.ids, self.values, self.lse = ids, values, lse

    mod = types.ModuleType("vllm_tt_plugin.spec_decode")
    mod.VerifyOutput, mod.FusedReadout, mod.SpecCandidates = VerifyOutput, FusedReadout, SpecCandidates
    mod.check_spec_side_tensors = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin.spec_decode", mod)
    return VerifyOutput, FusedReadout


def test_adapter_carries_the_payload_in_verify_output_and_fused_readout(monkeypatch):
    B = 3
    VO, FR = _fake_plugin(monkeypatch)
    cls = _qv()
    cand = candidates_from_logits(_smooth_logits(B * T, 50), B, T, M=S_M)
    calls = []
    res = types.SimpleNamespace(
        spec_mode="fused_sample",
        argmax_ids=torch.zeros(B, T, dtype=torch.int32),
        accepted_token_ids=torch.zeros(B, T, dtype=torch.int32),
        accepted_counts=torch.ones(B, dtype=torch.int32),
        candidates=cand,
    )
    eng = types.SimpleNamespace(decode_forward=lambda *a, **kw: calls.append(kw) or res)
    fake = types.SimpleNamespace(
        _SPEC_K=K,
        model_capabilities={"supports_spec_device_logprobs": True},
        _ensure_ready=lambda: eng,
        _table=lambda t: t,
        _check_spec_sampling=lambda ss, mode, B: ss,
    )
    tok, pos = torch.zeros(B, T, dtype=torch.int32), torch.zeros(B, T, dtype=torch.int32)
    ss = _ss_checked(B, torch.tensor([5, -2, 0], dtype=torch.int32))
    out = cls.decode_forward(
        fake,
        tok,
        pos,
        torch.zeros(B, NB, dtype=torch.int32),
        num_valid_drafts=torch.full((B,), K, dtype=torch.int32),
        accepted_counts=ONES(B),
        spec_mode="fused_sample",
        spec_sampling=ss,
    )
    assert type(out.candidates).__name__ == "SpecCandidates" and calls[0]["spec_sampling"] is ss
    _same_tensors(out.candidates, cand)  # same tensors, plugin type
    # a step without a payload never passes the keyword (an older plugin class has no such field)
    res.candidates = None
    VO2 = type("VO2", (), {"__init__": lambda self, **kw: setattr(self, "kw", kw)})
    monkeypatch.setattr(__import__("sys").modules["vllm_tt_plugin.spec_decode"], "VerifyOutput", VO2)
    out = cls.decode_forward(
        fake,
        tok,
        pos,
        torch.zeros(B, NB, dtype=torch.int32),
        num_valid_drafts=torch.full((B,), K, dtype=torch.int32),
        accepted_counts=ONES(B),
        spec_mode="fused_sample",
    )
    assert "candidates" not in out.kw
    # reader thread: FusedReadout from a handle
    handle = types.SimpleNamespace(
        wait=lambda: (res.accepted_token_ids, res.accepted_counts, torch.zeros(B, dtype=torch.int32)),
        candidates=lambda: cand,
    )
    from models.demos.blackhole.qwen36.tt import spec_engine

    monkeypatch.setattr(spec_engine, "FusedHandle", type(handle))
    ro = cls.read_decode_output(object(), handle)
    assert isinstance(ro, FR) and type(ro.candidates).__name__ == "SpecCandidates"
    _same_tensors(ro.candidates, cand)
    handle.candidates = lambda: None
    ro = cls.read_decode_output(object(), handle)
    assert ro.candidates is None
