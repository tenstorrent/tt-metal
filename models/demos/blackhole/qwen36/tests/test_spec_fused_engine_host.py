# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the fused greedy spec step in MTPSpecEngine (QWEN36_SPEC_FUSED, spec_mode "fused_sample"), over fake
persistent buffers and a call log (the approach of test_spec_stage_split_host.py): the real Qwen36Model verify split and
the real engine run; only the device is faked. The fake accept trace computes its result with the host reference
(spec_fused.fused_greedy_reference) from what the staged buffers hold. No device.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_fused_engine_host.py
"""

import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_hostset_host import _drafter_engine, _hostset_vfy_model
from models.demos.blackhole.qwen36.tests.test_spec_specstep_host import _Buf, _fake_ttnn
from models.demos.blackhole.qwen36.tt import spec_engine as se_mod
from models.demos.blackhole.qwen36.tt.spec_engine import DraftResult, mask_unused_columns
from models.demos.blackhole.qwen36.tt.spec_fused import FusedGreedyDevice, fused_greedy_reference
from models.demos.blackhole.qwen36.tt.tp_common import spec_fused_enabled

NB, K, T = 8, 3, 4
VFY, FA, DRAFT = 7, 12, 11  # trace ids


class _Rig:
    """Engine + model + fake device. ``ids`` (the verify argmax row the device 'computes') is set per step."""

    def __init__(self, monkeypatch, B, fused=True, scratch=5000):
        self.B, self.R = B, B * T
        self.log = []
        e = _drafter_engine(B, K, packed=True, nb=NB)
        m, _ = _hostset_vfy_model(B, T, fused=True, nb=NB)
        m._vfy_sharded, m._vfy_logits_trace_id, m._vfy_topk_trace_id, m._vfy_narrow_ready = False, None, None, False
        m._fit_spec_page_table = lambda pt, n: torch.as_tensor(pt, dtype=torch.int32)[:, :n]
        m._vfy_ids_out.name = "ids"
        e.model = m
        del e.__dict__["propose"], e.__dict__["_ids_to_host"]
        e.recorded_drafter, e.captured, e.prepared = True, True, True
        e._dr_trace, e._vfeed, e.eager_reseed_len, e._phys_perm = DRAFT, "vfeed", 10**9, None
        e._row_active, e._last_pt, e._last_cap_blocks, e._scratch_block = None, None, NB, scratch
        e._active, e._fresh, e._parked, e._prompts, e._fresh_dirty = set(range(B)), {}, {}, {}, set()
        e._speculated, e._withheld_streak, e.no_draft_streak, e._timing = set(), {}, 16, False
        e._gdn, e.fast_join, e._kc, e._fj_sel = [], False, 4, None
        e._stage_cache, e._no_draft, e._hp_buf = {}, set(), "hp"
        e._mtp_pt_rows = [_Buf(f"ptrow{u}") for u in range(B)]
        e._dr_outs = [_Buf(f"out{k}") for k in range(K)]
        for k, o in enumerate(e._dr_outs):
            o.data = torch.arange(32, dtype=torch.int32) * (k + 3) % 1000 + 1
        e._eager_called = []
        e._set_anchor = e._reseed_mtp = e._reseed_mtp_batched = e._draft = lambda *a, **k: e._eager_called.append(1)
        e._stage_mtp_tables = e._warm_mtp_last = lambda *a, **k: None
        self.e, self.m = e, m
        self.corrupt = False
        log = self.log
        _fake_ttnn(monkeypatch, log, m)
        pending = []

        class _HostCopy:  # a non-blocking D2H copy: its data lands only when a synchronize runs after the enqueue
            def __init__(self, t):
                self.name, self.src, self.data = t.name, t, None

        def from_device(t, blocking=True):
            log.append(("from_device", t.name, blocking))
            if blocking:
                return t
            h = _HostCopy(t)
            pending.append(h)
            return h

        def sync(d):
            log.append(("sync",))
            for h in pending:
                h.data = h.src.data
            del pending[:]

        def to_torch(t, *a, **k):
            log.append(("to_torch", t.name))
            assert t.data is not None, f"to_torch({t.name}): non-blocking read never synchronized"
            return t.data

        monkeypatch.setattr(ttnn, "from_device", from_device)
        monkeypatch.setattr(ttnn, "to_torch", to_torch)
        monkeypatch.setattr(ttnn, "synchronize_device", sync)
        monkeypatch.setattr(ttnn, "execute_trace", self._exec)
        monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [t])
        self.fa = None
        if fused:
            fa = FusedGreedyDevice()
            fa.mesh, fa.B, fa.K, fa.T, fa.R, fa.nb, fa.W = "mesh", B, K, T, B * T, NB, 32
            fa.aux_in, fa.pt_rows_in, fa.result_out = _Buf("fa_aux"), _Buf("fa_ptrows"), _Buf("fa_result")
            fa.ids_in, fa.tok_in, fa.pack_out, fa.rs_pt_out = m._vfy_ids_out, m._vfy_token_buf, e._dr_pack, e._dr_rs_pt
            e._fa, e._fa_trace = fa, FA
            self.fa = fa
        monkeypatch.setattr(se_mod, "note_withheld_drafts", lambda *a, **k: self.no_draft)
        self.no_draft = set()

    def _exec(self, dev, tid, cq_id=0, blocking=False):
        self.log.append(("trace", tid))
        if tid == FA:
            fa, B = self.fa, self.B
            aux = fa.aux_in.data
            pt = fa.pt_rows_in.data[::T]
            ref = fused_greedy_reference(
                self.m._vfy_ids_out.data.reshape(-1), self.m._vfy_token_buf.data, aux, pt, B=B, K=K
            )
            res = torch.zeros(2, 32, dtype=torch.int32)
            res[0, : self.R] = ref.committed.reshape(-1)
            res[1, :B] = ref.counts
            if self.corrupt:
                res[1, 0] = 1 if int(res[1, 0]) != 1 else 2
            fa.result_out.data = res

    def set_ids(self, rows):
        ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        ids[0, 0, 0, : self.R] = torch.as_tensor(rows, dtype=torch.int32).reshape(-1)
        self.m._vfy_ids_out.data = ids

    def step(self, tok, pos, nvd, acc, pt, spec_mode="fused_sample"):
        del self.log[:]
        return self.e.decode_forward(
            tok, pos, num_valid_drafts=nvd, accepted_counts=acc, page_table=pt, spec_mode=spec_mode
        )

    def names(self, kind):
        return [x[1] for x in self.log if x[0] == kind]


def _block(B, nv, base=50, stride=10):
    """Runner block: row u has nv[u] drafts (nv < 0: padding row); tokens u*100 + j + 1."""
    tok = torch.arange(1, T + 1, dtype=torch.int32)[None, :] + 100 * torch.arange(B, dtype=torch.int32)[:, None]
    pos = torch.full((B, T), -1, dtype=torch.int32)
    nvd = torch.zeros(B, dtype=torch.int32)
    for u in range(B):
        if nv[u] >= 0:
            pos[u, : nv[u] + 1] = base + stride * u + torch.arange(nv[u] + 1, dtype=torch.int32)
            nvd[u] = nv[u]
    return tok, pos, nvd


def _ids_for(tok, nvd, accept):
    """verify argmax ids [B, T] such that row u accepts accept[u] drafts: ids[u, j] == tok[u, j+1] for j < accept[u]."""
    ids = (tok.clone() + 1000) % 997 + 5
    for u in range(tok.shape[0]):
        for j in range(min(int(accept[u]), int(nvd[u]))):
            ids[u, j] = tok[u, j + 1]
    return ids


def _pt(B, seed=0):
    return torch.randint(1, 4000, (B, NB), generator=torch.Generator().manual_seed(seed), dtype=torch.int32)


def _expected_commit(ids, tok, nvd):
    committed = torch.full_like(ids, -1)
    counts = torch.ones(ids.shape[0], dtype=torch.int32)
    for u in range(ids.shape[0]):
        n = 0
        while n < int(nvd[u]) and int(ids[u, n]) == int(tok[u, n + 1]):
            n += 1
        counts[u] = n + 1
        committed[u, : n + 1] = ids[u, : n + 1]
    return committed, counts


def _plugin_propose_args(res, tok, pos):
    """What the plugin passes to propose_draft_tokens after its walk: the committed block, positions, counts."""
    cnt = res.accepted_counts
    cpos = torch.full_like(res.accepted_token_ids, -1)
    for u in range(cnt.shape[0]):
        if int(pos[u, 0]) >= 0:
            cpos[u, : int(cnt[u])] = int(pos[u, 0]) + 1 + torch.arange(int(cnt[u]), dtype=torch.int32)
    return res.accepted_token_ids.clone(), cpos, cnt.clone()


# --------------------------------------------------------------------------- #
# (i) call sequence
# --------------------------------------------------------------------------- #
def test_fused_step_call_sequence(monkeypatch):
    B = 3
    r = _Rig(monkeypatch, B)
    tok, pos, nvd = _block(B, [3, 2, 1])
    acc, pt = torch.ones(B, dtype=torch.int32), _pt(B)
    r.set_ids(_ids_for(tok, nvd, [3, 1, 0]))
    res = r.step(tok, pos, nvd, acc, pt)
    kinds = [x[0] for x in r.log]
    # aux, pt rows, mtp table; the verify's own copies (3 tables, tokens, aux words); the 3 traces; ONE sync; reads
    assert [x[:2] for x in r.log] == [
        ("h2d", "fa_aux"),
        ("h2d", "fa_ptrows"),
        ("h2d", "mtp"),
        ("h2d", "kvpt"),
        ("h2d", "kvpt_users"),
        ("h2d", "kvpt1"),
        ("h2d", "token"),
        ("h2d", "aux"),
        ("trace", VFY),
        ("trace", FA),
        ("trace", DRAFT),
        *[("from_device", f"out{k}") for k in range(K)],  # every non-blocking read precedes the sync
        ("sync",),
        ("to_torch", "ids"),
        ("to_torch", "fa_result"),
        *[("to_torch", f"out{k}") for k in range(K)],
    ]
    assert all(x[2] is False for x in r.log if x[0] == "from_device")
    assert kinds.count("sync") == 1
    assert not {"pack", "rs_pt"} & set(r.names("h2d")), "the device wrote the pack and the reseed table"
    assert r.m._vfy_gdn[0]._spec_win_par == 1  # parity flipped once by verify_after_external_sync
    assert r.e._dr_rs_dirty and r.e.fused_steps == 1 and not r.e.__dict__.get("fused_fallbacks")
    # same table, same declined rows: neither the pt rows nor the mtp table nor the verify tables are copied again
    r.step(tok, pos + 1, nvd, res.accepted_counts, pt)
    assert r.names("h2d") == ["fa_aux", "token", "aux"]
    assert [x[0] for x in r.log].count("sync") == 1
    # a changed table is copied everywhere
    r.step(tok, pos + 1, nvd, res.accepted_counts, _pt(B, 1))
    assert r.names("h2d")[:3] == ["fa_aux", "fa_ptrows", "mtp"]
    assert "kvpt" in r.names("h2d")


def test_fused_result_equals_host_walk_and_propose_bookkeeping(monkeypatch, expect_error):
    B = 3
    r = _Rig(monkeypatch, B)
    tok, pos, nvd = _block(B, [3, 2, 1])
    ids = _ids_for(tok, nvd, [3, 1, 0])
    r.set_ids(ids)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B))
    committed, counts = _expected_commit(ids, tok, nvd)
    assert torch.equal(res.accepted_token_ids, committed) and torch.equal(res.accepted_counts, counts)
    assert counts.tolist() == [4, 2, 1]
    assert r.e._offered == {0, 1, 2}
    assert r.e.mtp_extra_steps == 1
    # the debug check fires on a device result that differs from the host walk
    r.corrupt = True
    with expect_error(RuntimeError, "fused accept != host greedy walk"):
        r.step(tok, pos + 1, nvd, counts, _pt(B))


# --------------------------------------------------------------------------- #
# (ii) fallbacks
# --------------------------------------------------------------------------- #
def _fallback_case(monkeypatch, reason):
    B = 1 if reason in ("eager_drafter", "skip_all") else 3
    r = _Rig(monkeypatch, B)
    e = r.e
    nv, base = [3, 2, 1][:B], 50
    if reason == "not_captured":
        e._fa_trace = None
    elif reason == "eager_drafter":
        e.eager_reseed_len = 40  # B == 1, p = 50 + m > 40
    elif reason == "skip_all":
        r.no_draft = {0}
    elif reason == "mixed_fit":
        nv, base = [0, 0, 0], 507  # fits at m = 0 only: draft_rows_fit depends on m
    elif reason == "all_declined":
        nv, base = [0, 0, 0], 508  # no m fits any row
    elif reason == "disabled":
        e._fa = None
    tok, pos, nvd = _block(B, nv, base=base, stride=0 if reason in ("mixed_fit", "all_declined") else 10)
    return r, tok, pos, nvd, B


@pytest.mark.parametrize(
    "reason", ["not_captured", "disabled", "eager_drafter", "skip_all", "mixed_fit", "all_declined"]
)
def test_fallback_runs_the_argmax_path(monkeypatch, reason):
    r, tok, pos, nvd, B = _fallback_case(monkeypatch, reason)
    e = r.e
    ids = _ids_for(tok, nvd, [2] * B)
    r.set_ids(ids)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B))
    want = "not_captured" if reason == "disabled" else reason
    assert e.__dict__["fused_fallbacks"] == {want: 1} and e.fused_steps == 1
    # the plain verify: its copies, ONE trace, one sync, the ids read; no accept / drafter work, no fa copies
    assert ("trace", FA) not in r.log and ("trace", DRAFT) not in r.log
    assert [x[0] for x in r.log].count("sync") == 1 and r.names("to_torch") == ["ids"]
    assert not any(n.startswith("fa_") for n in r.names("h2d"))
    committed, counts = _expected_commit(ids, tok, nvd)
    assert torch.equal(res.accepted_token_ids, committed) and torch.equal(res.accepted_counts, counts)
    assert e._fused_cache is None
    # propose then runs normally (full pack copy and the drafter replay, unless every row declines)
    cp, cpos, cc = _plugin_propose_args(res, tok, pos)
    del r.log[:]
    out = e.propose_draft_tokens(K, cp, cpos, cc)
    assert isinstance(out, DraftResult)
    if reason in ("skip_all", "all_declined"):
        assert ("trace", DRAFT) not in r.log and out.num_valid.tolist() == [0] * B
    elif reason == "eager_drafter":  # the eager drafter runs (its device pieces are stand-ins here), no trace replay
        assert ("trace", DRAFT) not in r.log and e._eager_called
    else:
        assert ("trace", DRAFT) in r.log and "pack" in r.names("h2d")


def test_fallback_counts_and_log_every_256_steps(monkeypatch):
    from loguru import logger

    B = 2
    r = _Rig(monkeypatch, B)
    r.e._fa_trace = None
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="INFO")
    try:
        tok, pos, nvd = _block(B, [1, 1])
        r.set_ids(_ids_for(tok, nvd, [1, 0]))
        for _ in range(256):
            r.e.decode_forward(
                tok,
                pos,
                num_valid_drafts=nvd,
                accepted_counts=torch.ones(B, dtype=torch.int32),
                page_table=_pt(B),
                spec_mode="fused_sample",
            )
    finally:
        logger.remove(sink)
    lines = [m for m in msgs if "fused steps" in m]
    assert len(lines) == 1 and "fallbacks 256: not_captured=256" in lines[0] and "\n" not in lines[0].strip()


# --------------------------------------------------------------------------- #
# (iii) cache
# --------------------------------------------------------------------------- #
def test_cache_hit_returns_the_same_draft_result_without_device_work(monkeypatch):
    B = 3
    r = _Rig(monkeypatch, B)
    tok, pos, nvd = _block(B, [3, 2, 1])
    ids = _ids_for(tok, nvd, [3, 1, 0])
    r.set_ids(ids)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B))
    cp, cpos, cc = _plugin_propose_args(res, tok, pos)
    cached = r.e._fused_cache
    assert cached is not None
    del r.log[:]
    out = r.e.propose_draft_tokens(K, cp, cpos, cc)
    assert r.log == [] and r.e._fused_cache is None
    assert torch.equal(out.draft_token_ids, cached[0].draft_token_ids) and torch.equal(
        out.num_valid, cached[0].num_valid
    )
    # what the normal propose makes for the same commit (same fake drafter outputs)
    ref = _Rig(monkeypatch, B, fused=False)
    ref.set_ids(ids)
    ref_res = ref.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), spec_mode="argmax_ids")
    ref_out = ref.e.propose_draft_tokens(K, cp, cpos, cc)
    assert torch.equal(out.draft_token_ids, ref_out.draft_token_ids) and torch.equal(out.num_valid, ref_out.num_valid)
    assert ref_out.draft_token_ids.dtype == out.draft_token_ids.dtype == torch.int32
    assert torch.equal(ref_res.argmax_ids, res.argmax_ids)


@pytest.mark.parametrize("what", ["count", "token", "position", "explicit_table"])
def test_cache_mismatch_runs_the_normal_propose(monkeypatch, what):
    B = 3
    r = _Rig(monkeypatch, B)
    tok, pos, nvd = _block(B, [3, 2, 1])
    r.set_ids(_ids_for(tok, nvd, [3, 1, 0]))
    pt = _pt(B)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), pt)
    cp, cpos, cc = _plugin_propose_args(res, tok, pos)
    kw = {}
    if what == "count":  # e.g. the plugin truncated the commit at a stop token
        cc[0], cp[0, 3], cpos[0, 3] = 2, -1, -1
    elif what == "token":
        cp[1, 0] += 1
    elif what == "position":
        cpos[2, 0] += 1
    else:
        kw["page_table"] = pt
    del r.log[:]
    out = r.e.propose_draft_tokens(K, cp, cpos, cc, **kw)
    assert ("trace", DRAFT) in r.log and "pack" in r.names("h2d")
    assert "rs_pt" in r.names("h2d"), "the device rewrote the reseed table: a normal propose restages it whole"
    assert r.e._fused_cache is None and out.num_valid.tolist() == [K] * B
    # a second propose never sees a stale cache
    del r.log[:]
    r.e.propose_draft_tokens(K, *_plugin_propose_args(res, tok, pos))
    assert ("trace", DRAFT) in r.log


def test_next_decode_forward_clears_the_cache(monkeypatch):
    B = 2
    r = _Rig(monkeypatch, B)
    tok, pos, nvd = _block(B, [3, 2])
    r.set_ids(_ids_for(tok, nvd, [1, 1]))
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B))
    assert r.e._fused_cache is not None
    r.set_ids(_ids_for(tok, nvd, [0, 0]))
    r.step(tok, pos, nvd, res.accepted_counts, _pt(B), spec_mode="argmax_ids")
    assert r.e._fused_cache is None


# --------------------------------------------------------------------------- #
# (iv) output fields
# --------------------------------------------------------------------------- #
def test_fused_output_fields_shapes_and_padding_rows(monkeypatch):
    B = 4
    r = _Rig(monkeypatch, B)
    tok, pos, nvd = _block(B, [3, -1, 1, 0])  # row 1 is a padding row
    ids = _ids_for(tok, nvd, [3, 2, 1, 0])
    r.set_ids(ids)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B))
    assert res.spec_mode == "fused_sample"
    assert res.argmax_ids.shape == (B, T) and res.argmax_ids.dtype == torch.int32
    assert res.accepted_token_ids.shape == (B, T) and res.accepted_token_ids.dtype == torch.int32
    assert res.accepted_counts.shape == (B,) and res.accepted_counts.dtype == torch.int32
    assert torch.equal(res.argmax_ids, mask_unused_columns(ids, nvd))
    assert res.accepted_counts.tolist() == [4, 1, 2, 1]  # padding row: count 1
    assert res.accepted_token_ids[1].tolist() == [int(ids[1, 0]), -1, -1, -1]  # column-0 argmax, then -1
    assert res.accepted_token_ids[0].tolist() == ids[0].tolist()
    assert res.accepted_token_ids[2].tolist() == [int(ids[2, 0]), int(ids[2, 1]), -1, -1]
    assert int(res.accepted_counts.min()) >= 1 and int(res.accepted_counts.max()) <= T
    # the padding row is declined by the propose: no drafts for it, the others draft
    cp, cpos, cc = _plugin_propose_args(res, tok, pos)
    out = r.e.propose_draft_tokens(K, cp, cpos, cc)
    assert out.num_valid.tolist() == [K, 0, K, K] and out.draft_token_ids[1].tolist() == [0] * K


def test_fused_aux_words_staged(monkeypatch):
    B = 3
    r = _Rig(monkeypatch, B)
    tok, pos, nvd = _block(B, [3, -1, 2])
    r.set_ids(_ids_for(tok, nvd, [1, 0, 0]))
    r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B))
    w = r.fa.aux_in.data.to(torch.int64) & 0xFFFFFFFF
    assert w[0, :B].tolist() == [3, 0, 2] and w[1, :B].tolist() == [1, 0, 1] and w[2, :B].tolist() == [0, 1, 0]
    assert w[4, :B].tolist() == [50, 0, 70] and w[5, :5].tolist() == [5000, 0, 1024, B, K]
    assert r.fa.pt_rows_in.data.shape == (B * T, NB)
    assert torch.equal(r.fa.pt_rows_in.data[::T], r.e._last_pt)


# --------------------------------------------------------------------------- #
# (v) flag off = nothing changes
# --------------------------------------------------------------------------- #
def test_flag_default_off_and_argmax_flow_untouched(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_FUSED", raising=False)
    assert not spec_fused_enabled()
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    assert spec_fused_enabled()
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "0")
    assert not spec_fused_enabled()
    B = 3
    logs = []
    for fused in (False, True):  # an engine without the fused objects and one with them, both asked for argmax_ids
        r = _Rig(monkeypatch, B, fused=fused)
        tok, pos, nvd = _block(B, [3, 2, 1])
        r.set_ids(_ids_for(tok, nvd, [3, 1, 0]))
        res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), spec_mode="argmax_ids")
        assert res.spec_mode == "argmax_ids" and res.accepted_token_ids is None and r.e._fused_cache is None
        cp, cpos, cc = _plugin_propose_args(_expected_res(tok, nvd, [3, 1, 0]), tok, pos)
        r.e.propose_draft_tokens(K, cp, cpos, cc)
        logs.append(r.log[:])
        assert r.e.fused_steps == 0 and not any(x[1] in ("fa_aux", "fa_ptrows") for x in r.log if x[0] == "h2d")
        assert ("trace", FA) not in r.log and not r.e._dr_rs_dirty
    assert logs[0] == logs[1]


def _expected_res(tok, nvd, accept):
    c, n = _expected_commit(_ids_for(tok, nvd, accept), tok, nvd)
    return types.SimpleNamespace(accepted_token_ids=c, accepted_counts=n)


def test_bound_buffer_checks(expect_error):
    ok = types.SimpleNamespace(shape=(1, 12), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    FusedGreedyDevice._check_bound("tok_in", ok, (1, 12), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    for bad in (
        types.SimpleNamespace(shape=(1, 16), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        types.SimpleNamespace(shape=(1, 12), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        types.SimpleNamespace(shape=(1, 12), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT),
    ):
        with expect_error(AssertionError, "bound tok_in"):
            FusedGreedyDevice._check_bound("tok_in", bad, (1, 12), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)


# =========================================================================== #
# Stage S2: the SAMPLED fused step (spec_sampling), same fakes plus a fake sampled accept trace
# =========================================================================== #
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import (  # noqa: E402
    FusedSampledDevice,
    fused_sampled_reference,
    host_redo_walk,
    pack_samp_host,
    predraw_uniforms,
)

FS, TOPK, LOGIT = 13, 14, 15
S_TP, S_M, S_V = 2, 8, 64  # TP shards, per-shard top-M, vocab (per shard 32)
S_PER = S_V // S_TP


def _union_from_logits(logits):
    """The topk trace's per-shard top-M blocks of bf16 logits [R, V]: vals bf16 [R, TP*M], local ids int32 (value desc, id asc)."""
    R = logits.shape[0]
    vals, ids = [], []
    for d in range(S_TP):
        blk = logits[:, d * S_PER : (d + 1) * S_PER].to(torch.bfloat16).float()
        o = torch.sort(blk, dim=1, descending=True, stable=True).indices[:, :S_M]
        vals.append(blk.gather(1, o).to(torch.bfloat16))
        ids.append(o.to(torch.int32))
    return torch.cat(vals, 1), torch.cat(ids, 1)


class _SampledRig(_Rig):
    """_Rig + the S2 objects: a FusedSampledDevice shell sharing the S1 buffers, the topk / logits trace ids and a fake sampled accept
    trace that runs fused_sampled_reference on the staged buffers. ``set_logits`` sets what the verify 'computed' (full logits;
    the union and the argmax ids derive from it)."""

    def __init__(self, monkeypatch, B, with_fs=True):
        super().__init__(monkeypatch, B)
        e, m, fa = self.e, self.m, self.fa
        e.vocab = S_V
        m.vocab_size, m.num_devices, m._vfy_topk, m._vfy_topk_m = S_V, S_TP, True, S_M
        m._vfy_topk_trace_id, m._vfy_logits_trace_id = TOPK, LOGIT
        m._vfy_topk_vals_out, m._vfy_topk_ids_out = _Buf("uvals"), _Buf("uids")
        self.fs = None
        if with_fs:
            fs = FusedSampledDevice()
            fs.mesh, fs.B, fs.K, fs.T, fs.R, fs.nb, fs.W = "mesh", B, K, T, B * T, NB, 32
            fs.TP, fs.M, fs.per_shard = S_TP, S_M, S_PER
            fs.aux_in, fs.pt_rows_in, fs.ids_in, fs.tok_in = fa.aux_in, fa.pt_rows_in, m._vfy_ids_out, m._vfy_token_buf
            fs.pack_out, fs.rs_pt_out = e._dr_pack, e._dr_rs_pt
            fs.samp_in, fs.result_out = _Buf("fs_samp"), _Buf("fs_result")
            fs.union_vals_in, fs.union_ids_in = m._vfy_topk_vals_out, m._vfy_topk_ids_out
            e._fs, e._fs_trace = fs, FS
            self.fs = fs

    def set_logits(self, logits):
        self.logits = logits.to(torch.bfloat16)
        m = self.m
        pad = torch.zeros(1, 1, 32, S_V, dtype=torch.bfloat16)
        pad[0, 0, : self.R] = self.logits
        m._vfy_logits_rm_out.data = pad
        vals, ids = _union_from_logits(self.logits)
        uv = torch.zeros(1, 1, 32, S_TP * S_M, dtype=torch.bfloat16)
        ui = torch.zeros(1, 1, 32, S_TP * S_M, dtype=torch.int32)
        uv[0, 0, : self.R], ui[0, 0, : self.R] = vals, ids
        m._vfy_topk_vals_out.data, m._vfy_topk_ids_out.data = uv, ui
        am = torch.zeros(self.R, dtype=torch.int32)
        for r in range(self.R):
            row = self.logits[r].float()
            am[r] = int((row == row.max()).nonzero()[0])
        self.set_ids(am)
        return am

    def _exec(self, dev, tid, cq_id=0, blocking=False):
        super()._exec(dev, tid, cq_id, blocking)
        if tid == FS:
            fs, B = self.fs, self.B
            sp = fs.samp_in.data
            ref = fused_sampled_reference(
                fs.union_vals_in.data.reshape(32, -1)[: self.R],
                fs.union_ids_in.data.reshape(32, -1)[: self.R],
                self.m._vfy_token_buf.data,
                self.m._vfy_ids_out.data.reshape(-1),
                fs.aux_in.data,
                fs.pt_rows_in.data[::T],
                B=B,
                K=K,
                per_shard=S_PER,
                M=S_M,
                greedy=sp[3, :B] > 0.5,
                temperature=sp[0, :B],
                top_k=sp[1, :B].to(torch.int64),
                top_p=sp[2, :B],
                uniforms=sp[4 : 4 + B, : K + 2],
            )
            res = torch.zeros(3, 32, dtype=torch.int32)
            res[0, : self.R] = ref.committed.reshape(-1)
            res[1, :B], res[2, :B] = ref.counts, ref.overflow
            fs.result_out.data = res

    def step(self, tok, pos, nvd, acc, pt, sampling=None, spec_mode="fused_sample"):
        del self.log[:]
        kw = {} if sampling is None else {"spec_sampling": sampling}
        return self.e.decode_forward(
            tok, pos, num_valid_drafts=nvd, accepted_counts=acc, page_table=pt, spec_mode=spec_mode, **kw
        )


def _sampling(B, nv, temps, topk=3, topp=1.0, seed=0, seeded=()):
    """The plugin's spec_sampling for a step: temps [B] (0 = greedy); uniforms from predraw_uniforms (live = all rows)."""
    t = torch.tensor(temps, dtype=torch.float32)
    g = {u: torch.Generator().manual_seed(100 + u) for u in seeded}
    torch.manual_seed(seed)
    return {
        "temperature": t,
        "top_k": torch.where(t > 0, torch.full((B,), topk, dtype=torch.int32), torch.ones(B, dtype=torch.int32)),
        "top_p": torch.full((B,), topp, dtype=torch.float32),
        "greedy": t <= 0,
        "uniforms": predraw_uniforms(g, B, K, torch.as_tensor(nv)),
    }


def _smooth_logits(R, seed):
    """Distinct bf16 logits (no ties), so no row overflows: a random permutation of a fixed spread."""
    g = torch.Generator().manual_seed(seed)
    base = torch.linspace(6.0, -6.0, S_V)
    return torch.stack([base[torch.randperm(S_V, generator=g)] for _ in range(R)])


def _draft_block(logits, B, nv, hit_rows=()):
    """Runner block whose drafts follow the logits' top ids (so some accept): draft j of user u = rank-(u+j)%3 of row u*T+j."""
    tok, pos, nvd = _block(B, nv)
    for u in range(B):
        for j in range(T - 1):
            order = torch.sort(logits[u * T + j].float(), descending=True, stable=True).indices
            tok[u, j + 1] = int(order[(u + j) % 3])
    return tok, pos, nvd


def test_sampled_step_call_sequence(monkeypatch):
    B = 3
    r = _SampledRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 1)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    ss = _sampling(B, nvd, [0.8, 0.0, 1.2])
    r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
    assert [x[:2] for x in r.log] == [
        ("h2d", "fa_aux"),
        ("h2d", "fs_samp"),
        ("h2d", "fa_ptrows"),
        ("h2d", "mtp"),
        ("h2d", "kvpt"),
        ("h2d", "kvpt_users"),
        ("h2d", "kvpt1"),
        ("h2d", "token"),
        ("h2d", "aux"),
        ("trace", VFY),
        ("trace", TOPK),
        ("trace", FS),
        ("trace", DRAFT),
        *[("from_device", f"out{k}") for k in range(K)],
        ("sync",),
        ("to_torch", "ids"),
        ("to_torch", "fs_result"),
        *[("to_torch", f"out{k}") for k in range(K)],
        ("to_torch", "uvals"),  # audit (default on): the union, read after the sync
        ("to_torch", "uids"),
    ]
    assert [x[0] for x in r.log].count("sync") == 1
    assert ("trace", FA) not in r.log and ("trace", LOGIT) not in r.log
    assert all(x[2] is False for x in r.log if x[0] == "from_device")
    assert not {"pack", "rs_pt"} & set(r.names("h2d"))
    assert r.m._vfy_gdn[0]._spec_win_par == 1 and r.e._dr_rs_dirty
    assert r.e.fused_steps == 1 and not r.e.__dict__.get("fused_fallbacks") and r.e.fused_boundary_diffs == 0
    # the staged samp tensor is pack_samp_host of the plugin's controls (engine rows = runner rows at the identity map)
    want = pack_samp_host(ss["temperature"], ss["top_k"], ss["top_p"], ss["greedy"], ss["uniforms"], 32)
    assert torch.equal(r.fs.samp_in.data, want)


def test_sampled_step_audit_off_reads_no_union(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "0")
    B = 2
    r = _SampledRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 2)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2])
    r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), _sampling(B, nvd, [0.9, 0.9]))
    assert "uvals" not in r.names("to_torch") and "uids" not in r.names("to_torch")


def test_sampled_audit_every_4_reads_the_union_on_every_4th_step(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT_EVERY", "4")
    B = 2
    r = _SampledRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 3)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2])
    ss = _sampling(B, nvd, [0.9, 0.9])
    audited = []
    for _ in range(8):
        r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
        audited.append("uvals" in r.names("to_torch") and "uids" in r.names("to_torch"))
    assert audited == [True, False, False, False, True, False, False, False]
    assert r.e.fused_audited_steps == 2 and r.e.fused_sampled_steps == 8


@pytest.mark.parametrize("seed", range(6))
def test_sampled_result_equals_the_host_walk_on_full_logits(monkeypatch, seed):
    B = 4
    r = _SampledRig(monkeypatch, B)
    logits = _smooth_logits(B * T, seed)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 0, 1])
    ss = _sampling(B, nvd, [0.7, 0.0, 1.1, 1.5], topk=4, topp=0.9, seed=seed, seeded=(0, 3))
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
    committed, counts = host_redo_walk(r.logits, tok, ss, ss["uniforms"], B=B, K=K, num_valid=nvd)
    assert torch.equal(res.accepted_token_ids, committed) and torch.equal(res.accepted_counts, counts)
    # the propose bookkeeping ran from the device commit: drafts cached for the plugin's propose
    assert r.e._fused_cache is not None and r.e.fused_steps == 1
    cp, cpos, cc = _plugin_propose_args(res, tok, pos)
    n_before = len(r.log)
    out = r.e.propose_draft_tokens(K, cp, cpos, cc)
    assert len(r.log) == n_before, "the cache hit does no device work"
    assert out.draft_token_ids.shape == (B, K)


def test_greedy_only_steps_still_use_the_s1_trace(monkeypatch):
    B = 3
    r = _SampledRig(monkeypatch, B)
    tok, pos, nvd = _block(B, [3, 2, 1])
    ids = _ids_for(tok, nvd, [3, 1, 0])
    r.set_ids(ids)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B))  # no spec_sampling
    assert ("trace", FA) in r.log and ("trace", FS) not in r.log and ("trace", TOPK) not in r.log
    assert "fs_samp" not in r.names("h2d")
    c, n = _expected_commit(ids, tok, nvd)
    assert torch.equal(res.accepted_token_ids, c) and torch.equal(res.accepted_counts, n)
    # spec_sampling with every row greedy is the same S1 step
    r.set_ids(ids)
    r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), _sampling(B, nvd, [0.0, 0.0, 0.0]))
    assert ("trace", FA) in r.log and ("trace", FS) not in r.log


def _plateau_logits(R, seed):
    """Every row: two high distinct logits, then a plateau of equal values over all other ids: with top_k = 3 the tie group at the
    k-th value reaches the last union candidate (overflow)."""
    out = torch.full((R, S_V), -2.0)
    g = torch.Generator().manual_seed(seed)
    for r_ in range(R):
        a, b = torch.randperm(S_V, generator=g)[:2].tolist()
        out[r_, a], out[r_, b] = 5.0, 4.0
    return out


def test_overflow_redoes_the_step_on_the_host_and_skips_the_drafter_cache(monkeypatch):
    B = 3
    r = _SampledRig(monkeypatch, B)
    logits = _plateau_logits(B * T, 3)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    ss = _sampling(B, nvd, [0.8, 1.0, 0.0], topk=3)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
    kinds = [x[0] for x in r.log]
    assert r.e.fused_fallbacks == {"overflow_redo": 1} and r.e.fused_steps == 1
    # after the fused step's single sync: the logits trace replay (+ its own sync) and the logits read
    i = r.log.index(("trace", LOGIT))
    assert [x[0] for x in r.log[i : i + 3]] == ["trace", "sync", "to_torch"] and r.log[i + 2][1] == "logits"
    assert kinds.count("sync") == 2
    assert "uvals" not in r.names("to_torch"), "no audit on a redo step"
    committed, counts = host_redo_walk(r.logits, tok, ss, ss["uniforms"], B=B, K=K, num_valid=nvd)
    assert torch.equal(res.accepted_token_ids, committed) and torch.equal(res.accepted_counts, counts)
    assert r.e._fused_cache is None, "the plugin's propose runs the normal path"
    # the normal propose, with the plugin's commit, then drafts for every row
    cp, cpos, cc = _plugin_propose_args(res, tok, pos)
    out = r.e.propose_draft_tokens(K, cp, cpos, cc)
    assert ("trace", DRAFT) in r.log[r.log.index(("sync",)) :] or r.e._eager_called is not None
    assert out.num_valid.tolist() == [K, K, K]


def test_no_sampled_trace_walks_the_full_logits_on_the_host(monkeypatch):
    B = 3
    r = _SampledRig(monkeypatch, B, with_fs=False)  # QWEN36_SPEC_FUSED_SAMPLED=0, or no topk trace
    logits = _smooth_logits(B * T, 5)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    ss = _sampling(B, nvd, [0.8, 0.0, 1.2])
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
    assert r.e.fused_fallbacks == {"no_sampled_trace": 1}
    assert ("trace", FS) not in r.log and ("trace", FA) not in r.log
    committed, counts = host_redo_walk(r.logits, tok, ss, ss["uniforms"], B=B, K=K, num_valid=nvd)
    assert torch.equal(res.accepted_token_ids, committed) and torch.equal(res.accepted_counts, counts)
    assert r.e._fused_cache is None


def test_top_k_outside_the_union_walks_on_the_host(monkeypatch):
    B = 2
    r = _SampledRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 6)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2])
    ss = _sampling(B, nvd, [0.8, 0.8], topk=S_M)  # M, not M - 1: cannot run fused
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
    assert r.e.fused_fallbacks == {"top_k_range": 1}
    committed, counts = host_redo_walk(r.logits, tok, ss, ss["uniforms"], B=B, K=K, num_valid=nvd)
    assert torch.equal(res.accepted_token_ids, committed) and torch.equal(res.accepted_counts, counts)


def test_the_audit_raises_on_a_device_result_the_reference_does_not_give(monkeypatch, expect_error):
    B = 2
    r = _SampledRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 7)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 3])
    ss = _sampling(B, nvd, [0.9, 0.9], topk=4)
    real = r._exec

    def corrupt(dev, tid, cq_id=0, blocking=False):
        real(dev, tid, cq_id, blocking)
        if tid == FS:
            res = r.fs.result_out.data
            res[1, 0] = 1 if int(res[1, 0]) != 1 else 2

    monkeypatch.setattr(ttnn, "execute_trace", corrupt)
    with expect_error(RuntimeError, "fused sampled accept != host reference"):
        r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)


def test_the_audit_counts_a_boundary_difference(monkeypatch):
    B = 1
    r = _SampledRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 8)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3])
    ss = _sampling(B, nvd, [0.9], topk=4)

    def probe(uniforms):
        return fused_sampled_reference(
            r.m._vfy_topk_vals_out.data.reshape(32, -1)[: B * T],
            r.m._vfy_topk_ids_out.data.reshape(32, -1)[: B * T],
            tok.reshape(-1),
            r.m._vfy_ids_out.data.reshape(-1),
            _aux_for(tok, pos, nvd, B),
            _pt(B),
            B=B,
            K=K,
            per_shard=S_PER,
            M=S_M,
            greedy=ss["greedy"],
            temperature=ss["temperature"],
            top_k=ss["top_k"],
            top_p=ss["top_p"],
            uniforms=uniforms,
            num_valid=nvd,
        )

    p_d = float(probe(ss["uniforms"]).p_d[0, 0])
    assert 0 < p_d < 1
    ss["uniforms"][0, 1] = p_d + 5e-6  # the reference REJECTS draft 0 (u > p_d) ...
    ref = probe(ss["uniforms"])
    assert int(ref.counts[0]) == 1
    real = r._exec

    def hook(dev, tid, cq_id=0, blocking=False):
        if tid == FS:  # ... the 'device' saw u a float32 hair lower and ACCEPTS it: a boundary difference
            r.fs.samp_in.data[4, 1] = p_d - 5e-6
        real(dev, tid, cq_id, blocking)

    monkeypatch.setattr(ttnn, "execute_trace", hook)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
    assert int(res.accepted_counts[0]) > 1 and r.e.fused_boundary_diffs == 1


def _aux_for(tok, pos, nvd, B):
    from models.demos.blackhole.qwen36.tt.spec_fused import fused_accept_aux_host

    start0 = pos[:, 0].clamp(min=0)
    return fused_accept_aux_host(nvd, start0, pos[:, 0] >= 0, torch.zeros(B, dtype=torch.bool), 4096, K, 5000, 0, 1024)


def test_prepare_shares_the_s1_buffers_and_capture_binds_the_topk_outputs(monkeypatch):
    B = 2
    r = _SampledRig(monkeypatch, B, with_fs=False)
    e, m, fa = r.e, r.m, r.fa
    e._fs = None
    seen = {}

    class _Stub:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.union_vals_in, self.union_ids_in = _Buf("standin_v"), _Buf("standin_i")
            self.calls = []

        def body(self):
            self.calls.append("body")

        def bind_ids(self, t):
            self.calls.append(("bind_ids", t))

        def bind_union(self, v, i):
            self.calls.append(("bind_union", v, i))

    def alloc(mesh, *a, **kw):
        seen["args"], seen["kw"] = a, kw
        return _Stub(**kw)

    monkeypatch.setattr(FusedSampledDevice, "allocate", staticmethod(alloc))
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda mesh, cq_id=0: 99, raising=False)
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda *a, **k: None, raising=False)
    monkeypatch.setenv("QWEN36_SPEC_FUSED_SAMPLED", "0")
    e._prepare_fused_sampled(32)
    assert e._fs is None
    monkeypatch.setenv("QWEN36_SPEC_FUSED_SAMPLED", "1")
    e._prepare_fused_sampled(32)
    fs = e._fs
    assert seen["args"] == (B, K, NB, 32, S_TP, S_M, S_V // S_TP)
    for name, want in (
        ("ids_in", fa.ids_in),
        ("tok_in", m._vfy_token_buf),
        ("pack_out", e._dr_pack),
        ("rs_pt_out", e._dr_rs_pt),
        ("aux_in", fa.aux_in),
        ("pt_rows_in", fa.pt_rows_in),
    ):
        assert seen["kw"][name] is want, name
    assert fs.calls == ["body"] and e._fs_standin == (fs.union_vals_in, fs.union_ids_in)
    e._capture_fused_sampled()
    assert fs.calls[1:] == [
        ("bind_ids", m._vfy_ids_out),
        ("bind_union", m._vfy_topk_vals_out, m._vfy_topk_ids_out),
        "body",
    ]
    assert e._fs_trace == 99
