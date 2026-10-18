# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the device-side top-k of the sampled verify (QWEN36_VERIFY_DEVICE_TOPK, note 26). No device.

The host merge of the per-shard candidate lists must be the exact global top-M under (value desc, token id asc)
on bf16 logits full of ties, with stable and with unstable per-shard order; the invariant the plugin's overflow
rule relies on (every token strictly above the merged M-th value is present) must hold; the pad lanes must be
caught; verify_traced must replay the topk trace in place of the logits trace and hand back the candidate pair;
decode_forward(spec_mode="topk") must deliver candidates, ids and the lazy logits in runner-slot order; and the
memory cost must count the three new buffers only under the flag. With the flag off nothing changes.

Fresh rows: under QWEN36_SPEC_FAST_JOIN=1 (default) a fresh row's first verify rides the normal replay as a 0-draft
row, so in topk mode its candidates come from the topk trace with every other row's (one replay, read_topk), its
argmax filler columns are masked like any row's and no host logits are cached; with the flag at 0 the eager seed /
join step's host logits feed _topk_from_logits and the lazy fetch returns that copy without a device call.

Run: pytest models/demos/blackhole/qwen36/tests/test_spec_devtopk_host.py -q
"""

import types

import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_spec_fastjoin_host import _admit
from models.demos.blackhole.qwen36.tests.test_spec_fastjoin_host import _engine as _fj_engine
from models.demos.blackhole.qwen36.tests.test_spec_fastjoin_host import _GdnStub as _FjGdnStub
from models.demos.blackhole.qwen36.tests.test_spec_slotmap_host import _engine as _slot_engine
from models.demos.blackhole.qwen36.tests.test_spec_specstep_host import _fake_ttnn, _vfy_model
from models.demos.blackhole.qwen36.tt.gdn.tp import fresh_row_select
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_engine import (
    PLACEHOLDER_TOKEN_ID,
    SPEC_MODES,
    MTPSpecEngine,
    TopKCandidates,
    VerifyResult,
    exact_topk,
    mask_unused_columns,
    merge_shard_topk,
    sanitize_verify_block,
    spec_memory_costs,
)
from models.demos.blackhole.qwen36.tt.tp_common import verify_device_topk_m

NB = 16


# --------------------------------------------------------------------------- #
# References (independent of the code under test)
# --------------------------------------------------------------------------- #
def _reference_topk(values, m):
    """(value desc, id asc) top-m per row by a full stable sort; -0.0 and +0.0 tie."""
    v = values.to(torch.float32)
    v = torch.where(v == 0, torch.zeros_like(v), v)
    sv, si = torch.sort(v, dim=1, descending=True, stable=True)
    return sv[:, :m], si[:, :m].to(torch.int32)


def _tied_bf16(seed, rows, vocab, groups=((0, 300), (1000, 1004), (5000, 5100))):
    """bf16 logits with injected tie groups (a contiguous id range set to one value), plus natural bf16 ties."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, vocab, generator=g) * 2).to(torch.bfloat16)
    for lo, hi in groups:
        hi = min(hi, vocab)
        x[:, lo:hi] = x[:, lo : lo + 1].clone()
    # a few ties that straddle shard boundaries, at the top of the row
    top = (x.float().max(dim=1).values + 1).to(torch.bfloat16)
    for r in range(rows):
        step = vocab // 8
        x[r, [3, step + 7, 2 * step + 11, 5 * step + 2]] = top[r]
    return x


def _shard_lists(x, D, m, unstable_seed=None):
    """What the device's per-shard top-m would return: stable (value desc, local id asc) lists, or, with
    ``unstable_seed``, lists where the members of a tie group at the shard's cut are chosen at random."""
    rows, vocab = x.shape
    per = vocab // D
    vals, ids = [], []
    g = None if unstable_seed is None else torch.Generator().manual_seed(unstable_seed)
    for d in range(D):
        shard = x[:, d * per : (d + 1) * per]
        if g is None:
            v, i = exact_topk(shard, m)
        else:
            v, i = [], []
            sv, si = torch.sort(shard.float(), dim=1, descending=True, stable=True)
            for r in range(rows):
                cut = sv[r, m - 1]
                above = si[r][sv[r] > cut]
                tied = si[r][sv[r] == cut]
                tied = tied[torch.randperm(len(tied), generator=g)][: m - len(above)]
                chosen = torch.cat([above, tied])
                chosen = chosen[torch.randperm(len(chosen), generator=g)]  # any order
                i.append(chosen)
                v.append(shard[r].float()[chosen])
            v, i = torch.stack(v), torch.stack(i).to(torch.int32)
        vals.append(v.to(torch.bfloat16))
        ids.append(i)
    return torch.cat(vals, dim=1), torch.cat(ids, dim=1)


# --------------------------------------------------------------------------- #
# exact_topk / merge_shard_topk
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("rows,vocab,m", [(32, 8192, 64), (4, 2048, 16), (1, 512, 64)])
def test_exact_topk_is_the_stable_sort_prefix(seed, rows, vocab, m):
    x = _tied_bf16(seed, rows, vocab)
    if seed == 0:
        x[0, 10], x[0, 11] = 0.0, -0.0  # signed zeros tie, lowest id first
        x[-1] = float("-inf")
        x[-1, 7] = 1.0
    v, i = exact_topk(x, m)
    rv, ri = _reference_topk(x, m)
    assert torch.equal(i, ri) and torch.equal(v, rv)
    assert v.dtype == torch.float32 and i.dtype == torch.int32


def test_exact_topk_with_explicit_ids_and_bounds(expect_error):
    x = torch.tensor([[1.0, 3.0, 3.0, 2.0]])
    ids = torch.tensor([[40, 9, 7, 100]])
    v, i = exact_topk(x, 3, ids)
    assert i.tolist() == [[7, 9, 100]] and v.tolist() == [[3.0, 3.0, 2.0]]
    with expect_error(ValueError, "ids must be"):
        exact_topk(x, 2, torch.tensor([[1, 2, 3, 1 << 20]]))
    with expect_error(ValueError, "ids must be"):
        exact_topk(x, 2, ids[:, :3])


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("D,vocab,m", [(8, 8192, 64), (8, 4096, 16), (4, 2048, 32)])
def test_merge_of_stable_shard_lists_is_the_exact_global_topk(seed, D, vocab, m):
    rows = 32
    x = _tied_bf16(seed, rows, vocab)
    vals, ids = _shard_lists(x, D, m)
    assert vals.dtype == torch.bfloat16 and vals.shape == (rows, D * m)
    mv, mi = merge_shard_topk(vals, ids, vocab // D, m)
    rv, ri = _reference_topk(x, m)
    assert torch.equal(mi, ri), "ids differ from the stable sort's top-m"
    assert torch.equal(mv, rv), "values must be the bf16 inputs unchanged"


@pytest.mark.parametrize("seed", range(8))
def test_merge_of_unstable_shard_lists_keeps_every_token_above_the_cut(seed):
    """The exactness argument of note 26, 3.1: whatever a shard returns among the ties at its cut, every token
    STRICTLY above the merged m-th value is present, and the merged m-th value is >= every shard's cut value, so
    the plugin's overflow rule (merged[m-1] >= k-th -> full logits) covers every missing tie member."""
    D, vocab, m, rows = 8, 4096, 16, 32
    x = _tied_bf16(seed, rows, vocab, groups=((0, 40), (600, 660), (2040, 2070), (3000, 3100)))
    vals, ids = _shard_lists(x, D, m, unstable_seed=seed)
    mv, mi = merge_shard_topk(vals, ids, vocab // D, m)
    rv, ri = _reference_topk(x, m)
    assert torch.equal(mv, rv), "the m values are the same whichever tie members a shard kept"
    for r in range(rows):
        cut = rv[r, -1]
        strictly_above = set((x[r].float() > cut).nonzero().view(-1).tolist())
        assert strictly_above <= set(mi[r].tolist())
        per = vocab // D
        shard_cuts = [vals[r, d * m : (d + 1) * m].float().min() for d in range(D)]
        assert all(cut >= c for c in shard_cuts)
        # identical whenever the tie group at the cut is complete inside the m
        if int((x[r].float() == cut).sum()) + len(strictly_above) <= m:
            assert torch.equal(mi[r], ri[r])
        del per


def test_merge_rejects_pad_lanes_and_bad_shapes(expect_error):
    D, per, m = 2, 64, 4
    vals = torch.zeros(1, D * m, dtype=torch.bfloat16)
    ids = torch.zeros(1, D * m, dtype=torch.int32)
    ids[0, 5] = per  # a -1e30 pad lane's id
    with expect_error(ValueError, "pad lanes"):
        merge_shard_topk(vals, ids, per, m)
    with expect_error(ValueError, "merge_shard_topk"):
        merge_shard_topk(vals, ids[:, :-1], per, m)
    ok_ids = torch.arange(D * m, dtype=torch.int32).reshape(1, -1) % per
    mv, mi = merge_shard_topk(vals, ok_ids, per, m)
    assert mi.tolist() == [[0, 1, 2, 3]]  # all-equal values: lowest global ids


# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #
def test_flag_default_off_and_m_read(monkeypatch):
    monkeypatch.delenv("QWEN36_VERIFY_DEVICE_TOPK", raising=False)
    monkeypatch.delenv("QWEN36_VERIFY_TOPK_M", raising=False)
    assert verify_device_topk_m() == 0
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "1")
    assert verify_device_topk_m() == 64
    monkeypatch.setenv("QWEN36_VERIFY_TOPK_M", "32")
    assert verify_device_topk_m() == 32
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "0")
    assert verify_device_topk_m() == 0
    assert "topk" in SPEC_MODES


def test_setup_device_topk_resolves_width_and_refuses_bad_m(monkeypatch, expect_error):
    m = Qwen36Model.__new__(Qwen36Model)
    m.vocab_size, m.num_devices = 248320, 8
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "1")
    m._vfy_sharded = False
    m._vfy_setup_device_topk()
    assert m._vfy_topk is False, "needs the sharded-argmax verify"
    m._vfy_sharded = True
    m._vfy_setup_device_topk()
    assert (m._vfy_topk, m._vfy_topk_m, m._vfy_topk_w) == (True, 64, 32768)
    monkeypatch.setenv("QWEN36_VERIFY_TOPK_M", "65")
    with expect_error(ValueError, "QWEN36_VERIFY_TOPK_M=65"):
        m._vfy_setup_device_topk()
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "0")
    m._vfy_setup_device_topk()
    assert m._vfy_topk is False and m._vfy_topk_m is None


# --------------------------------------------------------------------------- #
# Qwen36Model.verify_traced(read_topk=True) over fake persistent buffers
# --------------------------------------------------------------------------- #
def _topk_model(B, T, D=8, m=64, per=64, with_trace=True):
    model, bufs = _vfy_model(B, T, fused=True, sharded=True)
    model.num_devices = D
    model._vfy_topk, model._vfy_topk_m = with_trace, m
    model._vfy_topk_trace_id = 9 if with_trace else None
    rows = B * T
    vals = torch.randn(1, 1, 32, D * m).to(torch.bfloat16)
    ids = torch.randint(0, per, (1, 1, 32, D * m), dtype=torch.int32)
    model._vfy_topk_vals_out = types.SimpleNamespace(data=vals)
    model._vfy_topk_ids_out = types.SimpleNamespace(data=ids)
    return model, bufs, vals.reshape(32, -1)[:rows], ids.reshape(32, -1)[:rows]


@pytest.mark.parametrize("B,T", [(2, 4), (8, 4), (1, 8)])
def test_verify_traced_read_topk_replays_the_topk_trace_and_returns_the_pair(monkeypatch, B, T):
    model, bufs, want_v, want_i = _topk_model(B, T)
    log = []
    _fake_ttnn(monkeypatch, log, model)
    ids, feed, rb = model.verify_traced([[1] * T] * B, [5] * B, [0] * B, read_topk=True)
    assert [x for x in log if x[0] == "trace"] == [("trace", 7), ("trace", 9)], "main trace, then the topk trace"
    assert log[-1][0] == "sync" and all(dn._spec_win_par == 1 for dn in model._vfy_gdn), "flip after the sync"
    assert ids == list(range(B * T)) and feed == "feed"
    assert isinstance(rb, tuple) and len(rb) == 2
    assert torch.equal(rb[0], want_v) and torch.equal(rb[1], want_i)
    assert rb[0].shape == (B * T, 8 * 64)


def test_verify_traced_default_reads_are_unchanged(monkeypatch):
    model, bufs, _, _ = _topk_model(2, 4)
    log = []
    _fake_ttnn(monkeypatch, log, model)
    ids, feed, rb = model.verify_traced([[1] * 4] * 2, [5, 9], [0, 3])
    assert [x for x in log if x[0] == "trace"] == [("trace", 7)] and rb is None
    del log[:]
    ids, feed, rb = model.verify_traced([[1] * 4] * 2, [5, 9], [0, 3], read_logits=True)
    assert [x for x in log if x[0] == "trace"] == [("trace", 7), ("trace", 8)]
    assert tuple(rb.shape) == (8, 64)


def test_verify_traced_refuses_topk_without_the_trace_or_with_both_reads(monkeypatch, expect_error):
    model, bufs, _, _ = _topk_model(2, 4, with_trace=False)
    log = []
    _fake_ttnn(monkeypatch, log, model)
    with expect_error(ValueError, "QWEN36_VERIFY_DEVICE_TOPK=1"):
        model.verify_traced([[1] * 4] * 2, [5, 9], [0, 3], read_topk=True)
    model, bufs, _, _ = _topk_model(2, 4)
    _fake_ttnn(monkeypatch, log, model)
    with expect_error(ValueError, "exclusive"):
        model.verify_traced([[1] * 4] * 2, [5, 9], [0, 3], read_logits=True, read_topk=True)
    assert not [x for x in log if x[0] == "trace"], "a refused replay touches nothing"


def test_fetch_verify_logits_replays_the_logits_trace(monkeypatch):
    model, bufs, _, _ = _topk_model(2, 4)
    log = []
    _fake_ttnn(monkeypatch, log, model)
    lt = model.fetch_verify_logits()
    assert [x[0] for x in log] == ["trace", "sync"] and log[0] == ("trace", 8)
    assert tuple(lt.shape) == (8, 64)


# --------------------------------------------------------------------------- #
# MTPSpecEngine.decode_forward(spec_mode="topk") with a fake model
# --------------------------------------------------------------------------- #
class _FakeTopkModel:
    """verify_traced stand-in: shard logits are a function of the row's tokens, so the candidates follow the
    row through the slot map. D shards of ``per`` ids each; bf16 values."""

    def __init__(self, B, T, D=4, per=64, m=8):
        self.B, self.T, self.num_devices, self.per, self._vfy_topk_m = B, T, D, per, m
        self.vocab_size = D * per
        self.calls = []
        self._last = None

    def _fit_spec_page_table(self, pt, nb):
        return torch.as_tensor(pt, dtype=torch.int32)

    def logits_of(self, tokens):
        flat = torch.tensor([int(t) for row in tokens for t in row])
        g = torch.Generator().manual_seed(1)
        base = torch.randn(self.vocab_size, generator=g)
        lt = base.unsqueeze(0).repeat(len(flat), 1) * 0.1
        lt[torch.arange(len(flat)), flat % self.vocab_size] = 9.0  # the row's token leads
        lt[torch.arange(len(flat)), (flat * 7 + 3) % self.vocab_size] = 8.0
        lt[:, :5] = 8.0  # a tie group everywhere
        return lt.to(torch.bfloat16)

    def verify_traced(self, tokens, positions, mi_prev, read_logits=False, page_tables=None, inactive=None, **kw):
        self.calls.append(dict(tokens=[list(map(int, r)) for r in tokens], kw=dict(kw), read_logits=read_logits))
        lt = self.logits_of(tokens)
        self._last = lt
        ids = [int(r.argmax()) for r in lt.float()]
        if kw.get("read_topk"):
            m, D, per = self._vfy_topk_m, self.num_devices, self.per
            vals, lids = [], []
            for d in range(D):
                v, i = exact_topk(lt[:, d * per : (d + 1) * per], m)
                vals.append(v.to(torch.bfloat16))
                lids.append(i)
            return ids, "feed", (torch.cat(vals, 1), torch.cat(lids, 1))
        return ids, "feed", (lt if read_logits else None)

    def fetch_verify_logits(self):
        self.calls.append("fetch")
        return self._last


def _topk_engine(B, K, model, slotmap=True):
    e = _slot_engine.__wrapped__(B, K, slotmap, model) if hasattr(_slot_engine, "__wrapped__") else None
    if e is None:
        e = MTPSpecEngine.__new__(MTPSpecEngine)
        e.B, e.K, e.T, e.block_size, e.nb = B, K, K + 1, 64, NB
        e.host_slotmap, e._phys, e._phys_perm = slotmap, list(range(B)), None
        e._active, e._fresh, e._offered = set(range(B)), {}, set()
        e._parked, e._prompts, e._fresh_dirty = {}, {}, set()
        e._no_draft, e._speculated, e._withheld_streak = set(), set(), {}
        e._row_active, e._last_pt, e._last_cap_blocks, e.skipped_propose_calls = None, None, NB, 0
        e.no_draft_streak, e.captured, e._timing = 16, True, False
        e._gdn = []
        e.model = model
        e.vocab = model.vocab_size
        e._last_hidden, e._mtp_pt_rows, e._vfeed, e._topk_logits = [None] * B, [None] * B, None, None
        e._stage_mtp_tables = e._warm_mtp_last = lambda *a, **k: None
    return e


def _block(B, T, nv, acc, base=50):
    tok = torch.arange(B * T, dtype=torch.int32).reshape(B, T) + 100
    pos = torch.full((B, T), -1, dtype=torch.int32)
    for u in range(B):
        pos[u, : nv[u] + 1] = base + 10 * u + torch.arange(nv[u] + 1, dtype=torch.int32)
    return tok, pos, torch.tensor(nv, dtype=torch.int32), torch.tensor(acc, dtype=torch.int32)


@pytest.mark.parametrize("remap", [None, [2, 0, 3, 1]])
def test_decode_forward_topk_delivers_candidates_and_lazy_logits_in_slot_order(remap):
    B, K = 4, 3
    T = K + 1
    model = _FakeTopkModel(B, T)
    e = _topk_engine(B, K, model)
    if remap is not None:
        e.remap_slots(remap)
    phys = list(e._phys)
    tok, pos, nv, acc = _block(B, T, nv=[3, 0, 1, 2], acc=[1, 4, 2, 3])
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB)
    res = e.decode_forward(tok, pos, num_valid_drafts=nv, accepted_counts=acc, page_table=pt, spec_mode="topk")
    assert isinstance(res, VerifyResult) and res.spec_mode == "topk" and res.logits is None
    assert model.calls[-1]["kw"] == {"read_topk": True} and model.calls[-1]["read_logits"] is False
    clean, start0, active = sanitize_verify_block(tok, pos, nv, K)
    # the engine saw runner slot s at engine row phys[s]
    assert all(model.calls[-1]["tokens"][phys[s]] == clean[s].tolist() for s in range(B))
    # candidates: exact top-m of each runner row's own logits, in runner order
    want = model.logits_of(clean.tolist()).float()
    rv, ri = _reference_topk(want, model._vfy_topk_m)
    assert torch.equal(res.topk_ids.reshape(B * T, -1), ri)
    assert torch.equal(res.topk_values.reshape(B * T, -1), rv)
    assert res.topk_values.dtype == torch.float32 and res.topk_ids.dtype == torch.int32
    assert torch.equal(res.argmax_ids, mask_unused_columns(want.argmax(-1).to(torch.int32).reshape(B, T), nv))
    # lazy full logits: one device round trip, runner order, fp32
    n_calls = len(model.calls)
    full = res.fetch_logits()
    assert model.calls[-1] == "fetch" and len(model.calls) == n_calls + 1
    assert full.dtype == torch.float32 and tuple(full.shape) == (B, T, model.vocab_size)
    assert torch.equal(full.reshape(B * T, -1), want)


def test_decode_forward_logits_and_argmax_modes_do_not_ask_for_topk():
    B, K = 2, 3
    model = _FakeTopkModel(B, K + 1)
    e = _topk_engine(B, K, model)
    tok, pos, nv, acc = _block(B, K + 1, nv=[1, 2], acc=[1, 1])
    pt = torch.zeros(B, NB, dtype=torch.int32)
    for mode in ("argmax_ids", "logits"):
        res = e.decode_forward(tok, pos, num_valid_drafts=nv, accepted_counts=acc, page_table=pt, spec_mode=mode)
        assert model.calls[-1]["kw"] == {}, "read_topk is passed only when asked (older verify_traced signatures)"
        assert res.topk_values is None and res.topk_ids is None and res.fetch_logits is None
        assert (res.logits is not None) == (mode == "logits")


def test_verify_refuses_both_reads(expect_error):
    B, K = 1, 3
    e = _topk_engine(B, K, _FakeTopkModel(B, K + 1))
    with expect_error(ValueError, "exclusive"):
        e.verify(
            torch.zeros(1, 4, dtype=torch.int32),
            torch.tensor([5]),
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, NB, dtype=torch.int32),
            read_logits=True,
            read_topk=True,
        )


def test_topk_from_logits_builds_the_same_format_and_caches_the_fallback():
    """Seed / join steps read host logits: the candidates must be the same exact top-m and the full logits are
    kept for fetch_verify_logits without a device call."""
    B, K = 2, 3
    T = K + 1
    model = _FakeTopkModel(B, T)
    e = _topk_engine(B, K, model)
    e.remap_slots([1, 0])
    logits = model.logits_of([[100 + u * T + j for j in range(T)] for u in range(B)]).float().reshape(B, T, -1)
    ids = torch.zeros(B, T, dtype=torch.int32)
    out_ids, cands = e._topk_from_logits((ids, logits))
    assert out_ids is ids and isinstance(cands, TopKCandidates)
    rv, ri = _reference_topk(logits.reshape(B * T, -1), model._vfy_topk_m)
    assert torch.equal(cands.ids.reshape(B * T, -1), ri) and torch.equal(cands.values.reshape(B * T, -1), rv)
    n_calls = len(model.calls)
    full = e.fetch_verify_logits()
    assert len(model.calls) == n_calls, "no device round trip after a seed / join"
    assert torch.equal(full, logits[torch.tensor([1, 0])]), "runner-slot order"


# --------------------------------------------------------------------------- #
# spec_memory_costs
# --------------------------------------------------------------------------- #
def test_spec_memory_costs_counts_the_topk_buffers_only_under_the_flag(monkeypatch):
    tc = dict(
        hidden_size=512,
        vocab_size=4096,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
    )
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "0")
    off = spec_memory_costs(tc, 4, 2, 3, nb=16)
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "1")
    monkeypatch.setenv("QWEN36_VERIFY_TOPK_M", "64")
    on = spec_memory_costs(tc, 4, 2, 3, nb=16)
    assert on["per_seq"] == off["per_seq"] and on["per_token"] == off["per_token"]
    rows = 2 * 4
    shard = 32 * 1024 * 2  # [1,1,rows->32, 4096/4=1024] bf16
    outs = 32 * 256 * 2 + 32 * 256 * 4  # [1,1,32, 4*64] bf16 + int32
    assert on["fixed"]["verify_io"] - off["fixed"]["verify_io"] == shard + outs
    del rows


# --------------------------------------------------------------------------- #
# Qwen36MTPContractForCausalLM.spec_plan advertises the mode only under the flag
# --------------------------------------------------------------------------- #
def test_spec_plan_offers_topk_only_under_the_flag(monkeypatch):
    pytest.importorskip("vllm")
    spec_decode = pytest.importorskip("vllm_tt_plugin.spec_decode")
    from models.demos.blackhole.qwen36.tt import qwen36_vllm

    seen = {}

    class _Plan:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(spec_decode, "SpecPlan", _Plan)
    monkeypatch.setattr(qwen36_vllm, "spec_memory_costs", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(
        "models.demos.blackhole.qwen36.tt.spec_engine.spec_memory_costs",
        lambda tc, tp, B, k, **kw: {
            "per_seq": {"total": 10, "gdn_ring": 1, "gdn_conv_windows": 2},
            "fixed": {"total": 6},
            "per_token": {"total": 3},
        },
    )
    tc = types.SimpleNamespace(mtp_num_hidden_layers=1)
    cfg = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(get_text_config=lambda: tc), max_model_len=4096
        ),
        speculative_config=None,
    )
    cls = qwen36_vllm.Qwen36MTPContractForCausalLM
    monkeypatch.setattr(cls, "_SPEC_K", 3)
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "0")
    cls.spec_plan(cfg, 1, 3)
    assert seen["accept_modes"] == ("argmax_ids", "logits") and "topk_candidates" not in seen
    seen.clear()
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "1")
    cls.spec_plan(cfg, 1, 3)
    assert seen["accept_modes"] == ("argmax_ids", "logits", "topk") and seen["topk_candidates"] == 64


# --------------------------------------------------------------------------- #
# Fresh rows in topk mode: QWEN36_SPEC_FAST_JOIN=1 (the normal replay) and =0 (the eager seed / join)
# --------------------------------------------------------------------------- #
class _FakeTopkJoinModel(_FakeTopkModel):
    """_FakeTopkModel that also records mi_prev and inactive, the per-row inputs a fresh row changes."""

    def verify_traced(self, tokens, positions, mi_prev, read_logits=False, page_tables=None, inactive=None, **kw):
        out = super().verify_traced(tokens, positions, mi_prev, read_logits, page_tables, inactive, **kw)
        self.calls[-1]["mi_prev"] = [int(m) for m in mi_prev]
        self.calls[-1]["inactive"] = None if inactive is None else [bool(x) for x in inactive]
        self.calls[-1]["positions"] = [int(p) for p in positions]
        return out


def _fj_topk_engine(B, K, model, gdn, **kw):
    """The fast-join host engine (test_spec_fastjoin_host._engine) with the topk-mode fields; the eager-seed topk
    helper fails the test unless the test replaces it."""
    e = _fj_engine(B, K, model, gdn, vocab=model.vocab_size, _topk_logits=None, **kw)
    e._topk_from_logits = lambda out: pytest.fail("_topk_from_logits ran: a fresh row left the normal replay")
    return e


@pytest.mark.parametrize("remap", [None, [2, 0, 3, 1]])
def test_fast_join_topk_fresh_rows_ride_the_same_replay(remap):
    """Two fresh rows join two running rows in ONE replay with read_topk: the fresh rows run at mi = 0 after the
    GDN seed, their candidates (column 0 real) are the exact top-m of the replay's logits like every other row's,
    their argmax filler columns are masked (num_valid_drafts 0), no host logits are cached and the lazy fetch is one
    device round trip; everything in runner-slot order through the slot map."""
    B, K = 4, 3
    T = K + 1
    model, gdn = _FakeTopkJoinModel(B, T), _FjGdnStub(T)
    e = _fj_topk_engine(B, K, model, gdn)
    if remap is not None:
        e.remap_slots(remap)
    phys = list(e._phys)  # runner slot s sits at engine row phys[s]
    tok, pos, nv, acc = _block(B, T, nv=[2, 0, 1, 0], acc=[2, 1, 3, 1])  # slot s starts at 50 + 10 s
    e._active = {phys[0], phys[2]}
    _admit(e, phys[1], 60)
    _admit(e, phys[3], 80)
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB) + 1
    res = e.decode_forward(tok, pos, num_valid_drafts=nv, accepted_counts=acc, page_table=pt, spec_mode="topk")
    assert isinstance(res, VerifyResult) and res.spec_mode == "topk" and res.logits is None
    # one replay for the whole step, the topk read asked, the fresh ENGINE rows at mi = 0, every row active
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["kw"] == {"read_topk": True} and call["read_logits"] is False
    assert call["inactive"] == [False] * B
    clean, start0, active = sanitize_verify_block(tok, pos, nv, K)
    for s in range(B):
        assert call["tokens"][phys[s]] == clean[s].tolist() and call["positions"][phys[s]] == int(start0[s])
    assert call["mi_prev"][phys[1]] == 0 and call["mi_prev"][phys[3]] == 0
    assert call["mi_prev"][phys[0]] == 1 and call["mi_prev"][phys[2]] == 2
    # the GDN seed named the fresh engine rows once (shared selector), the MTP slot Tp-1 warmed, bookkeeping cleared
    fresh_rows = sorted([phys[1], phys[3]])
    assert gdn.selects == [(fresh_rows, "SEL")]
    assert torch.equal(e.log["stage"][0][0], fresh_row_select(fresh_rows, B, T, 4)) and len(e.log["stage"]) == 1
    assert sorted(w[2] for w in e.log["warm"]) == [59, 79] and len(e.log["warm"]) == 2
    assert e._fresh == {} and e._fresh_dirty == set() and e.log["reprefill"] == []
    # candidates: exact top-m of each runner row's own replay logits, fresh rows included, runner order
    want = model.logits_of(clean.tolist()).float()
    rv, ri = _reference_topk(want, model._vfy_topk_m)
    assert torch.equal(res.topk_ids.reshape(B * T, -1), ri)
    assert torch.equal(res.topk_values.reshape(B * T, -1), rv)
    assert res.topk_values.dtype == torch.float32 and res.topk_ids.dtype == torch.int32
    # argmax: column 0 of a fresh row is its first generated token, its filler columns are masked like any row's
    assert torch.equal(res.argmax_ids, mask_unused_columns(want.argmax(-1).to(torch.int32).reshape(B, T), nv))
    assert res.argmax_ids[[1, 3], 1:].tolist() == [[PLACEHOLDER_TOKEN_ID] * K] * 2
    assert res.argmax_ids[1, 0] == int(want.reshape(B, T, -1)[1, 0].argmax())
    # no host logits were cached: the lazy fetch replays the logits trace (one device round trip), runner order
    assert e._topk_logits is None
    n_calls = len(model.calls)
    full = res.fetch_logits()
    assert model.calls[-1] == "fetch" and len(model.calls) == n_calls + 1
    assert full.dtype == torch.float32 and torch.equal(full.reshape(B * T, -1), want)


def test_fast_join_topk_seed_step_with_a_padding_row_merges_every_row():
    """Every active row fresh (a seed step) next to a padding row: one replay with read_topk, the whole-slot GDN
    write, the padding row runs over the null block and its candidate rows merge like any row's (real logits, no
    pad lanes: the readback is sliced to the B*T rows before the merge); the fresh rows keep column 0 only."""
    B, K = 3, 2
    T = K + 1
    model, gdn = _FakeTopkJoinModel(B, T), _FjGdnStub(T)
    e = _fj_topk_engine(B, K, model, gdn)
    _admit(e, 0, 50)
    _admit(e, 1, 60)
    tok = torch.full((B, T), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    pos = torch.full((B, T), -1, dtype=torch.int32)
    tok[0, 0], pos[0, 0] = 100, 50
    tok[1, 0], pos[1, 0] = 101, 60  # row 2 stays padding (start_pos -1)
    nv, acc = torch.zeros(B, dtype=torch.int32), torch.ones(B, dtype=torch.int32)
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB) + 1
    res = e.decode_forward(tok, pos, num_valid_drafts=nv, accepted_counts=acc, page_table=pt, spec_mode="topk")
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["kw"] == {"read_topk": True} and call["mi_prev"] == [0, 0, 0]
    assert call["inactive"] == [False, False, True]
    assert gdn.selects == [([0, 1, 2], "SEL")], "every active row is fresh: the whole-slot write"
    assert torch.equal(e.log["stage"][0][0], fresh_row_select([0, 1], B, T, 4)), "the selector names only the fresh"
    assert sorted(w[2] for w in e.log["warm"]) == [49, 59]
    clean, start0, active = sanitize_verify_block(tok, pos, nv, K)
    assert active.tolist() == [True, True, False] and call["tokens"] == clean.tolist()
    want = model.logits_of(clean.tolist()).float()
    rv, ri = _reference_topk(want, model._vfy_topk_m)
    assert torch.equal(res.topk_ids.reshape(B * T, -1), ri) and torch.equal(res.topk_values.reshape(B * T, -1), rv)
    assert res.argmax_ids[:, 0].tolist() == want.argmax(-1).reshape(B, T)[:, 0].tolist()
    assert bool((res.argmax_ids[:, 1:] == PLACEHOLDER_TOKEN_ID).all())
    assert e._fresh == {} and e._active == {0, 1} and e._topk_logits is None
    full = res.fetch_logits()
    assert model.calls[-1] == "fetch" and torch.equal(full.reshape(B * T, -1), want)


def test_fast_join_topk_left_out_fresh_row_waits_without_a_reprefill():
    """A fresh row left out of a topk step (padding) keeps its prefill state and joins the next replay as a fresh
    row at mi = 0, with read_topk on both steps and the candidates of both steps from the replay."""
    B, K = 2, 2
    T = K + 1
    model, gdn = _FakeTopkJoinModel(B, T), _FjGdnStub(T)
    e = _fj_topk_engine(B, K, model, gdn)
    _admit(e, 0, 50)
    _admit(e, 1, 70)
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB) + 1
    tok = torch.full((B, T), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    pos = torch.full((B, T), -1, dtype=torch.int32)
    tok[0, 0], pos[0, 0] = 100, 50
    nv, acc = torch.zeros(B, dtype=torch.int32), torch.ones(B, dtype=torch.int32)
    res = e.decode_forward(tok, pos, num_valid_drafts=nv, accepted_counts=acc, page_table=pt, spec_mode="topk")
    assert model.calls[0]["kw"] == {"read_topk": True} and model.calls[0]["inactive"] == [False, True]
    assert e._fresh == {1: 70} and e._fresh_dirty == set() and e.log["reprefill"] == []
    assert res.topk_ids.shape == (B, T, model._vfy_topk_m)
    # next step: row 0 running (1 accepted), row 1 comes in fresh
    tok2, pos2, nv2, acc2 = _block(B, T, nv=[1, 0], acc=[1, 1], base=51)  # slot 0 at 51, slot 1 at 61
    tok2[1, 0], pos2[1, 0] = 300, 70  # slot 1 verifies its first token at its prompt length
    res2 = e.decode_forward(tok2, pos2, num_valid_drafts=nv2, accepted_counts=acc2, page_table=pt, spec_mode="topk")
    call = model.calls[-1]
    assert call["kw"] == {"read_topk": True} and call["mi_prev"] == [0, 0] and call["inactive"] == [False, False]
    assert gdn.selects == [([0, 1], "SEL"), ([1], "SEL")] and e._fresh == {} and e.log["reprefill"] == []
    clean, _, _ = sanitize_verify_block(tok2, pos2, nv2, K)
    want = model.logits_of(clean.tolist()).float()
    rv, ri = _reference_topk(want, model._vfy_topk_m)
    assert torch.equal(res2.topk_ids.reshape(B * T, -1), ri) and torch.equal(res2.topk_values.reshape(B * T, -1), rv)
    assert torch.equal(res2.argmax_ids, mask_unused_columns(want.argmax(-1).to(torch.int32).reshape(B, T), nv2))


@pytest.mark.parametrize("join", [False, True])
def test_flag_off_topk_fresh_rows_take_the_eager_seed_and_cache_the_host_logits(join):
    """QWEN36_SPEC_FAST_JOIN=0: the fresh rows go through _seed (every active row fresh) or _join, asked for the
    logits (read_logits or read_topk), and _topk_from_logits turns those host logits into the same exact candidates;
    no replay, no fast GDN seed, and the lazy fetch returns the cached copy without a device call."""
    B, K = 2, 3
    T = K + 1
    model, gdn = _FakeTopkJoinModel(B, T), _FjGdnStub(T)
    e = _fj_engine(B, K, model, gdn, fast_join=False, vocab=model.vocab_size, _topk_logits=None)
    seen = []

    def fake_seed(tokens, start_pos, pt, read):
        seen.append(("seed", bool(read)))
        return torch.zeros(B, T, dtype=torch.int32), model.logits_of(tokens.tolist()).float().reshape(B, T, -1)

    def fake_join(tokens, start_pos, mi_prev, pt, active, fresh, read):
        seen.append(("join", bool(read), sorted(fresh)))
        return torch.zeros(B, T, dtype=torch.int32), model.logits_of(tokens.tolist()).float().reshape(B, T, -1)

    e._seed, e._join = fake_seed, fake_join
    tok, pos, nv, acc = _block(B, T, nv=[0, 0], acc=[1, 1])  # slot 0 at 50, slot 1 at 60
    if join:
        e._active = {0}
        nv[0], acc[0] = 2, 2
        pos[0, :3] = torch.tensor([50, 51, 52], dtype=torch.int32)
    else:
        _admit(e, 0, 50)
    _admit(e, 1, 60)
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB) + 1
    res = e.decode_forward(tok, pos, num_valid_drafts=nv, accepted_counts=acc, page_table=pt, spec_mode="topk")
    assert seen == ([("join", True, [1])] if join else [("seed", True)])
    assert model.calls == [] and gdn.selects == [] and e.log["stage"] == [], "no replay, no fast seed"
    clean, _, _ = sanitize_verify_block(tok, pos, nv, K)
    want = model.logits_of(clean.tolist()).float()
    rv, ri = _reference_topk(want, model._vfy_topk_m)
    assert torch.equal(res.topk_ids.reshape(B * T, -1), ri) and torch.equal(res.topk_values.reshape(B * T, -1), rv)
    assert torch.equal(res.argmax_ids, mask_unused_columns(torch.zeros(B, T, dtype=torch.int32), nv))
    assert e._topk_logits is not None and e._fresh == {}
    full = res.fetch_logits()
    assert model.calls == [], "the eager step's host logits serve the fallback: no device round trip"
    assert full.dtype == torch.float32 and torch.equal(full.reshape(B * T, -1), want)
