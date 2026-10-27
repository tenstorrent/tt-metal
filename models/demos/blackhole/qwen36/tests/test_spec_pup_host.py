# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the FLUX 3 Klein pup spec-decode engine's host side (tt/spec_pup.py, model.topn_from_gathered).
No device: the KV plan, the top-N bound math, the MTP tensor checks and PupSpecEngine's call translation onto
MTPSpecEngine (a recording fake).

Run: pytest models/demos/blackhole/qwen36/tests/test_spec_pup_host.py -q
"""
import types

import pytest
import torch

from models.demos.blackhole.qwen36.tt import spec_pup
from models.demos.blackhole.qwen36.tt.model import topn_from_gathered
from models.demos.blackhole.qwen36.tt.spec_engine import accepted_plan, draft_rows_fit, verify_write_table
from models.demos.blackhole.qwen36.tt.spec_pup import MTP_LAYER_SUFFIXES, SpecKVPlan, check_mtp_tensors, spec_kv_plan

VOCAB, N_DEV = 248320, 8
SHARD = VOCAB // N_DEV


# --------------------------------------------------------------------------------------------- #
# KV plan
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("K, nb", [(1, 129), (3, 129), (7, 129), (63, 129), (64, 130)])
def test_spec_kv_plan_8192(K, nb):
    plan = spec_kv_plan(8192, K)
    assert plan.table_blocks == nb
    assert plan.num_blocks == nb + 1
    assert plan.capacity == nb * 64
    assert plan.page_table.dtype == torch.int32 and tuple(plan.page_table.shape) == (1, nb)
    assert torch.equal(plan.page_table[0], torch.arange(1, nb + 1, dtype=torch.int32))
    assert int(plan.page_table.min()) == 1  # never names the null block 0
    assert int(plan.page_table.max()) < plan.num_blocks  # inside the base cache
    mtp_blocks = plan.num_blocks + 1  # allocate_kv_caches: base shape + 1
    assert mtp_blocks - 1 > int(plan.page_table.max())  # MTP scratch past every named block (_check_scratch)
    # the longest request: an 8191-token context's last verify window still fits
    assert 8191 + K < plan.capacity


def test_spec_kv_plan_rejects(expect_error):
    with expect_error(ValueError, "bad plan args"):
        spec_kv_plan(0, 3)
    with expect_error(ValueError, "bad plan args"):
        spec_kv_plan(8192, -1)


def test_verify_write_table_padding_hits_null_block_only():
    plan = spec_kv_plan(8192, 3)
    pt = plan.page_table
    for start, nv in [(100, 0), (127, 3), (63, 1), (8188, 3), (8191, 0)]:
        vpt = verify_write_table(pt, torch.tensor([start]), torch.tensor([nv]), plan.block_size)
        last_blk = (start + nv) // plan.block_size
        assert torch.equal(vpt[0, : last_blk + 1], pt[0, : last_blk + 1])
        assert int(vpt[0, last_blk + 1 :].abs().sum()) == 0  # padded columns past the last real block -> block 0


# --------------------------------------------------------------------------------------------- #
# Top-N bound math
# --------------------------------------------------------------------------------------------- #
def _device_topn(full_rows, top_n, n_dev=N_DEV):
    """What the trace tail returns: per device the top_n (value, local col) of its shard, devices concatenated."""
    R, V = full_rows.shape
    shard = V // n_dev
    vs, ls = [], []
    for d in range(n_dev):
        part = full_rows[:, d * shard : (d + 1) * shard]
        v, i = torch.topk(part.float(), top_n, dim=-1)  # any tie order is allowed by the contract
        vs.append(v.to(torch.bfloat16))
        ls.append(i.to(torch.int32))
    return torch.cat(vs, -1).reshape(1, 1, R, -1), torch.cat(ls, -1).reshape(1, 1, R, -1), shard


def _random_rows(R, V, gen, ties=True):
    x = (torch.randn(R, V, generator=gen) * 3).to(torch.bfloat16)
    if ties:  # plant exact ties at the top, across and within devices
        for r in range(R):
            top = x[r].float().max()
            idx = torch.randint(0, V, (5,), generator=gen)
            x[r, idx] = top.to(torch.bfloat16)
    return x


@pytest.mark.parametrize("top_n", [1, 8, 32])
def test_topn_from_gathered_bound_and_values(top_n):
    gen = torch.Generator().manual_seed(1234 + top_n)
    V = 8 * 4096  # a smaller vocab keeps the test fast; the math is width-independent
    for _ in range(5):
        full = _random_rows(4, V, gen)
        v, loc, shard = _device_topn(full, top_n)
        vals, ids, thresh = topn_from_gathered(v, loc, N_DEV, top_n, shard, 4)
        assert vals.dtype == torch.float32 and ids.dtype == torch.int64 and thresh.shape == (4,)
        for r in range(4):
            assert len(set(ids[r].tolist())) == ids.shape[1]  # unique
            assert int(ids[r].min()) >= 0 and int(ids[r].max()) < V
            assert torch.equal(vals[r], full[r, ids[r]].float())  # bit-exact candidate values
            mask = torch.ones(V, dtype=torch.bool)
            mask[ids[r]] = False
            assert bool((full[r][mask].float() <= thresh[r]).all())  # the certified bound


def _presence_greedy(row, generated, p):
    v = row.reshape(-1).float().clone()
    if p > 0 and generated:
        v[torch.tensor(sorted(set(int(t) for t in generated)), dtype=torch.long)] -= float(p)
    return int(torch.argmax(v).item())


def _certified(vals, ids, thresh, penalize, p):
    y = vals.clone()
    if p > 0 and penalize:
        y[torch.isin(ids, torch.tensor(sorted(penalize), dtype=torch.int64))] -= float(p)
    best = y.max()
    if not bool(best > thresh):
        return None
    return int(ids[y == best].min())


def test_bound_certifies_the_exact_penalized_argmax():
    """The design's host rule over this module's output: whenever certification succeeds it equals presence_greedy
    on the full row (lowest id on ties); it fails only when the penalty pushes every candidate to the bound."""
    gen = torch.Generator().manual_seed(7)
    V = 8 * 2048
    fails = 0
    for trial in range(200):
        full = _random_rows(4, V, gen, ties=bool(trial % 2))
        top_n = (1, 4, 32)[trial % 3]
        v, loc, shard = _device_topn(full, top_n)
        vals, ids, thresh = topn_from_gathered(v, loc, N_DEV, top_n, shard, 4)
        for r in range(4):
            # penalize the row's own top candidates (the hard case) plus random ids
            pen = set(ids[r][: int(torch.randint(0, ids.shape[1] + 1, (1,), generator=gen))].tolist())
            pen |= set(torch.randint(0, V, (20,), generator=gen).tolist())
            for p in (0.0, 1.5):
                got = _certified(vals[r], ids[r], thresh[r], pen, p)
                if got is None:
                    fails += 1
                    continue
                assert got == _presence_greedy(full[r], sorted(pen), p)
    assert fails < 200 * 4 * 2  # the property, not a rate; the rate is measured on real rows


def test_topn_from_gathered_rejects_padded_column(expect_error):
    v = torch.zeros(1, 1, 1, N_DEV * 2, dtype=torch.bfloat16)
    loc = torch.zeros(1, 1, 1, N_DEV * 2, dtype=torch.int32)
    loc[..., 3] = SHARD  # a pad column (>= shard width) won
    with expect_error(ValueError, "top-N local ids outside"):
        topn_from_gathered(v, loc, N_DEV, 2, SHARD, 1)


# --------------------------------------------------------------------------------------------- #
# MTP tensor checks
# --------------------------------------------------------------------------------------------- #
_REF = {
    "self_attn.q_proj.weight": (64, 16),
    "self_attn.k_proj.weight": (8, 16),
    "self_attn.v_proj.weight": (8, 16),
    "self_attn.o_proj.weight": (16, 32),
    "self_attn.q_norm.weight": (4,),
    "self_attn.k_norm.weight": (4,),
    "mlp.gate_proj.weight": (48, 16),
    "mlp.up_proj.weight": (48, 16),
    "mlp.down_proj.weight": (16, 48),
    "input_layernorm.weight": (16,),
    "post_attention_layernorm.weight": (16,),
}


def _mtp_tensors(dim=16):
    t = {
        "mtp.fc.weight": torch.zeros(dim, 2 * dim, dtype=torch.bfloat16),
        "mtp.pre_fc_norm_embedding.weight": torch.zeros(dim, dtype=torch.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": torch.zeros(dim, dtype=torch.bfloat16),
        "mtp.norm.weight": torch.zeros(dim, dtype=torch.bfloat16),
    }
    for s, shape in _REF.items():
        t[f"mtp.layers.0.{s}"] = torch.zeros(*shape, dtype=torch.bfloat16)
    return t


def test_check_mtp_tensors_ok_and_count():
    t = _mtp_tensors()
    assert len(t) == 15 and set(_REF) == set(MTP_LAYER_SUFFIXES)
    check_mtp_tensors(t, dim=16, ref_layer_shapes=_REF)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.pop("mtp.fc.weight"),
        lambda t: t.pop("mtp.layers.0.mlp.up_proj.weight"),
        lambda t: t.__setitem__("mtp.fc.weight", torch.zeros(16, 16, dtype=torch.bfloat16)),
        lambda t: t.__setitem__("mtp.layers.0.self_attn.q_proj.weight", torch.zeros(32, 16, dtype=torch.bfloat16)),
        lambda t: t.__setitem__("mtp.layers.1.mlp.up_proj.weight", torch.zeros(48, 16, dtype=torch.bfloat16)),
        lambda t: t.__setitem__("mtp.norm.weight", torch.zeros(16, dtype=torch.float16)),
    ],
)
def test_check_mtp_tensors_rejects(mutate, expect_error):
    t = _mtp_tensors()
    mutate(t)
    with expect_error(ValueError, "MTP"):
        check_mtp_tensors(t, dim=16, ref_layer_shapes=_REF)


# --------------------------------------------------------------------------------------------- #
# PupSpecEngine -> MTPSpecEngine call translation (recording fake, no device)
# --------------------------------------------------------------------------------------------- #
class _FakeInner:
    def __init__(self, model, B, K, **kw):
        self.B, self.K, self.T, self.kw = B, K, K + 1, kw
        self.calls = []
        self._fresh = {}
        self.captured = True
        self.prefill_prepared = True
        self.last_verify_timing = None
        self._timing = False

    def verify(self, tokens, start_pos, accepted, page_table, read_logits=False, num_valid_drafts=None):
        self.calls.append(("verify", tokens, start_pos, accepted, num_valid_drafts, read_logits))
        if 0 in self._fresh:
            del self._fresh[0]
            lt = torch.zeros(1, self.T, VOCAB)
            lt[0, 0, 11] = 1.0
            return torch.zeros(1, self.T, dtype=torch.int32), lt
        C = N_DEV * self.kw["top_n"]
        return (torch.zeros(self.T, C), torch.arange(C).repeat(self.T, 1).to(torch.int64), torch.zeros(self.T)), None

    def propose(self, ctok, cpos, acc, page_table):
        self.calls.append(("propose", ctok, cpos, acc))
        m, _, pending, p = accepted_plan(torch.tensor(ctok), torch.tensor(cpos), acc, 1, self.T)
        self.last_plan = (m, pending, p)
        return torch.arange(self.K, dtype=torch.int32).reshape(1, self.K) + 100

    def read_verify_rows(self, rows):
        return torch.zeros(len(rows), VOCAB)

    def release(self, slot):
        self.calls.append(("release", slot))

    def shutdown(self):
        self.calls.append(("shutdown",))


def _fake_model(max_seq_len=8192):
    args = types.SimpleNamespace(max_batch_size=1, max_seq_len=max_seq_len, gdn_chunk_size=128)
    return types.SimpleNamespace(
        args=args, num_devices=N_DEV, vocab_size=VOCAB, mtp=object(), _paged_kv_caches=None, _vfy_trace_id=None
    )


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(spec_pup, "MTPSpecEngine", _FakeInner)
    model = _fake_model()
    plan = spec_kv_plan(8192, 3)
    eng = spec_pup.PupSpecEngine(model, plan, draft_len=3, use_mtp=True, top_n=32)
    assert model._spec_engine is eng.inner
    kw = eng.inner.kw
    assert kw["external_prefill"] and kw["verify_output"] == "topn" and kw["top_n"] == 32
    assert kw["use_mtp"] and not kw["require_fused_decode"]
    return eng


def test_engine_seed_verify_propose_bookkeeping(engine):
    inner = engine.inner
    Tp = 1028
    engine._Tp = Tp
    inner._fresh[0] = Tp
    row = engine.seed(5)
    assert int(row.argmax()) == 11 and row.shape == (VOCAB,)
    assert inner.calls[-1][:5] == ("verify", [[5, 5, 5, 5]], [Tp], [1], None)
    # after the seed commits nxt at Tp+1: propose([nxt], Tp+1) -> anchor slot Tp, pending nxt
    d = engine.propose([42], Tp + 1)
    assert d == [100, 101, 102]
    m, pending, p = inner.last_plan
    assert m == [0] and pending == [42] and p == [Tp]
    # verify at start Tp+1 with 2 valid drafts
    vals, ids, thresh = engine.verify([42, 100, 101, 42], Tp + 1, 1, 2)
    assert inner.calls[-1][:5] == ("verify", [[42, 100, 101, 42]], [Tp + 1], [1], [2])
    assert vals.shape == (4, 256) and ids.shape == (4, 256) and thresh.shape == (4,)
    # m = 2 accepted + bonus 7: committed [100, 101, 7] at Tp+2.. ; anchor = Tp+3 (slot before the bonus)
    engine.propose([100, 101, 7], Tp + 2)
    m, pending, p = inner.last_plan
    assert m == [2] and pending == [7] and p == [Tp + 3]
    assert inner.calls[-1][2] == [[Tp + 2 + i for i in range(4)]]


def test_engine_propose_capacity_and_errors(engine, expect_error):
    engine._Tp = 100
    cap = engine.plan.capacity
    assert engine.propose([1], cap - 4) == [100, 101, 102]  # p = cap-5: p + 1 + K = cap - 1 < cap -> drafts
    assert engine.propose([1], cap - 3) == []  # p + 1 + K >= cap
    with expect_error(ValueError, "committed has 0 tokens"):
        engine.propose([], 10)
    with expect_error(ValueError, "verify takes exactly K"):
        engine.verify([1, 2, 3], 10, 1, 0)  # not K+1 tokens
    with expect_error(ValueError, "accepted_prev 0 outside"):
        engine.verify([1, 2, 3, 4], 10, 0, 0)  # accepted_prev 0
    with expect_error(ValueError, "num_valid 4 outside"):
        engine.verify([1, 2, 3, 4], 10, 1, 4)  # num_valid > K


def test_engine_propose_fits_matches_engine_rule(engine):
    """PupSpecEngine.propose's capacity rule is MTPSpecEngine's draft_rows_fit (else inner.propose would assert)."""
    cap = engine.plan.capacity
    for first in range(cap - 12, cap):
        for n in (1, 2, 4):
            p = first + n - 2
            ours = not (p + 1 + engine.K >= cap)
            assert ours == draft_rows_fit([p], engine.K, cap)[0]


def test_engine_rejects_bad_models(monkeypatch, expect_error):
    monkeypatch.setattr(spec_pup, "MTPSpecEngine", _FakeInner)
    plan = spec_kv_plan(8192, 3)
    m = _fake_model()
    m.args.max_batch_size = 2
    with expect_error(ValueError, "PupSpecEngine is B=1"):
        spec_pup.PupSpecEngine(m, plan, draft_len=3, use_mtp=True)
    m = _fake_model()
    m.mtp = None
    with expect_error(ValueError, "use_mtp needs model.mtp"):
        spec_pup.PupSpecEngine(m, plan, draft_len=3, use_mtp=True)
    spec_pup.PupSpecEngine(m, plan, draft_len=3, use_mtp=False)  # n-gram: no head needed
    with expect_error(ValueError, "plan has .* blocks"):
        spec_pup.PupSpecEngine(_fake_model(), spec_kv_plan(4096, 3), draft_len=3, use_mtp=True)  # plan too small


def test_engine_no_mtp_propose_raises(monkeypatch, expect_error):
    monkeypatch.setattr(spec_pup, "MTPSpecEngine", _FakeInner)
    m = _fake_model()
    eng = spec_pup.PupSpecEngine(m, spec_kv_plan(8192, 7), draft_len=7, use_mtp=False)
    assert not eng.inner.kw["use_mtp"]
    with expect_error(RuntimeError, "propose needs the MTP drafter"):
        eng.propose([1], 10)


def test_plan_is_frozen(expect_error):
    plan = spec_kv_plan(8192, 3)
    assert isinstance(plan, SpecKVPlan)
    with expect_error(Exception, "cannot assign to field"):
        plan.capacity = 1
