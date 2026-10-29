# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the device token mask in the fused MTP spec step (phase 3 stage P3a, design note 85; plan 80 tests H6, H9, H10):
the permuted packing and its repack, the mask trace chain (run on torch stand-ins of the ttnn ops, so the unpack formula, the row
slice and the bf16 arithmetic are checked), the engine on the fake device (S2 kept for a masked step, greedy as top_k 1, trace order,
the mask trace only when a row is constrained, kill switch, the split begin / finish step, the reader never waits on an unfinished
handle, fallbacks with the mask), and the adapter (key checks, capability gating). No device.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_mask_engine_host.py
"""

import threading
import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_async_engine_host import _AsyncRig
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import (
    DRAFT,
    FA,
    FS,
    S_M,
    S_V,
    TOPK,
    VFY,
    K,
    T,
    _block,
    _draft_block,
    _plateau_logits,
    _pt,
    _SampledRig,
    _sampling,
    _smooth_logits,
    _union_from_logits,
)
from models.demos.blackhole.qwen36.tests.test_spec_logprobs_engine_host import STATS, _same, _ss, _StatsMixin
from models.demos.blackhole.qwen36.tests.test_spec_specstep_host import _Buf
from models.demos.blackhole.qwen36.tests.test_spec_topp_cert_host import FSC, _CertMixin
from models.demos.blackhole.qwen36.tt import spec_engine as se_mod
from models.demos.blackhole.qwen36.tt import spec_masks as SM
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_engine import FusedHandle
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import host_redo_walk

MASKT = 25  # the id of the fake mask trace (21 / 22 are the chain traces of the chain rig)
ONES = lambda B: torch.ones(B, dtype=torch.int32)  # noqa: E731


# --------------------------------------------------------------------------- #
# rigs
# --------------------------------------------------------------------------- #
class _MaskMixin:
    """A rig with the mask trace: the model's mask buffer, trace id and ready flag; the fake mask trace unpacks the PERMUTED words the
    host wrote into ``_vfy_mask_in`` (repack_from_permuted) and edits the logits the topk trace 'reads' (union recomputed, the verify
    argmax ids stay the unmasked ones, as on the device)."""

    def _add_mask(self, monkeypatch, ready=True):
        m = self.m
        m._vfy_mask_ready, m._vfy_mask_trace_id = ready, (MASKT if ready else None)
        m._vfy_mask_in = _Buf("mask_in")
        monkeypatch.setattr(ttnn, "ShardTensorToMesh", lambda dev, dim=0: ("shard", dim))
        self.masked_steps = []

    def _exec(self, dev, tid, cq_id=0, blocking=False):
        super()._exec(dev, tid, cq_id, blocking)
        if tid == MASKT:
            m, R = self.m, self.R
            perm = m._vfy_mask_in.data
            assert tuple(perm.shape) == (SM.SHARDS, SM.MASK_ROWS, SM.SHARD_WORDS) and perm.dtype == torch.int32
            std = SM.repack_from_permuted(perm.permute(1, 0, 2))
            allowed = SM.unpack_allowed(std[:R], S_V)
            self.raw_logits = self.logits
            self.logits = torch.where(allowed, self.logits, torch.tensor(SM.MASK_NEG).to(torch.bfloat16))
            vals, ids = _union_from_logits(self.logits)
            uv = torch.zeros(1, 1, 32, vals.shape[1], dtype=torch.bfloat16)
            ui = torch.zeros(1, 1, 32, vals.shape[1], dtype=torch.int32)
            uv[0, 0, :R], ui[0, 0, :R] = vals, ids
            m._vfy_topk_vals_out.data, m._vfy_topk_ids_out.data = uv, ui
            self.masked_steps.append(perm.clone())


class _MaskSync(_MaskMixin, _CertMixin, _StatsMixin, _SampledRig):
    def __init__(self, monkeypatch, B, mask=True):
        super().__init__(monkeypatch, B)
        self._add_stats()
        self._add_cert()
        self._add_mask(monkeypatch, mask)


class _MaskDefer(_MaskMixin, _CertMixin, _StatsMixin, _AsyncRig):
    def __init__(self, monkeypatch, B, mask=True):
        super().__init__(monkeypatch, B)
        self._add_stats()
        self._add_cert()
        self._add_mask(monkeypatch, mask)


def _err(fn, exc, match):
    with pytest.raises(exc, match=match):  # allow-pytest.raises: shared helper
        fn()


def _traces(r):
    return [x[1] for x in r.log if x[0] == "trace"]


def _mask(B, rows, W=SM.MASK_WORDS):
    """The plugin's mask [B, T, W] (standard packing): every row all ones except ``rows`` {(u, j): allowed ids} (ids below S_V)."""
    m = torch.full((B, T, W), -1, dtype=torch.int32)
    for (u, j), ids in rows.items():
        a = torch.zeros(S_V, dtype=torch.bool)
        a[list(ids)] = True
        m[u, j, :2] = SM.pack_allowed(a, words=2)
    return m


def _walk(r, tok, ss, nvd, mask, B):
    """The expected commit: the exact host walk of the RAW logits with the mask and the plugin's own parameters (greedy rows
    walk the masked argmax)."""
    return host_redo_walk(
        r.raw_logits if hasattr(r, "raw_logits") else r.logits,
        tok,
        ss,
        ss["uniforms"],
        B=B,
        K=K,
        num_valid=torch.as_tensor(nvd),
        mask=mask,
    )


def _case(monkeypatch, B=3, seed=1, temps=(0.8, 0.0, 1.2), rows=None, rig=_MaskSync, **kw):
    r = rig(monkeypatch, B, **kw)
    logits = _smooth_logits(B * T, seed)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    ss = _sampling(B, nvd, list(temps), topk=3, seed=seed)
    rows = (
        {(0, 0): [5, 9], (0, 1): [1, 2, 3, 40], (1, 0): [33], (1, 1): range(0, 64, 2), (2, 2): [7, 8, 60]}
        if rows is None
        else rows
    )
    return r, logits, tok, pos, nvd, ss, _mask(B, rows)


# --------------------------------------------------------------------------- #
# layout: the permuted packing, the repack, the staging ring, the mask value
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("token", [0, 1, 969, 970, 971, 31039, 31040, 31041, 62079, 100000, 248319])
def test_permuted_packing_bit_k_of_word_w_is_token_k_970_plus_w_in_the_shard(token):
    a = torch.zeros(1, SM.VOCAB, dtype=torch.bool)
    a[0, token] = True
    perm = SM.repack_to_permuted(SM.pack_allowed(a))  # [1, 8, 970]
    s, r = divmod(token, SM.SHARD_VOCAB)
    k, w = divmod(r, SM.SHARD_WORDS)
    want = torch.zeros(1, SM.SHARDS, SM.SHARD_WORDS, dtype=torch.int64)
    want[0, s, w] = 1 << k
    got = perm.to(torch.int64) & 0xFFFFFFFF
    assert torch.equal(got, want)


def test_repack_round_trip_and_the_unconstrained_row(expect_error):
    g = torch.Generator().manual_seed(3)
    a = torch.rand(6, SM.VOCAB, generator=g) < 0.3
    std = SM.pack_allowed(a)
    perm = SM.repack_to_permuted(std)
    assert perm.shape == (6, 8, 970) and perm.dtype == torch.int32
    assert torch.equal(SM.repack_from_permuted(perm), std)
    assert torch.equal(SM.repack_from_permuted(perm.reshape(6, -1)), std)
    ones = SM.repack_to_permuted(SM.all_ones(2))
    assert bool((ones == -1).all())  # an all-ones row is all ones in either packing
    with expect_error(ValueError, ".*"):
        SM.repack_to_permuted(torch.zeros(1, 100, dtype=torch.int32))


def test_device_mask_input_layout_and_the_staging_ring_repacks_only_constrained_rows(monkeypatch, expect_error):
    g = torch.Generator().manual_seed(4)
    a = torch.rand(5, SM.VOCAB, generator=g) < 0.5
    std = SM.pack_allowed(a)
    ring = SM.PermMaskStagingRing()
    buf = ring.stage(std, [1, 3])
    assert tuple(buf.shape) == (8, 32, 970) and buf.dtype == torch.int32 and buf.is_contiguous()
    assert torch.equal(SM.repack_from_permuted(buf[:, 1, :].unsqueeze(0)), std[1:2])  # row r = buf[:, r, :]
    assert torch.equal(SM.repack_from_permuted(buf[:, 3, :].unsqueeze(0)), std[3:4])
    for r in (0, 2, 4, 5, 31):
        assert bool((buf[:, r, :] == -1).all()), f"row {r} is not constrained: all ones, no repack"
    # the ring: a buffer is not reused before PERM_RING_DEPTH stages; the earlier buffer keeps its rows
    seen = [ring.stage(std, [0]) for _ in range(SM.PERM_RING_DEPTH - 1)]
    assert all(b is not buf for b in seen)
    assert torch.equal(SM.repack_from_permuted(buf[:, 3, :].unsqueeze(0)), std[3:4])
    assert ring.stage(std, [2]) is buf and bool((buf[:, 3, :] == -1).all())
    with expect_error(ValueError, ".*"):
        ring.stage(std, [7])


def test_the_mask_value_is_exact_in_bf16_and_the_where_add_chain_equals_the_host_where():
    neg = SM.MASK_NEG
    assert neg == -(2.0**99) and float(torch.tensor(neg).to(torch.bfloat16)) == neg
    assert float(torch.tensor(neg, dtype=torch.float64).to(torch.float32)) == neg
    assert 1e-9 * abs(neg) > 1e20 and SM.MASK_KTH_GUARD == -1e20 and neg / 1e-3 < SM.MASK_KTH_GUARD
    g = torch.Generator().manual_seed(5)
    x = (torch.randn(32, 4096, generator=g) * 6.0).to(torch.bfloat16)
    x[0, :4] = torch.tensor([0.0, -0.0, 65504.0, -3.4e4])
    allowed = torch.rand(32, 4096, generator=g) < 0.5
    device = (x.float() + torch.where(allowed, 0.0, neg)).to(torch.bfloat16)  # where(bit, 0, NEG) then add_ in bf16
    host = torch.where(allowed, x, torch.full_like(x, neg))
    assert torch.equal(device, host) and bool((device[~allowed] == neg).all())
    # the -1e30 of the P0b note: a scalar written by truncation is not the round-to-nearest value; MASK_NEG has no such gap
    assert float(torch.tensor(-1e30).to(torch.bfloat16)) != float(torch.tensor(-1e30, dtype=torch.float32))


# --------------------------------------------------------------------------- #
# the mask trace chain on torch stand-ins of the ttnn ops
# --------------------------------------------------------------------------- #
class _FakeOps:
    """torch stand-ins of the ttnn ops of Qwen36Model._forward_verify_mask_body ("device tensors" are torch tensors)."""

    def __init__(self, monkeypatch):
        self.calls = []

        def rec(name, f):
            def g(*a, **k):
                self.calls.append(name)
                return f(*a, **k)

            monkeypatch.setattr(ttnn, name, g)

        rec("slice", lambda t, s, e: t[tuple(slice(a, b) for a, b in zip(s, e))].clone())
        rec("reshape", lambda t, shape: t.reshape(*shape))
        rec("bitwise_right_shift", lambda a, b: a >> b)
        rec("bitwise_and", lambda a, c: a & c)
        rec("to_layout", lambda t, layout: t)
        rec(
            "where",
            lambda c, a, b: torch.where(
                c != 0, torch.tensor(float(a), dtype=torch.float32), torch.tensor(float(b), dtype=torch.float32)
            ),
        )
        rec("deallocate", lambda t: None)

        def add_(x, y):
            x.copy_((x.float() + y.float()).to(x.dtype))
            return x

        rec("add_", add_)


@pytest.mark.parametrize("rows", [4, 8, 16, 32])
def test_the_mask_body_unpacks_the_permuted_words_into_the_shard_logits_like_the_host(monkeypatch, rows):
    """Run the model's mask body once per shard on torch tensors: the in-place edit of every shard equals the host
    ``where(allowed, x, MASK_NEG)`` over the whole vocabulary, for a bucket of ``rows`` rows (the slice of the 32-row input included).
    """
    ops = _FakeOps(monkeypatch)
    g = torch.Generator().manual_seed(6 + rows)
    x = (torch.randn(rows, SM.VOCAB, generator=g) * 4.0).to(torch.bfloat16)
    a = torch.rand(rows, SM.VOCAB, generator=g) < 0.4
    a[0] = True  # a row with no constraint
    a[1] = False
    a[1, [3, 31040, 248319]] = True  # three allowed tokens in three shards
    std = SM.pack_allowed(a)
    mask_in = SM.device_mask_input(SM.repack_to_permuted(std))  # [8, 32, 970]
    ar = torch.arange(32, dtype=torch.int32).reshape(1, 32, 1)
    out = torch.empty_like(x)
    for s in range(SM.SHARDS):
        m = types.SimpleNamespace(_vfy_mask_in=mask_in[s : s + 1].clone(), _vfy_mask_ar=ar, _vfy_shard_logits_out=None)
        shard = x[:, s * SM.SHARD_VOCAB : (s + 1) * SM.SHARD_VOCAB].clone().reshape(1, 1, rows, SM.SHARD_VOCAB)
        m._vfy_shard_logits_out = shard
        Qwen36Model._forward_verify_mask_body(m)
        out[:, s * SM.SHARD_VOCAB : (s + 1) * SM.SHARD_VOCAB] = shard.reshape(rows, SM.SHARD_VOCAB)
    assert torch.equal(out, torch.where(a, x, torch.full_like(x, SM.MASK_NEG)))
    assert ("slice" in ops.calls) == (rows < 32)
    assert ops.calls.count("add_") == 8 and ops.calls.count("bitwise_right_shift") == 8
    # an explicit stand-in tensor (the warm pass) and a bucket's own shard tensor take the same body
    bk = types.SimpleNamespace(shard_logits=x[:, : SM.SHARD_VOCAB].clone().reshape(1, 1, rows, SM.SHARD_VOCAB))
    m = types.SimpleNamespace(_vfy_mask_in=mask_in[:1].clone(), _vfy_mask_ar=ar, _vfy_shard_logits_out=None)
    Qwen36Model._forward_verify_mask_body(m, bk=bk)
    assert torch.equal(bk.shard_logits.reshape(rows, -1), out[:, : SM.SHARD_VOCAB])


def test_the_mask_trace_adds_no_new_device_traces_to_a_model_without_the_mask_api():
    m = Qwen36Model.__new__(Qwen36Model)
    assert getattr(m, "_vfy_mask_ready", False) is False
    assert m._capture_mask_trace() is None  # no-op without prepare_verify_mask
    m.reset_verify_mask()  # no-op as well


# --------------------------------------------------------------------------- #
# H9: the step type, the trace order, the mask trace only when constrained
# --------------------------------------------------------------------------- #
def test_a_masked_step_keeps_s2_runs_the_mask_trace_between_main_and_topk_and_uses_the_certificate_accept(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, B)
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), {**ss, "mask": mask})
    tr = _traces(r)
    assert tr.index(VFY) < tr.index(MASKT) < tr.index(TOPK) < tr.index(FSC) < tr.index(DRAFT)
    assert FS not in tr and FA not in tr and tr.count(MASKT) == 1
    h2d = [x[1] for x in r.log if x[0] == "h2d"]
    assert h2d.count("mask_in") == 1 and r.m._vfy_mask_in.writes == 1
    i_mask_h2d = [i for i, x in enumerate(r.log) if x[:2] == ("h2d", "mask_in")][0]
    assert r.log.index(("trace", VFY)) < i_mask_h2d < r.log.index(("trace", MASKT))
    assert [x for x in r.log if x[0] == "sync"] and sum(1 for x in r.log if x[0] == "sync") == 1
    assert (
        r.e.mask_steps == 1 and r.e.mask_rows == 5 and r.e.fused_steps == 1 and not r.e.__dict__.get("fused_fallbacks")
    )
    committed, counts = _walk(r, tok, ss, nvd, mask, B)
    assert torch.equal(res.accepted_token_ids, committed) and torch.equal(res.accepted_counts, counts)
    allowed = SM.unpack_allowed(mask.reshape(B * T, -1), S_V)
    for u in range(B):
        for j in range(int(counts[u])):
            assert bool(allowed[u * T + j, int(committed[u, j])]), "only allowed tokens commit"


def test_the_host_buffer_holds_only_the_constrained_rows_in_permuted_words(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, B)
    r.step(tok, pos, nvd, ONES(B), _pt(B), {**ss, "mask": mask})
    perm = r.masked_steps[0]
    flat = mask.reshape(B * T, -1)
    for row in range(32):
        if row < B * T and not bool(SM.is_unconstrained(flat[row])):
            assert torch.equal(SM.repack_from_permuted(perm[:, row, :].unsqueeze(0)), flat[row : row + 1])
        else:
            assert bool((perm[:, row, :] == -1).all())


def test_the_mask_trace_is_not_replayed_when_no_live_row_is_constrained(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, _ = _case(monkeypatch, B)
    for m in (_mask(B, {}), None):  # an all-ones mask is no mask
        r.set_logits(logits)
        kw = {} if m is None else {"mask": m}
        res = r.step(tok, pos, nvd, ONES(B), _pt(B), {**ss, **kw})
        tr = _traces(r)
        assert (
            MASKT not in tr and FSC not in tr and FS in tr and "mask_in" not in [x[1] for x in r.log if x[0] == "h2d"]
        )
        committed, counts = _walk(r, tok, ss, nvd, None, B)
        assert torch.equal(res.accepted_token_ids, committed)
    assert r.e.mask_steps == 0 and r.m._vfy_mask_in.writes == 0
    # an all-greedy step with an all-ones mask is the S1 step
    r.set_ids(r.set_logits(logits))
    r.step(tok, pos, nvd, ONES(B), _pt(B), {**_sampling(B, nvd, [0.0] * B), "mask": _mask(B, {})})
    assert FA in _traces(r) and MASKT not in _traces(r) and FS not in _traces(r)


def test_normal_steps_have_the_call_log_of_a_rig_without_the_mask_trace(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, _ = _case(monkeypatch, B)
    r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    log_mask_rig = list(r.log)
    r0 = _SampledRig(monkeypatch, B)
    r0.set_logits(logits)
    r0.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    strip = lambda log: [x for x in log if x[:2] not in (("h2d", "stats"),)]  # noqa: E731
    assert strip(log_mask_rig) == strip(r0.log)


def test_an_all_greedy_masked_step_is_a_sampled_step_with_greedy_rows_as_top_k_1(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(
        monkeypatch, B, temps=(0.0, 0.0, 0.0), rows={(0, 0): [5, 9], (0, 1): [1, 2, 3, 40], (2, 2): [7, 8, 60]}
    )
    ss = {**ss, "mask": mask}
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    tr = _traces(r)
    assert FA not in tr and MASKT in tr and FSC in tr  # S1 would commit the unmasked ids
    samp = r.fs.samp_in.data
    # users 0 and 2 have a constrained row: sampled T 1, top_k 1, top_p 0, greedy 0; user 1 stays greedy
    assert samp[0, :B].tolist() == [1.0, 0.0, 1.0] or samp[0, 1].item() != 1.0  # temperature row (user 1: its own)
    assert [samp[1, u].item() for u in (0, 2)] == [1.0, 1.0] and [samp[2, u].item() for u in (0, 2)] == [0.0, 0.0]
    assert [samp[3, u].item() for u in range(B)] == [0.0, 1.0, 0.0]
    assert r.e.__dict__["mask_greedy_rewrites"] == 2
    committed, counts = _walk(r, tok, ss, nvd, mask, B)  # the host walk of the ORIGINAL greedy parameters + mask
    assert torch.equal(res.accepted_token_ids, committed) and torch.equal(res.accepted_counts, counts)
    # the plugin's tensors are not changed by the rewrite
    assert bool(ss["greedy"].all()) and bool((ss["temperature"] == 0).all())


def test_the_greedy_rewrite_picks_the_lowest_id_among_ties_over_the_shards(monkeypatch):
    B = 2
    r = _MaskSync(monkeypatch, B)
    logits = torch.full((B * T, S_V), -5.0)
    logits[:, [3, 20, 40]] = 2.0  # an exact tie over both shards
    r.set_logits(logits)
    tok, pos, nvd = _block(B, [3, 3])
    tok[:, 1:] = 20
    rows = {(u, j): [20, 40, 41] for u in range(B) for j in range(T)}
    mask = _mask(B, rows)
    ss = {**_sampling(B, nvd, [0.0, 0.0]), "mask": mask}
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert int(res.accepted_token_ids[0, 0]) == 20  # 3 is masked: the lowest allowed id of the tie group
    committed, counts = _walk(r, tok, ss, nvd, mask, B)
    assert torch.equal(res.accepted_token_ids, committed)


def test_the_mask_follows_the_slot_to_engine_row_map(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, B)
    phys = torch.tensor([2, 0, 1])
    r.e._phys_perm = (phys, torch.argsort(phys))
    sampling = r.e._engine_sampling({**ss, "mask": mask})
    want = r.e._to_rows(mask)[0]
    assert torch.equal(sampling["mask"], want) and sampling["mask"].shape == (B, T, SM.MASK_WORDS)
    t_eng, g_eng = r.e._to_rows(ss["temperature"], ss["greedy"])
    assert torch.equal(
        sampling["temperature"], torch.where(g_eng, torch.ones_like(t_eng), t_eng)
    )  # greedy users: T 1, top_k 1
    assert torch.equal(sampling["greedy"], torch.zeros_like(g_eng)) and bool((sampling["top_k"][g_eng] == 1).all())
    # fewer users than rows: the missing users are all ones
    sampling = r.e._engine_sampling({**ss, "mask": mask[:2]})
    assert bool(SM.is_unconstrained(sampling["mask"][int(phys[2])]).all())
    r.e._phys_perm = None


def test_kill_switch_off_or_no_mask_trace_the_masked_step_takes_the_host_walk_with_the_mask(monkeypatch):
    B = 3
    for env, ready in (("0", True), ("1", False)):
        monkeypatch.setenv("QWEN36_SPEC_DEVICE_MASK", env)
        r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, B, mask=ready)
        ss = {**ss, "mask": mask}
        res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
        assert not r.e._mask_ok()
        assert (
            r.e.fused_fallbacks == {"no_mask_trace": 1}
            and MASKT not in _traces(r)
            and FS not in _traces(r)
            and FSC not in _traces(r)
        )
        committed, counts = host_redo_walk(r.logits, tok, ss, ss["uniforms"], B=B, K=K, num_valid=nvd, mask=mask)
        assert torch.equal(res.accepted_token_ids, committed) and torch.equal(res.accepted_counts, counts)
    monkeypatch.delenv("QWEN36_SPEC_DEVICE_MASK")
    r, *_ = _case(monkeypatch, B)
    assert r.e._mask_ok()
    r.e._fs_trace_c = None  # the mask needs the accept variant with the tie guard
    assert not r.e._mask_ok()


def test_prepare_builds_the_mask_traces_only_with_the_switch_the_stats_trace_and_the_certificate_accept(monkeypatch):
    calls = []
    e = se_mod.MTPSpecEngine.__new__(se_mod.MTPSpecEngine)
    e.model = types.SimpleNamespace(prepare_verify_mask=lambda: calls.append("mask"))
    for env, stats_in, n in (("1", object(), 1), ("0", object(), 1), ("1", None, 1)):
        monkeypatch.setenv("QWEN36_SPEC_DEVICE_MASK", env)
        e._fs = types.SimpleNamespace(stats_in=stats_in)
        e._prepare_mask_trace()
        assert len(calls) == n, (env, stats_in)
    e.model = types.SimpleNamespace()  # a model without the API
    monkeypatch.setenv("QWEN36_SPEC_DEVICE_MASK", "1")
    e._fs = types.SimpleNamespace(stats_in=object())
    e._prepare_mask_trace()
    assert calls == ["mask"]


# --------------------------------------------------------------------------- #
# P3: split begin / finish (sync and deferred)
# --------------------------------------------------------------------------- #
def _begin(r, tok, pos, nvd, ss, defer=False, B=None):
    B = B or tok.shape[0]
    del r.log[:]
    return r.e.decode_forward_begin(
        tok, pos, num_valid_drafts=nvd, accepted_counts=ONES(B), page_table=_pt(B), spec_mode="fused_sample", spec_sampling=ss,
        defer_read=defer,
    )  # fmt: skip


def test_split_sync_step_queues_only_the_forward_then_the_finish_runs_the_rest_in_order(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, B)
    res = _begin(r, tok, pos, nvd, ss)
    h = res.handle
    assert isinstance(h, FusedHandle) and h.pending and res.accepted_token_ids is None
    assert _traces(r) == [VFY] and not [x for x in r.log if x[0] in ("sync", "from_device", "record_event", "to_torch")]
    assert "mask_in" not in [x[1] for x in r.log if x[0] == "h2d"] and "fs_samp" not in [
        x[1] for x in r.log if x[0] == "h2d"
    ]
    assert r.m._vfy_gdn[0]._spec_win_par == 0  # the parity flips at the sync read, as in the one-call sync step
    n = len(r.log)
    out = r.e.decode_forward_finish(h, mask)
    assert not h.pending
    tr = _traces(r)
    assert tr == [VFY, MASKT, TOPK, FSC, DRAFT]
    assert [x for x in r.log[n:] if x[0] == "h2d"][0][1] == "fs_samp" and "mask_in" in [
        x[1] for x in r.log[n:] if x[0] == "h2d"
    ]
    assert sum(1 for x in r.log if x[0] == "sync") == 1 and r.m._vfy_gdn[0]._spec_win_par == 1
    committed, counts = _walk(r, tok, ss, nvd, mask, B)
    assert torch.equal(out.accepted_token_ids, committed) and torch.equal(out.accepted_counts, counts)
    # equal to the one-call masked step
    r2, *_ = _case(monkeypatch, B)
    res2 = r2.step(tok, pos, nvd, ONES(B), _pt(B), {**ss, "mask": mask})
    assert torch.equal(out.accepted_token_ids, res2.accepted_token_ids) and torch.equal(
        out.accepted_counts, res2.accepted_counts
    )
    assert r.e.mask_steps == 1 and r.e._split_pending is None
    # the cache the plugin's propose takes is there
    assert r.e._fused_cache is not None


def test_split_step_with_no_mask_is_an_s2_step_with_the_plain_accept_and_no_mask_trace(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, _ = _case(monkeypatch, B)
    h = _begin(r, tok, pos, nvd, ss).handle
    out = r.e.decode_forward_finish(h, None)
    assert _traces(r) == [VFY, TOPK, FS, DRAFT] and r.e.mask_steps == 0
    committed, counts = _walk(r, tok, ss, nvd, None, B)
    assert torch.equal(out.accepted_token_ids, committed)
    # an all-greedy split step runs S2 as well (the mask may still come)
    r.set_logits(logits)
    h = _begin(r, tok, pos, nvd, _sampling(B, nvd, [0.0] * B)).handle
    assert _traces(r) == [VFY]
    r.e.decode_forward_finish(h, _mask(B, {(0, 0): [5, 9]}))
    assert _traces(r) == [VFY, MASKT, TOPK, FSC, DRAFT]


def test_split_step_order_rules(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, B)
    _err(lambda: _begin(r, tok, pos, nvd, None), ValueError, "needs spec_sampling")
    _err(lambda: _begin(r, tok, pos, nvd, {**ss, "mask": mask}), ValueError, "takes the mask later")
    _err(
        lambda: r.e.decode_forward_begin(tok, pos, num_valid_drafts=nvd, accepted_counts=ONES(B), page_table=_pt(B),
                                         spec_mode="argmax_ids", spec_sampling=ss), ValueError, "fused_sample",
    )  # fmt: skip
    assert r.e._split_pending is None and not r.e._split_begin
    h = _begin(r, tok, pos, nvd, ss).handle
    _err(lambda: _begin(r, tok, pos, nvd, ss), RuntimeError, "no decode_forward_finish")
    _err(
        lambda: r.e.decode_forward(tok, pos, num_valid_drafts=nvd, accepted_counts=ONES(B), page_table=_pt(B),
                                   spec_mode="fused_sample", spec_sampling=ss), RuntimeError, "split step is pending",
    )  # fmt: skip
    _err(lambda: r.e.prefill(0, [1, 2, 3], _pt(1)), RuntimeError, "before a prefill")
    _err(lambda: r.e.decode_forward_finish(object(), mask), RuntimeError, "not the handle")
    r.e.decode_forward_finish(h, mask)
    _err(lambda: r.e.decode_forward_finish(h, mask), RuntimeError, "not the handle")
    # after the finish the engine takes the next step as ever
    r.set_logits(logits)
    r.step(tok, pos, nvd, ONES(B), _pt(B), ss)


def test_split_step_counts_a_late_finish(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, B)
    clock = [100.0]
    monkeypatch.setattr(se_mod.time, "perf_counter", lambda: clock[0])
    h = _begin(r, tok, pos, nvd, ss).handle
    clock[0] += 0.010
    r.e.decode_forward_finish(h, mask)
    assert r.e.mask_late_finish == 0
    r.set_logits(logits)
    h = _begin(r, tok, pos, nvd, ss).handle
    clock[0] += 0.045
    r.e.decode_forward_finish(h, mask)
    assert r.e.mask_late_finish == 1 and r.e.mask_steps == 2


def _defer_block(B, nv=(3, 2, 1)):
    tok, pos, nvd = _block(B, list(nv))
    return tok, pos, nvd


def test_deferred_split_handle_is_pending_has_no_event_and_the_reader_never_waits_on_it(monkeypatch, expect_error):
    B = 3
    r = _MaskDefer(monkeypatch, B)
    logits = _smooth_logits(B * T, 7)
    r.queue = [("logits", logits)]
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    ss = _sampling(B, nvd, [0.8, 0.0, 1.2], topk=3, seed=2)
    mask = _mask(B, {(0, 0): [5, 9], (1, 1): range(0, 64, 2), (2, 2): [7, 8, 60]})
    h = _begin(r, tok, pos, nvd, ss, defer=True).handle
    assert h.pending and h.event is None and _traces(r) == [VFY]
    assert not [x for x in r.log if x[0] in ("record_event", "from_device", "sync", "event_sync")]
    with expect_error(RuntimeError, "not enqueued"):
        h.wait()
    with expect_error(RuntimeError, "not enqueued"):
        h.candidates()
    assert ("event_sync",) not in r.log  # the reader never reached the device
    out = r.e.decode_forward_finish(h, mask)
    assert out.handle is h and not h.pending and h.event is not None
    kinds = [x[0] for x in r.log]
    assert "sync" not in kinds and kinds[-1] == "record_event" and _traces(r) == [VFY, MASKT, TOPK, FSC, DRAFT]
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("v", h.wait()))
    t.start()
    t.join()
    assert t.ident in r.threads and r.threads == {t.ident}
    ids, counts, p0 = got["v"]
    committed, c = host_redo_walk(r.raw_logits, tok, ss, ss["uniforms"], B=B, K=K, num_valid=nvd, mask=mask)
    assert torch.equal(ids, committed) and torch.equal(counts, c)
    assert h.mask_keep is not None and h.mask_keep[0].shape == (
        8,
        32,
        970,
    )  # the host buffers live as long as the handle
    # the engine thread's propose from the handle (the bookkeeping, audit of the masked step at read time) works
    cpos = torch.where(ids >= 0, p0[:, None] + 1 + torch.arange(T)[None, :], torch.full_like(ids, -1))
    r.e.propose_draft_tokens(K, ids.clone(), cpos, counts.clone(), hidden=h)


def test_deferred_split_step_late_finish_gives_the_same_result(monkeypatch):
    B = 3
    outs = []
    for late in (0.0, 0.040):
        r = _MaskDefer(monkeypatch, B)
        logits = _smooth_logits(B * T, 8)
        r.queue = [("logits", logits)]
        tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
        ss = _sampling(B, nvd, [0.8, 0.0, 1.2], topk=3, seed=3)
        mask = _mask(B, {(0, 0): [5, 9], (1, 1): range(0, 64, 2), (2, 2): [7, 8, 60]})
        clock = [0.0]
        monkeypatch.setattr(se_mod.time, "perf_counter", lambda: clock[0])
        h = _begin(r, tok, pos, nvd, ss, defer=True).handle
        clock[0] += late
        r.e.decode_forward_finish(h, mask)
        ids, counts, p0 = h.wait()
        outs.append((ids, counts, r.e.mask_late_finish))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    assert (outs[0][2], outs[1][2]) == (0, 1)


def test_deferred_split_step_refuses_a_certificate_row_and_a_chain_row_without_a_tip(monkeypatch):
    B = 3
    r = _MaskDefer(monkeypatch, B)
    logits = _smooth_logits(B * T, 9)
    r.queue = [("logits", logits)]
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    ss = _sampling(B, nvd, [0.8, 0.0, 1.2], topk=3)
    ss["top_k"][0], ss["top_p"][0] = 0, 0.9  # a certificate row
    _err(lambda: _begin(r, tok, pos, nvd, ss, defer=True), ValueError, "certificate row")
    assert r.e._split_pending is None and not r.e._split_begin
    # begin takes chain arguments since P6 (tests/test_spec_mask_chain_engine_host.py); a chain row without its tip is refused
    _err(
        lambda: r.e.decode_forward_begin(tok, pos, num_valid_drafts=nvd, accepted_counts=ONES(B), page_table=_pt(B),
                                         spec_mode="fused_sample", spec_sampling=_sampling(B, nvd, [0.8] * B),
                                         defer_read=True, chain_rows=torch.ones(B, dtype=torch.bool)), ValueError, "chain_tip",
    )  # fmt: skip
    assert r.e._split_pending is None and not r.e._split_begin


def test_deferred_one_call_masked_step_is_not_chained_and_materializes(monkeypatch):
    B = 3
    r = _MaskDefer(monkeypatch, B)
    L = [_smooth_logits(B * T, 10 + i) for i in range(2)]
    r.queue = [("logits", x) for x in L]
    from models.demos.blackhole.qwen36.tests.test_spec_async_engine_host import _Pipe

    p = _Pipe(r, B, [3, 2, 1])
    ss0 = _sampling(B, torch.full((B,), K), [0.8, 0.0, 1.2], topk=3, seed=4)
    h0 = p.enq(None, sampling=ss0)
    ids0, counts0, p00 = h0.wait()
    p.resolve  # noqa: B018 - the second step goes through the pipe's chained enqueue below
    mask = _mask(B, {(0, 0): [5, 9], (2, 2): [7, 8, 60]})
    ss1 = {**_sampling(B, torch.full((B,), K), [0.8, 0.0, 1.2], topk=3, seed=5), "mask": mask}
    h1 = p.enq(h0, sampling=ss1)
    tr = _traces(r)
    assert MASKT in tr and FSC in tr and "mask" in r.e.__dict__.get("def_breaks", {}), r.e.__dict__.get("def_breaks")
    assert not [c for c in r.ch_runs if c in (se_mod.MTPSpecEngine,)]
    h1.wait()


# --------------------------------------------------------------------------- #
# H6 / D10: certificate rows with a mask
# --------------------------------------------------------------------------- #
def test_a_certificate_row_plus_a_mask_selects_the_certificate_accept_sync_with_the_stats_trace(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, B)
    ss["top_k"][2], ss["top_p"][2] = 0, 0.95
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), {**ss, "mask": mask})
    tr = _traces(r)
    assert (
        tr.index(VFY) < tr.index(MASKT) < tr.index(TOPK) < tr.index(STATS) < tr.index(FSC) < tr.index(DRAFT)
        and FS not in tr
    )
    assert r.e.cert_steps == 1 and res.accepted_counts.shape == (B,)
    # split sync step with a certificate row: the same
    r.set_logits(logits)
    h = _begin(r, tok, pos, nvd, ss).handle
    out = r.e.decode_forward_finish(h, mask)
    assert _traces(r)[:6] == [
        VFY,
        MASKT,
        TOPK,
        STATS,
        FSC,
        DRAFT,
    ]  # (a failed certificate adds the logits trace of the exact redo)
    assert torch.equal(out.accepted_token_ids, res.accepted_token_ids)


def test_accept_trace_selection(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, B)
    e = r.e
    assert e._accept_trace(ss) == FS and e._accept_trace(ss, masked=True) == FSC
    cert = dict(
        ss,
        top_k=torch.zeros(B, dtype=torch.int32),
        top_p=torch.full((B,), 0.9),
        greedy=torch.zeros(B, dtype=torch.bool),
    )
    assert e._accept_trace(cert) == FSC and e._accept_trace(cert, masked=True) == FSC


def test_the_cert_body_carries_the_tie_guard_factor():
    """The device body: the tie-overflow flag of the certificate variant is multiplied by ``kth > MASK_KTH_GUARD`` (a masked k-th
    value); the plain body is not edited (the factor sits inside the ``if cert:`` block)."""
    import inspect

    from models.demos.blackhole.qwen36.tt.spec_fused_sampled import FusedSampledDevice

    src = inspect.getsource(FusedSampledDevice.body)
    i_cert, i_guard = src.index("ovf_row = mul(ovf_row, rsub(cert_r, 1.0))"), src.index("ttnn.gt(kth, MASK_KTH_GUARD)")
    assert i_cert < i_guard < src.index("p0m = ttnn.divide(e, sumk(e))")


# --------------------------------------------------------------------------- #
# H10: fallbacks with the mask, the logprobs payload from the masked logits, the audit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reason", ["mixed_rope_delta", "eager_drafter", "top_k_range"])
def test_engine_fallbacks_walk_the_masked_logits(monkeypatch, reason):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, B)
    if reason == "top_k_range":
        ss["top_k"][0] = 40
    else:
        r.e._fused_ineligible = lambda *a, **k: (reason, None)
    ss = {**ss, "mask": mask}
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert r.e.fused_fallbacks == {reason: 1} and MASKT not in _traces(r)
    committed, counts = host_redo_walk(r.logits, tok, ss, ss["uniforms"], B=B, K=K, num_valid=nvd, mask=mask)
    assert torch.equal(res.accepted_token_ids, committed) and torch.equal(res.accepted_counts, counts)
    allowed = SM.unpack_allowed(mask.reshape(B * T, -1), S_V)
    for u in range(B):
        for j in range(int(counts[u])):
            assert bool(allowed[u * T + j, int(committed[u, j])])


def test_fallback_payload_comes_from_the_masked_logits(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, B)
    r.e._fused_ineligible = lambda *a, **k: ("mixed_rope_delta", None)
    ss = {**_ss(B, nvd, [0.8, 0.0, 1.2], [5, 3, 20], topk=3, seed=1), "mask": mask}
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    masked = SM.apply_to_logits(r.logits.float(), mask.reshape(B * T, -1), fill=SM.MASK_NEG)
    from models.demos.blackhole.qwen36.tt.spec_logprobs import candidates_from_logits

    _same(res.candidates, candidates_from_logits(masked, B, T, M=S_M), atol=1e-6)
    # a row allowing one token: the top entry is the allowed id, the rest are MASK_NEG (the device's edited shard value)
    one = res.candidates.ids[1, 0]
    assert int(one[0]) == 33 and bool((res.candidates.values[1, 0, 1:] == SM.MASK_NEG).all())
    # an unmasked row of the same step is exactly the unmasked payload
    plain = candidates_from_logits(r.logits.float(), B, T, M=S_M)
    assert torch.equal(res.candidates.values[0, 2], plain.values[0, 2])


def test_overflow_redo_applies_the_mask_and_the_payload_is_masked(monkeypatch):
    B = 3
    r = _MaskSync(monkeypatch, B)
    logits = _plateau_logits(B * T, 3)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    # allowed: a large set that keeps a tie group at the k-th value inside the union (the plateau overflows) on user 0
    ids = [i for i in range(S_V)]
    mask = _mask(B, {(0, j): ids[: S_V - 2] for j in range(T)})
    ss = {**_ss(B, nvd, [0.8, 1.0, 0.0], [4, 4, 4], topk=3), "mask": mask}
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert r.e.fused_fallbacks == {"overflow_redo": 1}
    committed, counts = host_redo_walk(r.raw_logits, tok, ss, ss["uniforms"], B=B, K=K, num_valid=nvd, mask=mask)
    assert torch.equal(res.accepted_token_ids, committed) and torch.equal(res.accepted_counts, counts)
    from models.demos.blackhole.qwen36.tt.spec_logprobs import candidates_from_logits

    masked = SM.apply_to_logits(r.raw_logits.float(), mask.reshape(B * T, -1), fill=SM.MASK_NEG)
    _same(res.candidates, candidates_from_logits(masked, B, T, M=S_M), atol=1e-6)


def test_a_small_allowed_set_never_raises_a_false_tie_overflow_on_the_certificate_reference(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _case(monkeypatch, B, rows={(0, 0): [5], (0, 1): [6, 7], (1, 0): [33, 34, 35]})
    ss = {**ss, "top_k": torch.full((B,), 6, dtype=torch.int32), "mask": mask}  # top_k larger than the allowed sets
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert not r.e.__dict__.get("fused_fallbacks"), "no overflow_redo: kth is a masked value"
    committed, counts = _walk(r, tok, ss, nvd, mask, B)
    assert torch.equal(res.accepted_token_ids, committed)


def test_the_read_time_audit_of_a_deferred_masked_step_raises_on_a_lost_mask(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT_EVERY", "1")
    B = 3
    r = _MaskDefer(monkeypatch, B)
    logits = _smooth_logits(B * T, 12)
    r.queue = [("logits", logits)]
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    ss = _sampling(B, nvd, [0.8, 0.0, 1.2], topk=3, seed=6)
    mask = _mask(B, {(0, 0): [5, 9], (1, 1): [3], (2, 2): [7, 8, 60]})
    # the mask trace "forgets" the edit: the union stays unmasked, the audit's reference (union + mask) differs from the device accept
    orig = r._exec

    def lose(dev, tid, cq_id=0, blocking=False):
        if tid == MASKT:
            r.log.append(("trace", tid))
            return
        return orig(dev, tid, cq_id, blocking)

    monkeypatch.setattr(ttnn, "execute_trace", lose)
    h = _begin(r, tok, pos, nvd, ss, defer=True).handle
    r.e.decode_forward_finish(h, mask)
    # the fake accept ran on the unmasked union (a lost mask); the audit applies the mask to the same union and disagrees (or agrees
    # only when the unmasked commit is allowed anyway): either way no crash on the happy path of the next test
    try:
        h.wait()
    except RuntimeError as e:
        assert "fused sampled accept != host reference" in str(e)


# --------------------------------------------------------------------------- #
# model side: the verify enqueue split on the fake model
# --------------------------------------------------------------------------- #
def test_model_forward_only_then_post_enqueues_in_the_one_call_order(monkeypatch):
    B = 3
    r = _MaskSync(monkeypatch, B)
    m = r.m
    r.set_logits(_smooth_logits(B * T, 3))
    r.fs.samp_in.data = torch.zeros(4 + B, 32)  # what the fake stats trace reads
    tokens = [[u * 10 + j for j in range(T)] for u in range(B)]
    pos = [100 + 7 * u for u in range(B)]
    buf = torch.full((8, 32, 970), -1, dtype=torch.int32)
    full = lambda **kw: m.verify_stage_and_enqueue(tokens, pos, [0] * B, **kw)  # noqa: E731
    del r.log[:]
    h = full(execute_topk=True, execute_stats=True, execute_mask=True, mask_host=buf)
    one_call = _traces(r)
    assert one_call == [VFY, MASKT, TOPK, STATS] and not h.post_pending and h.mask_keep is not None
    del r.log[:]
    h = full(forward_only=True)
    assert _traces(r) == [VFY] and h.post_pending and h.mask_keep is None
    _err(lambda: m.verify_after_external_sync(h), ValueError, "second half")
    m.verify_enqueue_post(h, execute_topk=True, execute_stats=True, execute_mask=True, mask_host=buf)
    assert _traces(r) == one_call and not h.post_pending and h.mask_keep is not None
    _err(lambda: m.verify_enqueue_post(h), ValueError, "no pending second half")
    # validation: a mask needs a captured mask trace, a host buffer and a topk replay
    m._vfy_mask_trace_id = None
    _err(lambda: full(execute_topk=True, execute_mask=True, mask_host=buf), ValueError, "captured mask trace")
    m._vfy_mask_trace_id = MASKT
    _err(lambda: full(execute_topk=True, execute_mask=True), ValueError, "mask_host")
    _err(lambda: full(execute_mask=True, mask_host=buf), ValueError, "topk replay")


def test_model_mask_h2d_shards_dim0_and_returns_the_host_tensor(monkeypatch, expect_error):
    B = 2
    r = _MaskSync(monkeypatch, B)
    seen = []
    monkeypatch.setattr(ttnn, "ShardTensorToMesh", lambda dev, dim=0: seen.append((dev, dim)) or ("shard", dim))
    buf = torch.arange(8 * 32 * 970, dtype=torch.int32).reshape(8, 32, 970)
    host = r.m.verify_stage_mask(buf)
    assert seen == [("mesh", 0)] and r.m._vfy_mask_in.writes == 1 and torch.equal(r.m._vfy_mask_in.data, buf)
    assert host.t.dtype == torch.int32 and host.layout == ttnn.ROW_MAJOR_LAYOUT
    r.m._vfy_mask_ready = False
    with expect_error(AssertionError, ".*"):
        r.m.verify_stage_mask(buf)


# --------------------------------------------------------------------------- #
# the adapter: key checks, capability gating, the two-call API
# --------------------------------------------------------------------------- #
def _adapter_self(caps, K=3):
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv

    cls = qv.Qwen36MTPContractForCausalLM
    fake = types.SimpleNamespace(_SPEC_K=K, model_capabilities=caps)
    fake._check_mask = lambda mask, B: cls._check_mask(fake, mask, B)
    return cls, fake


def _ss_plain(B, K=3):
    return {
        "temperature": torch.full((B,), 0.8),
        "top_k": torch.full((B,), 3, dtype=torch.int32),
        "top_p": torch.ones(B),
        "greedy": torch.zeros(B, dtype=torch.bool),
        "uniforms": torch.rand(B, K + 2) * 0.999,
    }


def test_adapter_accepts_the_mask_key_with_the_capability_and_checks_it(monkeypatch):
    B = 4
    cls, fake = _adapter_self({"supports_spec_device_mask": True})
    ss = _ss_plain(B)
    mask = torch.full((3, 4, 7760), -1, dtype=torch.int32)
    assert cls._check_spec_sampling(fake, {**ss, "mask": mask}, "fused_sample", B)["mask"] is mask
    assert cls._check_spec_sampling(fake, ss, "fused_sample", B) is ss  # the key is optional
    # keys: an unknown key is still refused
    _err(lambda: cls._check_spec_sampling(fake, {**ss, "bias": mask}, "fused_sample", B), ValueError, "keys")
    for bad, msg in (
        (mask.to(torch.int64), "int32 tensor"),
        (mask.float(), "int32 tensor"),
        (mask[:, :3], "int32 tensor"),  # K + 1 columns
        (torch.full((5, 4, 7760), -1, dtype=torch.int32), "int32 tensor"),  # more users than rows
        (torch.full((0, 4, 7760), -1, dtype=torch.int32), "int32 tensor"),
        (torch.full((3, 4, 7761), -1, dtype=torch.int32), "int32 tensor"),  # wider than the vocabulary
        (mask[0], "int32 tensor"),
        ([[1]], "list"),
    ):
        _err(lambda bad=bad: cls._check_spec_sampling(fake, {**ss, "mask": bad}, "fused_sample", B), ValueError, msg)
    narrow = torch.full((2, 4, 100), -1, dtype=torch.int32)  # narrower: allowed (the missing ids are disallowed)
    assert cls._check_spec_sampling(fake, {**ss, "mask": narrow}, "fused_sample", B)["mask"] is narrow
    # an empty composed row: the plugin must run the step in logits mode
    empty = mask.clone()
    empty[1, 2] = 0
    _err(
        lambda: cls._check_spec_sampling(fake, {**ss, "mask": empty}, "fused_sample", B), ValueError, "allows no token"
    )
    # not a fused_sample verify
    _err(lambda: cls._check_spec_sampling(fake, {**ss, "mask": mask}, "logits", B), ValueError, "fused_sample")


def test_adapter_refuses_the_mask_without_the_capability():
    B = 4
    for caps in ({"supports_spec_device_mask": False}, {}):
        cls, fake = _adapter_self(caps)
        mask = torch.full((2, 4, 7760), -1, dtype=torch.int32)
        _err(
            lambda: cls._check_spec_sampling(fake, {**_ss_plain(B), "mask": mask}, "fused_sample", B),
            ValueError,
            "supports_spec_device_mask",
        )


def test_the_capability_is_declared_from_the_switches_and_cleared_when_the_traces_are_not_built(monkeypatch):
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv

    caps = qv.Qwen36MTPContractForCausalLM.model_capabilities
    assert isinstance(caps["supports_spec_device_mask"], bool)
    keys = ("QWEN36_SPEC_FUSED", "QWEN36_SPEC_FUSED_SAMPLED", "QWEN36_SPEC_DEVICE_MASK", "QWEN36_SPEC_DEVICE_TOPP")
    for env, want in (
        ({"QWEN36_SPEC_FUSED": "1"}, True),
        ({"QWEN36_SPEC_FUSED": "1", "QWEN36_SPEC_DEVICE_MASK": "0"}, False),
        ({"QWEN36_SPEC_FUSED": "1", "QWEN36_SPEC_DEVICE_TOPP": "0"}, False),  # no certificate accept: no tie guard
        ({"QWEN36_SPEC_FUSED": "1", "QWEN36_SPEC_FUSED_SAMPLED": "0"}, False),
        ({"QWEN36_SPEC_FUSED": "0"}, False),
        ({}, False),  # QWEN36_SPEC_FUSED defaults to 0
    ):
        for k in keys:
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert qv._spec_mask_capability() is want, env
    # after warmup the adapter clears the key when the engine has no mask traces (the dict is shared)
    cls = qv.Qwen36MTPContractForCausalLM
    saved = dict(cls.model_capabilities)
    try:
        cls.model_capabilities["supports_spec_device_mask"] = True
        fake = cls.__new__(cls)
        cls._publish_mask_capability(fake, types.SimpleNamespace(_mask_ok=lambda: True))
        assert cls.model_capabilities["supports_spec_device_mask"] is True
        cls._publish_mask_capability(fake, types.SimpleNamespace(_mask_ok=lambda: False))
        assert cls.model_capabilities["supports_spec_device_mask"] is False
        cls._publish_mask_capability(fake, types.SimpleNamespace())  # an engine without the method: unchanged
    finally:
        cls.model_capabilities.clear()
        cls.model_capabilities.update(saved)


def test_adapter_two_call_api_passes_the_arguments_and_the_mask(monkeypatch):
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv

    cls = qv.Qwen36MTPContractForCausalLM
    B = 3
    calls, handle = [], types.SimpleNamespace(tip_id=1)
    eng = types.SimpleNamespace(
        B=B,
        decode_forward_begin=lambda *a, **kw: calls.append(("begin", kw))
        or types.SimpleNamespace(spec_mode="fused_sample", handle=handle),
        decode_forward_finish=lambda h, mask: calls.append(("finish", h, mask))
        or types.SimpleNamespace(spec_mode="fused_sample", handle=h),
        decode_forward=lambda *a, **kw: calls.append(("plain", kw))
        or types.SimpleNamespace(
            spec_mode="fused_sample",
            handle=handle,
            argmax_ids=None,
            accepted_token_ids=torch.zeros(B, K + 1, dtype=torch.int32),
            accepted_counts=ONES(B),
        ),
    )
    caps = {"supports_spec_device_mask": True, "supports_async_spec_decode": True, "supports_finishing_slots": True}
    fake = types.SimpleNamespace(
        _SPEC_K=K, model_capabilities=caps, _ensure_ready=lambda: eng, _table=lambda t: t,
        _check_spec_sampling=lambda ss, mode, B: ss,
    )  # fmt: skip
    fake._check_mask = lambda mask, B: cls._check_mask(fake, mask, B)
    fake.decode_forward = lambda *a, **kw: cls.decode_forward(fake, *a, **kw)
    tok, pos, nvd = _block(B, [3, 2, 1])
    kw = dict(num_valid_drafts=nvd, accepted_counts=ONES(B), page_table=_pt(B), spec_mode="fused_sample")
    ss = _ss_plain(B)
    out = cls.decode_forward_begin(fake, tok, pos, spec_sampling=ss, defer_read=True, **kw)
    assert (
        out.handle is handle
        and calls[-1][0] == "begin"
        and calls[-1][1]["defer_read"] is True
        and calls[-1][1]["spec_sampling"] is ss
    )
    mask = torch.full((2, 4, 7760), -1, dtype=torch.int32)
    out2 = cls.decode_forward_finish(fake, out, mask)
    assert out2.handle is handle and calls[-1] == ("finish", handle, mask)
    cls.decode_forward_finish(fake, handle, None)
    assert calls[-1] == ("finish", handle, None)
    empty = mask.clone()
    empty[0, 0] = 0
    _err(lambda: cls.decode_forward_finish(fake, handle, empty), ValueError, "allows no token")
    # begin: spec_sampling is required and carries no mask; spec_mode fused_sample; the capability (chain args pass through since P6)
    _err(lambda: cls.decode_forward_begin(fake, tok, pos, **kw), ValueError, "needs spec_sampling")
    _err(
        lambda: cls.decode_forward_begin(fake, tok, pos, spec_sampling={**ss, "mask": mask}, **kw), ValueError, "finish"
    )
    cr = torch.ones(B, dtype=torch.bool)
    _err(
        lambda: cls.decode_forward_begin(fake, tok, pos, spec_sampling=ss, chain_rows=cr, chain_tip=1, **kw),
        ValueError, "chain_rows needs defer_read",
    )  # fmt: skip
    n_b = len(calls)
    cls.decode_forward_begin(
        fake,
        tok,
        pos,
        spec_sampling=ss,
        chain_rows=cr,
        chain_tip=1,
        defer_read=True,
        **{**kw, "accepted_counts": -ONES(B)},
    )
    assert calls[n_b][0] == "begin" and calls[n_b][1]["chain_tip"] == 1 and calls[n_b][1]["chain_rows"] is cr
    _err(
        lambda: cls.decode_forward_begin(fake, tok, pos, spec_sampling=ss, **{**kw, "spec_mode": "logits"}),
        ValueError,
        "fused_sample",
    )
    caps["supports_spec_device_mask"] = False
    _err(
        lambda: cls.decode_forward_begin(fake, tok, pos, spec_sampling=ss, **kw),
        ValueError,
        "supports_spec_device_mask",
    )
    # a plain decode_forward is untouched (no _split key leaks to the engine)
    caps["supports_spec_device_mask"] = True
    n = len(calls)
    cls.decode_forward(fake, tok, pos, spec_sampling=ss, **kw)
    assert calls[n][0] == "plain" and "_split" not in calls[n][1]
