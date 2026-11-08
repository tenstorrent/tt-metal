# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the model side of the KV-pair split's context gate (QWEN36_ATTN_KV_SPLIT=user + _MIN_CTX), over the fake buffers
and fake ttnn of the bucket tests:

* verify_stage_and_enqueue(kv_split=True) replays the split variant of the step's width (B included) with exactly the staging of
  the unsplit replay, a width without a variant replays its unsplit traces (counted), the handle / readbacks / lazy logits follow
  the variant;
* prepare / record / release of the variants: warmed after the buckets with stand-in hand-off targets, captured in descending
  order with the live shared outputs as hand-off targets (the B-row variant too), released with the buckets;
* which widths get a variant and which bodies split (_vfy_ksp_avail / _vfy_ksp_want / ksp_verify_variant_widths): packed split
  SDPA only (the adaptive-K wide family's unpacked call needs QWEN36_ATTN_KV_SPLIT_UNPACKED=1), canonical traces split only in
  the K1 "always" mode.
No device. Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_kv_split_gate_model_host.py
"""

import re
import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_bucket_model_host import (
    LOGITS,
    MAIN,
    _args,
    _bucket_model,
    _flow_model,
    _names,
    _T,
    _traces,
)
from models.demos.blackhole.qwen36.tests.test_spec_specstep_host import _Buf, _fake_ttnn
from models.demos.blackhole.qwen36.tt import kv_split as ks
from models.demos.blackhole.qwen36.tt.model import _VfyBucket

B, T, V = 8, 4, 64
KSP = {8: (60, 61, 62), 4: (70, 71, 72)}  # split variant trace ids (main, logits, topk) per width


def _ksp_model(widths=(8, 4)):
    m, bufs = _bucket_model()
    m._vfy_bk_ksp = {}
    for b in widths:
        bk = _VfyBucket(b, b * T)
        bk.kv_split = True
        bk.trace_id, bk.logits_trace_id, bk.topk_trace_id = KSP[b]
        bk.logits_rm_out = _Buf(f"ksp_logits{b}")
        bk.logits_rm_out.data = torch.full((1, 1, b * T, V), 10.0 * b, dtype=torch.bfloat16)
        bk.narrow_ready = True
        m._vfy_bk_ksp[b] = bk
    return m, bufs


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bucket,b", [(None, 8), (8, 8), (4, 4)])
def test_kv_split_replays_the_variant_of_the_width(monkeypatch, bucket, b):
    m, _ = _ksp_model()
    log = []
    _fake_ttnn(monkeypatch, log, m)
    h = m.verify_stage_and_enqueue(*_args(), read_logits=True, execute_topk=True, bucket=bucket, kv_split=True)
    assert _traces(log) == list(KSP[b]) and h.bucket == b and h.kv_split
    assert m._vfy_last_bucket is m._vfy_bk_ksp[b]
    ids, feed, lt = m.verify_sync_and_read(h)
    assert ids == list(range(B * T)), "ids come from the shared B-row buffer the variant handed off to"
    assert tuple(lt.shape) == (B * T, V) and bool((lt[: b * T] == 10.0 * b).all()) and bool((lt[b * T :] == 0).all())


def test_staging_is_the_same_split_or_not(monkeypatch):
    tokens, positions, mi = _args()
    ref_m, ref_bufs = _ksp_model()
    _fake_ttnn(monkeypatch, [], ref_m)
    ref_m.verify_stage_and_enqueue(tokens, positions, mi, page_tables=torch.ones(B, 8, dtype=torch.int32))
    m, bufs = _ksp_model()
    _fake_ttnn(monkeypatch, [], m)
    m.verify_stage_and_enqueue(tokens, positions, mi, page_tables=torch.ones(B, 8, dtype=torch.int32), kv_split=True)
    for k, v in bufs.items():
        assert v.writes == ref_bufs[k].writes, k
        if ref_bufs[k].data is not None:
            assert torch.equal(v.data, ref_bufs[k].data), k
    assert [dn._spec_win_par for dn in m._vfy_gdn] == [dn._spec_win_par for dn in ref_m._vfy_gdn]


def test_width_without_a_variant_replays_unsplit_and_counts(monkeypatch):
    m, _ = _ksp_model(widths=(8,))
    log = []
    _fake_ttnn(monkeypatch, log, m)
    h = m.verify_stage_and_enqueue(*_args(), bucket=2, kv_split=True)  # bucket 2 has no variant
    assert _traces(log) == [30] and h.bucket == 2 and not h.kv_split and m._vfy_ksp_fallbacks == 1
    del log[:]
    m._vfy_bk_ksp[8].trace_id = None  # not captured: same fallback, to the B-row trace
    h = m.verify_stage_and_enqueue(*_args(), kv_split=True)
    assert _traces(log) == [MAIN] and h.bucket is None and not h.kv_split and m._vfy_ksp_fallbacks == 2


def test_flag_off_calls_are_unchanged(monkeypatch):
    """No kv_split argument (every caller with the gate off): the bucket / B-row traces, as before; no counter appears."""
    m, _ = _ksp_model()
    log = []
    _fake_ttnn(monkeypatch, log, m)
    h = m.verify_stage_and_enqueue(*_args(), read_logits=True)
    assert _traces(log) == [MAIN, LOGITS] and h.bucket is None and h.kv_split is False
    assert "_vfy_ksp_fallbacks" not in m.__dict__


def test_lazy_logits_follow_the_split_replay(monkeypatch):
    m, _ = _ksp_model()
    log = []
    _fake_ttnn(monkeypatch, log, m)
    m.verify_traced(*_args(), kv_split=True)
    del log[:]
    lt = m.fetch_verify_logits()
    assert _traces(log) == [KSP[8][1]] and bool((lt == 80.0).all())


def test_parity_flips_once_per_split_replay(monkeypatch):
    m, _ = _ksp_model()
    _fake_ttnn(monkeypatch, [], m)
    m.verify_traced(*_args(), kv_split=True)
    assert all(dn._spec_win_par == 1 for dn in m._vfy_gdn)
    m.verify_traced(*_args(), bucket=4, kv_split=True)
    assert all(dn._spec_win_par == 0 for dn in m._vfy_gdn)


# --------------------------------------------------------------------------- #
# prepare / record / release
# --------------------------------------------------------------------------- #
def _flow(monkeypatch):
    m, log = _flow_model(monkeypatch)
    orig_body = m._forward_verify_bucket_tp

    def body(n_users, bk=None, kv_split=None):  # the bucket fake, recording the split choice
        log.append(("ksp", n_users, kv_split))
        return orig_body(n_users, bk)

    monkeypatch.setattr(m, "_forward_verify_bucket_tp", body)
    return m, log


def test_variants_warm_after_the_buckets_and_capture_in_order(monkeypatch):
    _T.live.clear()
    m, log = _flow(monkeypatch)
    m.prepare_verify_buckets((1, 2, 4))
    m.prepare_verify_ksp_variants((4, 8))
    assert sorted(m._vfy_bk_ksp) == [4, 8] and sorted(m._vfy_bk) == [1, 2, 4], "two registries"
    assert not _names(log, "begin"), "nothing is captured while warming"
    assert [x[1:] for x in _names(log, "ksp")] == [(4, None), (2, None), (1, None), (8, True), (4, True)]
    for b, bk in m._vfy_bk_ksp.items():
        assert bk.kv_split and bk.b == b and bk.rows == b * T and len(bk.standins) == 4
        assert all(t.name == "standin" for t in bk.handoff.values())
    del log[:]
    m.record_verify_buckets((1, 2, 4))
    m.record_verify_ksp_variants((4, 8))
    assert [x[1] for x in _names(log, "mem")] == ["V4", "V2", "V1", "V8s", "V4s"]
    bodies = [x for x in log if x[0] == "body"]
    for x in bodies[3:]:  # the variants: inside a trace, handing off to the LIVE shared outputs (the B-row one too)
        assert x[3] is not None and x[2]["ids"] is m._vfy_ids_out and x[2]["rows"] is m._vfy_rows_out
    for bk in m._vfy_bk_ksp.values():
        assert None not in (bk.trace_id, bk.logits_trace_id, bk.topk_trace_id) and bk.standins == []
    tids = [bk.trace_id for bk in list(m._vfy_bk.values()) + list(m._vfy_bk_ksp.values())]
    assert len(set(tids)) == 5
    # release: the variants go with the buckets
    ids = [t for bk in m._vfy_bk_ksp.values() for t in (bk.trace_id, bk.logits_trace_id, bk.topk_trace_id)]
    m._vfy_last_bucket = m._vfy_bk_ksp[8]
    del log[:]
    m.release_verify_trace()
    released = {x[1] for x in _names(log, "release")}
    assert set(ids) <= released and m._vfy_bk_ksp == {} and m._vfy_bk == {} and m._vfy_last_bucket is None


def test_variant_width_rules(monkeypatch, expect_error):
    m, log = _flow(monkeypatch)
    with expect_error(AssertionError, "must be even"):
        m.prepare_verify_bucket(3, kv_split=True)
    with expect_error(AssertionError, "must be even"):
        m.prepare_verify_bucket(16, kv_split=True)
    with expect_error(AssertionError, re.escape("prepare_verify_bucket(8, kv_split=True) first")):
        m.record_verify_bucket(8, kv_split=True)


# --------------------------------------------------------------------------- #
# which widths / bodies split
# --------------------------------------------------------------------------- #
class _Plan:
    def __init__(self, keys, parts=("verify", "draft")):
        self.keys, self.parts = set(keys), set(parts)

    def has(self, n, T):
        return (int(n), int(T)) in self.keys


class _Att0:
    """The packed-SDPA rule of TPAttention: groups of Tg candidates x nh heads fit one q tile when Tg * nh <= 32."""

    NH = 3

    def __init__(self, one_group_wide):
        self.one_group_wide = one_group_wide

    def _spec_sdpa_plan(self, T, n_users):
        if T == 4:
            return ("cfg", n_users, 4)
        if T in (6, 8) and self.one_group_wide:  # adaptive-K wide family: all T candidates in one group per user
            return ("cfg", n_users, T)
        if T == 8:
            return ("cfg", 2 * n_users, 4)
        return None

    def _spec_sdpa_pack_heads_n(self, Tg, nh):
        return nh if Tg * (nh + nh % 2) <= 32 else 0


def _avail_model(monkeypatch, gate_env=None, one_group_wide=True, keys=None):
    m, _ = _bucket_model()
    for k, v in (gate_env or {}).items():
        monkeypatch.setenv(k, v)
    m._kv_split_gate = ks.KVSplitGate.from_env()
    m._kv_split_plan = _Plan(keys or {(b, t) for b in (2, 4, 8) for t in (1, 4, 6)})
    m._vfy_kvpt_own = "own"
    m._vfy_att0 = _Att0(one_group_wide)
    m._vfy_prepared_key = (B, T, 8)
    return m


def test_avail_needs_the_packed_split_sdpa(monkeypatch):
    for k in ("QWEN36_ATTN_KV_SPLIT_MIN_CTX", "QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT", "QWEN36_ATTN_KV_SPLIT_UNPACKED"):
        monkeypatch.delenv(k, raising=False)
    m = _avail_model(monkeypatch)
    assert m._vfy_ksp_avail(8, 4) and m._vfy_ksp_avail(4, 4) and m._vfy_ksp_avail(2, 4)
    assert not m._vfy_ksp_avail(4, 6), "wide family T=6 in one group: 6 x 6 rows > 32, the unpacked split SDPA"
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT_UNPACKED", "1")
    assert m._vfy_ksp_avail(4, 6)
    monkeypatch.delenv("QWEN36_ATTN_KV_SPLIT_UNPACKED")
    assert not m._vfy_ksp_avail(6, 4), "no plan constants for the width"
    monkeypatch.setenv("QWEN36_VERIFY_ATTN_LEAN", "0")
    assert not m._vfy_ksp_avail(8, 4), "the split branch lives in the lean verify"
    monkeypatch.delenv("QWEN36_VERIFY_ATTN_LEAN")
    m._kv_split_plan.parts = {"draft"}
    assert not m._vfy_ksp_avail(8, 4), "QWEN36_ATTN_KV_SPLIT_PARTS=draft"
    m._kv_split_plan.parts = {"verify", "draft"}
    m._vfy_kvpt_own = None
    assert not m._vfy_ksp_avail(8, 4), "no per-device own table"


def test_want_canonical_only_in_always_mode(monkeypatch, expect_error):
    for k in ("QWEN36_ATTN_KV_SPLIT_MIN_CTX", "QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT", "QWEN36_ATTN_KV_SPLIT_UNPACKED"):
        monkeypatch.delenv(k, raising=False)
    m = _avail_model(monkeypatch)  # default gate: gated
    assert not m._vfy_ksp_want(8, 4, None), "gated: the canonical traces are r5's unsplit programs"
    assert m._vfy_ksp_want(8, 4, True) and not m._vfy_ksp_want(8, 4, False)
    with expect_error(AssertionError, "no split verify"):
        m._vfy_ksp_want(4, 6, True)
    m = _avail_model(monkeypatch, {"QWEN36_ATTN_KV_SPLIT_MIN_CTX": "0"})  # the K1 setting
    assert m._kv_split_gate.mode == "always" and m._vfy_ksp_want(8, 4, None) and m._vfy_ksp_want(2, 4, None)
    assert not m._vfy_ksp_want(4, 6, None), "always mode: still only where the packed split SDPA runs"
    m._kv_split_plan = None  # flag off
    assert not m._vfy_ksp_want(8, 4, None) and not m._vfy_ksp_avail(8, 4)


def test_variant_widths(monkeypatch):
    for k in ("QWEN36_ATTN_KV_SPLIT_MIN_CTX", "QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT", "QWEN36_ATTN_KV_SPLIT_UNPACKED"):
        monkeypatch.delenv(k, raising=False)
    m = _avail_model(monkeypatch)
    assert m.ksp_verify_variant_widths((8, 4, 2, 1)) == (8, 4), "default table: no 2-user verify variant"
    m = _avail_model(monkeypatch, {"QWEN36_ATTN_KV_SPLIT_MIN_CTX": "8:100,4:200,2:300"})
    assert m.ksp_verify_variant_widths((8, 4, 2, 1)) == (8, 4, 2)
    assert m.ksp_verify_variant_widths((8,)) == (8,), "buckets off: the B-row variant only"
    m._vfy_T = 6  # an adaptive-K wide family at T = 6 (one group per user): no packed split SDPA -> no variant
    assert m.ksp_verify_variant_widths((4, 2)) == ()
    m = _avail_model(monkeypatch, {"QWEN36_ATTN_KV_SPLIT_MIN_CTX": "0"})
    assert m.ksp_verify_variant_widths((8, 4, 2)) == (), "always mode captures the split as the canonical traces"
    m._kv_split_gate = None
    assert m.ksp_verify_variant_widths((8, 4)) == ()


def test_dram_budget_of_the_model(monkeypatch):
    """kv_split_dram_budget: pair weights of every attention module with them (+ MTP) and the verify variants' outputs."""
    for k in ("QWEN36_ATTN_KV_SPLIT_MIN_CTX", "QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT", "QWEN36_SPEC_K"):
        monkeypatch.delenv(k, raising=False)
    m = _avail_model(monkeypatch)
    att = types.SimpleNamespace(
        _kv_split_ready=True, tw={"wqkvg_pair": types.SimpleNamespace(dtype="DataType.BFLOAT8_B")}
    )
    m.layers = [types.SimpleNamespace(is_full_attention=i % 4 == 3, attention=att) for i in range(64)]
    m.mtp = types.SimpleNamespace(attention=att)
    m.args = types.SimpleNamespace(
        dim=5120, n_local_heads=3, n_local_kv_heads=1, head_dim=256, max_batch_size=8, vocab_size=151936
    )
    m.num_devices = 8
    bud = m.kv_split_dram_budget()
    assert bud["layers"] == 17 and bud["weights"] == ks.kv_split_pair_weight_bytes(5120, 3, 1, 256, 17, "bfloat8_b")
    assert bud["variants"] == ks.kv_split_variant_out_bytes([32, 16], 5120, 151936, 8, topk=True)
    assert bud["total"] == bud["weights"] + bud["variants"]
    m._kv_split_plan = None
    assert m.kv_split_dram_budget() is None
