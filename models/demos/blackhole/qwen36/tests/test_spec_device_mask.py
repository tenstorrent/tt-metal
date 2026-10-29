# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of the token mask in the fused spec step (phase 3 stage P3a, work/research_1026/85; plan 80 section 7.2). One Galaxy
slot (TP mesh, MESH_DEVICE=P150x8), ONE pytest node per process (run_p3_device.sh). PREPARED, NOT RUN when written.

D1   test_d1_mask_trace_bitwise_all_buckets: the shard logits edited by the mask trace equal the host ``where(allowed, x, MASK_NEG)`` bit
     for bit (dense, 2 allowed, all ones, one allowed per shard, allowed only outside the top-64), buckets 1 / 2 / 4 / 8.
D2   test_d2_masked_union_and_stats: the topk union of a masked step = the global top-64 of the masked logits; the stats lse = float64 lse
     of the masked logits.
D3   test_d3_normal_rows_bit_identical: the plain accept and the certificate variant give the same result_out / pack_out / rs_pt_out on
     normal rows; committed streams equal with the mask switch on and off; shared addresses unchanged (checked at capture).
D4   test_d4_masked_accept_vs_reference: structure rows (2-8 allowed), dense rows, invalid drafts at columns 0..2, top_k 1..63, greedy as
     top_k 1, certificate + mask; the engine audit (every step, fused_sampled_reference with the mask) must not raise; commits are allowed.
D5   test_d5_split_step_late_finish: begin / finish with a host sleep of 0-40 ms between: the same committed tokens as the one-call step.
D6   test_d6_no_compile_after_capture_and_trace_memory: every (bucket x mask on/off x cert on/off) with program-cache misses forbidden; the
     trace-region bytes after each capture are logged.
D8m  test_d8m_fallback_with_mask: engine fallbacks (forced reasons) with a mask: committed tokens equal the exact host walk of the logits
     trace rows with the mask.
D9   test_d9_mask_timing: mask trace device time per bucket (gate 1.0 ms), H2D of the mask (gate 0.4 ms).
D10  test_d10_cert_mask_logprobs_one_step: a certificate row, a masked row and a logprobs row in one sync step: allowed commits, masked payload.
D5b  test_d5b_chain_soak_masked_row (stage P6, work/research_1026/92 section C): 1000 chained split steps, one masked row (toy grammar).
D5c  test_d5c_late_mask_in_chain: the finish delayed by a host sleep of 0 / 10 / 20 / 35 / 45 ms: same streams, late counter.
D5d  test_d5d_preempt_while_pending: the masked row leaves / all masked rows leave / the bucket falls while the split step is pending.
D9c  test_d9c_phase2_enqueue_latency: host time of the finish (phase 2 enqueue) per bucket.

Run (one node per process): work/scratch/phase3/run_p3_device.sh <tag> <node name>
"""

import json
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_async_engine import AsyncDriver
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import BIG as _CHAIN_BIG
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _marks as _cmarks
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine import _admit_all
from models.demos.blackhole.qwen36.tests.test_spec_fused_sampled_engine import SampledDriver
from models.demos.blackhole.qwen36.tt import spec_masks as SM
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import host_redo_walk, merge_redo
from models.demos.blackhole.qwen36.tt.spec_logprobs import _stable_topm_row, candidates_from_union, stats_from_logits

OUT_DIR = os.environ.get("P3_OUT_DIR", "/home/ttuser/atupe/qwen-or/work/scratch/phase3")
M = 64
STEPS = int(os.environ.get("P3_STEPS", "10"))
CASES = ("dense", "two", "ones", "per_shard", "outside_top64", "structure", "tail")


def _case_rows(kind, x, g):
    """Allowed bool [R, V] of a mask case for logits x [R, V] (rows are independent)."""
    R, V = x.shape
    if kind == "dense":
        return torch.rand(R, V, generator=g) < 0.5
    if kind == "ones":
        return torch.ones(R, V, dtype=torch.bool)
    a = torch.zeros(R, V, dtype=torch.bool)
    for r in range(R):
        if kind == "two":
            a[r, torch.randint(0, V, (2,), generator=g)] = True
        elif kind == "per_shard":
            a[
                r,
                torch.arange(SM.SHARDS) * SM.SHARD_VOCAB + torch.randint(0, SM.SHARD_VOCAB, (SM.SHARDS,), generator=g),
            ] = True
        elif kind == "outside_top64":
            low = torch.topk(x[r].float(), 200, largest=False).indices
            a[r, low[torch.randperm(200, generator=g)[:5]]] = True
        elif kind == "structure":
            a[r, torch.randint(0, V, (int(torch.randint(2, 9, (1,), generator=g)),), generator=g)] = True
        elif kind == "tail":  # the last ids of the vocabulary only (shard 7, last words)
            a[r, V - 1 - torch.randperm(40, generator=g)[: 1 + r % 7]] = True
    return a


class MaskDriver(SampledDriver):
    """SampledDriver whose spec_sampling carries a ``mask`` ([B, 1+K, 7760] standard packing) built by ``mask_fn(step, B)`` (None = no
    mask), per-user greedy / top_k from ``params``, and a record of the sampling of the last step."""

    def __init__(self, eng, seed, mask_fn=None, params=None, split_sleep=None, **kw):
        super().__init__(eng, seed, **kw)
        self.mask_fn, self.params, self.split_sleep = mask_fn, params, split_sleep
        self.last_samp = self.last_mask = None

    def _spec_sampling(self):
        s = super()._spec_sampling()
        B = tc.B
        if self.params is not None:
            for k, v in self.params.items():
                s[k] = torch.as_tensor(v).clone()[:B] if torch.as_tensor(v).numel() >= B else s[k]
        if self.mask_fn is not None:
            self.last_mask = self.mask_fn(self.steps, B)
            if self.last_mask is not None and self.split_sleep is None:
                s["mask"] = self.last_mask
        self.last_samp = s
        return s

    def step(self, skip=frozenset()):
        eng = self.eng
        if self.split_sleep is None:
            return super().step(skip)
        orig = eng.decode_forward  # split: begin, sleep, finish

        def split(*a, **kw):
            eng.__dict__.pop(
                "decode_forward", None
            )  # decode_forward_begin calls self.decode_forward: no recursion into this patch
            ss = self._spec_sampling()
            kw = {k: v for k, v in kw.items() if k not in ("chain_rows", "chain_tip", "spec_sampling")}
            h = eng.decode_forward_begin(*a, **kw, spec_sampling=ss)
            time.sleep(self.split_sleep(self.steps))
            return eng.decode_forward_finish(h.handle, self.last_mask)

        eng.decode_forward = split
        try:
            super(SampledDriver, self).step(skip)
        finally:
            eng.__dict__.pop("decode_forward", None)
        self.n_sampled += 1


def _random_mask_fn(seed, kinds=CASES, rows_per_user=2):
    g = torch.Generator().manual_seed(seed)

    def fn(step, B):
        K1 = tc.K + 1
        m = torch.full((B, K1, SM.MASK_WORDS), -1, dtype=torch.int32)
        for u in range(B):
            kind = kinds[(step + u) % len(kinds)]
            for j in range(min(rows_per_user, K1)):
                a = _case_rows(kind, torch.randn(1, SM.VOCAB, generator=g), g)
                a[0, 0:1] |= (u + j) % 5 == 0  # a few rows allow token 0 as well
                m[u, j] = SM.pack_allowed(a)[0]
        return m

    return fn


def _setup(mesh_device, monkeypatch, env=None):
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    model, eng, tokenizer, prompts = tc._setup(mesh_device)
    assert eng._fa_trace is not None and eng._fs_trace is not None, "fused traces missing"
    return model, eng, prompts


def _run(drv, prompts, n, max_new, steps, finish=True):
    _admit_all(drv, prompts, n, max_new)
    try:
        for _ in range(steps):
            if not drv.live():
                break
            drv.step()
    finally:
        if finish:
            for row in drv.live():
                drv._finish(row)


def _shard_logits(model, bucket):
    t = model._vfy_shard_logits_out if bucket in (None, model._vfy_B) else model._vfy_bk[bucket].shard_logits
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(model.device, dim=3)).reshape(-1, model.vocab_size)


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_d1_mask_trace_bitwise_all_buckets(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "0")
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    assert model._vfy_mask_trace_id is not None and all(b.mask_trace_id is not None for b in model._vfy_bk.values())
    bad, seen = [], set()
    for n in (1, 2, 4, 8):
        for kind in CASES:
            drv = MaskDriver(eng, 50 + n, mask_fn=_random_mask_fn(n, kinds=(kind,), rows_per_user=tc.K + 1))
            _admit_all(drv, prompts, n, 100)
            try:
                for _ in range(3):
                    if not drv.live():
                        break
                    drv.step()
                    bucket = model._vfy_last_bucket.b if model._vfy_last_bucket is not None else model._vfy_B
                    if drv.last_mask is None or "mask" not in (drv.last_samp or {}):
                        continue
                    edited = _shard_logits(model, bucket)
                    rows = bucket * (tc.K + 1)
                    raw = model.fetch_verify_logits()[
                        :rows
                    ].float()  # logits trace: unmasked recompute of the same rows
                    flat = eng._to_rows(drv.last_mask)[0].reshape(-1, SM.MASK_WORDS)[:rows]
                    want = torch.where(
                        SM.unpack_allowed(flat, model.vocab_size),
                        raw.to(torch.bfloat16),
                        torch.tensor(SM.MASK_NEG).to(torch.bfloat16),
                    )
                    got = edited[:rows]
                    if not torch.equal(got.float(), want.float()):
                        bad.append((n, kind, bucket, int((got.float() != want.float()).sum())))
                    seen.add(bucket)
            finally:
                for row in drv.live():
                    drv._finish(row)
    logger.info(f"[mask D1] buckets seen {sorted(seen)}; mismatches {bad}")
    assert not bad, f"edited shard != host where(): {bad}"
    assert {1, 2, 4, 8} <= seen, f"buckets covered {sorted(seen)}"


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_d2_masked_union_and_stats(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "0")
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    errs, bad_rows = [], 0
    for kind in ("dense", "structure", "outside_top64", "two"):
        drv = MaskDriver(eng, 61, mask_fn=_random_mask_fn(3, kinds=(kind,), rows_per_user=tc.K + 1))
        _admit_all(drv, prompts, 4, 100)
        try:
            for _ in range(4):
                if not drv.live():
                    break
                orig = model.verify_stage_and_enqueue
                model.verify_stage_and_enqueue = lambda *a, _o=orig, **k: _o(
                    *a, **{**k, "execute_stats": bool(k.get("execute_topk"))}
                )
                try:
                    drv.step()
                finally:
                    del model.verify_stage_and_enqueue
                if "mask" not in (drv.last_samp or {}):
                    continue
                bucket = model._vfy_last_bucket.b if model._vfy_last_bucket is not None else model._vfy_B
                rows = bucket * (tc.K + 1)
                flat = eng._to_rows(drv.last_mask)[0].reshape(-1, SM.MASK_WORDS)[:rows]
                raw = model.fetch_verify_logits()[:rows].float()
                masked = SM.apply_to_logits(raw.to(torch.bfloat16).float(), flat, fill=SM.MASK_NEG)
                width = model.num_devices * model._vfy_topk_m
                uv = ttnn.to_torch(ttnn.get_device_tensors(model._vfy_topk_vals_out)[0]).reshape(-1, width)[:rows]
                ui = ttnn.to_torch(ttnn.get_device_tensors(model._vfy_topk_ids_out)[0]).reshape(-1, width)[:rows]
                cv, ci = candidates_from_union(uv, ui, model.vocab_size // model.num_devices, M)
                for r in range(rows):
                    wv, wi = _stable_topm_row(masked[r], M)
                    bad_rows += not (torch.equal(cv[r], wv) and torch.equal(ci[r], wi))
                l_raw, _ = model.fetch_verify_stats()
                m64, lr64, _ = stats_from_logits(masked, torch.ones(rows))
                errs.append(float(((m64 + l_raw[:rows].double().log()) - (m64 + lr64.log())).abs().max()))
        finally:
            for row in drv.live():
                drv._finish(row)
    logger.info(
        f"[mask D2] union bad rows {bad_rows}, max lse error {max(errs) if errs else None} over {len(errs)} steps"
    )
    assert errs and bad_rows == 0 and max(errs) <= 1e-4


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_d3_normal_rows_bit_identical(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "0")
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    assert eng._fs_trace_c is not None
    fs, dev = eng._fs, eng.mesh
    drv = MaskDriver(eng, 71)  # sampled steps, no mask
    _admit_all(drv, prompts, 8, 100)
    diffs = []
    try:
        for i in range(6):
            drv.step()
            rd = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()  # noqa: E731
            outs = {}
            for name, tid in (("plain", eng._fs_trace), ("cert", eng._fs_trace_c)):
                ttnn.execute_trace(
                    dev, tid, cq_id=0, blocking=True
                )  # same staged inputs, same union: replay both variants
                outs[name] = [rd(fs.result_out), rd(fs.pack_out), rd(fs.rs_pt_out)]
            for k, (a, b) in enumerate(zip(outs["plain"], outs["cert"])):
                if not torch.equal(a, b):
                    diffs.append((i, ("result_out", "pack_out", "rs_pt_out")[k]))
    finally:
        for row in drv.live():
            drv._finish(row)
    assert not diffs, f"plain vs certificate variant differ on normal rows: {diffs}"
    assert eng.mask_steps == 0
    # committed streams with the mask switch on (no mask row) and the shared-address check ran at capture (_check_shared_addrs)
    assert eng._addr_snap is not None


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_d4_masked_accept_vs_reference(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "1")
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT_EVERY", "1")
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    B = tc.B
    scenarios = {
        "structure": dict(kinds=("structure",), params=None),
        "dense": dict(kinds=("dense",), params=None),
        "top_k_1": dict(kinds=("structure", "dense"), params={"top_k": torch.ones(B, dtype=torch.int32)}),
        "top_k_63": dict(kinds=("dense", "two"), params={"top_k": torch.full((B,), 63, dtype=torch.int32)}),
        "greedy_top1": dict(kinds=("structure", "two"), params={"greedy": torch.ones(B, dtype=torch.bool)}),
        "cert_mask": dict(
            kinds=("dense", "structure"),
            params={"top_k": torch.zeros(B, dtype=torch.int32), "top_p": torch.full((B,), 0.9)},
        ),
    }
    redo_by_name = {}
    for name, sc in scenarios.items():
        drv = MaskDriver(
            eng, 80, mask_fn=_random_mask_fn(9, kinds=sc["kinds"], rows_per_user=tc.K + 1), params=sc["params"]
        )
        before = eng.fused_audited_steps
        redo_before = eng.fused_fallbacks.get("overflow_redo", 0)
        checked = []
        if name == "top_k_63":
            # bf16 ties at rank 63/64 are frequent: a step is either audited clean or takes the exact overflow redo.
            # Every redo step is checked against an independent masked host reference (same logits, tokens, uniforms, mask).
            orig_walk = eng._host_walk_step

            def checked_walk(logits, tokens, nv, sampling, users=None, dev=None, mask=None):
                fz = orig_walk(logits, tokens, nv, sampling, users=users, dev=dev, mask=mask)
                m = sampling.get("mask") if mask is None else mask
                lg = torch.as_tensor(logits).reshape(eng.B * (eng.K + 1), -1)[:, : eng.vocab]
                c, n = host_redo_walk(
                    lg, tokens, sampling, sampling["uniforms"], B=eng.B, K=eng.K, num_valid=nv, users=users, mask=m
                )
                if users is not None:
                    c, n = merge_redo(dev[0], dev[1], c, n, users)
                assert torch.equal(torch.as_tensor(fz.committed).to(torch.int64), torch.as_tensor(c).to(torch.int64))
                assert torch.equal(torch.as_tensor(fz.counts).to(torch.int64), torch.as_tensor(n).to(torch.int64))
                checked.append(1)
                return fz

            eng._host_walk_step = checked_walk
        try:
            _run(drv, prompts, 4, 60, STEPS)  # the audit raises RuntimeError on any device != reference commit
        finally:
            eng.__dict__.pop("_host_walk_step", None)
        audited = eng.fused_audited_steps - before
        redos = eng.fused_fallbacks.get("overflow_redo", 0) - redo_before
        logger.info(
            f"[mask D4:{name}] audited {audited} overflow_redo {redos} (checked vs host reference {len(checked)}) mask_steps {eng.mask_steps} fallbacks {dict(eng.fused_fallbacks)}"
        )
        redo_by_name[name] = redos
        if name == "top_k_63":
            assert audited + redos > 0 and len(checked) >= redos and eng.mask_steps > 0
        else:
            assert audited > 0 and eng.mask_steps > 0
    redo_other = sum(r for k, r in redo_by_name.items() if k != "top_k_63")
    assert redo_other <= 2, "a masked row must not raise false tie overflows"


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_d5_split_step_late_finish(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "0")
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    streams = {}
    for tag, sleep in (("early", lambda s: 0.0), ("late", lambda s: (s % 5) * 0.010), ("one_call", None)):
        drv = MaskDriver(
            eng, 90, mask_fn=_random_mask_fn(5, kinds=("structure", "dense"), rows_per_user=tc.K + 1), split_sleep=sleep
        )
        if sleep is None:
            drv.split_sleep = None
        _admit_all(drv, prompts, 4, 40)
        try:
            for _ in range(STEPS):
                if not drv.live():
                    break
                drv.step()
        finally:
            streams[tag] = {k: list(v) for k, v in drv.outs.items()}
            for row in drv.live():
                drv._finish(row)
    logger.info(f"[mask D5] late finishes {eng.mask_late_finish}")
    assert streams["early"] == streams["late"], "a late second half changed the result"
    assert streams["early"] == streams["one_call"], "split step != one-call step"


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_d6_no_compile_after_capture_and_trace_memory(mesh_device, monkeypatch):
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    mem = dict(getattr(model, "_vfy_mem_log", {}))
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        for n in (1, 2, 4, 8):
            for masked in (False, True):
                for cert in (False, True):
                    params = (
                        {"top_k": torch.zeros(tc.B, dtype=torch.int32), "top_p": torch.full((tc.B,), 0.9)}
                        if cert
                        else None
                    )
                    drv = MaskDriver(eng, 100 + n, mask_fn=_random_mask_fn(2) if masked else None, params=params)
                    _run(drv, prompts, n, 40, 4)
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    with open(os.path.join(OUT_DIR, "d6_mask_mem.json"), "w") as f:
        json.dump(mem, f, indent=1)
    logger.info(
        f"[mask D6] trace / DRAM bytes after each capture: {mem} (the mask traces' cost is the V* / accept_c differences)"
    )


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_d8m_fallback_with_mask(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "0")
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    bad = []
    for reason in ("mixed_rope_delta", "eager_drafter", "top_k_range"):
        orig = eng._fused_ineligible
        if reason != "top_k_range":
            eng._fused_ineligible = lambda *a, _r=reason, **k: (_r, None)
        params = {"top_k": torch.full((tc.B,), 64 if reason == "top_k_range" else 20, dtype=torch.int32)}
        drv = MaskDriver(
            eng, 110, mask_fn=_random_mask_fn(4, kinds=("structure", "dense"), rows_per_user=tc.K + 1), params=params
        )
        walked = []
        orig_walk = eng._host_walk_step
        eng._host_walk_step = (
            lambda logits, tokens, nv, sampling, *a, _o=orig_walk, **k: walked.append(
                (sampling.get("mask"), _o(logits, tokens, nv, sampling, *a, **k))
            )
            or walked[-1][1]
        )
        try:
            _run(drv, prompts, 4, 40, 4)
        finally:
            eng._fused_ineligible, eng._host_walk_step = orig, orig_walk
        for mask, fz in walked:
            if mask is None:
                bad.append((reason, "walk without a mask"))
                continue
            flat = mask.reshape(-1, SM.MASK_WORDS)
            allowed = SM.unpack_allowed(flat, eng.vocab)
            for u in range(tc.B):
                for j in range(int(fz.counts[u])):
                    if bool((~SM.is_unconstrained(flat[u * (tc.K + 1) + j])).item()) and not bool(
                        allowed[u * (tc.K + 1) + j, int(fz.committed[u, j])]
                    ):
                        bad.append((reason, u, j))
        assert walked, f"no fallback step ran for {reason}"
    assert not bad, f"fallback commits a disallowed token: {bad[:5]}"


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_d9_mask_timing(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "0")
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    dev, reps, res = model.device, 100, {"mask_trace_ms": {}}

    def replay_ms(tid):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / reps * 1e3

    drv = MaskDriver(eng, 120, mask_fn=_random_mask_fn(1))
    _admit_all(drv, prompts, 8, 100)
    drv.step()
    drv.step()
    tids = {8: model._vfy_mask_trace_id, **{b: bk.mask_trace_id for b, bk in model._vfy_bk.items()}}
    for b, tid in sorted(tids.items()):
        res["mask_trace_ms"][b] = replay_ms(tid)
        logger.info(f"[mask D9] bucket {b}: mask trace {res['mask_trace_ms'][b]:.3f} ms")
    buf = SM.PermMaskStagingRing().stage(SM.all_ones(4, words=SM.MASK_WORDS), [0, 1])
    ts = []
    for _ in range(50):
        t0 = time.perf_counter()
        host = model.verify_stage_mask(buf)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.synchronize_device(dev)
        del host
    res["h2d_call_ms_median"] = statistics.median(ts)
    t0 = time.perf_counter()
    for _ in range(20):
        SM.PermMaskStagingRing().stage(drv.last_mask.reshape(-1, SM.MASK_WORDS), [0, 1, 2, 3])
    res["host_stage_4_rows_ms"] = (time.perf_counter() - t0) / 20 * 1e3
    with open(os.path.join(OUT_DIR, "d9_timing.json"), "w") as f:
        json.dump(res, f, indent=1)
    for b, ms in res["mask_trace_ms"].items():
        assert ms <= 1.0, f"bucket {b}: mask trace {ms:.3f} ms > 1.0 ms gate"
    assert res["h2d_call_ms_median"] <= 0.4


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_d10_cert_mask_logprobs_one_step(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "1")
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT_EVERY", "1")
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    B = tc.B
    top_k = torch.full((B,), 20, dtype=torch.int32)
    top_k[0], top_p = 0, torch.ones(B)
    top_p[0] = 0.9  # user 0: certificate row
    nl = torch.full((B,), -2, dtype=torch.int32)
    nl[1] = 5
    drv = MaskDriver(eng, 130, mask_fn=_random_mask_fn(6, kinds=("structure", "dense"), rows_per_user=tc.K + 1),
                     params={"top_k": top_k, "top_p": top_p, "num_logprobs": nl})  # fmt: skip
    seen = []
    orig = eng.decode_forward

    def watch(*a, **kw):
        r = orig(*a, **kw)
        if r.candidates is not None:
            seen.append(r.candidates)
        return r

    eng.decode_forward = watch
    try:
        _run(drv, prompts, 4, 60, STEPS)
    finally:
        eng.__dict__.pop("decode_forward", None)  # SampledDriver.step already removed its own wrapper
    assert eng.cert_steps > 0 and eng.mask_steps > 0 and seen, (eng.cert_steps, eng.mask_steps, len(seen))
    for c in seen:
        assert bool(torch.isfinite(c.lse).all())
    logger.info(
        f"[mask D10] cert steps {eng.cert_steps}, mask steps {eng.mask_steps}, cert failed {eng.cert_failed_steps}"
    )


# --------------------------------------------------------------------------- #
# Stage P6: the two-phase CHAINED step (begin on the previous step's tip, mask built from the previous step's result, finish)
# --------------------------------------------------------------------------- #
_P6_NB = 80  # page-table width in blocks (5120 tokens): 1000 steps of up to 4 tokens + a 600-token prompt
_TOY_N = 24  # allowed ids per toy grammar state


def _toy_allowed(t, n=_TOY_N):
    """Toy grammar: the ids allowed after token ``t`` (a fixed pseudo-random set of n ids, a function of t only)."""
    base = (int(t) * 2654435761 + 12345) % SM.VOCAB
    return sorted({(base + i * 7919) % SM.VOCAB for i in range(n)})


def _pack_ids(ids):
    w = torch.zeros(SM.MASK_WORDS, dtype=torch.int64)
    for v in ids:
        w[v // 32] |= 1 << (v % 32)
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)


def _p6_env(monkeypatch, audit_every=8, audit=True, buckets_hold0=False):
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    monkeypatch.setenv("QWEN36_SPEC_CHAIN", "1")
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_SAMPLED", "1")
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT", "1" if audit else "0")
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT_EVERY", str(audit_every))
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "1" if audit else "0")
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT_EVERY", str(audit_every))
    if buckets_hold0:
        monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "0")
    monkeypatch.setattr(tc, "B", 8)
    monkeypatch.setattr(tc, "NB", _P6_NB)
    monkeypatch.setattr(tc, "NUM_BLOCKS", 8 * _P6_NB + 40)


def _p6_setup(mesh_device):
    model, eng, tokenizer, prompts = tc._setup(mesh_device)
    assert eng._ch_g is not None and eng._ch_trace_g is not None, "chain greedy trace missing"
    assert eng._fa_trace is not None and eng._fs_trace is not None and eng._mask_ok(), "fused / mask traces missing"
    return model, eng, prompts


class MaskChainDriver(AsyncDriver):
    """AsyncDriver whose steps are TWO-PHASE chained steps: ``decode_forward_begin`` (chained on the previous handle), then the
    previous step is read (``wait``), the mask of the masked rows is built from ITS commits and ``handle.drafts()``, then
    ``decode_forward_finish``. Masked rows follow the toy grammar: the token committed at a position must be in
    ``_toy_allowed(previous token)``. ``masking=False`` = the control run: the same split steps, finish(None)."""

    def __init__(self, eng, seed, masked_rows=(0,), masking=True, sleep_s=0.0, **kw):
        super().__init__(eng, seed, sampled=lambda s: True, **kw)
        self.masked_rows, self.masking, self.sleep_s = set(masked_rows), masking, sleep_s
        self.fin_slots = None  # finishing_slots for the next begin (runner slots)
        self.last = None  # (ids, counts, drafts) of the newest resolved step
        self.tail = {}  # row -> last committed token of the masked row (grammar state)
        self.bad = []  # (row, step, token, previous token) of a masked commit outside its allowed set
        self.n_checked = 0
        self.finish_ms, self.begin_ms, self.gap_ms = [], [], []

    # -- begin through decode_forward_begin
    def begin(self):
        e = self.eng
        orig = e.decode_forward

        def route(*a, **k):
            if e._split_begin:  # decode_forward_begin calls self.decode_forward again from inside
                return orig(*a, **k)
            if self.fin_slots:
                k["finishing_slots"] = list(self.fin_slots)
                self.fin_slots = None
            return e.decode_forward_begin(*a, **k)

        e.decode_forward = route
        try:
            t0 = time.perf_counter()
            self.enq()
            self.begin_ms.append(1e3 * (time.perf_counter() - t0))
        finally:
            del e.decode_forward
        self.t_begin_done = time.perf_counter()
        return self.pend[-1][0]

    # -- read the oldest step, check the masked commits against the grammar
    def resolve(self):
        h, rows, rids = self.pend[0]
        ids, counts, p0 = h.wait()
        drafts = h.drafts()
        for r in rows:
            q = self.reqs[r]
            if q is None or q["rid"] != rids[r] or r not in self.masked_rows or not self.masking:
                continue
            prev = self.tail.get(r, q["out"][-1])
            for t in ids[r, : int(counts[r])].tolist():
                self.n_checked += 1
                if t not in _toy_allowed(prev):
                    self.bad.append((r, self.steps, t, prev))
                prev = t
            self.tail[r] = prev
        self.last = (ids.clone(), counts.clone(), drafts.clone())
        super().resolve()

    # -- the mask of the step just begun, from the newest resolved step (or the host proposal for the first step)
    def mask_for(self, chained):
        if not self.masking:
            return None
        B, K = tc.B, tc.K
        rows = [r for r in sorted(self.masked_rows) if self.reqs[r] is not None]
        if not rows:
            return None
        m = torch.full((B, K + 1, SM.MASK_WORDS), -1, dtype=torch.int32)
        for r in rows:
            if chained:
                ids, counts, drafts = self.last
                toks = [int(ids[r, int(counts[r]) - 1])] + drafts[r].tolist()
            else:
                q = self.reqs[r]
                d = list(q["proposal"])[:K]
                toks = [q["out"][-1]] + d + [0] * (K - len(d))
            for j in range(K + 1):
                m[r, j] = _pack_ids(_toy_allowed(toks[j]))
        return m

    def run(self, steps, events=None, mask_override=None):
        """``steps`` two-phase steps. events: {iteration: fn(driver)} runs after begin(N+1) and the read of N, before the finish.
        mask_override: {iteration: value} replaces the mask of the finish (value None is a real None)."""
        events = events or {}
        mask_override = mask_override or {}
        for it in range(steps):
            if not self.live():
                break
            chained = bool(self.pend)
            h = self.begin()
            if len(self.pend) == 2:
                self.resolve()
            if it in events:
                events[it](self)
            mask = self.mask_for(chained and self.last is not None)
            if it in mask_override:
                mask = mask_override[it]
            if self.sleep_s:
                time.sleep(self.sleep_s)
            t0 = time.perf_counter()
            self.eng.decode_forward_finish(h, mask)
            t1 = time.perf_counter()
            self.finish_ms.append(1e3 * (t1 - t0))
            self.gap_ms.append(1e3 * (t1 - self.t_begin_done))
        self.drain()

    def finish_all(self):
        for r in self.live():
            self._finish(r)
        return {rid: list(o) for rid, o in self.drv_outs().items()}

    def drv_outs(self):
        return self.outs


_P6_COUNTERS = (
    "chain_steps",
    "chain_audits",
    "fused_audited_steps",
    "fused_boundary_diffs",
    "mask_steps",
    "mask_chain_steps",
    "mask_late_finish",
)


def _snap(eng):
    d = {k: int(getattr(eng, k, 0)) for k in _P6_COUNTERS}
    d["overflow_redo"] = int(dict(getattr(eng, "fused_fallbacks", {})).get("overflow_redo", 0))
    d["chain_breaks"] = dict(eng.chain_breaks)
    d["def_breaks"] = dict(eng.__dict__.get("def_breaks", {}))
    return d


def _delta(a, b):
    d = {k: b[k] - a[k] for k in b if isinstance(b[k], int)}
    for k in ("chain_breaks", "def_breaks"):
        d[k] = {r: b[k].get(r, 0) - a[k].get(r, 0) for r in set(b[k]) | set(a[k]) if b[k].get(r, 0) != a[k].get(r, 0)}
    return d


def _p6_run(eng, prompts, tag, steps, seed=5, n_rows=7, masking=True, sleep_s=0.0, events=None, mask_override=None, prep=None,
            max_new=_CHAIN_BIG):  # fmt: skip
    """One two-phase pipeline run on a fresh driver. RuntimeError is logged with the counters (overflow risk) and re-raised."""
    drv = MaskChainDriver(eng, seed, masking=masking, sleep_s=sleep_s)
    _admit_all(drv, prompts, n_rows, max_new)
    if prep is not None:
        prep(drv)
    s0 = _snap(eng)
    t0 = time.perf_counter()
    try:
        drv.run(steps, events=events, mask_override=mask_override)
    except RuntimeError as e:
        logger.error(
            f"[p3b:{tag}] RuntimeError (overflow / tie / audit mismatch?): {e}; counters {_delta(s0, _snap(eng))}"
        )
        raise
    wall = time.perf_counter() - t0
    d = _delta(s0, _snap(eng))
    d.update(
        steps=drv.steps, wall_s=round(wall, 2), ms_per_step=round(1e3 * wall / max(drv.steps, 1), 2),
        masked_commits_checked=drv.n_checked, bad=len(drv.bad),
        finish_ms_median=round(statistics.median(drv.finish_ms), 3) if drv.finish_ms else None,
    )  # fmt: skip
    logger.info(f"[p3b:{tag}] {d}")
    streams = drv.finish_all()
    return drv, streams, d


@_cmarks
def test_d5b_chain_soak_masked_row(mesh_device, monkeypatch):
    """1000 two-phase chained steps, 7 rows (row 0 masked by the toy grammar, 1-6 normal, row 7 spare), audits every 8 steps."""
    _p6_env(monkeypatch)
    model, eng, prompts = _p6_setup(mesh_device)
    N = 1000
    # control first: the same split chained steps, no masked row (finish(None))
    _, ctl, dc = _p6_run(eng, prompts, "d5b-control", N, masking=False)
    drv, got, dm = _p6_run(eng, prompts, "d5b-masked", N, masking=True)
    assert not drv.bad, f"masked row committed a token outside its allowed set: {drv.bad[:5]}"
    assert drv.n_checked >= N, drv.n_checked
    assert dm["chain_steps"] >= 0.95 * N, f"chained steps {dm['chain_steps']} of {N}"
    assert dm["mask_chain_steps"] >= 0.95 * N, dm
    assert dm["mask_steps"] >= 0.95 * N, dm
    assert abs(dm["chain_audits"] - N // 8) <= 3 and abs(dm["fused_audited_steps"] - N // 8) <= 3, dm
    assert dm["fused_boundary_diffs"] == 0, dm
    for rid in range(1, 7):
        assert (
            got[rid] == ctl[rid]
        ), f"normal row {rid} differs from the control run (len {len(got[rid])} vs {len(ctl[rid])})"
    logger.info(
        f"[p3b:d5b] chained share {dm['chain_steps'] / N:.3f}, late {dm['mask_late_finish']} of {dm['mask_chain_steps']}, "
        f"overflow_redo {dm['overflow_redo']}, audits {dm['chain_audits']}/{dm['fused_audited_steps']}, ms/step {dm['ms_per_step']} "
        f"(control {dc['ms_per_step']})"
    )


@_cmarks
def test_d5c_late_mask_in_chain(mesh_device, monkeypatch):
    """The finish waits a host sleep of 0 / 10 / 20 / 35 / 45 ms: the streams do not change; the late counter follows the 31 ms rule."""
    _p6_env(monkeypatch)
    model, eng, prompts = _p6_setup(mesh_device)
    N = 150
    ref, res = None, {}
    for ms in (0, 10, 20, 35, 45):
        drv, got, d = _p6_run(eng, prompts, f"d5c-sleep{ms}", N, masking=True, sleep_s=ms / 1e3)
        assert not drv.bad, f"sleep {ms}: {drv.bad[:5]}"
        res[ms] = d
        if ref is None:
            ref = got
        else:
            for rid in sorted(ref):
                assert got[rid] == ref[rid], f"sleep {ms} ms: stream of row {rid} differs from the early run"
    logger.info(
        "[p3b:d5c] late finishes: "
        + ", ".join(f"{ms} ms: {d['mask_late_finish']}/{d['mask_chain_steps']}" for ms, d in res.items())
    )
    for ms in (0, 10, 20):
        assert res[ms]["mask_late_finish"] == 0, f"sleep {ms} ms counted late: {res[ms]}"
    for ms in (35, 45):
        assert res[ms]["mask_late_finish"] > 0, f"sleep {ms} ms not counted late: {res[ms]}"


def _ones_mask():
    return torch.full((tc.B, tc.K + 1, SM.MASK_WORDS), -1, dtype=torch.int32)


@_cmarks
def test_d5d_preempt_while_pending(mesh_device, monkeypatch):
    """Rows leave while a split chained step is pending (after begin(N+1), before its finish). Each scenario has a control run with
    the same leave schedule and no mask; the rows that remain must equal the control."""
    _p6_env(monkeypatch, buckets_hold0=True)
    model, eng, prompts = _p6_setup(mesh_device)
    L, N = 12, 40

    def leave_finishing(
        rows,
    ):  # the plugin flow: the row is dropped by the host, the engine frees it by finishing_slots at N+2
        def ev(d):
            for r in rows:
                q = d.reqs[r]
                d.outs[q["rid"]] = q["out"]
                d.reqs[r] = None
            d.fin_slots = list(rows)

        return ev

    def leave_release(rows):
        def ev(d):
            for r in rows:
                d._finish(r)

        return ev

    scen = {
        # name: (event, mask override at the leave iteration, rows compared, expected break reason or None)
        "masked_row_leaves_all_ones_finishing_slot": (leave_finishing([0]), {L: _ones_mask()}, range(1, 7), None),
        "all_masked_rows_leave_finish_none": (leave_release([0]), {L: None}, range(1, 7), None),
        "bucket_8_to_4_at_n2": (leave_release([4, 5, 6]), {}, range(1, 7), "bucket_change"),
    }
    out = {}
    for name, (ev, ov, cmp_rows, reason) in scen.items():
        c_drv, ctl, dc = _p6_run(eng, prompts, f"d5d-{name}-control", N, masking=False, events={L: ev})
        drv, got, d = _p6_run(eng, prompts, f"d5d-{name}", N, masking=True, events={L: ev}, mask_override=ov)
        assert not drv.bad, f"{name}: {drv.bad[:5]}"
        for rid in cmp_rows:
            assert got[rid] == ctl[rid], f"{name}: row {rid} differs from the control run"
        if reason is not None:
            seen = d["chain_breaks"].get(reason, 0) + d["def_breaks"].get(reason, 0)
            assert seen >= 1, f"{name}: no chain break '{reason}' counted: {d['chain_breaks']} {d['def_breaks']}"
        out[name] = d
    logger.info(f"[p3b:d5d] {out}")


@_cmarks
def test_d9c_phase2_enqueue_latency(mesh_device, monkeypatch):
    """Host time of decode_forward_finish (phase 2 enqueue: mask H2D, mask trace, topk, accept, drafter, reads) per bucket, plus
    begin return -> finish done, and the mask trace device time (as D9)."""
    _p6_env(monkeypatch, audit=False, buckets_hold0=True)
    model, eng, prompts = _p6_setup(mesh_device)
    dev, res = model.device, {}
    for n_rows, bucket in ((7, 8), (4, 4), (2, 2), (1, 1)):
        drv, _, d = _p6_run(eng, prompts, f"d9c-b{bucket}", 80, n_rows=n_rows, masking=True)
        assert not drv.bad
        fm, gm = drv.finish_ms[8:], drv.gap_ms[8:]
        res[bucket] = dict(finish_median=statistics.median(fm), finish_p90=sorted(fm)[int(0.9 * len(fm))],
                           begin_to_finish_done_median=statistics.median(gm), begin_median=statistics.median(drv.begin_ms[8:]),
                           late=d["mask_late_finish"], buckets=dict(getattr(eng, "bucket_steps", {})))  # fmt: skip
        logger.info(f"[p3b:d9c] bucket {bucket}: {res[bucket]}")

    def replay_ms(tid, reps=100):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / reps * 1e3

    tids = {8: model._vfy_mask_trace_id, **{b: bk.mask_trace_id for b, bk in model._vfy_bk.items()}}
    mask_ms = {b: replay_ms(t) for b, t in sorted(tids.items())}
    logger.info(f"[p3b:d9c] mask trace ms per bucket {mask_ms}")
    with open(os.path.join(OUT_DIR, "d9c_timing.json"), "w") as f:
        json.dump({"finish": res, "mask_trace_ms": mask_ms}, f, indent=1)
    for b, r in res.items():
        assert r["finish_median"] < 3.0, f"bucket {b}: finish enqueue median {r['finish_median']:.2f} ms >= 3 ms"
    for b, ms in mask_ms.items():
        assert ms <= 1.0, f"bucket {b}: mask trace {ms:.3f} ms"
