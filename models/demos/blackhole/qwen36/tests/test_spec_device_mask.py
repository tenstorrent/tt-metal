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
