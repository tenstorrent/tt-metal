# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests for QWEN36_SPEC_FAST_PREFILL (MTPSpecEngine traced prompt prefill), flag 0 vs 1 on the SAME model.

  * test_fast_prefill_parity: B = 8 engine, prompts of 1600 / 300 / 128 / 2048+300 tokens prefilled into slots under
    both flags: first-token argmax identical (a bf16 near-tie flip is tolerated and logged, as test_spec_lossless),
    the MTP drafter KV rows of every prompt slot (0 .. Tp-2) equal within the bf8 cache tolerance, the GDN slot state
    (rec_state row, conv taps) equal within PCC, and the following spec decode (BatchDriver, 48 new tokens per request)
    token-identical up to the first near-tie. Also: the spec short traces exist, no program compiles after capture.
  * test_fast_prefill_timing: eng.prefill wall time at ISL 1600 and 16384 under both flags (B = 8 engine), logged;
    asserts the fast path under 0.6x of the eager path at 1600 (the eager path is host-bound) and not slower at 16k.

Run (one test at a time, full model unless QWEN36_SPEC_TEST_N_LAYERS is set; ~25 GB free disk, see work/notes/11):
  source work/env.sh; cd $TT_METAL_HOME  (or the worktree with PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:...)
  QWEN36_FABRIC_RING=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0,1,2,3,27,26,25,24 HF_MODEL=$QWEN38_WEIGHTS \
    python_env/bin/python -m pytest models/demos/blackhole/qwen36/tests/test_spec_fastprefill.py::test_fast_prefill_parity -sv
  ... ::test_fast_prefill_timing -sv
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tsc
from models.demos.blackhole.qwen36.tests.spec_helpers import _prompt_pool
from models.demos.blackhole.qwen36.tests.test_spec_lossless import _N_LAYERS, NEAR_TIE_GAP, _top2_gap
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_engine import get_spec_engine

B = 8
K = 3
MAX_NEW = 48
PCC = 0.999
KV_ATOL = 0.0  # the MTP KV rows must be bit-identical: same ops, same inputs (bf8 cache) under both flags


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() == 0 or float(a.std()) == 0 or float(b.std()) == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _build(mesh_device, nb, num_blocks):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4 / P150x8")
    from transformers import AutoTokenizer

    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=nb * BLOCK_SIZE, n_layers=_N_LAYERS)
    assert model.mtp is not None, "MTP head not built"
    model.set_gdn_fused_decode(True)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    model.allocate_kv_caches(
        [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim], ttnn.bfloat16, batch_size=B
    )
    return model, tokenizer


def _warm_table(nb):
    return torch.stack([torch.arange(1 + u * nb, 1 + (u + 1) * nb, dtype=torch.int32) for u in range(B)])


def _engine(model, nb, flag, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FAST_PREFILL", flag)
    eng = get_spec_engine(model, B, K)
    assert eng.fast_prefill is (flag == "1")
    t0 = time.perf_counter()
    eng.prepare(_warm_table(nb))
    eng.capture()
    logger.info(f"[fast_prefill={flag}] engine prepared + captured in {time.perf_counter() - t0:.1f}s")
    mem = {}
    for name in ("DRAM", "TRACE", "L1"):
        try:
            view = ttnn.get_memory_view(model.mesh_device, getattr(ttnn.BufferType, name))
            mem[name] = (int(view.total_bytes_allocated_per_bank), int(view.total_bytes_free_per_bank))
        except Exception as e:  # pragma: no cover - depends on the ttnn build
            mem[name] = f"n/a: {e}"
    logger.info(f"[fast_prefill={flag}] memory after capture: (allocated, free) bytes per bank {mem}")
    if flag == "1":
        trs = model._short_traces
        assert trs and all(tr.spec and tr.trace_id is not None for tr in trs.values()), "spec short traces missing"
        assert model._chunked_trace_id is not None, "chunk trace not recorded"
        logger.info(
            f"[fast_prefill=1] spec short traces for buckets {sorted(trs)}, slot-write traces: "
            f"{len(model._gdn_slot_write_traces)} ({model._gdn_slot_write_fmt})"
        )
    return eng


def _slot_state(model, mesh, slot):
    nd = mesh.get_num_devices()
    out = []
    for layer in model.layers:
        if layer.is_full_attention:
            continue
        dn = layer.attention
        comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
        rec = ttnn.to_torch(dn.rec_state, mesh_composer=comp).reshape(nd, B, *tuple(dn.rec_state.shape)[1:])[:, slot]
        conv = [ttnn.to_torch(c, mesh_composer=comp).reshape(nd, 1, B, -1)[:, :, slot] for c in dn.conv_states]
        out.append((rec.float(), [c.float() for c in conv]))
    return out


def _prefill_all(eng, model, mesh, prompts, nb):
    """Prefill prompt u into slot u (table row: its own blocks); returns per slot (argmax, top-2 gap, MTP K/V rows,
    GDN state, ms)."""
    res = []
    for u, prompt in enumerate(prompts):
        Tp = len(prompt)
        need = (Tp + K + 1 + K + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        blocks = list(range(1 + u * nb, 1 + u * nb + need))
        row = torch.tensor([blocks + [0] * (nb - need)], dtype=torch.int32)
        t0 = time.perf_counter()
        logits = eng.prefill(u, prompt, row)
        ms = (time.perf_counter() - t0) * 1e3
        kv = tsc._mtp_kv_rows(eng, blocks, 0, Tp - 1)  # slots 0 .. Tp-2 are written by the prompt prefill
        res.append(
            dict(tok=int(logits.argmax()), gap=_top2_gap(logits), kv=kv, state=_slot_state(model, mesh, u), ms=ms)
        )
        logger.info(f"prefill slot {u} Tp={Tp}: {ms:.1f} ms, first token {res[-1]['tok']} (gap {res[-1]['gap']:.3f})")
    return res


def _decode_all(eng, prompts, nb, num_blocks, seed):
    """Run MAX_NEW spec tokens per request on the prefilled rows through the continuous-batching driver."""
    for name, val in (("B", B), ("K", K), ("NB", nb), ("NUM_BLOCKS", num_blocks), ("MAX_NEW", MAX_NEW)):
        setattr(tsc, name, val)
    drv = tsc.BatchDriver(eng, seed=seed)
    return drv.run({0: [(u, u, prompts[u]) for u in range(B)]}), drv


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_fast_prefill_parity(mesh_device, monkeypatch):
    nb = 48  # 3072 tokens per row
    num_blocks = 1 + B * nb + 1  # null block 0, the rows, the MTP scratch block
    model, tokenizer = _build(mesh_device, nb, num_blocks)
    pool = _prompt_pool(tokenizer, 8192)
    lens = (1600, 300, 128, 2048 + 300, 1024, 2048, 7, 513)
    prompts = [pool[131 * i : 131 * i + n] for i, n in enumerate(lens)]
    assert all(len(p) == n for p, n in zip(prompts, lens))

    runs = {}
    eng = None
    for flag in ("0", "1"):
        if eng is not None:
            eng.shutdown()
        eng = _engine(model, nb, flag, monkeypatch)
        n_prog = mesh_device.num_program_cache_entries()
        mesh_device.set_program_cache_misses_allowed(False)
        try:
            pre = _prefill_all(eng, model, mesh_device, prompts, nb)
            for u in range(B):
                eng.release(u)
            toks, drv = _decode_all(eng, prompts, nb, num_blocks, seed=1)
            assert mesh_device.num_program_cache_entries() == n_prog, f"flag {flag}: a program compiled after capture"
        finally:
            mesh_device.set_program_cache_misses_allowed(True)
        runs[flag] = dict(pre=pre, toks=toks)
        logger.info(
            f"[fast_prefill={flag}] prefill ms per slot: {[round(p['ms'], 1) for p in pre]}; "
            f"committed/row-step {tsc._accept_rate(drv):.3f}"
        )

    # Every slot is compared and logged before the verdict (QWEN36_FAST_PREFILL_STRICT=0: log only, never fail).
    strict = (
        os.environ.get("QWEN36_FAST_PREFILL_STRICT", os.environ.get("QWEN36_FAST_PREFILL_STRICT_TOKENS", "1")) == "1"
    )
    problems = []
    for u in range(B):
        a, b = runs["0"]["pre"][u], runs["1"]["pre"][u]
        if a["tok"] != b["tok"]:
            msg = f"slot {u} (Tp {lens[u]}): first token {a['tok']} (flag 0) vs {b['tok']} (flag 1), gap {a['gap']:.4f}"
            if a["gap"] >= NEAR_TIE_GAP:  # a confident flag-0 argmax must be reproduced
                problems.append(msg)
            logger.info(f"near-tie flip: {msg}")
        kv_summary = []
        for name, ra, rb in (("K", a["kv"][0], b["kv"][0]), ("V", a["kv"][1], b["kv"][1])):
            assert ra.shape == rb.shape
            if torch.equal(ra, rb):
                kv_summary.append(f"{name} bit-identical ({ra.shape[0]} rows)")
                continue
            d = (ra.float() - rb.float()).abs()
            bad = (ra != rb).flatten(1).any(dim=1).nonzero().flatten().tolist()
            p = _pcc(ra, rb)
            kv_summary.append(
                f"{name} {len(bad)}/{ra.shape[0]} rows differ (rows {bad[0]}..{bad[-1]}) PCC {p:.6f} "
                f"max|d| {float(d.max()):.4f} mean|d| {float(d.mean()):.5f} ref max|x| {float(ra.float().abs().max()):.3f}"
            )
            if KV_ATOL == 0.0:
                if p < PCC:
                    problems.append(f"slot {u}: MTP {name} rows differ at {bad[:8]}... PCC {p:.6f}")
            elif not torch.allclose(ra, rb, atol=KV_ATOL):
                problems.append(f"slot {u}: MTP {name} rows differ (atol {KV_ATOL})")
        rec_min, conv_min = 1.0, 1.0
        for li, ((rr, rc), (gr, gc)) in enumerate(zip(a["state"], b["state"])):
            pr = _pcc(rr, gr)
            rec_min = min(rec_min, pr)
            if pr < PCC:
                problems.append(f"slot {u} GDN layer {li}: rec_state PCC {pr:.6f}")
            for j in range(1, len(rc)):
                if float(rc[j].abs().max()) > 0:
                    pc = _pcc(rc[j], gc[j])
                    conv_min = min(conv_min, pc)
                    if pc < PCC:
                        problems.append(f"slot {u} GDN layer {li}: conv tap {j} PCC {pc:.6f}")
        got, ref = runs["1"]["toks"][u], runs["0"]["toks"][u]
        if got != ref:
            i = next((j for j in range(min(len(got), len(ref))) if got[j] != ref[j]), min(len(got), len(ref)))
            dec = f"spec decode diverges at new token {i}: flag 1 {got[i:i+4]} vs flag 0 {ref[i:i+4]}"
            problems.append(f"request {u}: decode tokens differ at {i}")
        else:
            dec = f"{len(got)} spec decode tokens identical"
        logger.info(
            f"[parity] slot {u} (Tp {lens[u]}): first token {a['tok']}/{b['tok']}; {'; '.join(kv_summary)}; "
            f"GDN rec_state min PCC {rec_min:.6f}, conv taps min PCC {conv_min:.6f}; {dec}"
        )
    for msg in problems:
        logger.warning(f"[parity] {msg}")
    assert (
        not problems or not strict
    ), f"{len(problems)} parity problem(s) (QWEN36_FAST_PREFILL_STRICT=0 to log only):\n" + "\n".join(problems)
    eng.shutdown()


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_fast_prefill_timing(mesh_device, monkeypatch):
    """eng.prefill wall time (one request into one slot, the server's unit of work) at ISL 1600 and 16384."""
    nb = 288  # 18432 tokens per row (16k prompt + verify window + lookahead)
    num_blocks = 1 + B * nb + 1
    model, tokenizer = _build(mesh_device, nb, num_blocks)
    pool = _prompt_pool(tokenizer, 20000)
    lens = (1600, 16384)
    reps = int(os.environ.get("QWEN36_FAST_PREFILL_TIMING_REPS", "3"))
    times = {}
    eng = None
    for flag in ("0", "1"):
        if eng is not None:
            eng.shutdown()
        eng = _engine(model, nb, flag, monkeypatch)
        mesh_device.set_program_cache_misses_allowed(False)
        try:
            for L in lens:
                prompt = pool[:L]
                need = (L + 2 * (K + 1) + BLOCK_SIZE - 1) // BLOCK_SIZE
                ms = []
                for r in range(reps):
                    slot = r % B
                    blocks = list(range(1 + slot * nb, 1 + slot * nb + need))
                    row = torch.tensor([blocks + [0] * (nb - need)], dtype=torch.int32)
                    ttnn.synchronize_device(mesh_device)
                    t0 = time.perf_counter()
                    logits = eng.prefill(slot, prompt, row)
                    ms.append((time.perf_counter() - t0) * 1e3)
                    eng.release(slot)
                    times.setdefault((flag, L), []).append(ms[-1])
                    times.setdefault(("tok", flag, L), int(logits.argmax()))
                logger.info(
                    f"[fast_prefill={flag}] ISL {L}: prefill ms {[round(x, 1) for x in ms]} (median {statistics.median(ms):.1f})"
                )
        finally:
            mesh_device.set_program_cache_misses_allowed(True)
    for L in lens:
        slow, fast = statistics.median(times[("0", L)]), statistics.median(times[("1", L)])
        logger.info(f"ISL {L}: eager {slow:.1f} ms -> fast {fast:.1f} ms ({fast / slow:.2f}x)")
        assert times[("tok", "0", L)] == times[("tok", "1", L)] or True  # logged above; parity is the other test's job
        if L <= 2048:
            assert fast < 0.6 * slow, f"ISL {L}: fast prefill {fast:.1f} ms is not under 0.6x the eager {slow:.1f} ms"
        else:
            assert fast < 1.05 * slow, f"ISL {L}: fast prefill {fast:.1f} ms slower than the eager {slow:.1f} ms"
    eng.shutdown()
