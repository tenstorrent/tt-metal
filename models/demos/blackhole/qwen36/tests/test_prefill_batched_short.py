# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grouped (batched) short-prompt prefill (QWEN36_BATCHED_PREFILL=1, EAGER, increment 2) vs the per-user path.

Device test (TP mesh, MESH_DEVICE=P150x4; the model is truncated to 8 layers = full-attn + GDN, 32 decode slots):

* ``test_prefill_batched_short_vs_per_user``  N in {4, 8, 9} prompts (mixed lengths inside buckets 128 / 256, distinct
  tokens) prefilled into SCATTERED decode slots with QWEN36_BATCHED_PREFILL=1 vs =0 on the same model: host logits
  PCC > 0.9999 and identical argmax per row, the target slots' rec_state rows equal, the greedy decode of those slots
  equal, and NEIGHBOUR SAFETY: every non-target slot's GDN state row is bit-identical before / after the batched call.
* ``test_prefill_batched_short_traced``       increments 3-5: prepare (warm) + record the grouped forward traces and the
  row-extract traces exactly as warmup_model_prefill does (plus the production B=1 short traces), then batched TRACED
  replay vs the eager per-user reference: bit-exact logits / rec_state rows / KV blocks / decode, neighbour safety,
  repeated replays. NO lazy allocation (QWEN36_BATCHED_PREFILL_EAGER_LAZY unset): the prepare phase must cover it all.
* ``test_host_plan_batched_groups``           host-only: the grouping plan (no device).

Run:
  source /home/ttuser/atupe/qwen38_work/env38_local.sh
  TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 MESH_DEVICE=P150x4 \
    pytest -svq models/demos/blackhole/qwen36/tests/test_prefill_batched_short.py
"""

import gc
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

BLOCK = 64
WIDTH = 32  # decode slots
BPU = 8  # blocks per slot (512 tokens: prompt <= 256 + decode)
PCC = 0.9999
DECODE_STEPS = 4
# Scattered slots (first N used) and lengths mixed inside buckets 128 (<=128) and 256 (129..256).
SLOTS = [3, 17, 5, 30, 11, 24, 0, 8, 21]
LENS = [100, 128, 200, 256, 70, 129, 250, 33, 127]


def _pcc(a, b):
    return float(comp_pcc(a.reshape(-1).float(), b.reshape(-1).float(), 0.0)[1])


def _gdn(model):
    return [l.attention for l in model.layers if not l.is_full_attention]


def _host(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def _rec_rows(model, mesh):
    """Per GDN layer rec_state as host [nd, WIDTH, ...]."""
    nd = mesh.get_num_devices()
    return [_host(mesh, dn.rec_state).reshape(nd, WIDTH, *tuple(dn.rec_state.shape)[1:]) for dn in _gdn(model)]


def _fake_model(trash=0):
    m = Qwen36Model.__new__(Qwen36Model)
    m.num_devices = 4
    m._lmhead_vocab_sharded = True
    m.prefill_trash_block = trash
    m._batched_short_unavailable = False
    m._paged_kv_caches = [[SimpleNamespace(shape=(8, 1, BLOCK, 1))]]
    m._short_trace_masks_supported = lambda: True
    return m


@pytest.mark.parametrize(
    "lens,expect",
    [
        ([100] * 8, [(128, 8, 8)]),
        ([100] * 9, [(128, 8, 8)]),  # 9th row alone -> per-user
        ([100] * 4, [(128, 4, 4)]),
        ([100] * 2, [(128, 4, 2)]),
        ([100] * 6, [(128, 8, 6)]),
        ([100] * 3 + [200] * 5, [(128, 4, 3), (256, 8, 5)]),
        ([100, 200, 300, 1], [(128, 4, 2)]),  # 100 and 1 share bucket 128; 200 alone; 300 > 256 per-user
    ],
)
def test_host_plan_batched_groups(lens, expect, monkeypatch):
    monkeypatch.delenv("QWEN36_BATCHED_PREFILL_BUCKETS", raising=False)
    monkeypatch.delenv("QWEN36_BATCHED_PREFILL_BG", raising=False)
    m = _fake_model()
    n = len(lens)
    pt = torch.arange(1, 1 + n * BPU, dtype=torch.int32).reshape(n, BPU)
    groups, rest = m._plan_batched_groups(lens, [0] * n, [False] * n, pt, list(range(n)))
    assert [(b, bg, len(idx)) for b, bg, idx in groups] == expect
    served = {u for _, _, idx in groups for u in idx}
    assert served.isdisjoint(rest) and served | set(rest) == set(range(n))
    # Ineligible rows: start > 0, vision, trash collision (block 0 in the real blocks).
    pt2 = torch.arange(1, 1 + 4 * BPU, dtype=torch.int32).reshape(4, BPU)
    pt2[0, 0] = 0
    g2, r2 = m._plan_batched_groups([100] * 4, [0, 64, 0, 0], [False, False, True, False], pt2, list(range(4)))
    assert g2 == [] and sorted(r2) == [0, 1, 2, 3]


@torch.no_grad()
@parametrize_mesh_tp()
def test_prefill_batched_short_vs_per_user(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    mesh = mesh_device
    assert mesh.get_num_devices() > 1
    # Eager test: buffers are allocated lazily on first use (no trace is ever captured here).
    monkeypatch.setenv("QWEN36_BATCHED_PREFILL_EAGER_LAZY", "1")
    model = Qwen36Model.from_pretrained(mesh, max_batch_size=WIDTH, max_seq_len=4096, n_layers=8)
    args, vocab = model.args, model.args.vocab_size
    total_blocks = WIDTH * BPU + 1
    model.allocate_kv_caches(
        (total_blocks, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat16, batch_size=WIDTH
    )
    model.prefill_trash_block = 0  # block 0 is never owned by a slot (vLLM null block convention)
    # Slot s owns blocks [1 + s*BPU, 1 + (s+1)*BPU).
    full_pt = torch.stack([torch.arange(1 + s * BPU, 1 + (s + 1) * BPU, dtype=torch.int32) for s in range(WIDTH)])
    torch.manual_seed(7)

    def prompts(n):
        return [torch.randint(1, vocab, (1, LENS[u]), dtype=torch.long) for u in range(n)]

    def prefill(toks, slots):
        lens = [t.shape[1] for t in toks]
        host = model.prefill_paged_slots(toks, full_pt[slots], slots, valid_lens=lens)
        return [h.reshape(-1)[:vocab].float() for h in host]

    def decode(first, slots, lens):
        """DECODE_STEPS greedy steps over ALL slots (non-target slots: dummy token 0 at position 0)."""
        toks = torch.zeros(WIDTH, 1, dtype=torch.int32)
        pos = torch.zeros(WIDTH, dtype=torch.int32)
        for f, s, L in zip(first, slots, lens):
            toks[s, 0], pos[s] = f, L
        outs = []
        for _ in range(DECODE_STEPS):
            dev = model.prepare_inputs_decode(toks, pos, full_pt)
            out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
            lg = model.process_output_decode(out, WIDTH)[:, 0, :vocab].float()
            nxt = lg.argmax(dim=-1)
            outs.append([int(nxt[s]) for s in slots])
            for s in slots:
                toks[s, 0] = nxt[s]
                pos[s] += 1
        return outs

    # Make every slot live first (per-user path) so neighbour rows hold non-trivial state.
    monkeypatch.setenv("QWEN36_BATCHED_PREFILL", "0")
    live = [torch.randint(1, vocab, (1, 128), dtype=torch.long) for _ in range(WIDTH)]
    prefill(live, list(range(WIDTH)))

    # The per-user REFERENCE must run the production per-user op sequence: the masked-bucket short-prefill body
    # (valid_len=None, persistent GDN masks, KDA conv), which the grouped body mirrors op for op. The eager
    # prefill_masked_bucket path instead runs the GDN with valid_len set (the MAC-FIR conv fallback: ~1 ulp apart).
    # So the short-trace objects are built by the production capture routine with the trace calls stubbed out
    # (no trace is ever captured or replayed), and "execute_trace" of a short trace runs the trace body EAGERLY on
    # the buffers _replay_short_trace staged. Everything else (rope, masks, page-table staging, slot write) is
    # production. The fake traces are released right after the reference so the batched path never sees them.
    warm_pt = torch.cat([full_pt.reshape(-1), torch.zeros(31, dtype=torch.int32)])
    warm_pt = warm_pt[: ((total_blocks + 31) // 32) * 32].reshape(1, -1)

    def prefill_eager_ref(toks, slots):
        fake_ids = {}
        counter = iter(range(1000, 2000))

        def fake_begin(*a, **k):
            return next(counter)

        def fake_execute_trace(device, tid, cq_id=0, blocking=False):
            tr = fake_ids[tid]
            tr.idx, tr.val, tr.logits = model._forward_prefill_short_tp(tr)

        prev = model._bind_gdn_prefill_scratch()
        try:
            with monkeypatch.context() as mp:
                mp.setattr(ttnn, "begin_trace_capture", fake_begin)
                mp.setattr(ttnn, "end_trace_capture", lambda *a, **k: None)
                model.capture_prefill_traces_short(mesh, warm_pt, buckets=(128, 256))
        finally:
            model._unbind_gdn_prefill_scratch(prev)
        for tr in model._short_traces.values():
            fake_ids[tr.trace_id] = tr
        try:
            with monkeypatch.context() as mp:
                mp.setattr(ttnn, "execute_trace", fake_execute_trace)
                return prefill(toks, slots)
        finally:
            with monkeypatch.context() as mp:
                mp.setattr(ttnn, "release_trace", lambda *a, **k: None)
                model.release_prefill_trace_short()

    for N in (4, 8, 9):
        slots, lens = SLOTS[:N], LENS[:N]
        toks = prompts(N)
        groups, rest = model._plan_batched_groups(lens, [0] * N, [False] * N, full_pt[slots], slots)
        n_served = sum(len(idx) for _, _, idx in groups)
        assert n_served > 0, f"N={N}: the plan produced no group"
        calls = []
        orig = model._run_batched_short

        def counting(cfg, rows, _orig=orig):
            calls.append((cfg.bucket, cfg.Bg, len(rows)))
            return _orig(cfg, rows)

        monkeypatch.setattr(model, "_run_batched_short", counting)

        # ---- batched
        before = _rec_rows(model, mesh)
        monkeypatch.setenv("QWEN36_BATCHED_PREFILL", "1")
        lg_b = prefill(toks, slots)
        assert sum(c[2] for c in calls) == n_served, (N, calls, groups)
        after = _rec_rows(model, mesh)
        others = [s for s in range(WIDTH) if s not in slots]
        for li, (b, a) in enumerate(zip(before, after)):
            assert torch.equal(b[:, others], a[:, others]), f"N={N}: layer {li}: a non-target slot's rec_state changed"
        rec_b = [a[:, slots] for a in after]
        first = [int(l.argmax()) for l in lg_b]
        dec_b = decode(first, slots, lens)

        # ---- per-user reference (same tokens, same slots, same blocks)
        monkeypatch.setenv("QWEN36_BATCHED_PREFILL", "0")
        ncalls = len(calls)
        lg_r = prefill_eager_ref(toks, slots)
        assert len(calls) == ncalls, "reference run must not use the grouped path"
        rec_r = [a[:, slots] for a in _rec_rows(model, mesh)]
        dec_r = decode([int(l.argmax()) for l in lg_r], slots, lens)

        for u in range(N):
            p = _pcc(lg_b[u], lg_r[u])
            logger.info(f"N={N} row {u}: logits PCC {p!r}, max|diff| {(lg_b[u] - lg_r[u]).abs().max().item():.3e}")
            assert p > PCC, f"N={N} row {u} (len {lens[u]}, slot {slots[u]}): logits PCC {p}"
            assert int(lg_b[u].argmax()) == int(lg_r[u].argmax()), f"N={N} row {u}: argmax differs"
        for li, (b, r) in enumerate(zip(rec_b, rec_r)):
            for j, s in enumerate(slots):
                p = _pcc(b[:, j], r[:, j])
                logger.info(f"N={N} layer {li} slot {s}: rec_state PCC {p!r}, equal={torch.equal(b[:, j], r[:, j])}")
                assert p > PCC, f"N={N} layer {li} slot {s}: rec_state PCC {p}"
        assert dec_b == dec_r, f"N={N}: greedy decode differs: batched={dec_b} ref={dec_r}"
        logger.info(f"N={N}: groups {calls}, {n_served} rows batched; logits/rec/decode match")
        monkeypatch.setattr(model, "_run_batched_short", orig)

    del model
    gc.collect()


def _kv_blocks(model, mesh, slots, lens=None):
    """Host copy of the KV blocks owned by `slots` (every attention layer, K and V): [(k, v)] with [n_blocks, ...].
    lens: only the ceil(L/BLOCK) blocks the prefill of each slot writes (blocks beyond hold whatever the eager
    reference decode of phase A left there, which the traced prefill rightly does not touch)."""
    nb = [BPU] * len(slots) if lens is None else [-(-L // BLOCK) for L in lens]
    ids = torch.cat([torch.arange(1 + s * BPU, 1 + s * BPU + n) for s, n in zip(slots, nb)])
    out = []
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    for k, v in model._paged_kv_caches:
        nd = mesh.get_num_devices()
        kk = ttnn.to_torch(k, mesh_composer=comp).float()
        vv = ttnn.to_torch(v, mesh_composer=comp).float()
        nb = k.shape[0]
        out.append(
            (kk.reshape(nd, nb, *kk.shape[1:])[:, ids].clone(), vv.reshape(nd, nb, *vv.shape[1:])[:, ids].clone())
        )
    return out


@torch.no_grad()
@parametrize_mesh_tp()
def test_prefill_batched_short_traced(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    """Traced grouped prefill (increments 3-5) vs the eager per-user reference.

    ORDERING (known hazard: an EAGER decode / compile / allocation while traces are parked hung the device before):
      phase A (no trace parked): all eager work - the live-state prefill, the per-user reference (prefill_eager_ref
              with stubbed trace calls) and its snapshots (logits, target rec_state rows, target KV blocks) and the
              reference greedy decode of the LAST N only.
      phase B (traces parked): prepare (warm) -> production B=1 short traces -> record grouped traces; ONLY replays,
              host copies and reads run here (no decode, no per-user prefill, no new shapes): batched traced prefill
              (+ the slot write, which is the production per-slot path), bit-exact compare against the phase-A
              snapshots, neighbour safety, repeated replays. The B=1 short traces are parked but unused (N is chosen
              so every row is grouped), mirroring production where they coexist with the grouped traces.
      phase C: release every trace FIRST, then the eager decode of the last N's slots vs the phase-A decode.
    The lazy allocation of the eager increment is disabled (QWEN36_BATCHED_PREFILL_EAGER_LAZY unset): a config the
    prepare phase missed would silently disable grouping and fail the served-rows assertion.
    """
    mesh = mesh_device
    assert mesh.get_num_devices() > 1
    monkeypatch.delenv("QWEN36_BATCHED_PREFILL_EAGER_LAZY", raising=False)
    model = Qwen36Model.from_pretrained(mesh, max_batch_size=WIDTH, max_seq_len=4096, n_layers=8)
    args, vocab = model.args, model.args.vocab_size
    total_blocks = WIDTH * BPU + 1
    model.allocate_kv_caches(
        (total_blocks, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat16, batch_size=WIDTH
    )
    model.prefill_trash_block = 0
    full_pt = torch.stack([torch.arange(1 + s * BPU, 1 + (s + 1) * BPU, dtype=torch.int32) for s in range(WIDTH)])
    warm_pt = torch.cat([full_pt.reshape(-1), torch.zeros(31, dtype=torch.int32)])
    warm_pt = warm_pt[: ((total_blocks + 31) // 32) * 32].reshape(1, -1)
    torch.manual_seed(7)
    NS = (
        4,
        5,
        8,
    )  # all rows grouped (5 -> Bg 8 with 3 dummy rows): no per-user eager slot write while traces are parked
    toks_by_n = {N: [torch.randint(1, vocab, (1, LENS[u]), dtype=torch.long) for u in range(N)] for N in NS}
    live = [torch.randint(1, vocab, (1, 128), dtype=torch.long) for _ in range(WIDTH)]

    def prefill(toks, slots):
        lens = [t.shape[1] for t in toks]
        host = model.prefill_paged_slots(toks, full_pt[slots], slots, valid_lens=lens)
        return [h.reshape(-1)[:vocab].float() for h in host]

    def decode(first, slots, lens):
        toks = torch.zeros(WIDTH, 1, dtype=torch.int32)
        pos = torch.zeros(WIDTH, dtype=torch.int32)
        for f, s, L in zip(first, slots, lens):
            toks[s, 0], pos[s] = f, L
        outs = []
        for _ in range(DECODE_STEPS):
            dev = model.prepare_inputs_decode(toks, pos, full_pt)
            out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
            lg = model.process_output_decode(out, WIDTH)[:, 0, :vocab].float()
            nxt = lg.argmax(dim=-1)
            outs.append([int(nxt[s]) for s in slots])
            for s in slots:
                toks[s, 0] = nxt[s]
                pos[s] += 1
        return outs

    def prefill_eager_ref(toks, slots):  # same stubbed-trace production per-user reference as the eager test
        fake_ids, counter = {}, iter(range(1000, 2000))

        def fake_execute_trace(device, tid, cq_id=0, blocking=False):
            tr = fake_ids[tid]
            tr.idx, tr.val, tr.logits = model._forward_prefill_short_tp(tr)

        prev = model._bind_gdn_prefill_scratch()
        try:
            with monkeypatch.context() as mp:
                mp.setattr(ttnn, "begin_trace_capture", lambda *a, **k: next(counter))
                mp.setattr(ttnn, "end_trace_capture", lambda *a, **k: None)
                model.capture_prefill_traces_short(mesh, warm_pt, buckets=(128, 256))
        finally:
            model._unbind_gdn_prefill_scratch(prev)
        for tr in model._short_traces.values():
            fake_ids[tr.trace_id] = tr
        try:
            with monkeypatch.context() as mp:
                mp.setattr(ttnn, "execute_trace", fake_execute_trace)
                return prefill(toks, slots)
        finally:
            with monkeypatch.context() as mp:
                mp.setattr(ttnn, "release_trace", lambda *a, **k: None)
                model.release_prefill_trace_short()

    # ------------------------------------------------------------------ phase A (eager only, no trace parked)
    monkeypatch.setenv("QWEN36_BATCHED_PREFILL", "0")
    prefill(live, list(range(WIDTH)))
    ref = {}
    for N in NS:
        slots, lens = SLOTS[:N], LENS[:N]
        lg_r = prefill_eager_ref(toks_by_n[N], slots)
        ref[N] = {
            "logits": lg_r,
            "rec": [a[:, slots] for a in _rec_rows(model, mesh)],
            "kv": _kv_blocks(model, mesh, slots, lens),
        }
        if N == NS[-1]:
            ref[N]["decode"] = decode([int(l.argmax()) for l in lg_r], slots, lens)
    assert not model._any_trace_captured()

    # ------------------------------------------------------------------ phase B (traces parked: replay + read only)
    # Exactly warmup_model_prefill's order: eager pass prepare (alloc + warm) BEFORE any trace, then the trace pass:
    # B=1 short traces (production per-user path for the rows the plan leaves out), then the grouped traces.
    monkeypatch.setenv("QWEN36_BATCHED_PREFILL", "1")
    n_cfg = model.prepare_prefill_batched_short(mesh, warm_pt)
    assert n_cfg == len(model.batched_prefill_buckets()) * len(model.batched_prefill_bgs())
    prev = model._bind_gdn_prefill_scratch()
    try:
        model.capture_prefill_traces_short(mesh, warm_pt, buckets=(128, 256))
    finally:
        model._unbind_gdn_prefill_scratch(prev)
    n_tr = model.record_prefill_trace_batched_short(mesh)
    assert n_tr == n_cfg and all(c.trace_id is not None for c in model._batched_short_cfgs.values())
    assert len(model._batched_extract_traces) == sum(model.batched_prefill_bgs())
    assert model._any_trace_captured() and not model._batched_alloc_allowed()
    monkeypatch.setenv("QWEN36_BATCHED_PREFILL", "1")
    for N in NS:
        slots, lens = SLOTS[:N], LENS[:N]
        groups, _rest = model._plan_batched_groups(lens, [0] * N, [False] * N, full_pt[slots], slots)
        served = {u for _, _, idx in groups for u in idx}
        assert served, f"N={N}: the plan produced no group"
        replays = []
        orig = model._run_batched_short

        def counting(cfg, rows, _orig=orig):
            replays.append((cfg.bucket, cfg.Bg, len(rows), cfg.trace_id is not None))
            return _orig(cfg, rows)

        monkeypatch.setattr(model, "_run_batched_short", counting)
        before = _rec_rows(model, mesh)
        others = [s for s in range(WIDTH) if s not in slots]
        for rep in range(2):  # repeated replays of the same traces must be identical
            lg_b = prefill(toks_by_n[N], slots)
            assert len(served) == N and sum(r[2] for r in replays) == N * (rep + 1), (N, replays, groups)
            assert all(r[3] for r in replays), f"N={N}: a group fell back to eager: {replays}"
            after = _rec_rows(model, mesh)
            for li, (b, a) in enumerate(zip(before, after)):
                assert torch.equal(b[:, others], a[:, others]), f"N={N} rep {rep} layer {li}: neighbour rec_state moved"
            R = ref[N]
            for u in range(N):
                if u in served:
                    assert torch.equal(lg_b[u], R["logits"][u]), (
                        f"N={N} rep {rep} row {u}: traced logits not bit-exact, "
                        f"max|diff| {(lg_b[u] - R['logits'][u]).abs().max().item():.3e}"
                    )
                else:  # per-user fallback row (production B=1 trace vs the stubbed eager reference)
                    assert _pcc(lg_b[u], R["logits"][u]) > PCC and int(lg_b[u].argmax()) == int(R["logits"][u].argmax())
            for li, a in enumerate(after):
                for j, s in enumerate(slots):
                    if j in served:
                        assert torch.equal(a[:, s], R["rec"][li][:, j]), f"N={N} layer {li} slot {s}: rec_state differs"
            kv = _kv_blocks(model, mesh, slots, lens)
            for li, ((kb, vb), (kr, vr)) in enumerate(zip(kv, R["kv"])):
                assert torch.equal(kb, kr) and torch.equal(vb, vr), f"N={N} rep {rep} attn layer {li}: KV blocks differ"
            before = after
        logger.info(f"N={N}: traced groups {replays[: len(replays) // 2]}: logits/rec/KV bit-exact, neighbours intact")
        monkeypatch.setattr(model, "_run_batched_short", orig)

    # ------------------------------------------------------------------ phase C (release FIRST, then eager decode)
    # (the state now is the last N's batched traced prefill, same as the reference's decode in phase A)
    model.release_prefill_trace_short()  # also releases the grouped forward + extract traces
    assert not model._batched_extract_traces and all(c.trace_id is None for c in model._batched_short_cfgs.values())
    N = NS[-1]
    slots, lens = SLOTS[:N], LENS[:N]
    dec = decode([int(l.argmax()) for l in ref[N]["logits"]], slots, lens)
    assert (
        dec == ref[N]["decode"]
    ), f"N={N}: greedy decode after traced batched prefill differs: {dec} vs {ref[N]['decode']}"
    del model
    gc.collect()
