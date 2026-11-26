# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The serving form of the decode model: the prefill stack stays BUILT beside it (both weight sets resident, the
prefill stack's trace and the decode step's trace in one process), prefill passes are replayed between decode steps,
and a conversation prefilled while the rows decode joins them. tests/test_prefill_decode.py is the same chain with the
weights swapped; tests/test_coexist.py is this form for the shared-indexer step with the host's selection.

    HY4_MOCK=1 pytest <this file>                                 (the decode side, the reserve, the restore program; no stack)
    HY4_TEST_TIMEOUT=2400 TT_METAL_OPERATION_TIMEOUT_SECONDS=30 pytest <this file> -x

Env: tests/test_prefill_decode.py's, and HY4_PD_RESERVE_GB (3: the DRAM held for the decode model, per chip; its
weights, the ends' tables, the restore image and both traces' temporaries of ours must fit), HY4_PD_EVERY (4: a
prefill pass before every n-th decode step of phase B), HY4_PD_PREFILL (the stack's layout: PREFILL_HY4_LAYOUT).

Order (PLAN.md section 24):
  a. the prefill stack in trace mode, its caches for 8 users, its programs compiled with nothing of ours in L1
  b. a DRAM ballast holds the range the decode model will take (model/reserve.py)
  c. the prefill stack's trace captured
  d. the golden's prompt into slots 0 .. 6 by replay; slot 7 stays empty
  e. plugs over all free DRAM, the ballast freed; DecodeModel(stores=the stack's caches) (its L1 state, its weights);
     rows 0 .. 6 adopted at N - 1; golden step 0 program by program (it compiles the step); the tensors the host
     writes before every step marked; the restore program made, launched and captured as a trace; golden step 1
     (the model captures its step there: the capture's temporaries inside the range); the plugs freed, every DRAM
     buffer of ours checked to lie in the range
  A. the golden's remaining steps on rows 0 .. 6: the logits kept
  B. rows 0 .. 6 rewound to N - 1 and the same steps again, with before every HY4_PD_EVERY-th step ONE prefill
     pass by replay of the prompt into slot 7 and one replay of the restore trace: every row's logits against
     phase A, BIT FOR BIT, and against the golden
  C. row 7 adopted at N - 1 (the conversation that arrived while the others decoded): the golden's steps on it,
     its logits against row 0's of phase A bit for bit
Lines start with HY4_PD.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.model import reserve as RS
from models.experimental.hy4_preview.fused.model.decode_model import DecodeModel
from models.experimental.hy4_preview.fused.model.program import ROWS, is_mock
from models.experimental.hy4_preview.fused.model.tests import test_decode_model as TD
from models.experimental.hy4_preview.fused.model.tests import test_prefill_decode as PD
from models.experimental.hy4_preview.tt import prefill_bridge as pb

RESERVE = int(float(os.environ.get("HY4_PD_RESERVE_GB", "3")) * (1 << 30))
EVERY = int(os.environ.get("HY4_PD_EVERY", "4"))
TRACE_REGION = 256 << 20  # (the prefill stack's trace and ours)

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION), TD.MESH)],
    indirect=True,
    ids=["8x4"],
)


def mark_per_step(model, tokens, positions):
    """The L1 tensors the host writes before EVERY step (the rows' positions, the RoPE tiles, the scan's contexts):
    found by writing a step's inputs once more and seeing which registered tensors were written. The restore leaves
    them alone and a restore image is not stale because of them."""
    hw = model.st.hw
    before = {a: e.writes for a, e in hw.live().items()}
    model._inputs(tokens, positions)
    ttnn.synchronize_device(model.mesh)
    found = [e for a, e in hw.live().items() if e.writes != before.get(a)]
    for e in found:
        e.per_step = True
    return [(e.where, e.bytes()) for e in found]


@galaxy
@pytest.mark.timeout(0)
def test_prefill_beside_decode(mesh_device, monkeypatch):
    mesh, n = mesh_device, TD.LAYERS
    g = TD.golden_of()
    N, steps, ids = g["meta"]["prompt_len"], g["meta"]["steps"], g["input_ids"]
    assert N <= pb.CHUNK, "one chunk pass a prompt in this test"
    prompt = [int(t) for t in ids[:N]]
    label = (
        f"prefill stack BESIDE the decode model [{n} layers, P{N}, append {PD.APPEND}, latent_scale {PD.SCALE}, "
        f"experts {TD.SPLIT}, prefill layout {PD.PREFILL_LAYOUT or 'dev'}, a prefill pass every {EVERY} steps]"
    )
    marks = []

    def mark(what, t0):
        marks.append((what, round(time.perf_counter() - t0, 1), round(pb.dram(mesh)["allocated_gb"], 3)))
        logger.info(f"HY4_PD {label}: {what}: {marks[-1][1]} s, DRAM {marks[-1][2]} GB per chip")

    bridge = reserve = prefill = restore_trace = None
    banks = RS.dram_view(mesh)["banks"]
    try:
        if is_mock():
            stores = PD.mock_stores(mesh, n)
        else:
            import models.demos.common.prefill.adapter as AD
            from models.experimental.hy4_preview.fused.kv_store.engine import StackPrefill

            # a. the stack in trace mode (the one parameter the bridge does not pass)
            PD.select_prefill_layout(monkeypatch)
            params = AD.PrefillRunParams
            monkeypatch.setattr(AD, "PrefillRunParams", lambda **kw: params(**dict(kw, use_trace=True)))
            t0 = time.perf_counter()
            bridge = pb.PrefillBridge(mesh, num_layers=n, num_users=ROWS, max_seq_len=PD.MAX_SEQ)
            bridge.build()
            monkeypatch.setattr(AD, "PrefillRunParams", params)
            mark("a prefill stack built in trace mode, warmed", t0)
            stores = (bridge.kv_caches.kvpe.storage, bridge.kv_caches.index)
        # b. the reserve
        t0 = time.perf_counter()
        reserve = RS.DramReserve(mesh, RESERVE // banks)
        mark(f"b ballast of {RESERVE >> 20} MB per chip", t0)
        if not is_mock():
            # c. the capture; d. the prompt into slots 0 .. 6 by replay
            t0 = time.perf_counter()
            runtime, caches = bridge.runtime, bridge.kv_caches
            runtime.capture_trace(caches)
            ttnn.synchronize_device(mesh)
            mark(
                f"c prefill trace captured ({runtime._controller.num_segments} segments, {runtime._controller.trace_bytes()} B)",
                t0,
            )
            prefill = StackPrefill(runtime, caches, mesh, pb.MESH, trace=True)
            t0 = time.perf_counter()
            passes = [prefill(user, 0, N, prompt) for user in range(ROWS - 1)]
            mark(f"d prefill of {N} tokens into slots 0 .. {ROWS - 2} by replay", t0)
            logger.info(
                f"HY4_PD {label}: the replayed passes before anything of ours exists, ms each (the pass, its input): "
                f"{[(round(1e3 * r['chunk_s'], 1), round(1e3 * r['prep_s'], 1)) for r in passes]}"
            )
        # e. ours, inside the reserved range
        before = RS.buffers(mesh, "DRAM")
        reserve.plug()
        reserve.open()
        t0 = time.perf_counter()
        model = DecodeModel(
            mesh,
            layers=n,
            max_positions=PD.MAX_SEQ,
            trace=True,
            keep_logits=True,
            check_every=TD.CHECK_EVERY,
            mlp_split=TD.SPLIT,
            append=PD.APPEND,
            latent_scale=PD.SCALE,
            stores=stores,
        )
        mark("e1 decode model built beside the prefill stack", t0)
        first = list(range(ROWS - 1))  # the rows that decode from the start; row 7 joins in phase C
        for r in first:
            model.adopt_row(r, N - 1)

        def feed(rows, j):
            return TD.idle_mix({r: int(ids[N - 1 + j]) for r in rows}, {r: N - 1 + j for r in rows})

        active = [r in first for r in range(ROWS)]
        t0 = time.perf_counter()
        tokens, positions = feed(first, 0)
        # (the mock runs nothing: there is no selection to check)
        probe = (
            (lambda name: None)
            if is_mock()
            else TD.Probe(model, f"{label} step 0", positions, active, TD.golden_picks(g, 0))
        )
        out = {0: model.step_probed(tokens, positions, after=probe)}
        logits = {} if is_mock() else {0: model.read_logits().clone()}
        per_step = mark_per_step(model, *feed(first, 1))  # (the next step's inputs: the rows' next positions)
        restore = model.st.make_restore()
        restore.launch()  # (compiled here, before our capture; the step is at rest: it changes nothing)
        ttnn.synchronize_device(mesh)
        logger.info(
            f"HY4_PD {label}: {restore.summary()}; written by the host before every step (not restored): {per_step}"
        )
        if not is_mock():
            restore_trace, _ = tracing.capture(mesh, restore.launch)
            ttnn.synchronize_device(mesh)
        out[1] = model.step(*feed(first, 1))  # (the model captures its step here)
        if not is_mock():
            logits[1] = model.read_logits().clone()
        used, outside = reserve.close(before)
        mark(
            f"e2 the first two steps, the restore program and both captures of ours ({used * banks >> 20} MB per chip of the reserve)",
            t0,
        )
        assert not outside, f"DRAM tensors of ours outside the reserved range: {outside}"
        if is_mock():
            logger.info(
                f"HY4_PD {label}: compiled on the mock: {len(model.programs) + 2} programs a step and the restore program, "
                f"program-cache entries {mesh.num_program_cache_entries()}, L1 tensors {model.st.l1_bytes()} B per core; "
                f"{used * banks >> 20} MB per chip of the reserve used"
            )
            return
        # A. the remaining steps, nothing in between
        for j in range(2, steps):
            out[j] = model.step(*feed(first, j))
            logits[j] = model.read_logits().clone()
        status = model.status()
        assert not status, f"phase A: {status}"
        want = [int(v) for v in g["cached_logits_argmax"][:steps]]
        logger.info(
            f"HY4_PD {label} PHASE A (no prefill between the steps): row 0's token = the golden's on "
            f"{sum(out[j][0] == want[j] for j in range(steps))} of {steps} steps; every row the same token on "
            f"{sum(len({out[j][r] for r in first}) == 1 for j in range(steps))} of {steps}"
        )
        # B. the same steps with prefill passes between them
        for r in first:
            model.reset_row(r, keep=N - 1)
        table, pass_ms, restore_ms, differ, n_pass = [], [], [], [], 0
        for j in range(steps):
            if j % EVERY == 0:
                r = prefill(ROWS - 1, 0, N, prompt)
                pass_ms.append(round(1e3 * r["chunk_s"], 1))
                assert not restore.stale(), "the restore image is stale"
                t1 = time.perf_counter()
                ttnn.execute_trace(mesh, restore_trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                restore_ms.append(round(1e3 * (time.perf_counter() - t1), 2))
                n_pass += 1
            got = model.step(*feed(first, j))
            now = model.read_logits()
            TD.compare_step(model, g, j, got, first, table, None)
            bad = [r for r in first if not torch.equal(now[r], logits[j][r])]
            if bad:
                differ.append((j, bad, float((now[bad] - logits[j][bad]).abs().max())))
        status = model.status()
        TD.summarise(table, label + " PHASE B")
        logger.info(
            f"HY4_PD {label} PHASE B: {n_pass} prefill passes by replay into slot {ROWS - 1} between the {steps} decode steps "
            f"(ms each {pass_ms}; the restore replay with its synchronize, ms {restore_ms}); logits of rows 0 .. {ROWS - 2} against "
            f"phase A: equal in every bit on {steps - len(differ)} of {steps} steps; steps that differ (step, rows, largest "
            f"difference) {differ[:12]}; status {status or 'at rest'}"
        )
        assert not status, f"phase B: {status}"
        # C. the conversation that arrived joins on row 7
        model.adopt_row(ROWS - 1, N - 1)
        joined, same = [], 0
        for j in range(steps):
            got = model.step(*feed([ROWS - 1], j))
            joined.append(got[ROWS - 1])
            same += bool(torch.equal(model.read_logits()[ROWS - 1], logits[j][0]))
        status = model.status()
        logger.info(
            f"HY4_PD {label} PHASE C: row {ROWS - 1} on the slot prefilled between the others' steps: token = the golden's on "
            f"{sum(a == b for a, b in zip(joined, want))} of {steps} steps, = row 0's of phase A on "
            f"{sum(joined[j] == out[j][0] for j in range(steps))}; logits equal to row 0's of phase A in every bit on {same} of {steps}; "
            f"status {status or 'at rest'}"
        )
        logger.info(f"HY4_PD {label} phases (what, s, DRAM GB per chip): {marks}")
        logger.info(f"HY4_PD {label} DRAM with both models resident and both traces captured: {pb.dram(mesh)}")
        assert not status, f"phase C: {status}"
        assert not differ, f"a prefill pass between decode steps changed the decode rows: {differ[:6]}"
        assert sum(joined[j] == out[j][0] for j in range(steps)) == steps, "the joined row leaves the others' tokens"
    finally:
        if reserve is not None:
            reserve.release()
        if bridge is not None:
            bridge.close()
