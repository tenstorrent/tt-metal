# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end gate for the TRACE-ACCELERATED fast path (tt/pipeline.py::run_tts_fast).

The committed `test_e2e_tts.py` gates the DYNAMIC path (`run_tts`); the perf harness
(`test_main_perf.py`) profiles the per-stage traces in isolation. Neither actually runs
`run_tts_fast` — the chained trace+bypass path we ship and quote RTF for. This test closes
that gap with two hard gates on the fast path itself:

  Correctness — log-spectrogram PCC of the fast waveform vs the determinized HF golden
                >= 0.95 (same phase-invariant metric as the dynamic gate; see test_e2e_tts).
  Performance — measured steady-state RTF (dynamic token/align prep + traced frame replay,
                trace captured once) below a floor with wide margin, so a change that
                re-inflates op count / breaks trace / drops back to the dynamic path fails CI.

Run: ./python_env/bin/python -m pytest models/demos/kokoro_82m/tests/e2e/test_tts_fast.py -s
"""
from __future__ import annotations

import os
import sys
import time

import pytest

import ttnn
from models.demos.kokoro_82m._stubs import _trace_alloc
from models.demos.kokoro_82m._stubs._lstm_scan import pop_trace_ctx, push_trace_ctx
from models.demos.kokoro_82m.tt import ops
from models.demos.kokoro_82m.tt import pipeline as P

SPEC_PCC_TARGET = 0.95
# Measured RTF is 0.21 on a single P150; gate at 0.45 so normal device/host jitter never
# flakes the CI, but a real regression (op-count re-inflation, trace break, or a silent
# fall-back to the ~0.87-RTF dynamic path) trips it.
RTF_MAX = 0.45
SR = 24000

_DEV_PARAMS = {"l1_small_size": 24576, "trace_region_size": (1 << 30), "num_command_queues": 2}


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_tts_fast(device_params, device):
    if not P._FRAME_TRACE:
        pytest.skip("KOKORO_TRACE_FRAME=0 forces the dynamic fallback; fast path not under test")

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pcc"))
    from _reference_loader import load_reference_model

    model = load_reference_model("hexgrad/Kokoro-82M").float().eval()
    input_ids, ref_s = P.build_input(model)
    gold, gold_dur = P.hf_reference_tts(model, input_ids, ref_s)

    pipe = P.build_pipeline(device, model=model)

    # ---- correctness: the shipped fast path vs the HF golden ----
    wav, pred_dur = P.run_tts_fast(pipe, input_ids, ref_s)
    dur_s = wav.numel() / SR
    spec_pcc = P.log_spectrogram_pcc(gold, wav)
    dur_match = bool((pred_dur == gold_dur).all())
    print(f"\n[fast] waveform samples: {wav.numel()} ({dur_s:.2f}s)")
    print(f"[fast] pred_dur matches HF exactly: {dur_match}")
    print(f"[fast] log-spectrogram PCC (vs gold) = {spec_pcc}")

    # ---- performance: measured steady-state RTF (trace captured once) ----
    ins = {"input_ids": input_ids, "ref_s": ref_s}
    ops.set_hp_bypass(True)
    try:

        def _timeit(fn, iters, warmup):
            for _ in range(warmup):
                fn()
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            ttnn.synchronize_device(device)
            return 1000.0 * (time.perf_counter() - t0) / iters

        prep_ms = _timeit(lambda: pipe._prep_frame_inputs(ins), iters=5, warmup=2)
        d = pipe._prep_frame_inputs(ins)
        en, asr, s, dstyle, ctx = d["en"], d["asr"], d["s"], d["dec_style"], d["ctx"]

        def _frame_fwd():
            F0, N = pipe._f0n_train(en, s)
            x = pipe._decode_features(asr, F0, N, dstyle)
            return pipe.generator(x, dstyle, F0)

        _trace_alloc.activate()
        push_trace_ctx(ctx)
        _frame_fwd()
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        _frame_fwd()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        replay_ms = _timeit(lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False), iters=10, warmup=3)
        ttnn.release_trace(device, tid)
        pop_trace_ctx()
        _trace_alloc.deactivate()
    finally:
        ops.set_hp_bypass(False)

    fast_ms = prep_ms + replay_ms
    rtf = fast_ms / 1000.0 / dur_s
    print(f"[fast] steady-state = {fast_ms:.1f} ms (prep {prep_ms:.1f} + frame replay {replay_ms:.1f})")
    print(f"[fast] RTF = {rtf:.3f}  ({1000.0 * dur_s / fast_ms:.2f}x real-time)")

    assert dur_match, f"pred_dur mismatch TT={pred_dur.tolist()} HF={gold_dur.tolist()}"
    assert spec_pcc >= SPEC_PCC_TARGET, f"fast-path log-spectrogram PCC {spec_pcc} < {SPEC_PCC_TARGET}"
    assert rtf < RTF_MAX, f"fast-path RTF {rtf:.3f} >= {RTF_MAX} (perf regression / trace fell back)"
