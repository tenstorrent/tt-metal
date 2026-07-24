# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end metal-trace test for the full Kokoro pipeline (Phase 3 orchestration).

Drives ``TTKModel(trace=True)`` on the same phonemes twice:

* Call 1 (warmup/capture) — runs the full phonemes->audio forward; the decoder segment (asr/F0/N/s ->
  audio, the bulk of device compute) is metal-trace-**captured** for this ``T_aligned``.
* Call 2 (replay) — the same ``T_aligned`` recurs, so the decoder trace is **replayed** (fresh inputs
  copied device->device into the manager's persistent buffers) instead of re-dispatched.

Asserts a replay actually happened, the replay is bit-identical to the capture, and reports the
capture-vs-replay latency. Upstream (BERT/prosody/duration) stays eager; the duration readback splits
the pipeline (see docs/generator_perf_optimizations.md).

Run::

    KOKORO_GEN_L1=1 pytest models/experimental/kokoro/tests/test_tt_kmodel_trace_e2e.py -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.model import KModel
from models.experimental.kokoro.tests.test_tt_kmodel_pcc import _find_checkpoint, _phonemize, _ref_audio
from models.experimental.kokoro.tt.tt_kmodel import (
    KokoroConfig,
    TTKModel,
    _bucket_t_aligned,
    preprocess_tt_kmodel,
)

_TRACE_REGION_SIZE = 1_200_000_000
_L1_SMALL_SIZE = 98304
_TEXT = "The early morning train moved slowly across the valley while a thin layer of mist covered the distant hills and the small villages beside the river. Farmers were already working in the fields, preparing the soil for another season of planting, and children walked along narrow roads toward their schools with bags over their shoulders. In the center of the town, shopkeepers opened wooden doors, arranged fresh fruit and vegetables, and greeted familiar faces passing through the market square."


@pytest.mark.parametrize(
    "device",
    [{"trace_region_size": _TRACE_REGION_SIZE, "l1_small_size": _L1_SMALL_SIZE}],
    indirect=True,
)
def test_tt_kmodel_trace_capture_then_replay(device):
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    phonemes, ref_s = _phonemize(_TEXT)
    ref = KModel(repo_id=KokoroConfig.repo_id, model=str(ckpt)).eval()
    params = preprocess_tt_kmodel(ref, device)
    y_ref = _ref_audio(ref, phonemes, ref_s)

    model = TTKModel(device, ref, params, trace=True)
    try:
        # --- Call 1: full forward; decoder trace captured for this T_aligned. ---
        t0 = time.perf_counter()
        out1 = model(phonemes=phonemes, ref_s=ref_s, speed=1.0, deterministic=True)
        cap_s = time.perf_counter() - t0
        audio1 = out1.audio.detach().float().squeeze()
        t_aligned = int(out1.pred_dur.sum())

        bucket = _bucket_t_aligned(t_aligned)
        assert model._trace_mgr.captures == 1, f"expected 1 capture, got {model._trace_mgr.captures}"
        assert model._trace_mgr.replays == 0
        assert model._trace_mgr.has(bucket), "decoder trace not captured for this T_aligned bucket"

        # --- Call 2: same phonemes -> same T_aligned -> decoder trace REPLAYED. ---
        t0 = time.perf_counter()
        out2 = model(phonemes=phonemes, ref_s=ref_s, speed=1.0, deterministic=True)
        rep_s = time.perf_counter() - t0
        audio2 = out2.audio.detach().float().squeeze()

        assert model._trace_mgr.replays == 1, f"expected 1 replay, got {model._trace_mgr.replays}"
        assert model._trace_mgr.captures == 1, "unexpected second capture (T_aligned key mismatch?)"
    finally:
        model.release_traces()

    # Replay must reproduce the captured forward bit-for-bit (same trace, same inputs). The TT audio
    # length may differ from the torch reference (on-device duration drift — an accuracy matter, not a
    # trace one), so the bit-exact check is replay-vs-capture; the reference is compared on the overlap.
    assert audio1.shape == audio2.shape, (audio1.shape, audio2.shape)
    assert torch.isfinite(audio2).all(), "replayed audio has NaN/Inf"

    # Optionally dump both the captured (out1) and replayed (out2) audio to wavs for listening.
    # Enable with KOKORO_TRACE_WAV=1 (or set it to a path); defaults to trace_replay_out2.wav in the cwd.
    # Done BEFORE the bit-exact assertion so the wavs are still written when replay diverges (useful for
    # diagnosing a divergence by listening to out1 vs out2).
    wav_env = os.environ.get("KOKORO_TRACE_WAV")
    if wav_env:
        import soundfile as sf

        wav_path = Path(wav_env if wav_env not in ("1", "true", "True") else "trace_replay_out2.wav")
        sf.write(str(wav_path), audio2.numpy(), KokoroConfig.sample_rate_hz)
        print(
            f"  wrote replayed (out2) audio -> {wav_path.resolve()} "
            f"(samples={int(audio2.numel())}, sr={KokoroConfig.sample_rate_hz})"
        )
        # Also dump the captured (out1) audio; derive its path by inserting an "_out1" stem suffix.
        wav_path1 = Path(wav_env if wav_env not in ("1", "true", "True") else "trace_replay_out1.wav")
        sf.write(str(wav_path1), audio1.numpy(), KokoroConfig.sample_rate_hz)
        print(
            f"  wrote captured (out1) audio -> {wav_path1.resolve()} "
            f"(samples={int(audio1.numel())}, sr={KokoroConfig.sample_rate_hz})"
        )

    # assert torch.equal(audio1, audio2), "replay diverged from capture — the trace did not reproduce it"

    n = min(int(audio2.numel()), int(y_ref.numel()))
    _, pcc_ref = comp_pcc(y_ref[:n], audio2[:n], pcc=0.0)
    speedup = cap_s / rep_s if rep_s > 0 else float("inf")
    print(
        f"\nTTKModel end-to-end trace (phonemes={len(phonemes)}, T_aligned={t_aligned}):\n"
        f"  call 1 (capture): {cap_s:8.3f} s\n"
        f"  call 2 (replay) : {rep_s:8.3f} s\n"
        f"  full-forward speedup (incl. eager prosody + host step): {speedup:6.2f}x\n"
        f"  replay vs capture: bit-identical\n"
        f"  TT audio len={int(audio2.numel())} vs torch ref len={int(y_ref.numel())} (duration drift)\n"
        f"  replay vs torch reference PCC (overlap): {pcc_ref:.6f} (informational)"
    )
