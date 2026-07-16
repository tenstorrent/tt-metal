# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full end-to-end validation of the TWO-TRACE path (Trace A prosody+ASR on CQ0 + folded decoder
Trace B on CQ1). Long text, capture then replay, bit-identical + torch-reference PCC.

Trace A (prosody->duration + ASR TextEncoder, T_tokens) and the decoder Trace B (en/F0N/asr + decoder,
T_aligned) coexist on SEPARATE command queues (a second trace on the same CQ hangs the decoder replay).
Sets ``KOKORO_TRACE_A=1`` itself and opens the device with ``num_command_queues=2``.

Run (cold first run recompiles ~800 kernels for the 2-CQ config, so allow a long timeout)::

    KOKORO_GEN_L1=1 pytest -s --timeout=1400 \
        models/experimental/kokoro/tests/test_tt_kmodel_two_trace_e2e.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.model import KModel
from models.experimental.kokoro.tests.test_tt_kmodel_pcc import _find_checkpoint, _phonemize
from models.experimental.kokoro.tt.tt_kmodel import KokoroConfig, TTKModel, preprocess_tt_kmodel

_TRACE_REGION_SIZE = 1_200_000_000
_L1_SMALL_SIZE = 98304
# ~499 phonemes as a SINGLE KPipeline chunk (verified via KPipeline af_heart): stays under the
# PLBERT context cap (512 - 2 BOS/EOS = 510 phonemes max), so the whole passage is one forward.
_TEXT = (
    "The early morning train moved slowly across the wide green valley while a thin layer of silver "
    "mist covered the distant hills and the small quiet villages beside the winding river. Farmers "
    "were already working in the fields, carefully preparing the dark soil for another season of "
    "planting, and children walked along the narrow country roads toward their schools with heavy "
    "bags over their shoulders, laughing softly as the cool autumn breeze carried the gentle sound "
    "of church bells across the calm and shining water."
)


@pytest.mark.parametrize(
    "device",
    [{"trace_region_size": _TRACE_REGION_SIZE, "l1_small_size": _L1_SMALL_SIZE, "num_command_queues": 2}],
    indirect=True,
)
def test_tt_kmodel_two_trace_capture_then_replay(device):
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    prev = os.environ.get("KOKORO_TRACE_A")
    os.environ["KOKORO_TRACE_A"] = "1"
    try:
        phonemes, ref_s = _phonemize(_TEXT)
        ref = KModel(repo_id=KokoroConfig.repo_id, model=str(ckpt)).eval()
        params = preprocess_tt_kmodel(ref, device)

        model = TTKModel(device, ref, params, trace=True)
        assert model._trace_mgr_a is not None, "two-trace path not enabled (KOKORO_TRACE_A/num CQs?)"
        try:
            # Call 1: capture both traces.
            t0 = time.perf_counter()
            out1 = model(phonemes=phonemes, ref_s=ref_s, speed=1.0, deterministic=True)
            cap_s = time.perf_counter() - t0
            audio1 = out1.audio.detach().float().squeeze()
            t_aligned = int(out1.pred_dur.sum())
            assert model._trace_mgr_a.captures == 1 and model._trace_mgr_a.replays == 0
            assert model._trace_mgr.captures == 1 and model._trace_mgr.replays == 0

            # Call 2: same input -> both traces replay.
            t0 = time.perf_counter()
            out2 = model(phonemes=phonemes, ref_s=ref_s, speed=1.0, deterministic=True)
            rep_s = time.perf_counter() - t0
            audio2 = out2.audio.detach().float().squeeze()
            assert model._trace_mgr_a.replays == 1 and model._trace_mgr_a.captures == 1
            assert model._trace_mgr.replays == 1 and model._trace_mgr.captures == 1
        finally:
            model.release_traces()
    finally:
        if prev is None:
            os.environ.pop("KOKORO_TRACE_A", None)
        else:
            os.environ["KOKORO_TRACE_A"] = prev

    assert audio1.shape == audio2.shape, (audio1.shape, audio2.shape)
    assert torch.isfinite(audio2).all(), "replayed audio has NaN/Inf"

    _, parity = comp_pcc(audio1, audio2, pcc=0.0)
    speedup = cap_s / rep_s if rep_s > 0 else float("inf")
    print(
        f"\nTTKModel TWO-TRACE end-to-end (phonemes={len(phonemes)}, T_aligned={t_aligned}):\n"
        f"  traceA cap/rep = {model._trace_mgr_a.captures}/{model._trace_mgr_a.replays}\n"
        f"  decoderB cap/rep = {model._trace_mgr.captures}/{model._trace_mgr.replays}\n"
        f"  call 1 (capture): {cap_s:8.3f} s\n"
        f"  call 2 (replay) : {rep_s:8.3f} s   ({speedup:5.2f}x full-forward)\n"
        f"  replay vs capture: PCC={parity:.8f} (want 1.0, bit-identical)\n"
        f"  TT audio len={int(audio2.numel())}"
    )

    # Dump both the captured (out1) and replayed (out2) audio to wavs for listening. ON BY DEFAULT;
    # set KOKORO_TRACE_WAV=0 to disable, or =/path/stem.wav for a custom location (out1/out2 derived
    # from the stem). Done BEFORE the bit-exact assertion so the wavs are still written when the
    # two-trace path diverges (the whole point right now — listen to out1/out2 to hear the corruption).
    wav_env = os.environ.get("KOKORO_TRACE_WAV", "1")
    if wav_env and wav_env not in ("0", "false", "False"):
        import soundfile as sf

        base = Path("two_trace_out.wav" if wav_env in ("1", "true", "True") else wav_env)
        wav2 = base.with_name(f"{base.stem}_out2{base.suffix}")
        wav1 = base.with_name(f"{base.stem}_out1{base.suffix}")
        sf.write(str(wav2), audio2.numpy(), KokoroConfig.sample_rate_hz)
        sf.write(str(wav1), audio1.numpy(), KokoroConfig.sample_rate_hz)
        print(
            f"  wrote replayed (out2) audio -> {wav2.resolve()} (samples={int(audio2.numel())}, "
            f"sr={KokoroConfig.sample_rate_hz})\n"
            f"  wrote captured (out1) audio -> {wav1.resolve()} (samples={int(audio1.numel())}, "
            f"sr={KokoroConfig.sample_rate_hz})"
        )

    assert torch.equal(audio2, audio1), f"two-trace replay diverged from capture (PCC={parity:.8f})"
