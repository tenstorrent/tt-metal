# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""End-to-end SINGLE-STREAM RTF on N300 with REAL generated audio.

Unlike a placeholder-frame timing test, this runs the *actual* traced generator
(real prefill, real traced decode, real sampling -> real audio codes) and then
the ported TTNN codec on those real codes -> real waveform, timing the true
generate->waveform pipeline. It reports an honest single-stream RTF (audio
seconds produced per wall second), NOT a multi-chip aggregate.

Metrics:
  decode_tok_per_s / decode_rtf  — steady-state decode (excludes one-time
                                   prefill + trace build), the model's real rate.
  e2e_rtf                        — full single-utterance experience
                                   (prefill + decode + codec) / audio_seconds.
"""
import os
import time

import pytest
import ttnn
from loguru import logger

from models.demos.audio.higgs_audio_v2.demo.generator import HiggsAudioTTSGenerator
from models.demos.audio.higgs_audio_v2.demo.demo import _tts_conversation

CODEC_FRAME_RATE_HZ = 25.0
SAMPLING_RATE = 24000
MAX_NEW = int(os.environ.get("E2E_MAX_NEW", "200"))
TEXT = "Tenstorrent hardware now runs text to speech on Wormhole."


@pytest.fixture(scope="module")
def mesh_device():
    # l1_small_size bumped so the TTNN codec can run co-resident with the LLM.
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=98304, trace_region_size=200000000)
    yield dev
    ttnn.close_mesh_device(dev)


def test_e2e_generated_audio_rtf(mesh_device):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["HIGGS_TTNN_CODEC"] = "1"  # decode the real codes on the ported TTNN codec

    gen = HiggsAudioTTSGenerator(mesh_device, precision="performance")
    conv = _tts_conversation(TEXT)

    # warmup: compile + build the decode trace AND build/warm the TTNN codec
    # (the first to_waveforms otherwise pays a one-time ~30s TtDacDecoder weight-prep).
    # Pinned to the HYBRID path (ondevice_sample=False) as a stable single-stream perf
    # reference: host reads logits + samples each step. The fully-on-device sampler is the
    # generate() default, but its blocking readback makes wall-clock RTF host-load-sensitive,
    # so the hybrid path is used here for a stable, tight gate.
    warm_seq = gen.generate(
        conv, max_new_tokens=MAX_NEW, temperature=0.7, top_k=50, top_p=0.95, seed=1, use_trace=True, ondevice_sample=False
    )
    _ = gen.to_waveforms(warm_seq, trim=False)

    # ---- timed real run: real tokens through the traced generator ----
    seq = gen.generate(
        conv, max_new_tokens=MAX_NEW, temperature=0.7, top_k=50, top_p=0.95, seed=2, use_trace=True, ondevice_sample=False
    )
    t = gen._last_timing
    rows = int(seq.shape[1] - 1)
    audio_seconds = rows / CODEC_FRAME_RATE_HZ

    # ---- real codec on the real generated codes (TTNN DAC decoder conv stack) ----
    t0 = time.perf_counter()
    wf = gen.to_waveforms(seq, trim=False)[0]
    codec_seconds = time.perf_counter() - t0
    wav_seconds = wf.numel() / SAMPLING_RATE

    decode_tps = t["decode_tok_per_s"]
    decode_rtf = CODEC_FRAME_RATE_HZ / decode_tps if decode_tps > 0 else float("inf")
    total_seconds = t["prefill_s"] + t["decode_s"] + codec_seconds
    e2e_rtf = total_seconds / audio_seconds if audio_seconds > 0 else float("inf")

    logger.info(f"REAL generated audio: {rows} frames = {audio_seconds:.2f}s of audio ({wav_seconds:.2f}s waveform)")
    logger.info(f"  steady-state decode: {decode_tps:.1f} tok/s  decode-RTF {decode_rtf:.3f}")
    logger.info(
        f"  prefill {t['prefill_s']*1e3:.0f}ms  trace-build {t['trace_build_s']*1e3:.0f}ms  "
        f"decode {t['decode_s']:.2f}s  codec {codec_seconds*1e3:.0f}ms (~{100*codec_seconds/total_seconds:.0f}% of e2e)"
    )
    logger.info(f"  END-TO-END single-utterance RTF (prefill+decode+codec) = {e2e_rtf:.3f}")
    print(
        f"PERF_E2E_REAL decode_tps={decode_tps:.1f} decode_rtf={decode_rtf:.4f} e2e_rtf={e2e_rtf:.4f} "
        f"rows={rows} prefill_ms={t['prefill_s']*1e3:.0f} codec_ms={codec_seconds*1e3:.0f}"
    )

    # Honest single-stream gates (env-overridable). decode-RTF is the model's real
    # steady-state rate; Stage-1 target is RTF < 0.5. e2e includes one-time prefill.
    max_decode_rtf = float(os.environ.get("HIGGS_MAX_DECODE_RTF", "0.55"))
    max_e2e_rtf = float(os.environ.get("HIGGS_MAX_E2E_RTF", "0.75"))
    assert rows > 8, f"generation too short ({rows} rows) to time meaningfully"
    assert decode_rtf <= max_decode_rtf, f"steady-state decode RTF {decode_rtf:.4f} > {max_decode_rtf} (regressed)"
    assert e2e_rtf <= max_e2e_rtf, f"e2e single-utterance RTF {e2e_rtf:.4f} > {max_e2e_rtf} (regressed)"
    assert codec_seconds < t["decode_s"], "codec should be a small fraction of decode"
