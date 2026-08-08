# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""Batched multi-stream throughput on one Wormhole chip (real, DIFFERENT prompts).

This is the honest batching experiment: B genuinely different TTS utterances are
prefilled into their own KV-cache rows and decoded in lockstep, so a SINGLE
~2.17 GB weight read per step serves all B streams. Decode is DRAM-bandwidth-bound
at batch 1, so batching is the one single-chip lever that raises aggregate
throughput toward B x — until it crosses into compute-bound.

Streams terminate independently (ragged EOS): a finished stream is frozen while
the rest keep decoding. This is NOT the deleted data-parallel test (which ran two
independent model copies, one per chip); here it is ONE model on ONE chip
amortizing one weight read across B different utterances.

Metrics:
  per_stream_tok_per_s  — decode steps / wall second (~ the single-stream rate).
  device_tok_per_s      — B x per_stream: aggregate tokens the chip sustains.
  scaling               — device_tok_per_s / single-stream reference (target > 1).

Honest caveat: aggregate throughput rises with B, but PER-STREAM RTF degrades as
B grows (more compute per weight read). This is a throughput/serving-capacity
metric, NOT a single-utterance latency win.
"""
import os

import pytest
from loguru import logger

import ttnn
from models.demos.audio.higgs_audio_v2.demo.demo import _tts_conversation
from models.demos.audio.higgs_audio_v2.demo.generator import HiggsAudioTTSGenerator

CODEC_FRAME_RATE_HZ = 25.0
SAMPLING_RATE = 24000
MAX_NEW = int(os.environ.get("BATCH_MAX_NEW", "160"))
BATCH = int(os.environ.get("HIGGS_BATCH", "4"))

# B genuinely different utterances (different lengths -> ragged EOS).
TEXTS = [
    "Tenstorrent hardware now runs text to speech on Wormhole.",
    "The quick brown fox jumps over the lazy dog.",
    "Batched decoding lets one weight read serve many streams at once.",
    "Good morning, and welcome to the demonstration.",
    "She sells sea shells by the sea shore.",
    "Artificial intelligence is transforming how we build software.",
    "A journey of a thousand miles begins with a single step.",
    "Please remember to save your work before you leave.",
]


def _rms(wf):
    return float(wf.detach().float().pow(2).mean().clamp_min(1e-12).sqrt())


@pytest.fixture(scope="module")
def mesh_device():
    # Single chip: batching amortizes the weight read within one chip (no CCL).
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=98304, trace_region_size=200000000)
    yield dev
    ttnn.close_mesh_device(dev)


def test_batched_multistream_throughput(mesh_device):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["HIGGS_TTNN_CODEC"] = "1"

    B = min(BATCH, len(TEXTS))
    convs = [_tts_conversation(t) for t in TEXTS[:B]]

    # Single-stream reference (scaling denominator). A full 3B model already fills ~10.4 GB
    # of the chip's DRAM, so a second batch-1 model can't co-reside; instead we use the
    # measured single-stream decode rate (~50 tok/s), which test_perf_e2e_n300 and the
    # generate_batch(N=1) path both confirm on this exact chip/precision. Env-overridable.
    single_stream_tps = float(os.environ.get("HIGGS_SINGLE_STREAM_TPS", "50.0"))

    # ---- batched run: B different utterances in lockstep on a max_batch_size=B model ----
    gen = HiggsAudioTTSGenerator(mesh_device, precision="performance", max_batch_size=B)
    seqs, t = gen.generate_batch(convs, max_new_tokens=MAX_NEW, temperature=0.7, top_k=50, top_p=0.95, seed=2)

    per_stream_tps = t["per_stream_tok_per_s"]
    device_tps = t["device_tok_per_s"]
    scaling = device_tps / single_stream_tps if single_stream_tps > 0 else 0.0
    per_stream_rtf = CODEC_FRAME_RATE_HZ / per_stream_tps if per_stream_tps > 0 else float("inf")

    # decode each stream's real codes -> waveform, check it's non-silent audio.
    rms_vals = []
    for b, seq in enumerate(seqs):
        wf = gen.to_waveforms(seq, trim=False)[0]
        rms_vals.append(_rms(wf))
        logger.info(f"  stream {b}: {int(seq.shape[1]-1)} rows, rms={rms_vals[-1]:.4f}")

    logger.info(f"BATCH B={B}: rows/user={t['rows_per_user']} steps={t['steps_run']}")
    logger.info(f"  single-stream ref: {single_stream_tps:.1f} tok/s")
    logger.info(f"  batched per-stream: {per_stream_tps:.1f} tok/s  per-stream RTF {per_stream_rtf:.3f}")
    logger.info(f"  device aggregate: {device_tps:.1f} tok/s  scaling {scaling:.2f}x vs single-stream")
    print(
        f"PERF_BATCH B={B} single_tps={single_stream_tps:.1f} per_stream_tps={per_stream_tps:.1f} "
        f"device_tps={device_tps:.1f} scaling={scaling:.2f} per_stream_rtf={per_stream_rtf:.4f}"
    )

    min_rms = float(os.environ.get("HIGGS_BATCH_MIN_RMS", "0.005"))
    min_scaling = float(os.environ.get("HIGGS_BATCH_MIN_SCALING", "1.5"))
    for b, r in enumerate(rms_vals):
        assert r > min_rms, f"stream {b} is silent (rms {r:.4f} <= {min_rms})"
    assert all(n > 8 for n in t["rows_per_user"]), f"a stream generated too few rows: {t['rows_per_user']}"
    assert (
        scaling >= min_scaling
    ), f"batch-{B} aggregate scaling {scaling:.2f}x < {min_scaling}x (batching not amortizing)"
