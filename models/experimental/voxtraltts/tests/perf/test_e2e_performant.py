# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end wall-clock performance test for Voxtral TTS on TTNN (Blackhole / P150).

Imports the trace/2CQ helpers from ``demo/decode_trace_2cq``, opens the device via ``device_params``
with ``num_command_queues_for_decode()`` + a trace region, and reports through ``prep_perf_report`` +
``BenchmarkData``, adapted to Voxtral's TTS AR loop.

Unlike a plain LLM, Voxtral decode is a discrete-feedback loop (text-decode -> acoustic FM ->
code -> embedding -> text-decode). The trace + 2CQ live inside
``VoxtralTTSPipeline.forward_device_resident`` (text-decode trace + acoustic-FM trace, CQ1 input
staging); this test drives one full traced generation and reports the steady-state per-frame decode
time, RTF, and throughput.

  1. Build the TT pipeline (untimed).
  2. Warm-up generation (compile + trace capture; untimed).
  3. Timed generation: full traced AR loop -> per-frame decode time + RTF + frames/s.

Decode trace + 2CQ (CQ1 input H2D, CQ0 trace replay) is on by default
(``VOXTRAL_DECODE_TRACE=1`` / ``VOXTRAL_DECODE_TRACE_2CQ=1``; set ``0`` for single-CQ / no-trace
baselines).

Run::

    pytest models/experimental/voxtraltts/tests/perf/test_e2e_performant.py -q -s
"""

from __future__ import annotations

import gc
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

# Enable traced decode for the perf run BEFORE device_params is evaluated (collection time), so the
# device opens with a trace region + 2 command queues. Override with VOXTRAL_DECODE_TRACE=0.
os.environ.setdefault("VOXTRAL_DECODE_TRACE", "1")

# The trace + 2CQ staging run *inside* VoxtralTTSPipeline.forward_device_resident (the TTS AR loop
# owns the loop), so this perf test only needs the two device-config helpers; the staging/replay
# helpers (DecodeTrace2CQ, stage_decode_inputs, signal_decode_step_done, ...) are exercised through
# forward_device_resident.
from models.experimental.voxtraltts.demo.decode_trace_2cq import (  # noqa: E402
    decode_trace_2cq_enabled,
    num_command_queues_for_decode,
)
from models.experimental.voxtraltts.reference.voxtral_config import (  # noqa: E402
    DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
)
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip  # noqa: E402
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline  # noqa: E402
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler  # noqa: E402
from models.perf.perf_utils import prep_perf_report  # noqa: E402

_PERF_TEXT = "The architecture of Voxtral combines a text backbone with a flow matching acoustic head."
_PERF_VOICE = "casual_male"
_WARMUP_TOKENS = 16  # short warm-up: compile + trace capture only (untimed)


def _e2e_perf_device_params():
    """Open the device with a trace region + N command queues."""
    return {
        "trace_region_size": int(os.environ.get("VOXTRAL_TRACE_REGION_SIZE", str(200_000_000))),
        "num_command_queues": num_command_queues_for_decode(),
    }


def _audio_seconds(pipe: VoxtralTTSPipeline, out) -> float:
    sr = int(pipe.config.audio_model_args.audio_encoding_args.sampling_rate)
    n = int(out.waveform.numel())
    return n / sr if sr > 0 else 0.0


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("decode_iters", [128], ids=["F128"])
@pytest.mark.parametrize("expected_compile_time, expected_inference_time", [(120.0, 0.10)])
@pytest.mark.parametrize("device_params", [_e2e_perf_device_params()], indirect=True)
def test_voxtral_tts_e2e_performant(
    device,
    reset_seeds,
    decode_iters,
    expected_compile_time,
    expected_inference_time,
):
    name = resolve_voxtral_model_name_or_skip()
    cq_note = "trace+2CQ" if decode_trace_2cq_enabled() else "trace+1CQ"
    logger.info(f"Voxtral e2e perf: decode_iters={decode_iters}, {cq_note}, CQs={num_command_queues_for_decode()}")

    t_build = time.time()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=name,
            text_max_seq_len=DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    build_time = time.time() - t_build
    logger.info(f"Pipeline built in {build_time:.1f}s")

    try:
        # 1) Warm-up generation: weight tilize + JIT + trace capture (untimed).
        t = time.time()
        _ = pipe.forward_device_resident(text=_PERF_TEXT, voice=_PERF_VOICE, max_tokens=_WARMUP_TOKENS, seed=0)
        ttnn.synchronize_device(device)
        warmup_time = time.time() - t
        logger.info(f"Warm-up (compile + trace capture): {warmup_time:.1f}s")

        # 2) Timed generation: full traced AR loop (trace replay steady-state).
        t = time.time()
        out = pipe.forward_device_resident(
            text=_PERF_TEXT, voice=_PERF_VOICE, max_tokens=decode_iters, seed=0, fixed_step_count=True
        )
        ttnn.synchronize_device(device)
        gen_time = time.time() - t

        assert torch.isfinite(out.waveform).all(), "TT forward produced non-finite waveform"
        n_frames = int(out.codes_b37t.shape[2])
        audio_s = _audio_seconds(pipe, out)
        per_frame_s = gen_time / max(n_frames, 1)
        rtf = gen_time / audio_s if audio_s > 0 else float("inf")
        frames_per_s = n_frames / gen_time if gen_time > 0 else 0.0

        logger.info(
            f"Voxtral decode ({cq_note}): {n_frames} frames in {gen_time:.2f}s "
            f"({per_frame_s*1000:.1f} ms/frame, {frames_per_s:.2f} frames/s); "
            f"audio {audio_s:.2f}s -> RTF {rtf:.3f}"
        )

        results = {
            "build_time_s": build_time,
            "warmup_time_s": warmup_time,
            "gen_time_s": gen_time,
            "n_frames": float(n_frames),
            "audio_s": audio_s,
            "per_frame_decode_s": per_frame_s,
            "frames_per_s": frames_per_s,
            "rtf": rtf,
            "decode_trace_2cq": float(decode_trace_2cq_enabled()),
            "num_command_queues": float(num_command_queues_for_decode()),
        }

        model_name = "voxtral_tts_4B"
        prep_perf_report(
            model_name=model_name,
            batch_size=1,
            inference_and_compile_time=build_time + warmup_time + gen_time,
            inference_time=per_frame_s,
            expected_compile_time=expected_compile_time,
            expected_inference_time=expected_inference_time,
            comments=f"decode{n_frames}_{cq_note}",
        )

        profiler = BenchmarkProfiler()
        profiler.start("run")
        profiler.end("run")
        step = "voxtral_tts_e2e"
        profiler.start(step)
        profiler.end(step)
        benchmark_data = BenchmarkData()
        for k, v in results.items():
            benchmark_data.add_measurement(profiler, 0, step, k, float(v))
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="end_to_end_perf",
            ml_model_name=model_name,
            batch_size=1,
        )

        logger.info(
            f"{model_name}: per-frame={per_frame_s*1000:.1f}ms, frames/s={frames_per_s:.2f}, "
            f"RTF={rtf:.3f} ({cq_note}) over {n_frames} frames"
        )
    finally:
        pipe.cleanup_all()
        del pipe
        gc.collect()
