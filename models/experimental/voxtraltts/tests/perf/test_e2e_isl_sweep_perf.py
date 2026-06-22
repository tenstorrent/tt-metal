# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-pipeline ISL sweep: text prefill + traced AR decode + waveform (Voxtral TTS).

Sweeps voice-wrapped **prompt sequence length** and includes the demo-standard
``VOXTRAL_STANDARD_CHAR_TEXT`` (~500 chars) baseline for apples-to-apples RTF comparison.

Metrics (demo-aligned — same as ``demo.py`` ``_log_perf``):

  - **Latency** — end-to-end generation wall time
  - **TTFA** — time to first acoustic frame (``first_frame_s``)
  - **RTF** — ``latency / audio_duration`` (lower is better; <1 is faster than realtime)
  - **Throughput** — input characters / latency (char/s)
  - **rt_x** — ``audio_duration / latency`` (realtime multiplier shown in demo RTF line)

Unlike the earlier fixed ``max_tokens=128`` budget, each case uses the demo
``_resolve_max_speech_tokens`` path so generation runs until END_AUDIO (or the demo
token ceiling), which is required for meaningful RTF on the 500-char standard prompt.

A summary table is printed at the end of the single test run.

Run::

    pytest models/experimental/voxtraltts/tests/perf/test_e2e_isl_sweep_perf.py -q -s
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.voxtraltts.demo.decode_trace_2cq import (  # noqa: E402
    configure_decode_trace,
    decode_trace_2cq_enabled,
    num_command_queues_for_decode,
)
from models.experimental.voxtraltts.demo.demo import (  # noqa: E402
    _resolve_max_speech_tokens,
)

configure_decode_trace(decode_trace=True, decode_trace_2cq=True)
from models.experimental.voxtraltts.reference.voxtral_config import (  # noqa: E402
    DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
)
from models.experimental.voxtraltts.utils.test_common import (  # noqa: E402
    VOXTRAL_STANDARD_CHAR_TEXT,
    resolve_voxtral_model_name_or_skip,
    speech_prompt_seq_len,
    tale_speech_text_for_min_prompt_len,
)
from models.experimental.voxtraltts.tests.perf.isl_sweep_reporting import log_e2e_isl_sweep_summary  # noqa: E402
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline  # noqa: E402
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler  # noqa: E402
from models.perf.perf_utils import prep_perf_report  # noqa: E402

_VOICE = "casual_male"
_DEMO_MAX_SPEECH_TOKENS = 5000  # demo default --max-speech-tokens
_WARMUP_DECODE = 8
_MODEL_NAME = "voxtral_tts_4B"


@dataclass(frozen=True)
class _E2EIslCase:
    label: str
    target_prompt_len: int | None = None  # None → use ``standard_text``


# Standard 500-char baseline first, then tale ISL targets (voice floor ≈155; skip redundant 128).
_TALE_ISL_TARGETS = (256, 512, 1024, 4096, 8192)
_E2E_ISL_CASES = (
    _E2EIslCase("standard_500c"),
    *(_E2EIslCase(f"tale_{n}", target_prompt_len=n) for n in _TALE_ISL_TARGETS),
)


def _e2e_perf_device_params():
    return {
        "trace_region_size": int(os.environ.get("VOXTRAL_TRACE_REGION_SIZE", str(200_000_000))),
        "num_command_queues": num_command_queues_for_decode(),
    }


def _sample_rate_hz(pipe: VoxtralTTSPipeline) -> int:
    return int(pipe.config.audio_model_args.audio_encoding_args.sampling_rate)


def _audio_duration_s(waveform: torch.Tensor, sample_rate_hz: int) -> float:
    n_samples = int(waveform.numel())
    if sample_rate_hz <= 0 or n_samples <= 0:
        return 0.0
    return n_samples / sample_rate_hz


def _resolve_case_text(case: _E2EIslCase, model_name: str) -> tuple[str, int]:
    if case.target_prompt_len is None:
        text = VOXTRAL_STANDARD_CHAR_TEXT
        prompt_len = speech_prompt_seq_len(text, model_name=model_name, voice=_VOICE)
        return text, prompt_len
    return tale_speech_text_for_min_prompt_len(
        case.target_prompt_len,
        model_name=model_name,
        voice=_VOICE,
    )


def _demo_max_tokens(text: str, model_name: str, text_max_seq_len: int) -> int:
    return _resolve_max_speech_tokens(
        text,
        _VOICE,
        model_name,
        _DEMO_MAX_SPEECH_TOKENS,
        text_max_seq_len,
        log_prefix="e2e_isl_sweep",
    )


@torch.no_grad()
@pytest.mark.timeout(14400)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("expected_compile_time, expected_inference_time", [(180.0, 5.0)])
@pytest.mark.parametrize("device_params", [_e2e_perf_device_params()], indirect=True)
def test_voxtral_tts_e2e_isl_sweep_perf(
    device,
    reset_seeds,
    expected_compile_time,
    expected_inference_time,
):
    model_name = resolve_voxtral_model_name_or_skip()
    text_max_seq_len = DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
    cq_note = "trace+2CQ" if decode_trace_2cq_enabled() else "trace+1CQ"

    cases = list(_E2E_ISL_CASES)
    if os.environ.get("CI") == "true":
        cases = [c for c in cases if c.target_prompt_len is None or c.target_prompt_len <= 1024]

    t_build = time.time()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=model_name,
            text_max_seq_len=text_max_seq_len,
            use_paged_kv_cache=True,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    build_time = time.time() - t_build
    sample_rate_hz = _sample_rate_hz(pipe)
    summary_rows: list[dict] = []

    try:
        warmup_max = _demo_max_tokens(VOXTRAL_STANDARD_CHAR_TEXT, model_name, text_max_seq_len)
        t_warmup = time.time()
        _ = pipe.forward_device_resident(
            text=VOXTRAL_STANDARD_CHAR_TEXT,
            voice=_VOICE,
            max_tokens=min(_WARMUP_DECODE, warmup_max),
            seed=0,
        )
        ttnn.synchronize_device(device)
        warmup_time = time.time() - t_warmup
        logger.info(f"Warm-up on standard 500c text (compile + trace): {warmup_time:.1f}s")

        for case in cases:
            text, prompt_seq_len = _resolve_case_text(case, model_name)
            n_chars = len(text)
            max_tokens = _demo_max_tokens(text, model_name, text_max_seq_len)
            logger.info(
                f"e2e ISL case {case.label}: prompt_seq_len={prompt_seq_len}, chars={n_chars}, "
                f"max_tokens={max_tokens}, {cq_note}"
            )

            t_gen = time.time()
            out = pipe.forward_device_resident(
                text=text,
                voice=_VOICE,
                max_tokens=max_tokens,
                seed=0,
            )
            ttnn.synchronize_device(device)
            gen_time = time.time() - t_gen

            assert torch.isfinite(out.waveform).all(), f"{case.label}: non-finite waveform"

            audio_s = _audio_duration_s(out.waveform, sample_rate_hz)
            ttfa_s = float(out.first_frame_s) if out.first_frame_s is not None else 0.0
            rtf = gen_time / audio_s if audio_s > 0 else 0.0
            chars_per_s = n_chars / gen_time if gen_time > 0 else 0.0
            n_frames = int(out.codes_b37t.shape[2]) if out.codes_b37t.numel() > 0 else 0
            rt_x = audio_s / gen_time if gen_time > 0 else 0.0

            row = {
                "label": case.label,
                "prompt_seq_len": prompt_seq_len,
                "target_prompt_len": case.target_prompt_len or prompt_seq_len,
                "n_chars": n_chars,
                "latency_s": gen_time,
                "latency_ms": gen_time * 1000.0,
                "ttfa_s": ttfa_s,
                "ttfa_ms": ttfa_s * 1000.0,
                "audio_duration_s": audio_s,
                "rtf": rtf,
                "chars_per_s": chars_per_s,
                "realtime_x": rt_x,
                "n_frames": n_frames,
                "hit_end_audio": bool(out.hit_end_audio),
                "max_tokens": max_tokens,
            }
            summary_rows.append(row)

            logger.info(
                f"{case.label}: seq={prompt_seq_len} lat={gen_time*1000:.1f}ms "
                f"TTFA={ttfa_s*1000:.1f}ms audio={audio_s:.2f}s RTF={rtf:.3f} "
                f"char/s={chars_per_s:.1f} rt_x={rt_x:.2f}x END_AUDIO={out.hit_end_audio}"
            )

            prep_perf_report(
                model_name=_MODEL_NAME,
                batch_size=1,
                inference_and_compile_time=build_time + warmup_time + gen_time,
                inference_time=gen_time,
                expected_compile_time=expected_compile_time,
                expected_inference_time=expected_inference_time,
                comments=f"{case.label}_isl{prompt_seq_len}_{cq_note}",
            )

            profiler = BenchmarkProfiler()
            profiler.start("run")
            profiler.end("run")
            step = "voxtral_tts_e2e_isl"
            profiler.start(step)
            profiler.end(step)
            benchmark_data = BenchmarkData()
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    benchmark_data.add_measurement(profiler, 0, step, key, float(value))
            benchmark_data.save_partial_run_json(
                profiler,
                run_type="e2e_isl_sweep",
                ml_model_name=_MODEL_NAME,
                batch_size=1,
                input_sequence_length=int(prompt_seq_len),
                output_sequence_length=int(n_frames),
                config_params={
                    "label": case.label,
                    "voice": _VOICE,
                    "max_tokens": max_tokens,
                    "cq_note": cq_note,
                },
            )

        log_e2e_isl_sweep_summary(summary_rows)
    finally:
        pipe.cleanup_all()
        del pipe
        gc.collect()
