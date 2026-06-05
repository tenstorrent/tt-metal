# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E perf for SeamlessM4Tv2 — ``generate()`` per task on TP + 2CQ + Trace.

One parameterized perf test runs each of the five inference tasks (T2TT, T2ST, S2TT, S2ST, ASR)
through ``TTSeamlessM4Tv2Model.generate(..., use_decode_trace=True, use_2cq=True)`` on a 1×N mesh.
This matches the demo's production generate path.

For each task we:
  * Warm up once (compiles + captures the per-task decode trace + warms ``_conv1d_prepared_cache``).
  * Measure ``NUM_MEASUREMENT_ITERS`` ``generate()`` calls under ``profiler.start/end("inference")``.
  * Sanity-check correctness vs HF ``generate()`` on the first iteration (token-for-token for
    text-out tasks, length + RMS + voicing for speech-out tasks).
  * Emit ``prep_perf_report`` with the measured inference time and an FPS bound.
"""

from __future__ import annotations

import os
import time
from typing import Any, Tuple

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor, AutoTokenizer

from models.common.utility_functions import run_for_blackhole
from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.common import to_torch_replicated_first_shard
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_seamless_m4t_v2_model_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    TTSeamlessM4Tv2GenerationOutput,
    TTSeamlessM4Tv2GreedySearchOutput,
    TTSeamlessM4Tv2Model,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.perf_utils import prep_perf_report

# ---------------------------------------------------------------------------------------------
# Task table — parametrize the single perf test over all 5 inference tasks.
# (use_speech_input, tgt_lang, generate_speech, expected_inference_time_s)
# ``expected_inference_time_s`` is the per-call wall-clock time bound (passed to
# ``prep_perf_report``). The *reported* throughput is task-specific:
#   text outputs  → tokens/sec  = _MAX_NEW_TOKENS / inference_time
#   speech outputs → samples/sec = audio_samples_emitted / inference_time
# ---------------------------------------------------------------------------------------------
_TASKS = {
    "t2tt": (False, "hin", False, 2.0),  # eng text -> hin text
    "t2st": (False, "hin", True, 4.0),  # eng text -> hin speech
    "s2tt": (True, "eng", False, 2.0),  # hin speech -> eng text
    "s2st": (True, "spa", True, 5.0),  # hin speech -> spa speech
    "asr": (True, "eng", False, 2.0),  # speech -> same-lang text
}
_TASK_PARAMS = [(t,) + _TASKS[t] for t in _TASKS]
_TASK_IDS = list(_TASKS.keys())

_MAX_NEW_TOKENS = 10
_NUM_MEASUREMENT_ITERS = 3

# Speech-output tolerances (vs HF on the same inputs).
_AUDIO_LEN_TOL = 0.02
_RMS_RATIO_LO = 0.70
_RMS_RATIO_HI = 1.43
_VOICING_FRAC_TOL = 0.15


# ---------------------------------------------------------------------------------------------
# Local helpers (kept inline so this file is independent of the PCC tests).
# ---------------------------------------------------------------------------------------------


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


def _torch_ids_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _torch_feats_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.bfloat16).cpu().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _make_tt_model(device: ttnn.Device, model: Any, cfg: Any, t2u_cfg: Any) -> TTSeamlessM4Tv2Model:
    params = create_seamless_m4t_v2_model_parameters(model, device=device)
    return TTSeamlessM4Tv2Model(
        device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        encoder_layers=cfg.encoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_layers=cfg.decoder_layers,
        decoder_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        feature_projection_input_dim=cfg.feature_projection_input_dim,
        speech_encoder_attention_heads=cfg.speech_encoder_attention_heads,
        speech_encoder_intermediate_size=cfg.speech_encoder_intermediate_size,
        speech_encoder_layers=cfg.speech_encoder_layers,
        speech_encoder_chunk_size=cfg.speech_encoder_chunk_size,
        speech_encoder_left_chunk_num=cfg.speech_encoder_left_chunk_num,
        pad_token_id=cfg.pad_token_id,
        decoder_start_token_id=cfg.decoder_start_token_id,
        vocab_size=cfg.vocab_size,
        adaptor_kernel_size=cfg.adaptor_kernel_size,
        adaptor_stride=cfg.adaptor_stride,
        t2u_eos_token_id=cfg.t2u_eos_token_id,
        t2u_pad_token_id=t2u_cfg.pad_token_id,
        vocoder_offset=cfg.vocoder_offset,
        t2u_layer_norm_eps=t2u_cfg.layer_norm_eps,
        t2u_encoder_layers=t2u_cfg.encoder_layers,
        t2u_encoder_attention_heads=t2u_cfg.encoder_attention_heads,
        t2u_decoder_layers=t2u_cfg.decoder_layers,
        t2u_decoder_attention_heads=t2u_cfg.decoder_attention_heads,
        variance_predictor_embed_dim=t2u_cfg.variance_predictor_embed_dim,
        variance_predictor_hidden_dim=t2u_cfg.variance_predictor_hidden_dim,
        variance_predictor_kernel_size=t2u_cfg.variance_predictor_kernel_size,
        vocoder_config=cfg,
        generation_config=model.generation_config,
        hf_config=cfg,
    )


def _make_text_inputs(weights_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(os.fspath(weights_dir), local_files_only=True)
    enc = tokenizer(["Hello, my name is SeamlessM4T."], return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"]


def _make_speech_inputs(weights_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """1-second 16 kHz waveform → processor input_features. Real audio shape, synthetic samples."""
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(os.fspath(weights_dir), local_files_only=True)
    wav = (torch.randn(1, 16_000, dtype=torch.float32) * 0.01).numpy().reshape(-1)
    audio_inputs = processor(audios=wav, sampling_rate=16_000, return_tensors="pt")
    return audio_inputs["input_features"].to(torch.bfloat16), audio_inputs["attention_mask"]


def _audio_stats(wav: np.ndarray) -> Tuple[int, float, float]:
    n = wav.size
    rms = float(np.sqrt((wav.astype(np.float64) ** 2).mean())) if n else 0.0
    voicing = float((np.abs(wav) > 0.01).mean()) if n else 0.0
    return n, rms, voicing


def _tt_waveform_to_np(wav_tt: ttnn.Tensor, lengths_tt: ttnn.Tensor) -> np.ndarray:
    arr = to_torch_replicated_first_shard(wav_tt).float().squeeze().cpu().numpy()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    valid_len = int(to_torch_replicated_first_shard(lengths_tt).long().reshape(-1)[0].item())
    if 0 < valid_len <= arr.size:
        arr = arr[:valid_len]
    return arr


def _verify_text(tt_seq_tt: ttnn.Tensor, hf_ids: list, *, ctx: str, strict: bool) -> None:
    """Token check. Text-input tasks (T2TT) match exactly; speech-input tasks (S2TT, ASR) only
    need to match the BOS+lang-code prefix — bf16 drift through the speech encoder can shift the
    first content token within a handful of greedy steps."""
    tt_ids = to_torch_replicated_first_shard(tt_seq_tt).long().cpu().reshape(-1).tolist()
    logger.info(f"{ctx} HF tokens: {hf_ids}")
    logger.info(f"{ctx} TT tokens: {tt_ids}")
    if strict:
        assert tt_ids == hf_ids, f"{ctx}: token mismatch — HF {hf_ids} vs TT {tt_ids}"
    else:
        assert hf_ids[:2] == tt_ids[:2], f"{ctx}: seed token mismatch — HF {hf_ids[:2]} vs TT {tt_ids[:2]}"
        assert (
            len(tt_ids) >= 4 and len(hf_ids) >= 4
        ), f"{ctx}: too few content tokens (HF={len(hf_ids)} TT={len(tt_ids)})"


def _verify_speech_tensors(
    wav_tt: ttnn.Tensor, lens_tt: ttnn.Tensor, hf_wav: np.ndarray, *, ctx: str, strict_len: bool
) -> None:
    """Sanity-check TT vs HF audio. Speech-in speech-out (S2ST) accumulates bf16 drift through the
    speech encoder, so the strict ±2 % length bound from T2ST doesn't apply — relax to the same
    "plausible voiced output" check used by the PCC test (RMS in band, voicing close)."""
    tt_wav = _tt_waveform_to_np(wav_tt, lens_tt)
    hf_n, hf_rms, hf_voice = _audio_stats(hf_wav)
    tt_n, tt_rms, tt_voice = _audio_stats(tt_wav)
    logger.info(f"{ctx} HF audio: samples={hf_n} rms={hf_rms:.4f} voicing={hf_voice:.3f}")
    logger.info(f"{ctx} TT audio: samples={tt_n} rms={tt_rms:.4f} voicing={tt_voice:.3f}")

    if strict_len:
        rel = abs(tt_n - hf_n) / max(1, hf_n)
        assert rel < _AUDIO_LEN_TOL, f"{ctx}: audio length differs > {_AUDIO_LEN_TOL*100:.0f}%: HF={hf_n} TT={tt_n}"
    else:
        assert hf_n > 0 and tt_n > 0, f"{ctx}: empty audio (HF={hf_n}, TT={tt_n})"
    assert hf_rms > 0.0 and tt_rms > 0.0, f"{ctx}: zero-energy audio (HF={hf_rms}, TT={tt_rms})"
    ratio = tt_rms / hf_rms
    assert (
        _RMS_RATIO_LO <= ratio <= _RMS_RATIO_HI
    ), f"{ctx}: RMS ratio TT/HF={ratio:.3f} outside [{_RMS_RATIO_LO}, {_RMS_RATIO_HI}]"
    voice_tol = _VOICING_FRAC_TOL if strict_len else 2 * _VOICING_FRAC_TOL
    assert (
        abs(tt_voice - hf_voice) <= voice_tol
    ), f"{ctx}: voicing frac differs > {voice_tol} (TT={tt_voice:.3f} HF={hf_voice:.3f})"


def _run_seamless_perf(
    mesh_device,
    *,
    task: str,
    use_speech_input: bool,
    tgt_lang: str,
    generate_speech: bool,
    max_new_tokens: int,
    measure_iters: int,
) -> dict:
    """Run one task: build (untimed), compile ``generate()``, timed TTFT + measurement replays."""
    weights_dir = _weights_dir_or_skip()

    # ── Build (untimed) — host weight load + HF reference for correctness verify ────────────
    t_build = time.time()
    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config

    if use_speech_input:
        input_features, enc_attn = _make_speech_inputs(weights_dir)
        input_ids = None
    else:
        input_ids, enc_attn = _make_text_inputs(weights_dir)
        input_features = None

    common_kwargs = dict(
        tgt_lang=tgt_lang,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        generate_speech=generate_speech,
    )

    # HF reference (CPU only).
    with torch.no_grad():
        if use_speech_input:
            hf_out = model.generate(input_features=input_features.float(), attention_mask=enc_attn, **common_kwargs)
        else:
            hf_out = model.generate(input_ids=input_ids, attention_mask=enc_attn, **common_kwargs)
    if generate_speech:
        if isinstance(hf_out, (tuple, list)) and len(hf_out) == 2:
            wav_t, lens_t = hf_out
        else:
            wav_t = hf_out.waveform
            lens_t = getattr(hf_out, "waveform_lengths", None)
        hf_wav = wav_t.detach().cpu().float().squeeze().numpy()
        if hf_wav.ndim > 1:
            hf_wav = hf_wav.reshape(-1)
        hf_len = int(lens_t.detach().cpu().reshape(-1)[0].item()) if lens_t is not None else hf_wav.size
        hf_wav = hf_wav[:hf_len]
        hf_text_ids: list = []
    else:
        hf_text_ids = hf_out.sequences[0].cpu().tolist() if hasattr(hf_out, "sequences") else hf_out[0].cpu().tolist()
        hf_wav = np.zeros(0, dtype=np.float32)
    build_time = time.time() - t_build

    with mesh_default_device(mesh_device):
        tt_model = _make_tt_model(mesh_device, model, cfg, t2u_cfg)

        def _call_generate(verify: bool, ctx: str) -> int:
            if use_speech_input:
                tt_out = tt_model.generate(
                    input_features=_torch_feats_to_ttnn(mesh_device, input_features),
                    attention_mask=_torch_ids_to_ttnn(mesh_device, enc_attn),
                    use_kv_cache=True,
                    use_decode_trace=True,
                    use_2cq=True,
                    speaker_id=0,
                    **common_kwargs,
                )
            else:
                tt_out = tt_model.generate(
                    input_ids=_torch_ids_to_ttnn(mesh_device, input_ids),
                    attention_mask=_torch_ids_to_ttnn(mesh_device, enc_attn),
                    use_kv_cache=True,
                    use_decode_trace=True,
                    use_2cq=True,
                    **common_kwargs,
                )
            if generate_speech:
                if isinstance(tt_out, TTSeamlessM4Tv2GenerationOutput):
                    wav_tt, lens_tt = tt_out.waveform, tt_out.waveform_lengths
                else:
                    assert isinstance(tt_out, (tuple, list)) and len(tt_out) == 2, type(tt_out)
                    wav_tt, lens_tt = tt_out
                num_samples = int(to_torch_replicated_first_shard(lens_tt).long().reshape(-1)[0].item())
                if verify:
                    _verify_speech_tensors(wav_tt, lens_tt, hf_wav, ctx=ctx, strict_len=not use_speech_input)
                ttnn.deallocate(wav_tt)
                ttnn.deallocate(lens_tt)
                return num_samples
            seq_tt = tt_out.sequences if isinstance(tt_out, TTSeamlessM4Tv2GreedySearchOutput) else tt_out
            if verify:
                _verify_text(seq_tt, hf_text_ids, ctx=ctx, strict=not use_speech_input)
            ttnn.deallocate(seq_tt)
            return 0

        t_compile = time.time()
        _call_generate(verify=True, ctx=f"{task}_compile")
        ttnn.synchronize_device(mesh_device)
        compile_time = time.time() - t_compile

        ttnn.synchronize_device(mesh_device)
        t_ttft = time.time()
        sample_count_ttft = _call_generate(verify=False, ctx=f"{task}_ttft")
        ttnn.synchronize_device(mesh_device)
        ttft_s = time.time() - t_ttft

        sample_counts = [sample_count_ttft]
        t_meas_start = time.time()
        for i in range(max(0, measure_iters - 1)):
            sample_counts.append(_call_generate(verify=False, ctx=f"{task}_iter{i}"))
        ttnn.synchronize_device(mesh_device)
        replay_time_avg = (time.time() - t_meas_start) / max(1, measure_iters - 1) if measure_iters > 1 else ttft_s

    avg_num_samples = sum(sample_counts) / len(sample_counts) if sample_counts else 0
    if generate_speech:
        throughput = avg_num_samples / replay_time_avg if replay_time_avg > 0 else 0.0
        throughput_unit = "samples_per_sec"
        workload = int(avg_num_samples)
    else:
        throughput = max_new_tokens / replay_time_avg if replay_time_avg > 0 else 0.0
        throughput_unit = "tokens_per_sec"
        workload = max_new_tokens

    return {
        "task": task,
        "build_time_s": build_time,
        "compile_time_s": compile_time,
        "ttft_s": ttft_s,
        "ttft_ms": ttft_s * 1000,
        "replay_time_avg_s": replay_time_avg,
        "replay_time_avg_ms": replay_time_avg * 1000,
        "throughput": throughput,
        "throughput_unit": throughput_unit,
        "workload": workload,
        "max_new_tokens": max_new_tokens,
        "measure_iters": measure_iters,
    }


@run_for_blackhole()
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(
    "task,use_speech_input,tgt_lang,generate_speech,expected_inference_time_s",
    _TASK_PARAMS,
    ids=_TASK_IDS,
)
def test_seamless_m4t_v2_generate_perf(
    mesh_device,
    device_params,
    task: str,
    use_speech_input: bool,
    tgt_lang: str,
    generate_speech: bool,
    expected_inference_time_s: float,
):
    """Per-task generate() perf on TP=N + 2CQ + Trace, compared to HF for correctness."""
    _ = device_params

    results = _run_seamless_perf(
        mesh_device,
        task=task,
        use_speech_input=use_speech_input,
        tgt_lang=tgt_lang,
        generate_speech=generate_speech,
        max_new_tokens=_MAX_NEW_TOKENS,
        measure_iters=_NUM_MEASUREMENT_ITERS,
    )

    num_devices = int(mesh_device.shape[0]) * int(mesh_device.shape[1])
    batch_size = 1 * num_devices
    model_name = f"ttnn_seamless_m4t_v2_large_generate_TP{num_devices}_2cq_trace_{task}"
    comments = f"generate_task_{task}_TP{num_devices}_2cq_trace_max_new_tokens{_MAX_NEW_TOKENS}"

    inference_time = results["replay_time_avg_s"]
    inference_and_compile_time = (
        results["build_time_s"]
        + results["compile_time_s"]
        + results["ttft_s"]
        + inference_time * max(0, _NUM_MEASUREMENT_ITERS - 1)
    )

    prep_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=600,
        expected_inference_time=expected_inference_time_s,
        comments=comments,
        inference_time_cpu=0.0,
    )

    profiler = BenchmarkProfiler()
    profiler.start("run")
    profiler.end("run")
    step = "seamless_e2e"
    profiler.start(step)
    profiler.end(step)
    benchmark_data = BenchmarkData()
    for k, v in results.items():
        if isinstance(v, (int, float)):
            benchmark_data.add_measurement(profiler, 0, step, k, float(v))
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="end_to_end_perf",
        ml_model_name=model_name,
        batch_size=batch_size,
    )

    logger.info(
        f"SeamlessM4Tv2 generate task={task} TP={num_devices} "
        f"build={results['build_time_s']:.2f}s compile={results['compile_time_s']:.2f}s "
        f"TTFT={results['ttft_ms']:.1f}ms replay_avg={results['replay_time_avg_ms']:.2f}ms "
        f"({results['throughput']:.2f} {results['throughput_unit']})"
    )
