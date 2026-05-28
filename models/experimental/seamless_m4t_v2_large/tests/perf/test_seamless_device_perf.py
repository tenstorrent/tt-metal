# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Seamless M4T v2 Large — device performance test (per-task tokens/sec, wall-clock with trace).

Mirrors the production ``generate()`` path on a 1×N mesh (TP + 2 CQ + decode-Trace) for each of
the five inference tasks (T2TT, T2ST, S2TT, S2ST, ASR). For each task we:

  1. Build the TT model.
  2. Warmup one ``generate()`` call (compile + capture decode trace + warm prepared caches).
  3. Time a single steady-state ``generate()`` call (the trace is already captured).
  4. Report ``tokens/sec = max_new_tokens / inference_time``.

No tracy / no device profiler — pure wall-clock around ``ttnn.execute_trace`` is the canonical
device-perf measurement once a trace is captured (host overhead is minimised, so wall-clock
tracks the device kernel critical path). This matches the devstral2 ``test_perf.py`` pattern.

Usage::

    pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \\
        -v -m models_device_performance_bare_metal
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import pytest
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor, AutoTokenizer

from models.common.utility_functions import profiler, run_for_blackhole
from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
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
from models.perf.device_perf_utils import prep_device_perf_report

# Task table — parametrize over all 5 inference tasks.
# (use_speech_input, tgt_lang, generate_speech)
_TASKS = {
    "t2tt": (False, "hin", False),  # eng text -> hin text
    "t2st": (False, "hin", True),  # eng text -> hin speech
    "s2tt": (True, "eng", False),  # hin speech -> eng text
    "s2st": (True, "spa", True),  # hin speech -> spa speech
    "asr": (True, "eng", False),  # speech -> same-lang text (rep-penalty=1.0)
}
_TASK_PARAMS = [(t,) + _TASKS[t] for t in _TASKS]
_TASK_IDS = list(_TASKS.keys())

_PROMPT = "Hello, my name is SeamlessM4T."
_MAX_NEW_TOKENS = 10


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
        raise
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")
        raise


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
    enc = tokenizer([_PROMPT], return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"]


def _make_speech_inputs(weights_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """1-second 16 kHz waveform → processor input_features. Real audio shape, synthetic samples."""
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(os.fspath(weights_dir), local_files_only=True)
    wav = (torch.randn(1, 16_000, dtype=torch.float32) * 0.01).numpy().reshape(-1)
    audio_inputs = processor(audios=wav, sampling_rate=16_000, return_tensors="pt")
    return audio_inputs["input_features"].to(torch.bfloat16), audio_inputs["attention_mask"]


def _release_tt_out(tt_out: Any, generate_speech: bool) -> None:
    if generate_speech:
        if isinstance(tt_out, TTSeamlessM4Tv2GenerationOutput):
            ttnn.deallocate(tt_out.waveform)
            ttnn.deallocate(tt_out.waveform_lengths)
        else:
            wav_tt, lens_tt = tt_out
            ttnn.deallocate(wav_tt)
            ttnn.deallocate(lens_tt)
    else:
        seq = tt_out.sequences if isinstance(tt_out, TTSeamlessM4Tv2GreedySearchOutput) else tt_out
        ttnn.deallocate(seq)


@run_for_blackhole()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("task,use_speech_input,tgt_lang,generate_speech", _TASK_PARAMS, ids=_TASK_IDS)
def test_perf_device_bare_metal_seamless(
    mesh_device,
    device_params,
    task: str,
    use_speech_input: bool,
    tgt_lang: str,
    generate_speech: bool,
):
    """Wall-clock per-task device perf on TP=N + 2CQ + Trace.

    Single warmup ``generate()`` call (compiles + captures decode trace + warms prepared caches),
    then a single timed steady-state replay. Reports tokens/sec.
    """
    _ = device_params
    weights_dir = _weights_dir_or_skip()

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
        max_new_tokens=_MAX_NEW_TOKENS,
        generate_speech=generate_speech,
    )
    # ASR (same-language transcription): disable rep-penalty so the decoder doesn't get pushed
    # off the target language token (matches the demo).
    if task == "asr":
        common_kwargs["repetition_penalty"] = 1.0

    with mesh_default_device(mesh_device):
        tt_model = _make_tt_model(mesh_device, model, cfg, t2u_cfg)

        def _call_generate() -> None:
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
            _release_tt_out(tt_out, generate_speech=generate_speech)

        # Warmup (compiles + captures decode trace + warms prepared caches).
        profiler.clear()
        profiler.start("warmup")
        _call_generate()
        profiler.end("warmup")

        # Single steady-state measurement (the trace is already captured).
        profiler.start("inference")
        _call_generate()
        profiler.end("inference")

    num_devices = int(mesh_device.shape[0]) * int(mesh_device.shape[1])
    batch_size = 1
    warmup_time = profiler.get("warmup")
    inference_time = profiler.get("inference")
    tokens_per_sec = _MAX_NEW_TOKENS / inference_time

    post_processed_results = {
        "INFERENCE TIME [s]": inference_time,
        "WARMUP TIME [s]": warmup_time,
        "AVG TOKENS/S": tokens_per_sec,
        "MAX_NEW_TOKENS": _MAX_NEW_TOKENS,
    }
    print(f"\n{'='*60}")
    print(f"Seamless M4T v2 Large Device Performance ({task.upper()})")
    print(f"{'='*60}")
    print(
        f"  Measured: {tokens_per_sec:.2f} tokens/s  "
        f"({inference_time * 1000:.1f} ms wall-clock, {_MAX_NEW_TOKENS} new tokens, TP={num_devices})"
    )
    print(f"{'='*60}\n")
    logger.info(
        f"SeamlessM4Tv2 device-perf task={task} TP={num_devices} "
        f"warmup={warmup_time:.2f}s inference={inference_time:.4f}s "
        f"tokens_per_sec={tokens_per_sec:.2f}"
    )

    prep_device_perf_report(
        model_name=f"ttnn_seamless_m4t_v2_large_batch{batch_size}_{task}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results={},
        comments=f"seamless_m4t_v2_large_{task}_TP{num_devices}_2cq_trace_max_new_tokens{_MAX_NEW_TOKENS}",
    )
