# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Forward-only per-task inner tests for tracy-driven device-perf profiling.

The outer driver ``test_seamless_device_perf.py`` spawns one of these under ``python3 -m tracy``
so the per-op kernel durations can be summed into a TP-aware per-device kernel time. To make
that measurement clean:

  * **Eager decode** — ``use_decode_trace=False``, ``use_2cq=False``. Tracy's per-op profiler
    can't reliably reconcile host/device records across metal trace replays (replays reuse the
    captured global-call-counts, host tracer logs new ones), so trace is intentionally disabled
    here even though it's the production path. The outer e2e perf test measures the trace path.
  * **HF correctness skipped** — pure TT-only forward with fixed dummy inputs. The PCC test
    covers correctness.
  * **Audio sample count side-channeled** — for speech-output tasks we write the
    ``waveform_lengths`` value to a file under ``/tmp/`` so the outer driver can read it back
    and compute ``samples/sec``. The outer process can't sync-read from the inner pytest's
    device tensors (different process), so a small text file is the simplest channel.
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import pytest
import torch
import ttnn
from transformers import AutoProcessor, AutoTokenizer

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.common import to_torch_replicated_first_shard
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_FULL,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_seamless_m4t_v2_model_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    TTSeamlessM4Tv2GenerationOutput,
    TTSeamlessM4Tv2GreedySearchOutput,
    TTSeamlessM4Tv2Model,
)

_PROMPT = "Hello, my name is SeamlessM4T."
# Text-output decode budget. Kept the same as e2e so per-token comparisons are meaningful.
_MAX_NEW_TOKENS_TEXT = 10
# Speech-output decode budget is smaller — the prefill text-decoder + T2U + vocoder fan out into
# many ops and the tracy profiler buffer fills if max_new_tokens is too large.
_MAX_NEW_TOKENS_SPEECH = 4

_TGT_HIN = "hin"
_TGT_ENG = "eng"
_TGT_SPA = "spa"

_TEXT_KWARGS = dict(do_sample=False, num_beams=1, max_new_tokens=_MAX_NEW_TOKENS_TEXT)
_SPEECH_KWARGS = dict(do_sample=False, num_beams=1, max_new_tokens=_MAX_NEW_TOKENS_SPEECH)
_ASR_KWARGS = {**_TEXT_KWARGS, "repetition_penalty": 1.0}
# Eager decode: tracy's per-op records don't reconcile across trace replays. KV cache stays on.
_TT_EXTRA = dict(use_kv_cache=True, use_decode_trace=False, use_2cq=False)

# Path where speech-output tasks write the actual generated audio-sample count so the outer
# tracy driver (different process) can pick it up and compute ``samples/sec`` from the
# per-device kernel time.
SAMPLES_PATH_FMT = "/tmp/seamless_dperf_{task}_samples.txt"


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


def _setup(weights_dir: str):
    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    path = os.fspath(weights_dir)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    return model, cfg, t2u_cfg, tokenizer, processor


def _text_inputs(tokenizer: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer([_PROMPT], return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"]


def _synthetic_speech_inputs(processor: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed 1-second 16 kHz waveform → processor input_features. Deterministic seed makes the
    audio identical across inner-test invocations, so the outer driver's per-task numbers don't
    drift between consecutive tracy runs."""
    torch.manual_seed(42)
    wav = (torch.randn(1, 16_000, dtype=torch.float32) * 0.01).numpy().reshape(-1)
    audio = processor(audios=wav, sampling_rate=16_000, return_tensors="pt")
    return audio["input_features"].to(torch.bfloat16), audio["attention_mask"]


def _release_and_record(task: str, tt_out: Any, generate_speech: bool) -> None:
    """Release TT generate output. For speech-output tasks, side-channel the audio-sample count
    to ``SAMPLES_PATH_FMT`` so the outer tracy driver can compute ``samples/sec``."""
    if generate_speech:
        if isinstance(tt_out, TTSeamlessM4Tv2GenerationOutput):
            wav_tt, lens_tt = tt_out.waveform, tt_out.waveform_lengths
        else:
            wav_tt, lens_tt = tt_out
        num_samples = int(to_torch_replicated_first_shard(lens_tt).long().reshape(-1)[0].item())
        with open(SAMPLES_PATH_FMT.format(task=task), "w") as f:
            f.write(str(num_samples))
        ttnn.deallocate(wav_tt)
        ttnn.deallocate(lens_tt)
        return
    seq = tt_out.sequences if isinstance(tt_out, TTSeamlessM4Tv2GreedySearchOutput) else tt_out
    ttnn.deallocate(seq)


# ---------------------------------------------------------------------------------------------
# Per-task forward tests — one TT ``generate()`` call on 1×N mesh, eager (no trace).
# ---------------------------------------------------------------------------------------------


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_FULL, indirect=["mesh_device", "device_params"])
def test_t2tt(mesh_device, device_params):
    _ = device_params
    weights_dir = _weights_dir_or_skip()
    model, cfg, t2u_cfg, tokenizer, _processor = _setup(weights_dir)
    input_ids, attn = _text_inputs(tokenizer)
    with mesh_default_device(mesh_device):
        tt_model = _make_tt_model(mesh_device, model, cfg, t2u_cfg)
        tt_out = tt_model.generate(
            input_ids=_torch_ids_to_ttnn(mesh_device, input_ids),
            attention_mask=_torch_ids_to_ttnn(mesh_device, attn),
            generate_speech=False,
            tgt_lang=_TGT_HIN,
            **_TEXT_KWARGS,
            **_TT_EXTRA,
        )
        _release_and_record("t2tt", tt_out, generate_speech=False)
        ttnn.synchronize_device(mesh_device)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_FULL, indirect=["mesh_device", "device_params"])
def test_t2st(mesh_device, device_params):
    _ = device_params
    weights_dir = _weights_dir_or_skip()
    model, cfg, t2u_cfg, tokenizer, _processor = _setup(weights_dir)
    input_ids, attn = _text_inputs(tokenizer)
    with mesh_default_device(mesh_device):
        tt_model = _make_tt_model(mesh_device, model, cfg, t2u_cfg)
        tt_out = tt_model.generate(
            input_ids=_torch_ids_to_ttnn(mesh_device, input_ids),
            attention_mask=_torch_ids_to_ttnn(mesh_device, attn),
            generate_speech=True,
            tgt_lang=_TGT_HIN,
            speaker_id=0,
            **_SPEECH_KWARGS,
            **_TT_EXTRA,
        )
        _release_and_record("t2st", tt_out, generate_speech=True)
        ttnn.synchronize_device(mesh_device)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_FULL, indirect=["mesh_device", "device_params"])
def test_s2tt(mesh_device, device_params):
    _ = device_params
    weights_dir = _weights_dir_or_skip()
    model, cfg, t2u_cfg, _tokenizer, processor = _setup(weights_dir)
    sp_features, sp_attn = _synthetic_speech_inputs(processor)
    with mesh_default_device(mesh_device):
        tt_model = _make_tt_model(mesh_device, model, cfg, t2u_cfg)
        tt_out = tt_model.generate(
            input_features=_torch_feats_to_ttnn(mesh_device, sp_features),
            attention_mask=_torch_ids_to_ttnn(mesh_device, sp_attn),
            generate_speech=False,
            tgt_lang=_TGT_ENG,
            **_TEXT_KWARGS,
            **_TT_EXTRA,
        )
        _release_and_record("s2tt", tt_out, generate_speech=False)
        ttnn.synchronize_device(mesh_device)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_FULL, indirect=["mesh_device", "device_params"])
def test_s2st(mesh_device, device_params):
    _ = device_params
    weights_dir = _weights_dir_or_skip()
    model, cfg, t2u_cfg, _tokenizer, processor = _setup(weights_dir)
    sp_features, sp_attn = _synthetic_speech_inputs(processor)
    with mesh_default_device(mesh_device):
        tt_model = _make_tt_model(mesh_device, model, cfg, t2u_cfg)
        tt_out = tt_model.generate(
            input_features=_torch_feats_to_ttnn(mesh_device, sp_features),
            attention_mask=_torch_ids_to_ttnn(mesh_device, sp_attn),
            generate_speech=True,
            tgt_lang=_TGT_SPA,
            speaker_id=0,
            **_SPEECH_KWARGS,
            **_TT_EXTRA,
        )
        _release_and_record("s2st", tt_out, generate_speech=True)
        ttnn.synchronize_device(mesh_device)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_FULL, indirect=["mesh_device", "device_params"])
def test_asr(mesh_device, device_params):
    _ = device_params
    weights_dir = _weights_dir_or_skip()
    model, cfg, t2u_cfg, _tokenizer, processor = _setup(weights_dir)
    sp_features, sp_attn = _synthetic_speech_inputs(processor)
    with mesh_default_device(mesh_device):
        tt_model = _make_tt_model(mesh_device, model, cfg, t2u_cfg)
        tt_out = tt_model.generate(
            input_features=_torch_feats_to_ttnn(mesh_device, sp_features),
            attention_mask=_torch_ids_to_ttnn(mesh_device, sp_attn),
            generate_speech=False,
            tgt_lang=_TGT_HIN,
            **_ASR_KWARGS,
            **_TT_EXTRA,
        )
        _release_and_record("asr", tt_out, generate_speech=False)
        ttnn.synchronize_device(mesh_device)
