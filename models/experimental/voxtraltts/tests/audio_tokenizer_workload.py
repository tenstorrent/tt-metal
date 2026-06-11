# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared Voxtral audio tokenizer decode workload for perf and optimization tests.

Single entry point: ``run_voxtral_audio_tokenizer_decode_benchmark`` loads the model once,
warmups at ``VOXTRAL_PERF_WARMUP_T`` (default 64), then times a full decode at
``VOXTRAL_PERF_DECODE_T`` (default 1600) between Tracy ``start``/``stop`` signposts.

Use ``test_audio_tokenizer_opt.py`` for iterative kernel optimization; the device-perf
wrapper in ``tests/perf/`` invokes the same harness.
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import audio_tokenizer_decode_reference
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.tests.common import (
    create_voxtral_audio_tokenizer_or_skip,
    resolve_voxtral_model_name_or_skip,
)
from models.experimental.voxtraltts.tt.audio_tokenizer.model import (
    _DECODE_CHUNK_T,
    extract_audio_tokenizer_state_dict,
)
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict
from models.experimental.voxtraltts.utils.audio_tokenizer_optimizations import (
    AudioTokenizerOptimizations,
    voxtral_audio_tokenizer_default_optimizations,
)

try:
    from tracy import signpost

    _USE_SIGNPOST = True
except ModuleNotFoundError:
    _USE_SIGNPOST = False


def tracy_start() -> None:
    if _USE_SIGNPOST:
        signpost(header="start")


def tracy_stop() -> None:
    if _USE_SIGNPOST:
        signpost(header="stop")


def resolve_perf_warmup_t() -> int:
    """Warmup acoustic frames — small enough to stay within Tracy profiler op budget."""
    return int(os.getenv("VOXTRAL_PERF_WARMUP_T", "64"))


def resolve_perf_decode_t() -> int:
    """Acoustic frame count for the timed pass (default: production chunk ``1600``)."""
    return int(os.getenv("VOXTRAL_PERF_DECODE_T", str(_DECODE_CHUNK_T)))


def codes_to_tt_b37t(device, codes: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        codes.to(torch.uint32).contiguous(),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def make_zero_codes(*, n_codebooks: int, time_frames: int) -> torch.Tensor:
    return torch.zeros(1, n_codebooks, time_frames, dtype=torch.long)


def decode_codes_to_wav(
    tok,
    mesh_device,
    codes_b37t: torch.Tensor,
    *,
    log_stages: bool = False,
) -> torch.Tensor:
    """Run codes → waveform on device; optional per-stage logs for long decodes."""

    def _stage(name: str, fn):
        if log_stages:
            logger.info("voxtral audio tokenizer: {} …", name)
            t0 = time.monotonic()
        result = fn()
        if log_stages:
            ttnn.synchronize_device(mesh_device)
            logger.info(
                "voxtral audio tokenizer: {} done ({:.1f}s)",
                name,
                time.monotonic() - t0,
            )
        return result

    if log_stages:
        logger.info(
            "voxtral audio tokenizer: decode T={} codes={}",
            int(codes_b37t.shape[2]),
            tuple(codes_b37t.shape),
        )

    codes_tt = _stage("codes_to_device", lambda: codes_to_tt_b37t(mesh_device, codes_b37t))
    latent_tt = _stage("latent_from_codes_tt", lambda: tok.latent_from_codes_tt(codes_tt))
    ttnn.deallocate(codes_tt)
    mel_tt = _stage("decode_latent_to_mel_b1tc", lambda: tok.decode_latent_to_mel_b1tc(latent_tt))
    ttnn.deallocate(latent_tt)
    wav_tt = _stage("pretransform_decode_tt", lambda: tok.pretransform_decode_tt(mel_tt))
    ttnn.deallocate(mel_tt)
    wav = _stage("to_torch", lambda: ttnn.to_torch(wav_tt).float())
    ttnn.deallocate(wav_tt)
    ttnn.synchronize_device(mesh_device)
    return wav


@dataclass(frozen=True)
class AudioTokenizerBenchmarkResult:
    codes_shape: tuple[int, ...]
    latent_dim: int
    mel_patch: int
    wav_shape: tuple[int, ...]
    warmup_frames: int
    timed_frames: int
    waveform_pcc: float | None = None


def load_voxtral_audio_tokenizer_for_benchmark(
    mesh_device,
    *,
    optimizations: AudioTokenizerOptimizations | None = None,
):
    """Load checkpoint + build ``VoxtralTTAudioTokenizer``; skip test on failure."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    cfg = load_voxtral_config(model_name).audio_tokenizer_args
    sd = extract_audio_tokenizer_state_dict(full)
    opt = optimizations or voxtral_audio_tokenizer_default_optimizations()
    tok = create_voxtral_audio_tokenizer_or_skip(
        mesh_device,
        state_dict=sd,
        tokenizer_cfg=cfg,
        full_checkpoint=full,
        optimizations=opt,
    )
    return tok, cfg, sd


@torch.no_grad()
def run_voxtral_audio_tokenizer_decode_benchmark(
    mesh_device,
    *,
    log_stages: bool = True,
    optimizations: AudioTokenizerOptimizations | None = None,
) -> AudioTokenizerBenchmarkResult:
    """Warmup decode, drain profiler, timed decode between Tracy signposts."""
    tok, cfg, sd = load_voxtral_audio_tokenizer_for_benchmark(mesh_device, optimizations=optimizations)

    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    time_frames = resolve_perf_decode_t()
    warmup_frames = min(resolve_perf_warmup_t(), time_frames)
    n_codebooks = 1 + int(cfg.acoustic_dim)
    codes_b37t = make_zero_codes(n_codebooks=n_codebooks, time_frames=time_frames)
    warmup_codes_b37t = codes_b37t[..., :warmup_frames] if warmup_frames < time_frames else codes_b37t

    logger.info(
        "voxtral audio tokenizer benchmark: warmup decode (T={}, outside Tracy signposts) …",
        warmup_frames,
    )
    decode_codes_to_wav(tok, mesh_device, warmup_codes_b37t, log_stages=log_stages)

    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    tracy_start()
    wav = decode_codes_to_wav(tok, mesh_device, codes_b37t, log_stages=log_stages)
    ttnn.synchronize_device(mesh_device)
    tracy_stop()

    if not torch.isfinite(wav).all():
        raise AssertionError("audio tokenizer decode produced non-finite samples")

    waveform_pcc: float | None = None
    if os.getenv("VOXTRAL_PERF_LOG_PCC", "1").lower() not in ("0", "false", "no", "off"):
        try:
            ref_wav = audio_tokenizer_decode_reference(codes_b37t, sd, cfg)
            ok, pcc_msg = comp_pcc(ref_wav.float(), wav.float(), pcc=0.0)
            waveform_pcc = float(pcc_msg)
            logger.info(
                "voxtral audio tokenizer benchmark: waveform PCC vs CPU golden = {} (pass_any={})",
                pcc_msg,
                ok,
            )
        except Exception as exc:
            logger.warning("voxtral audio tokenizer benchmark: waveform PCC skipped ({})", exc)

    result = AudioTokenizerBenchmarkResult(
        codes_shape=tuple(codes_b37t.shape),
        latent_dim=int(cfg.semantic_dim + cfg.acoustic_dim),
        mel_patch=int(cfg.pretransform_patch_size),
        wav_shape=tuple(wav.shape),
        warmup_frames=warmup_frames,
        timed_frames=time_frames,
        waveform_pcc=waveform_pcc,
    )
    logger.info(
        "voxtral audio tokenizer benchmark: codes={} latent_dim={} mel_patch={} wav={} "
        "warmup_T={} timed_T={} pcc={}",
        result.codes_shape,
        result.latent_dim,
        result.mel_patch,
        result.wav_shape,
        result.warmup_frames,
        result.timed_frames,
        result.waveform_pcc,
    )

    del tok
    gc.collect()
    return result
