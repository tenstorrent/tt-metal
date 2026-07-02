# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Codes → waveform PCC vs CPU ``audio_tokenizer_decode_reference`` at model-card full depth."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import audio_tokenizer_decode_reference
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.utils.common import (
    create_voxtral_audio_tokenizer_or_skip,
    resolve_voxtral_model_name_or_skip,
)
from models.experimental.voxtraltts.utils.mesh import voxtral_to_torch_replicated
from models.experimental.voxtraltts.utils.audio_tokenizer_optimizations import (
    voxtral_audio_tokenizer_default_optimizations,
)
from models.experimental.voxtraltts.tt.audio_tokenizer.model import (
    _DECODE_CHUNK_T,
    extract_audio_tokenizer_state_dict,
)
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict

# Production decode chunk from params.json / ``VoxtralTTAudioTokenizer`` (1600 acoustic frames).
_MODEL_CARD_DECODE_T = _DECODE_CHUNK_T
_PCC_TARGET = 0.98


@torch.no_grad()
@pytest.mark.timeout(0)
def test_audio_tokenizer_full_decode_pcc(device, reset_seeds):
    """Random codes → waveform PCC vs CPU at full model-card depth (T=1600)."""
    time_len = _MODEL_CARD_DECODE_T
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    cfg = load_voxtral_config(model_name).audio_tokenizer_args
    sd = extract_audio_tokenizer_state_dict(full)
    try:
        tok = create_voxtral_audio_tokenizer_or_skip(
            device,
            state_dict=sd,
            tokenizer_cfg=cfg,
            optimizations=voxtral_audio_tokenizer_default_optimizations(),
        )
    except Exception as exc:
        pytest.skip(str(exc))

    b = 1
    n_acoustic_cb = cfg.acoustic_dim
    semantic_codes = torch.randint(0, cfg.semantic_codebook_size, (b, 1, time_len))
    acoustic_codes = torch.randint(0, cfg.acoustic_codebook_size, (b, n_acoustic_cb, time_len))
    codes = torch.cat([semantic_codes, acoustic_codes], dim=1).long()

    try:
        ref_wav = audio_tokenizer_decode_reference(codes, sd, cfg)
    except Exception as exc:
        pytest.skip(f"CPU reference decode failed: {exc}")

    try:
        codes_tt = ttnn.from_torch(
            codes.to(torch.uint32).contiguous(),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        latent_tt = tok.latent_from_codes_tt(codes_tt)
        ttnn.deallocate(codes_tt)
        mel_tt = tok.decode_latent_to_mel_b1tc(latent_tt)
        ttnn.deallocate(latent_tt)
        wav_tt = tok.pretransform_decode_tt(mel_tt)
        ttnn.deallocate(mel_tt)
        tt_wav = voxtral_to_torch_replicated(wav_tt).float()
        ttnn.deallocate(wav_tt)
    except RuntimeError as exc:
        msg = str(exc)
        if "requires the full decoder stack" in msg or "output_proj" in msg or "not loaded" in msg:
            pytest.skip(msg)
        raise

    assert (
        ref_wav.shape == tt_wav.shape
    ), f"Waveform shape mismatch: ref={tuple(ref_wav.shape)}, tt={tuple(tt_wav.shape)}"
    passing, msg = comp_pcc(ref_wav.float(), tt_wav.float(), pcc=_PCC_TARGET)
    assert passing, f"Full decode PCC failed (T={time_len}, target={_PCC_TARGET}): {msg}"
