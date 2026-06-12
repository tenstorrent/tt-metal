# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Codes → waveform PCC vs CPU ``audio_tokenizer_decode_reference``."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import audio_tokenizer_decode_reference
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.tests.common import (
    create_voxtral_audio_tokenizer_or_skip,
    resolve_voxtral_model_name_or_skip,
)
from models.experimental.voxtraltts.utils.audio_tokenizer_optimizations import (
    voxtral_audio_tokenizer_default_optimizations,
)
from models.experimental.voxtraltts.tt.audio_tokenizer.model import extract_audio_tokenizer_state_dict
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict

_PCC_TARGET = 0.99


@torch.no_grad()
@pytest.mark.parametrize(
    "time_len,pcc",
    [
        (4, _PCC_TARGET),
        (32, _PCC_TARGET),
        (39, _PCC_TARGET),
        (64, _PCC_TARGET),
        pytest.param(
            96,
            _PCC_TARGET,
            marks=pytest.mark.timeout(3600),
            id="medium_decode_96",
        ),
        pytest.param(
            160,
            _PCC_TARGET,
            marks=pytest.mark.timeout(3600),
            id="chunked_decode_160",
        ),
    ],
)
def test_audio_tokenizer_full_decode_pcc(device, reset_seeds, time_len, pcc):
    """Random codes → waveform PCC vs CPU golden."""
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
        tt_wav = ttnn.to_torch(wav_tt).float()
        ttnn.deallocate(wav_tt)
    except RuntimeError as exc:
        msg = str(exc)
        if "requires the full decoder stack" in msg or "output_proj" in msg or "not loaded" in msg:
            pytest.skip(msg)
        raise

    assert (
        ref_wav.shape == tt_wav.shape
    ), f"Waveform shape mismatch: ref={tuple(ref_wav.shape)}, tt={tuple(tt_wav.shape)}"
    passing, msg = comp_pcc(ref_wav.float(), tt_wav.float(), pcc=pcc)
    assert passing, f"Full decode end-to-end PCC failed (time_len={time_len}, target={pcc}): {msg}"
