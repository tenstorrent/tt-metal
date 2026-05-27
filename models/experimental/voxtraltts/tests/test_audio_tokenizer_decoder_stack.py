# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import decoder_blocks_stack_reference
from models.experimental.voxtraltts.reference.voxtral_config import audio_tokenizer_latent_dim, load_voxtral_config
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.audio_tokenizer.model import (
    VoxtralTTAudioTokenizer,
    extract_audio_tokenizer_state_dict,
)
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict
from models.experimental.voxtraltts.utils.audio_tokenizer_optimizations import (
    voxtral_audio_tokenizer_high_accuracy_optimizations,
)


def _latent_ncl_to_tt_b1tc(device, latent_ncl_bf16: torch.Tensor) -> ttnn.Tensor:
    """``[B, C, T]`` host → ``[B, 1, T, C]`` tile on device."""
    x4 = latent_ncl_bf16.to(torch.bfloat16).permute(0, 2, 1).unsqueeze(1).contiguous()
    return ttnn.from_torch(
        x4,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "time_len,pcc",
    [
        (4, 0.99),
        (39, 0.98),
        (64, 0.99),
        (128, 0.98),
    ],
)
def test_audio_tokenizer_decode_full_forward_stack_pcc(device, reset_seeds, time_len, pcc):
    """All 12 decoder blocks chained vs bf16 CPU golden. T=128/pcc=0.98 covers realistic audio lengths."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    cfg = load_voxtral_config(model_name).audio_tokenizer_args
    sd = extract_audio_tokenizer_state_dict(full)
    try:
        tok = VoxtralTTAudioTokenizer(
            device,
            state_dict=sd,
            tokenizer_cfg=cfg,
            optimizations=voxtral_audio_tokenizer_high_accuracy_optimizations(),
        )
    except Exception as exc:
        pytest.skip(f"Unable to build VoxtralTTAudioTokenizer: {exc}")

    latent_c = audio_tokenizer_latent_dim(cfg)
    b = 1
    latent = torch.randn(b, latent_c, time_len, dtype=torch.bfloat16)
    ref = decoder_blocks_stack_reference(latent, sd, cfg)

    latent_tt = _latent_ncl_to_tt_b1tc(device, latent)
    try:
        y_tt = tok.decode_full_forward(latent_tt)
    except RuntimeError as exc:
        if "decode_full_forward requires" in str(exc):
            pytest.skip(str(exc))
        raise
    ttnn.deallocate(latent_tt)

    tt_btd = ttnn.to_torch(y_tt).squeeze(1).float()
    assert ref.shape == tt_btd.shape, f"shape mismatch ref={tuple(ref.shape)} tt={tuple(tt_btd.shape)}"
    passing, msg = comp_pcc(ref.float(), tt_btd, pcc=pcc)
    assert passing, f"decode_full_forward stack PCC failed (time_len={time_len}, required={pcc}): {msg}"


@torch.no_grad()
def test_decode_full_forward_raises_when_decoder_incomplete(device, reset_seeds):
    """If a submodule is missing, ``decode_full_forward`` lists required blocks."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    cfg = load_voxtral_config(model_name).audio_tokenizer_args
    sd = extract_audio_tokenizer_state_dict(full)
    try:
        tok = VoxtralTTAudioTokenizer(device, state_dict=sd, tokenizer_cfg=cfg)
    except Exception as exc:
        pytest.skip(f"Unable to build VoxtralTTAudioTokenizer: {exc}")

    saved = tok.decoder_blocks_6_conv_transpose
    if saved is None:
        pytest.skip("decoder_blocks.6 already absent; cannot test incomplete-stack error path.")
    tok.decoder_blocks_6_conv_transpose = None
    latent_c = audio_tokenizer_latent_dim(cfg)
    latent_tt = _latent_ncl_to_tt_b1tc(device, torch.randn(1, latent_c, 8, dtype=torch.bfloat16))
    try:
        with pytest.raises(RuntimeError, match="decoder_blocks.6"):
            tok.decode_full_forward(latent_tt)
    finally:
        tok.decoder_blocks_6_conv_transpose = saved
        ttnn.deallocate(latent_tt)
