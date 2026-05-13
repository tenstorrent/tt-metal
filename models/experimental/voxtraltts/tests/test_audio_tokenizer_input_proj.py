# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import causal_conv1d_left_pad_reference
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.audio_tokenizer.conv import (
    VoxtralTTAudioTokenizerInputProj,
    resolve_input_proj_conv_weight,
)
from models.experimental.voxtraltts.tt.audio_tokenizer.model import (
    VoxtralTTAudioTokenizer,
    extract_audio_tokenizer_state_dict,
)
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict


def _mel_torch_to_tt_b1tc(device, mel_ncl_bf16: torch.Tensor) -> ttnn.Tensor:
    """``[B, C, T]`` host → ``[B, 1, T, C]`` tile on device."""
    x = mel_ncl_bf16.to(torch.bfloat16).contiguous()
    # [B, C, T] -> [B, T, C] -> [B, 1, T, C]
    x4 = x.permute(0, 2, 1).unsqueeze(1)
    return ttnn.from_torch(
        x4,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@torch.no_grad()
@pytest.mark.parametrize("time_len", [64, 128])
def test_audio_tokenizer_input_proj_pcc(device, reset_seeds, time_len):
    """``input_proj`` TT ``conv1d`` vs CPU golden (checkpoint weights)."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    sd = extract_audio_tokenizer_state_dict(full)
    try:
        proj = VoxtralTTAudioTokenizerInputProj(device, state_dict=sd)
    except KeyError as exc:
        pytest.skip(f"Missing audio tokenizer weights: {exc}")

    w = resolve_input_proj_conv_weight(sd).to(torch.bfloat16)
    b, c = 1, proj.in_channels
    mel = torch.randn(b, c, time_len, dtype=torch.bfloat16)
    ref = causal_conv1d_left_pad_reference(mel, w, left_pad=proj.left_pad, stride=1)

    mel_tt = _mel_torch_to_tt_b1tc(device, mel)
    tt_out = proj(mel_tt)
    ttnn.deallocate(mel_tt)

    tt_ncl = torch.permute(ttnn.to_torch(tt_out).squeeze(1), (0, 2, 1)).contiguous().float()
    passing, msg = comp_pcc(ref.float(), tt_ncl, pcc=0.99)
    assert passing, f"input_proj PCC failed: {msg}"


@torch.no_grad()
def test_voxtral_tt_audio_tokenizer_create_smoke(device, reset_seeds):
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        tok = VoxtralTTAudioTokenizer.create_from_model_name(device, model_name_or_path=model_name)
    except Exception as exc:
        pytest.skip(f"Unable to create VoxtralTTAudioTokenizer: {exc}")

    # Public checkpoints ship decoder weights; encoder ``input_proj`` / ``encoder_blocks`` may be absent.
    # Keep this as a construction smoke test; per-op PCC tests exercise each TT op separately.
    assert tok.decoder_blocks_0_conv is not None
    assert tok.decoder_blocks_1_layer0 is not None
    assert tok.decoder_blocks_1_layer1 is not None
    assert tok.decoder_blocks_2_conv_transpose is not None
    assert tok.decoder_blocks_4_conv_transpose is not None
    assert tok.decoder_blocks_6_conv_transpose is not None
    assert tok.decoder_blocks_3_layer0 is not None
    assert tok.decoder_blocks_3_layer1 is not None
    assert tok.decoder_blocks_5_layer0 is not None
    assert tok.decoder_blocks_5_layer1 is not None
    assert tok.decoder_blocks_7_layer0 is not None
    assert tok.decoder_blocks_7_layer1 is not None
