# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import causal_conv1d_left_pad_reference
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.audio_tokenizer.conv import (
    VoxtralTTAudioTokenizerEncoderDownsampleConv,
    resolve_encoder_block_strided_conv_weight,
)
from models.experimental.voxtraltts.tt.audio_tokenizer.model import extract_audio_tokenizer_state_dict
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict


def _feat_ncl_to_tt_b1tc(device, x_ncl_bf16: torch.Tensor) -> ttnn.Tensor:
    x = x_ncl_bf16.to(torch.bfloat16).contiguous()
    x4 = x.permute(0, 2, 1).unsqueeze(1)
    return ttnn.from_torch(
        x4,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@torch.no_grad()
@pytest.mark.parametrize("time_len", [32, 64])
def test_audio_tokenizer_encoder_downsample_conv_pcc(device, reset_seeds, time_len):
    """``encoder_blocks.1`` strided conv: TT vs CPU (checkpoint weights)."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")
    sd = extract_audio_tokenizer_state_dict(full)
    try:
        conv = VoxtralTTAudioTokenizerEncoderDownsampleConv(device, state_dict=sd, block_index=1)
    except KeyError as exc:
        pytest.skip(f"Missing encoder downsample weights: {exc}")

    w = resolve_encoder_block_strided_conv_weight(sd, 1).to(torch.bfloat16)
    b, c = 1, conv.in_channels
    feat = torch.randn(b, c, time_len, dtype=torch.bfloat16)
    ref = causal_conv1d_left_pad_reference(feat, w, left_pad=conv.left_pad, stride=conv.stride)

    x_tt = _feat_ncl_to_tt_b1tc(device, feat)
    y_tt = conv(x_tt)
    ttnn.deallocate(x_tt)

    t_out = int(y_tt.shape[2])
    assert t_out == conv.expected_output_length(time_len)
    tt_ncl = torch.permute(ttnn.to_torch(y_tt).squeeze(1), (0, 2, 1)).contiguous().float()
    passing, msg = comp_pcc(ref.float(), tt_ncl, pcc=0.97)
    assert passing, f"encoder downsample PCC failed: {msg}"
