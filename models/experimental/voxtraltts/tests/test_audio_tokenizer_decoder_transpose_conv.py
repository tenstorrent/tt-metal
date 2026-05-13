# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import causal_conv_transpose1d_reference_bf16
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config, parse_csv_ints
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.audio_tokenizer.conv import (
    VoxtralTTAudioTokenizerDecoderCausalConvTranspose1d,
    resolve_decoder_block_conv_transpose_fused_weight,
)
from models.experimental.voxtraltts.tt.audio_tokenizer.model import extract_audio_tokenizer_state_dict
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict


def _feat_ncl_to_tt_b1tc(device, x_ncl_bf16: torch.Tensor) -> ttnn.Tensor:
    x4 = x_ncl_bf16.to(torch.bfloat16).contiguous().permute(0, 2, 1).unsqueeze(1)
    return ttnn.from_torch(
        x4,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "decoder_block_index,kern_stride_index",
    [(2, 1), (4, 2), (6, 3)],
)
@pytest.mark.parametrize("time_len", [16, 32])
def test_audio_tokenizer_decoder_transpose_conv_pcc(
    device, reset_seeds, time_len, decoder_block_index, kern_stride_index
):
    """``decoder_blocks.{2,4,6}`` ``CausalConvTranspose1d`` vs ``causal_conv_transpose1d_reference``."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    cfg = load_voxtral_config(model_name).audio_tokenizer_args
    kerns = parse_csv_ints(cfg.decoder_convs_kernels_str)
    strides = parse_csv_ints(cfg.decoder_convs_strides_str)
    ks = kerns[kern_stride_index]
    st = strides[kern_stride_index]
    sd = extract_audio_tokenizer_state_dict(full)
    try:
        w = resolve_decoder_block_conv_transpose_fused_weight(sd, decoder_block_index)
        conv_t = VoxtralTTAudioTokenizerDecoderCausalConvTranspose1d(
            device,
            state_dict=sd,
            block_index=decoder_block_index,
            kernel_size=ks,
            stride=st,
            in_channels=cfg.dim,
            out_channels=cfg.dim,
        )
    except KeyError as exc:
        pytest.skip(f"No decoder_blocks.{decoder_block_index} transpose weights: {exc}")

    x = torch.randn(1, cfg.dim, time_len, dtype=torch.bfloat16)
    ref = causal_conv_transpose1d_reference_bf16(
        x,
        w.to(torch.bfloat16),
        kernel_size=ks,
        stride=st,
        trim_ratio=1.0,
    )

    x_tt = _feat_ncl_to_tt_b1tc(device, x)
    y_tt = conv_t(x_tt)
    ttnn.deallocate(x_tt)

    assert int(y_tt.shape[2]) == ref.shape[2]
    tt_ncl = torch.permute(ttnn.to_torch(y_tt).squeeze(1), (0, 2, 1)).contiguous().float()
    passing, msg = comp_pcc(ref.float(), tt_ncl, pcc=0.99)
    assert passing, f"decoder_blocks.{decoder_block_index} conv transpose PCC failed: {msg}"
