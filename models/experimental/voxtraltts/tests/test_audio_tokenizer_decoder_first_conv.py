# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import causal_conv1d_reference_bf16
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.audio_tokenizer.conv import (
    VoxtralTTAudioTokenizerDecoderCausalConv1d,
    resolve_decoder_block_causal_conv_fused_weight,
)
from models.experimental.voxtraltts.tt.audio_tokenizer.model import extract_audio_tokenizer_state_dict
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict


def _latent_ncl_to_tt_b1tc(device, x_ncl_bf16: torch.Tensor) -> ttnn.Tensor:
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
def test_audio_tokenizer_decoder_blocks_0_conv_pcc(device, reset_seeds, time_len):
    """``decoder_blocks.0`` ``CausalConv1d`` same math as ``vllm_omni`` (replicate pad + fused weight norm)."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    cfg = load_voxtral_config(model_name).audio_tokenizer_args

    sd = extract_audio_tokenizer_state_dict(full)
    try:
        w = resolve_decoder_block_causal_conv_fused_weight(sd, 0)
        kerns = [int(x.strip()) for x in cfg.decoder_convs_kernels_str.split(",") if x.strip()]
        strides = [int(x.strip()) for x in cfg.decoder_convs_strides_str.split(",") if x.strip()]
        conv = VoxtralTTAudioTokenizerDecoderCausalConv1d(
            device,
            state_dict=sd,
            block_index=0,
            kernel_size=kerns[0],
            stride=strides[0],
            pad_mode="replicate",
            in_channels=cfg.semantic_dim + cfg.acoustic_dim,
            out_channels=cfg.dim,
        )
    except KeyError as exc:
        pytest.skip(f"No decoder_blocks.0 weights: {exc}")

    b, c_in = 1, int(cfg.semantic_dim + cfg.acoustic_dim)
    x_ncl = torch.randn(b, c_in, time_len, dtype=torch.bfloat16)
    ref = causal_conv1d_reference_bf16(
        x_ncl,
        w.to(torch.bfloat16),
        kernel_size=kerns[0],
        stride=strides[0],
        dilation=1,
        pad_mode="replicate",
    )

    x_tt = _latent_ncl_to_tt_b1tc(device, x_ncl)
    y_tt = conv(x_tt)
    ttnn.deallocate(x_tt)

    assert int(y_tt.shape[2]) == ref.shape[2]
    tt_ncl = torch.permute(ttnn.to_torch(y_tt).squeeze(1), (0, 2, 1)).contiguous().float()
    passing, msg = comp_pcc(ref.float(), tt_ncl, pcc=0.99)
    assert passing, f"decoder_blocks.0 PCC failed: {msg}"


@torch.no_grad()
def test_audio_tokenizer_config_codec_eps_matches_reference():
    """Codec (tokenizer transformer / FFN) RMSNorm ``norm_eps`` is ``0.01`` in reference config, not text ``1e-5``."""
    model_name = resolve_voxtral_model_name_or_skip()
    cfg = load_voxtral_config(model_name)
    assert cfg.audio_tokenizer_args.norm_eps == pytest.approx(0.01)
    assert cfg.audio_tokenizer_args.qk_norm_eps == pytest.approx(1e-6)
    assert cfg.norm_eps == pytest.approx(1e-5)
