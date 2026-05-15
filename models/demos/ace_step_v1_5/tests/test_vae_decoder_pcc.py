# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the TTNN Oobleck VAE decoder.

Each test builds a torch-reference module, derives the matching TTNN module
from the same state dict, runs both, and checks PCC. The tests use small
channel counts and short time axes to keep them cheap; the full-decoder test
is gated to a reduced configuration so it can run on a single Wormhole.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

import ttnn

logger = logging.getLogger(__name__)
from models.demos.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.demos.ace_step_v1_5.torch_ref.vae.oobleck_decoder import (
    OobleckDecoder,
    OobleckDecoderBlock,
    OobleckResidualUnit,
    Snake1d,
)
from models.demos.ace_step_v1_5.ttnn_impl.vae import (
    TtOobleckDecoder,
    TtOobleckDecoderBlock,
    TtOobleckResidualUnit,
    TtSnake1d,
)
from models.demos.ace_step_v1_5.ttnn_impl.vae.weight_utils import _fused_or_passthrough, _maybe_bias


@dataclass(frozen=True)
class TinyDecoderConfig:
    """Reduced Oobleck config that exercises every stage at a tractable size."""

    decoder_channels: int = 32
    decoder_input_channels: int = 16
    audio_channels: int = 2
    upsampling_ratios: tuple = (2, 2)
    channel_multiples: tuple = (1, 2)


def _btc_to_bct(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def _bct_to_btc(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def _param_to_f32_numpy(t: torch.Tensor):
    """Host float32 ndarray for Snake / metadata (avoids BF16 `.numpy()` issues)."""
    return t.detach().float().cpu().numpy()


def test_snake1d_pcc(device, torch_seed):
    _ = torch_seed
    c = 64
    t = 33
    snake_t = Snake1d(c).eval()
    # Randomize parameters so the activation is not the identity (where logscale=>alpha=1).
    with torch.no_grad():
        snake_t.alpha.copy_(torch.randn_like(snake_t.alpha) * 0.5)
        snake_t.beta.copy_(torch.randn_like(snake_t.beta) * 0.5)

    x_bct = torch.randn(1, c, t, dtype=torch.bfloat16).float()
    y_ref_bct = snake_t(x_bct)
    y_ref_btc = _bct_to_btc(y_ref_bct)

    sd = snake_t.state_dict()
    tt_snake = TtSnake1d(
        alpha_host=_param_to_f32_numpy(sd["alpha"]),
        beta_host=_param_to_f32_numpy(sd["beta"]),
        device=device,
    )

    x_btc = _bct_to_btc(x_bct)
    x_tt = ttnn.from_torch(x_btc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_tt = tt_snake(x_tt)
    y_tt = ttnn.to_layout(y_tt, ttnn.ROW_MAJOR_LAYOUT)
    y_tt_torch = ttnn.to_torch(y_tt).float()

    assert_pcc_print("vae_snake1d", y_ref_btc, y_tt_torch)


def _residual_weights_dict(mod: OobleckResidualUnit) -> dict:
    sd = mod.state_dict()
    return {
        "snake1.alpha": _param_to_f32_numpy(sd["snake1.alpha"]),
        "snake1.beta": _param_to_f32_numpy(sd["snake1.beta"]),
        "conv1.weight": _fused_or_passthrough(sd, "conv1"),
        "conv1.bias": _maybe_bias(sd, "conv1"),
        "snake2.alpha": _param_to_f32_numpy(sd["snake2.alpha"]),
        "snake2.beta": _param_to_f32_numpy(sd["snake2.beta"]),
        "conv2.weight": _fused_or_passthrough(sd, "conv2"),
        "conv2.bias": _maybe_bias(sd, "conv2"),
    }


def test_residual_unit_pcc(device, torch_seed):
    _ = torch_seed
    c = 32
    t = 64
    mod = OobleckResidualUnit(c, dilation=3).eval()
    x_bct = torch.randn(1, c, t, dtype=torch.bfloat16).float()
    y_ref_bct = mod(x_bct)
    y_ref_btc = _bct_to_btc(y_ref_bct)

    tt_mod = TtOobleckResidualUnit(
        weights=_residual_weights_dict(mod),
        dimension=c,
        dilation=3,
        device=device,
    )
    x_tt = ttnn.from_torch(_bct_to_btc(x_bct), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_tt = tt_mod(x_tt)
    y_tt_torch = ttnn.to_torch(y_tt).float()

    assert_pcc_print("vae_residual_unit", y_ref_btc, y_tt_torch)


def _block_weights_dict(mod: OobleckDecoderBlock) -> dict:
    sd = mod.state_dict()
    out: dict = {
        "snake1.alpha": _param_to_f32_numpy(sd["snake1.alpha"]),
        "snake1.beta": _param_to_f32_numpy(sd["snake1.beta"]),
        "conv_t1.weight": _fused_or_passthrough(sd, "conv_t1"),
        "conv_t1.bias": _maybe_bias(sd, "conv_t1"),
    }
    for ru in (1, 2, 3):
        rp = f"res_unit{ru}"
        out[f"{rp}.snake1.alpha"] = _param_to_f32_numpy(sd[f"{rp}.snake1.alpha"])
        out[f"{rp}.snake1.beta"] = _param_to_f32_numpy(sd[f"{rp}.snake1.beta"])
        out[f"{rp}.snake2.alpha"] = _param_to_f32_numpy(sd[f"{rp}.snake2.alpha"])
        out[f"{rp}.snake2.beta"] = _param_to_f32_numpy(sd[f"{rp}.snake2.beta"])
        out[f"{rp}.conv1.weight"] = _fused_or_passthrough(sd, f"{rp}.conv1")
        out[f"{rp}.conv1.bias"] = _maybe_bias(sd, f"{rp}.conv1")
        out[f"{rp}.conv2.weight"] = _fused_or_passthrough(sd, f"{rp}.conv2")
        out[f"{rp}.conv2.bias"] = _maybe_bias(sd, f"{rp}.conv2")
    return out


def test_decoder_block_pcc(device, torch_seed):
    _ = torch_seed
    in_c, out_c, stride, t = 32, 16, 2, 32
    mod = OobleckDecoderBlock(in_c, out_c, stride=stride).eval()
    x_bct = torch.randn(1, in_c, t, dtype=torch.bfloat16).float()
    y_ref_bct = mod(x_bct)
    y_ref_btc = _bct_to_btc(y_ref_bct)

    tt_mod = TtOobleckDecoderBlock(
        weights=_block_weights_dict(mod),
        input_dim=in_c,
        output_dim=out_c,
        stride=stride,
        device=device,
    )
    x_tt = ttnn.from_torch(_bct_to_btc(x_bct), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_tt = tt_mod(x_tt)
    y_tt_torch = ttnn.to_torch(y_tt).float()

    assert_pcc_print("vae_decoder_block", y_ref_btc, y_tt_torch)


def test_decoder_tiny_pcc(device, torch_seed):
    _ = torch_seed
    cfg = TinyDecoderConfig()
    torch_dec = OobleckDecoder(
        channels=cfg.decoder_channels,
        input_channels=cfg.decoder_input_channels,
        audio_channels=cfg.audio_channels,
        upsampling_ratios=cfg.upsampling_ratios,
        channel_multiples=cfg.channel_multiples,
    ).eval()

    t_lat = 32
    x_bct = torch.randn(1, cfg.decoder_input_channels, t_lat, dtype=torch.bfloat16).float()
    with torch.inference_mode():
        y_ref_bct = torch_dec(x_bct)
    y_ref_btc = _bct_to_btc(y_ref_bct)

    full_sd = {f"decoder.{k}": v for k, v in torch_dec.state_dict().items()}
    tt_dec = TtOobleckDecoder(
        state_dict=full_sd,
        device=device,
        decoder_prefix="decoder.",
        channels=cfg.decoder_channels,
        input_channels=cfg.decoder_input_channels,
        audio_channels=cfg.audio_channels,
        upsampling_ratios=cfg.upsampling_ratios,
        channel_multiples=cfg.channel_multiples,
    )
    x_tt = ttnn.from_torch(_bct_to_btc(x_bct), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_tt = tt_dec(x_tt)
    y_tt_torch = ttnn.to_torch(y_tt).float()

    assert_pcc_print("vae_decoder_tiny", y_ref_btc, y_tt_torch)
