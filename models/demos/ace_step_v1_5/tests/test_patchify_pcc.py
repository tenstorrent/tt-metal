# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.demos.ace_step_v1_5.ttnn_impl.patchify import TtAceStepDePatchify1D, TtAceStepPatchEmbed1D
from tests.ttnn.utils_for_testing import assert_with_pcc


@dataclass(frozen=True)
class AceStepLikeConfig:
    patch_size: int = 2
    in_channels: int = 192
    hidden_size: int = 2048
    audio_acoustic_hidden_dim: int = 64


class TorchProjIn(nn.Module):
    def __init__(self, config: AceStepLikeConfig, conv: nn.Conv1d):
        super().__init__()
        self.patch_size = config.patch_size
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _c = x.shape
        if t % self.patch_size != 0:
            pad_length = self.patch_size - (t % self.patch_size)
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_length), mode="constant", value=0.0)
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)


class TorchProjOut(nn.Module):
    def __init__(self, config: AceStepLikeConfig, convt: nn.ConvTranspose1d):
        super().__init__()
        self.patch_size = config.patch_size
        self.convt = convt

    def forward(self, x: torch.Tensor, *, original_seq_len: int) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.convt(x)
        x = x.transpose(1, 2)
        return x[:, :original_seq_len, :]


def _state_dict(proj_in: nn.Conv1d, proj_out: nn.ConvTranspose1d) -> dict:
    return {
        "proj_in.1.weight": proj_in.weight.detach().clone(),
        "proj_in.1.bias": proj_in.bias.detach().clone(),
        "proj_out.1.weight": proj_out.weight.detach().clone(),
        "proj_out.1.bias": proj_out.bias.detach().clone(),
    }


def _make_modules(config: AceStepLikeConfig, device):
    torch_proj_in_conv = nn.Conv1d(
        in_channels=config.in_channels,
        out_channels=config.hidden_size,
        kernel_size=config.patch_size,
        stride=config.patch_size,
        padding=0,
        bias=True,
    )
    torch_proj_out_convt = nn.ConvTranspose1d(
        in_channels=config.hidden_size,
        out_channels=config.audio_acoustic_hidden_dim,
        kernel_size=config.patch_size,
        stride=config.patch_size,
        padding=0,
        bias=True,
    )

    state_dict = _state_dict(torch_proj_in_conv, torch_proj_out_convt)

    tt_proj_in = TtAceStepPatchEmbed1D(
        config=config, state_dict=state_dict, base_address="proj_in", device=device, activation_dtype=ttnn.bfloat16
    )
    tt_proj_out = TtAceStepDePatchify1D(
        config=config, state_dict=state_dict, base_address="proj_out", device=device, activation_dtype=ttnn.bfloat16
    )

    torch_proj_in = TorchProjIn(config, torch_proj_in_conv).eval()
    torch_proj_out = TorchProjOut(config, torch_proj_out_convt).eval()

    return torch_proj_in, torch_proj_out, tt_proj_in, tt_proj_out


def test_patch_embed_matches_torch(device, torch_seed):
    _ = torch_seed
    config = AceStepLikeConfig()
    torch_proj_in, _torch_proj_out, tt_proj_in, _tt_proj_out = _make_modules(config, device)

    b, t = 2, 257
    x = torch.randn((b, t, config.in_channels), dtype=torch.bfloat16).float()

    y_ref = torch_proj_in(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_tt, _meta = tt_proj_in(x_tt)
    y_tt_torch = ttnn.to_torch(y_tt).float()

    _pcc_ok, pcc_msg = assert_with_pcc(y_ref, y_tt_torch, pcc=0.99)
    logger.info(f"[ace_step_v1_5][patch_embed] {pcc_msg} (threshold=0.99, ok={_pcc_ok})")


def test_depatchify_matches_torch(device, torch_seed):
    _ = torch_seed
    config = AceStepLikeConfig()
    torch_proj_in, torch_proj_out, tt_proj_in, tt_proj_out = _make_modules(config, device)

    b, t = 2, 257
    x = torch.randn((b, t, config.in_channels), dtype=torch.bfloat16).float()

    y_patches_ref = torch_proj_in(x)
    y_ref = torch_proj_out(y_patches_ref, original_seq_len=t)

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_tt_patches, meta = tt_proj_in(x_tt)
    y_tt = tt_proj_out(y_tt_patches, meta)
    y_tt_torch = ttnn.to_torch(y_tt).float()

    _pcc_ok, pcc_msg = assert_with_pcc(y_ref, y_tt_torch, pcc=0.99)
    logger.info(f"[ace_step_v1_5][depatchify] {pcc_msg} (threshold=0.99, ok={_pcc_ok})")
