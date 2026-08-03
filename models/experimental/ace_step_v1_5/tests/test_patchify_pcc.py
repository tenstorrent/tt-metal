# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import ttnn
from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.experimental.ace_step_v1_5.torch_ref.patchify import depatchify_1d, patchify_1d
from models.experimental.ace_step_v1_5.ttnn_impl.patchify import TtAceStepDePatchify1D, TtAceStepPatchEmbed1D


@dataclass(frozen=True)
class AceStepLikeConfig:
    patch_size: int = 2
    in_channels: int = 192
    hidden_size: int = 2048
    audio_acoustic_hidden_dim: int = 64


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
    return torch_proj_in_conv.eval(), torch_proj_out_convt.eval(), tt_proj_in, tt_proj_out


def test_patch_embed_matches_torch(device, torch_seed):
    _ = torch_seed
    config = AceStepLikeConfig()
    proj_in_conv, _proj_out_convt, tt_proj_in, _tt_proj_out = _make_modules(config, device)

    b, t = 2, 257
    x = torch.randn((b, t, config.in_channels), dtype=torch.bfloat16).float()

    y_ref, _meta_ref = patchify_1d(x, conv=proj_in_conv, patch_size=config.patch_size)

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_tt, _meta = tt_proj_in(x_tt)
    y_tt_torch = ttnn.to_torch(y_tt).float()

    assert_pcc_print("patch_embed", y_ref, y_tt_torch)


def test_depatchify_matches_torch(device, torch_seed):
    _ = torch_seed
    config = AceStepLikeConfig()
    proj_in_conv, proj_out_convt, tt_proj_in, tt_proj_out = _make_modules(config, device)

    b, t = 2, 257
    x = torch.randn((b, t, config.in_channels), dtype=torch.bfloat16).float()

    y_patches_ref, meta_ref = patchify_1d(x, conv=proj_in_conv, patch_size=config.patch_size)
    y_ref = depatchify_1d(y_patches_ref, convt=proj_out_convt, meta=meta_ref, original_seq_len=t)

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_tt_patches, meta = tt_proj_in(x_tt)
    y_tt = tt_proj_out(y_tt_patches, meta)
    y_tt_torch = ttnn.to_torch(y_tt).float()

    assert_pcc_print("depatchify", y_ref, y_tt_torch)
