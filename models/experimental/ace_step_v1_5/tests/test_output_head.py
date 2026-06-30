# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import ttnn
from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.experimental.ace_step_v1_5.torch_ref.output_head import OutputHeadConfig, TorchAceStepDiTOutputHead
from models.experimental.ace_step_v1_5.ttnn_impl.output_head import TtAceStepDiTOutputHead
from models.experimental.ace_step_v1_5.ttnn_impl.patchify import PatchifyMetadata


@dataclass(frozen=True)
class AceStepLikeConfig:
    patch_size: int = 2
    in_channels: int = 192
    hidden_size: int = 2048
    audio_acoustic_hidden_dim: int = 64
    rms_norm_eps: float = 1e-6


def _make_state_and_torch(
    config: AceStepLikeConfig,
) -> tuple[dict, torch.Tensor, torch.Tensor, TorchAceStepDiTOutputHead, PatchifyMetadata]:
    torch.manual_seed(0)
    norm_w = torch.randn(config.hidden_size, dtype=torch.bfloat16)
    sst = torch.randn(1, 2, config.hidden_size, dtype=torch.bfloat16)
    convt = nn.ConvTranspose1d(
        in_channels=config.hidden_size,
        out_channels=config.audio_acoustic_hidden_dim,
        kernel_size=config.patch_size,
        stride=config.patch_size,
        padding=0,
        bias=True,
    ).to(torch.bfloat16)

    state_dict = {
        "decoder.norm_out.weight": norm_w.detach().clone(),
        "decoder.scale_shift_table": sst.detach().clone(),
        "decoder.proj_out.1.weight": convt.weight.detach().clone(),
        "decoder.proj_out.1.bias": convt.bias.detach().clone(),
    }

    torch_head = TorchAceStepDiTOutputHead(
        config=OutputHeadConfig(
            hidden_size=config.hidden_size,
            audio_acoustic_hidden_dim=config.audio_acoustic_hidden_dim,
            patch_size=config.patch_size,
            rms_norm_eps=config.rms_norm_eps,
        ),
        state_dict=state_dict,
        base_address="decoder",
        dtype=torch.bfloat16,
    ).eval()

    original_seq_len = 257
    pad_length = config.patch_size - (original_seq_len % config.patch_size)
    t_p = (original_seq_len + pad_length) // config.patch_size
    meta = PatchifyMetadata(original_seq_len=original_seq_len, pad_length=pad_length, patch_size=config.patch_size)

    x = torch.randn((2, t_p, config.hidden_size), dtype=torch.bfloat16)
    temb = torch.randn((2, config.hidden_size), dtype=torch.bfloat16)
    return state_dict, x, temb, torch_head, meta


def test_output_head_matches_torch(device, torch_seed):
    _ = torch_seed
    config = AceStepLikeConfig()
    state_dict, x, temb, torch_head, meta = _make_state_and_torch(config)

    y_ref = torch_head(x, temb, meta)

    tt_head = TtAceStepDiTOutputHead(
        config=config,
        state_dict=state_dict,
        base_address="decoder",
        device=device,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
    )

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    temb_tt = ttnn.from_torch(temb, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_tt = tt_head(x_tt, temb_tt, meta)
    y_tt_torch = ttnn.to_torch(y_tt).float()

    assert_pcc_print("output_head", y_ref, y_tt_torch)
