# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.demos.ace_step_v1_5.ttnn_impl.output_head import TtAceStepDiTOutputHead
from models.demos.ace_step_v1_5.ttnn_impl.patchify import PatchifyMetadata
from tests.ttnn.utils_for_testing import assert_with_pcc


@dataclass(frozen=True)
class AceStepLikeConfig:
    patch_size: int = 2
    in_channels: int = 192
    hidden_size: int = 2048
    audio_acoustic_hidden_dim: int = 64
    rms_norm_eps: float = 1e-6


def _torch_rmsnorm_qwen3(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Match ``transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm`` / ttnn RMS golden."""
    x_f = x.float()
    x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f * weight.float()).to(x.dtype)


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


def _torch_output_head(
    x: torch.Tensor,
    temb: torch.Tensor,
    *,
    norm_w: torch.Tensor,
    scale_shift_table: torch.Tensor,
    proj_out: TorchProjOut,
    eps: float,
    original_seq_len: int,
) -> torch.Tensor:
    """HF ``AceStepDiTModel`` tail: norm_out → scale/shift → proj_out → length crop."""
    normed = _torch_rmsnorm_qwen3(x, norm_w, eps)
    shift = scale_shift_table[:, 0:1, :] + temb.unsqueeze(1)
    scale = scale_shift_table[:, 1:2, :] + temb.unsqueeze(1)
    modulated = (normed * (1 + scale) + shift).type_as(x)
    return proj_out(modulated, original_seq_len=original_seq_len)


def _make_state_and_torch(
    config: AceStepLikeConfig,
) -> tuple[dict, torch.Tensor, torch.Tensor, TorchProjOut, PatchifyMetadata]:
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
    proj = TorchProjOut(config, convt).eval()

    state_dict = {
        "decoder.norm_out.weight": norm_w.detach().clone(),
        "decoder.scale_shift_table": sst.detach().clone(),
        "decoder.proj_out.1.weight": convt.weight.detach().clone(),
        "decoder.proj_out.1.bias": convt.bias.detach().clone(),
    }

    original_seq_len = 257
    pad_length = config.patch_size - (original_seq_len % config.patch_size)
    t_p = (original_seq_len + pad_length) // config.patch_size
    meta = PatchifyMetadata(original_seq_len=original_seq_len, pad_length=pad_length, patch_size=config.patch_size)

    x = torch.randn((2, t_p, config.hidden_size), dtype=torch.bfloat16)
    temb = torch.randn((2, config.hidden_size), dtype=torch.bfloat16)
    return state_dict, x, temb, proj, meta


def test_output_head_matches_torch(device, torch_seed):
    _ = torch_seed
    config = AceStepLikeConfig()
    state_dict, x, temb, torch_proj, meta = _make_state_and_torch(config)

    y_ref = _torch_output_head(
        x,
        temb,
        norm_w=state_dict["decoder.norm_out.weight"],
        scale_shift_table=state_dict["decoder.scale_shift_table"],
        proj_out=torch_proj,
        eps=config.rms_norm_eps,
        original_seq_len=meta.original_seq_len,
    )

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

    _pcc_ok, pcc_msg = assert_with_pcc(y_ref, y_tt_torch, pcc=0.99)
    logger.info(f"[ace_step_v1_5][output_head] {pcc_msg} (threshold=0.99, ok={_pcc_ok})")
