from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# Allow running as a standalone script (without installing tt-metal as a package)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TT_METAL_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_TTNN_ROOT = os.path.join(_TT_METAL_ROOT, "ttnn")
for _p in (_TT_METAL_ROOT, _TTNN_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

import ttnn
from models.demos.ace_step_v1_5.ttnn_impl.patchify import TtAceStepDePatchify1D, TtAceStepPatchEmbed1D
from models.demos.ace_step_v1_5.validation.pcc_check import print_report


@dataclass(frozen=True)
class AceStepLikeConfig:
    # Patchify-related fields only
    patch_size: int = 2
    in_channels: int = 192
    hidden_size: int = 2048
    audio_acoustic_hidden_dim: int = 64


class TorchProjIn(nn.Module):
    def __init__(self, config: AceStepLikeConfig, conv: nn.Conv1d):
        super().__init__()
        self.patch_size = config.patch_size
        self.conv = conv

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        # x: [B, T, C]
        b, t, c = x.shape
        pad_length = 0
        if t % self.patch_size != 0:
            pad_length = self.patch_size - (t % self.patch_size)
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_length), mode="constant", value=0.0)
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.conv(x)  # [B, hidden, T//p]
        x = x.transpose(1, 2)  # [B, T//p, hidden]
        return x, pad_length


class TorchProjOut(nn.Module):
    def __init__(self, config: AceStepLikeConfig, convt: nn.ConvTranspose1d):
        super().__init__()
        self.patch_size = config.patch_size
        self.convt = convt

    def forward(self, x: torch.Tensor, *, original_seq_len: int) -> torch.Tensor:
        # x: [B, T_p, hidden]
        x = x.transpose(1, 2)  # [B, hidden, T_p]
        x = self.convt(x)  # [B, out_ch, T]
        x = x.transpose(1, 2)  # [B, T, out_ch]
        return x[:, :original_seq_len, :]


def _build_state_dict_from_torch_modules(proj_in: nn.Conv1d, proj_out: nn.ConvTranspose1d) -> dict:
    # Match the key patterns used by ttnn_impl/patchify.py (Sequential index 1)
    return {
        "proj_in.1.weight": proj_in.weight.detach().clone(),
        "proj_in.1.bias": proj_in.bias.detach().clone(),
        "proj_out.1.weight": proj_out.weight.detach().clone(),
        "proj_out.1.bias": proj_out.bias.detach().clone(),
    }


def main():
    torch.manual_seed(0)

    # Create TTNN device
    device = ttnn.open_device(device_id=0, trace_region_size=128 << 20)
    device.enable_program_cache()

    try:
        config = AceStepLikeConfig()

        # Torch reference modules with identical structure to HF's proj_in/proj_out
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

        torch_proj_in = TorchProjIn(config, torch_proj_in_conv).eval()
        torch_proj_out = TorchProjOut(config, torch_proj_out_convt).eval()

        state_dict = _build_state_dict_from_torch_modules(torch_proj_in_conv, torch_proj_out_convt)

        # TTNN modules using the same weights
        tt_proj_in = TtAceStepPatchEmbed1D(
            config=config, state_dict=state_dict, base_address="proj_in", device=device, activation_dtype=ttnn.bfloat16
        )
        tt_proj_out = TtAceStepDePatchify1D(
            config=config, state_dict=state_dict, base_address="proj_out", device=device, activation_dtype=ttnn.bfloat16
        )

        # Inputs
        b, t = 2, 257  # choose a length not divisible by patch_size to test padding/trim
        x = torch.randn((b, t, config.in_channels), dtype=torch.bfloat16).float()

        # Torch reference
        y_ref_patches, _pad = torch_proj_in(x)
        y_ref = torch_proj_out(y_ref_patches, original_seq_len=t)

        # TTNN path (Host -> TTNN once at start; TTNN -> Host once at end)
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        y_tt_patches, meta = tt_proj_in(x_tt)
        y_tt = tt_proj_out(y_tt_patches, meta)
        y_tt_host = ttnn.to_torch(y_tt)

        # Reports + hard asserts (must end with 2 passing checks)
        print_report("proj_in (patch embed) output", y_ref_patches, ttnn.to_torch(y_tt_patches), pcc_threshold=0.99)
        print_report("proj_out (de-patchify) output", y_ref, y_tt_host, pcc_threshold=0.99)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
