from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .patchify import PatchifyMetadata, depatchify_1d


def _state_key(base_address: str, suffix: str) -> str:
    return f"{base_address}.{suffix}" if base_address else suffix


def _maybe_get_state_dict_key(state_dict: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for k in candidates:
        if k in state_dict:
            return k
    raise KeyError(f"None of the candidate keys were found in state_dict: {candidates}")


def rmsnorm_qwen3(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Match ``transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm``.

    HF uses RMSNorm (no bias):
      y = x * rsqrt(mean(x^2) + eps) * weight
    """
    x_f = x.float()
    x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + float(eps))
    return (x_f * weight.float()).to(dtype=x.dtype)


@dataclass(frozen=True)
class OutputHeadConfig:
    """
    Minimal config surface for the output head.

    This matches fields used by HF `AceStepDiTModel` tail:
    - hidden_size: patch-token embedding dim (inner_dim in HF snippet)
    - audio_acoustic_hidden_dim: output channels after de-patchify
    - patch_size: upsample factor (kernel_size=stride=patch_size)
    - rms_norm_eps: RMSNorm epsilon
    """

    hidden_size: int
    audio_acoustic_hidden_dim: int
    patch_size: int
    rms_norm_eps: float = 1e-6


class TorchAceStepDiTOutputHead(nn.Module):
    """
    Torch reference of HF `AceStepDiTModel` output head:

      shift, scale = (scale_shift_table + temb[:,None,:]).chunk(2, dim=1)
      x = (norm_out(x) * (1 + scale) + shift).type_as(x)
      x = proj_out(x)   # ConvTranspose1d stack (depatchify)
      x = x[:, :original_seq_len, :]  # crop padding

    Inputs:
      hidden_states: [B, T_p, hidden_size] patch tokens
      temb:          [B, hidden_size] diffusion timestep embedding ("temb" branch)
      meta:          PatchifyMetadata(original_seq_len, pad_length, patch_size)
    Output:
      [B, original_seq_len, audio_acoustic_hidden_dim]
    """

    def __init__(
        self,
        *,
        config: OutputHeadConfig,
        state_dict: Optional[dict[str, Any]] = None,
        base_address: str = "",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.patch_size = int(config.patch_size)
        self.out_channels = int(config.audio_acoustic_hidden_dim)
        self.eps = float(config.rms_norm_eps)

        # Parameters (populated from state_dict if provided)
        self.norm_weight = nn.Parameter(torch.ones(self.hidden_size), requires_grad=False)
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 2, self.hidden_size), requires_grad=False)

        self.proj_out = nn.ConvTranspose1d(
            in_channels=self.hidden_size,
            out_channels=self.out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=True,
        )

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

        if state_dict is not None:
            self.load_from_state_dict(state_dict=state_dict, base_address=base_address)

    def load_from_state_dict(self, *, state_dict: dict[str, Any], base_address: str = "") -> None:
        norm_w_key = _maybe_get_state_dict_key(
            state_dict,
            (
                _state_key(base_address, "norm_out.weight"),
                "norm_out.weight",
            ),
        )
        sst_key = _maybe_get_state_dict_key(
            state_dict,
            (
                _state_key(base_address, "scale_shift_table"),
                "scale_shift_table",
            ),
        )

        proj_w_key = _maybe_get_state_dict_key(
            state_dict,
            (
                _state_key(base_address, "proj_out.1.weight"),
                _state_key(base_address, "proj_out.weight"),
            ),
        )
        proj_b_key = _maybe_get_state_dict_key(
            state_dict,
            (
                _state_key(base_address, "proj_out.1.bias"),
                _state_key(base_address, "proj_out.bias"),
            ),
        )

        norm_w = state_dict[norm_w_key]
        sst = state_dict[sst_key]
        proj_w = state_dict[proj_w_key]
        proj_b = state_dict[proj_b_key]

        if int(norm_w.shape[0]) != self.hidden_size:
            raise ValueError(f"norm_out.weight length mismatch: got {norm_w.shape[0]}, expected {self.hidden_size}")
        if tuple(sst.shape) != (1, 2, self.hidden_size):
            raise ValueError(
                f"scale_shift_table must be [1, 2, hidden_size], got {tuple(sst.shape)} vs expected (1, 2, {self.hidden_size})"
            )

        # ConvTranspose1d weight is [in_channels, out_channels, kernel_size]
        if tuple(proj_w.shape) != (self.hidden_size, self.out_channels, self.patch_size):
            raise ValueError(
                "proj_out weight shape mismatch: "
                f"got {tuple(proj_w.shape)} expected ({self.hidden_size}, {self.out_channels}, {self.patch_size})"
            )
        if int(proj_b.shape[0]) != self.out_channels:
            raise ValueError(f"proj_out bias length mismatch: got {proj_b.shape[0]} expected {self.out_channels}")

        with torch.no_grad():
            self.norm_weight.copy_(norm_w.to(device=self.norm_weight.device, dtype=self.norm_weight.dtype))
            self.scale_shift_table.copy_(
                sst.to(device=self.scale_shift_table.device, dtype=self.scale_shift_table.dtype)
            )
            self.proj_out.weight.copy_(proj_w.to(device=self.proj_out.weight.device, dtype=self.proj_out.weight.dtype))
            self.proj_out.bias.copy_(proj_b.to(device=self.proj_out.bias.device, dtype=self.proj_out.bias.dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        meta: PatchifyMetadata,
        *,
        debug: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden_states [B, T_p, H], got {tuple(hidden_states.shape)}")
        if temb.ndim != 2:
            raise ValueError(f"Expected temb [B, H], got {tuple(temb.shape)}")
        if int(hidden_states.shape[-1]) != self.hidden_size:
            raise ValueError(f"Expected hidden_states last dim {self.hidden_size}, got {hidden_states.shape[-1]}")
        if int(temb.shape[-1]) != self.hidden_size:
            raise ValueError(f"Expected temb last dim {self.hidden_size}, got {temb.shape[-1]}")
        if int(meta.patch_size) != self.patch_size:
            raise ValueError(
                f"Patch size mismatch: meta.patch_size={meta.patch_size} vs module.patch_size={self.patch_size}"
            )

        normed = rmsnorm_qwen3(hidden_states, self.norm_weight, self.eps)
        shift = self.scale_shift_table[:, 0:1, :] + temb.unsqueeze(1)
        scale = self.scale_shift_table[:, 1:2, :] + temb.unsqueeze(1)
        modulated = (normed * (1.0 + scale) + shift).type_as(hidden_states)
        if debug is not None and debug.get("enabled", False):
            debug["head.after_norm"] = normed.detach().float().cpu().clone()
            debug["head.modulated_patches"] = modulated.detach().float().cpu().clone()
        # ConvTranspose1d requires activation dtype to match weight dtype.
        if modulated.dtype != self.proj_out.weight.dtype:
            modulated = modulated.to(dtype=self.proj_out.weight.dtype)

        out = depatchify_1d(modulated, convt=self.proj_out, meta=meta, original_seq_len=meta.original_seq_len)
        if debug is not None and debug.get("enabled", False):
            debug["pipe.acoustic"] = out.detach().float().cpu().clone()
        return out
