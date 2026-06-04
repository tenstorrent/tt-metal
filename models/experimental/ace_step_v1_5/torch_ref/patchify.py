from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PatchifyMetadata:
    """
    Metadata needed to undo patchify padding.

    Matches the behavior used by ACE-Step v1.5's HF model:
    - pad sequence length T to be divisible by patch_size (pad with zeros)
    - run Conv1d with kernel_size=stride=patch_size
    - later de-patchify and crop back to original_seq_len
    """

    original_seq_len: int
    pad_length: int
    patch_size: int


def _patchify_pad_meta(seq_len: int, patch_size: int) -> PatchifyMetadata:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    remainder = int(seq_len) % int(patch_size)
    pad_length = 0 if remainder == 0 else (int(patch_size) - remainder)
    return PatchifyMetadata(original_seq_len=int(seq_len), pad_length=int(pad_length), patch_size=int(patch_size))


def pad_seq_len_to_patch_size(
    x: torch.Tensor, patch_size: int, *, value: float = 0.0
) -> Tuple[torch.Tensor, PatchifyMetadata]:
    """
    Pad `x` along the sequence dimension so its length is divisible by `patch_size`.

    Expected shape: [B, T, C]
    Returns: (padded_x, meta)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x rank-3 [B, T, C], got shape={tuple(x.shape)}")
    meta = _patchify_pad_meta(int(x.shape[1]), int(patch_size))
    if meta.pad_length == 0:
        return x, meta
    # F.pad pads the last dimensions; for [B,T,C], pad tuple is (C_left, C_right, T_left, T_right)
    x = F.pad(x, (0, 0, 0, meta.pad_length), mode="constant", value=float(value))
    return x, meta


def patchify_1d(x: torch.Tensor, *, conv: nn.Conv1d, patch_size: int) -> Tuple[torch.Tensor, PatchifyMetadata]:
    """
    Torch reference for ACE-Step v1.5 `proj_in` patch embedding.

    HF behavior reference (from `AceStepDiTModel.proj_in`):
      x: [B, T, C]
      pad T to multiple of patch_size (zeros)
      x = x.transpose(1, 2)        # [B, C, T]
      x = Conv1d(...)(x)           # [B, H, T_p]
      x = x.transpose(1, 2)        # [B, T_p, H]

    Returns: (patch_tokens [B, T_p, H], meta)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x shape [B, T, C], got {tuple(x.shape)}")
    padded, meta = pad_seq_len_to_patch_size(x, patch_size, value=0.0)
    y = conv(padded.transpose(1, 2)).transpose(1, 2)
    return y, meta


def depatchify_1d(
    x_patches: torch.Tensor,
    *,
    convt: nn.ConvTranspose1d,
    meta: PatchifyMetadata,
    original_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Torch reference for ACE-Step v1.5 `proj_out` de-patchify.

    HF behavior reference (from `AceStepDiTModel.proj_out` + cropping):
      x_patches: [B, T_p, H]
      x = x.transpose(1, 2)              # [B, H, T_p]
      x = ConvTranspose1d(...)(x)        # [B, C_out, T_p * patch_size]
      x = x.transpose(1, 2)              # [B, T, C_out]
      x = x[:, :original_seq_len, :]     # crop away patchify padding
    """
    if x_patches.ndim != 3:
        raise ValueError(f"Expected x_patches shape [B, T_p, H], got {tuple(x_patches.shape)}")
    if meta.patch_size <= 0:
        raise ValueError(f"Invalid meta.patch_size={meta.patch_size}")

    y = convt(x_patches.transpose(1, 2)).transpose(1, 2)
    seq_len = int(meta.original_seq_len if original_seq_len is None else original_seq_len)
    return y[:, :seq_len, :]


class TorchAceStepPatchEmbed1D(nn.Module):
    """
    Convenience wrapper around :func:`patchify_1d`.
    """

    def __init__(self, *, conv: nn.Conv1d, patch_size: int):
        super().__init__()
        self.conv = conv
        self.patch_size = int(patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, PatchifyMetadata]:
        return patchify_1d(x, conv=self.conv, patch_size=self.patch_size)


class TorchAceStepDePatchify1D(nn.Module):
    """
    Convenience wrapper around :func:`depatchify_1d`.
    """

    def __init__(self, *, convt: nn.ConvTranspose1d, patch_size: int):
        super().__init__()
        self.convt = convt
        self.patch_size = int(patch_size)

    def forward(self, x_patches: torch.Tensor, meta: PatchifyMetadata) -> torch.Tensor:
        if int(meta.patch_size) != self.patch_size:
            raise ValueError(
                f"Patch size mismatch: meta.patch_size={meta.patch_size} vs module.patch_size={self.patch_size}"
            )
        return depatchify_1d(x_patches, convt=self.convt, meta=meta)
