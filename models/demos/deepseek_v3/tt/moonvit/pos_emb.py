# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Learnable 2D positional embedding with bicubic interpolation.

The HF reference is `Learnable2DInterpPosEmb` in modeling_kimi_vl.py:
the weight lives at a fixed (H_base, W_base, D) grid (default 64x64 for
MoonViT), and at inference time it is bicubic-interpolated to each
image's actual (H_i, W_i) patch grid. The interpolated tensor is then
added to the patch projection output.

We do the interpolation in torch on CPU and ship a flat
`(sum(H_i*W_i), D)` tensor to the device, where the patch-embedding
add is a plain `ttnn.add`. There is no on-device kernel for bicubic
interpolation — see plan Deferred #2 for the (H, W) cache + device
kernel upgrade path.

Reference (HF Learnable2DInterpPosEmb.forward, paraphrased):
    for shape in grid_hws:
        if shape == self.weight.shape[:-1]:
            pos_embs.append(self.weight.flatten(end_dim=1))
        else:
            pos_embs.append(
                F.interpolate(self.weight.permute(2, 0, 1).unsqueeze(0),
                              size=shape, mode="bicubic")
                .squeeze(0).permute(1, 2, 0).flatten(end_dim=1)
            )
    out = x + torch.cat(pos_embs)
"""
from __future__ import annotations

from typing import Iterable, Sequence, Union

import torch
import torch.nn.functional as F

GridShape = Sequence[int]  # length-2 (H, W)
GridHws = Union[torch.Tensor, Iterable[GridShape]]


def _grid_hws_to_list(grid_hws: GridHws) -> list[tuple[int, int]]:
    """Normalize grid_hws into a list of (H, W) int tuples."""
    if isinstance(grid_hws, torch.Tensor):
        rows = grid_hws.tolist()
    else:
        rows = [tuple(s) for s in grid_hws]
    out: list[tuple[int, int]] = []
    for shape in rows:
        if len(shape) != 2:
            raise ValueError(f"each grid_hws row must have 2 ints (H, W); got {shape}")
        h, w = int(shape[0]), int(shape[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"grid_hws entries must be positive; got ({h}, {w})")
        out.append((h, w))
    return out


class Learnable2DInterpPosEmb:
    """Host-side bicubic interpolation of a (H_base, W_base, D) learned posemb.

    Not a LightweightModule — there is no device state. The output of
    `compute(grid_hws)` is a torch CPU tensor the caller is responsible
    for transferring to device alongside the patch-projection output.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        interpolation_mode: str = "bicubic",
    ):
        if weight.ndim != 3:
            raise ValueError(f"weight must be 3D (H, W, D); got shape {tuple(weight.shape)}")
        # Keep on CPU as fp32 for interpolation accuracy. F.interpolate's bicubic
        # implementation is exact in fp32 and there's no benefit to running this
        # in lower precision — the weight is small (~4.7M elements at D=1152) and
        # we do the cast back to bf16 just before device transfer.
        self.weight_fp32 = weight.detach().to(torch.float32).contiguous()
        self.h_base, self.w_base, self.dim = self.weight_fp32.shape
        self.interpolation_mode = interpolation_mode

    @classmethod
    def from_torch(
        cls,
        ref: torch.nn.Module,
    ) -> "Learnable2DInterpPosEmb":
        """Construct from the HF Learnable2DInterpPosEmb module.

        Pulls `ref.weight` (the nn.Parameter) and `ref.interpolation_mode`.
        """
        if not hasattr(ref, "weight"):
            raise AttributeError(f"expected an HF Learnable2DInterpPosEmb-like module, got {type(ref).__name__}")
        mode = getattr(ref, "interpolation_mode", "bicubic")
        return cls(weight=ref.weight.data, interpolation_mode=mode)

    def _interp_one(self, h: int, w: int) -> torch.Tensor:
        """Interpolate the base weight to (h, w, D), return as (h*w, D)."""
        if (h, w) == (self.h_base, self.w_base):
            # Fast path: matches the HF check `shape == self.weight.shape[:-1]`.
            return self.weight_fp32.flatten(end_dim=1)
        # F.interpolate expects NCHW. We've stored as (H, W, D); permute to
        # (1, D, H, W), interpolate, then permute back to (h, w, D) and flatten.
        chw = self.weight_fp32.permute(2, 0, 1).unsqueeze(0)  # (1, D, H_base, W_base)
        interp = F.interpolate(chw, size=(h, w), mode=self.interpolation_mode)
        return interp.squeeze(0).permute(1, 2, 0).flatten(end_dim=1)  # (h*w, D)

    def compute(self, grid_hws: GridHws, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Compute the concatenated per-image interpolated posemb.

        Args:
            grid_hws: per-image (H, W) shapes — torch tensor of shape [N, 2]
                or any iterable of length-2 sequences.
            dtype: output dtype. Defaults to bfloat16 to match the activation
                stream the result will be added into on device.

        Returns:
            torch CPU tensor of shape (sum(H_i * W_i), D).
        """
        shapes = _grid_hws_to_list(grid_hws)
        if not shapes:
            return torch.empty((0, self.dim), dtype=dtype)
        chunks = [self._interp_one(h, w) for h, w in shapes]
        out = torch.cat(chunks, dim=0)
        if dtype != out.dtype:
            out = out.to(dtype)
        return out

    def apply_torch(
        self,
        x: torch.Tensor,
        grid_hws: GridHws,
    ) -> torch.Tensor:
        """Reference-style apply: add interpolated posembs to a torch input.

        Mirrors the HF `forward(x, grid_hws)` for unit testing.
        """
        pos_embs = self.compute(grid_hws, dtype=x.dtype)
        return x + pos_embs
