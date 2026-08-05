# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""InterpolateRegulator: resample token-rate features to mel rate.

The reference resamples with `F.interpolate(mode='linear')` to an *arbitrary*
output length, then runs 4x (Conv1d k=3, GroupNorm(1, C), Mish) and a final
Conv1d k=1.

`ttnn.upsample` is the obvious candidate for the resample and it is the wrong
one: it takes a scale factor, whereas this needs an arbitrary (T_in -> T_out)
mapping — 174 -> 348 for the prompt and 164 -> 260 for the generated part in the
captured utterance, neither an integer ratio.

**So linear interpolation is expressed as a matmul.** For fixed (T_in, T_out) it
is a fixed, extremely sparse `[T_out, T_in]` matrix with two non-zeros per row,
built once on host. That is the same move the iSTFT makes with the DFT basis: an
awkward op becomes a constant matrix, and matmul is the widest unit on the core.

The head/mid/tail split is preserved exactly. The reference interpolates the
first 20 and last 20 tokens separately from the middle whenever the segment
exceeds 40 tokens, so that streaming chunks have a clean seam. Collapsing that
into one resample would change the output at the boundaries — subtly, and only
in streaming, which is precisely the failure R3 exists to catch.
"""
from __future__ import annotations

import torch

import ttnn


def linear_resample_matrix(t_in: int, t_out: int, dtype=torch.float32) -> torch.Tensor:
    """`F.interpolate(..., size=t_out, mode='linear', align_corners=False)` as a
    `[t_out, t_in]` matrix.

    align_corners=False maps output index i to source coordinate
    `(i + 0.5) * t_in/t_out - 0.5`, clamped at the edges. Two non-zeros per row.
    """
    if t_in == t_out:
        return torch.eye(t_out, dtype=dtype)
    scale = t_in / t_out
    m = torch.zeros(t_out, t_in, dtype=dtype)
    idx = (torch.arange(t_out, dtype=torch.float64) + 0.5) * scale - 0.5
    idx = idx.clamp(min=0.0)
    lo = idx.floor().long().clamp(max=t_in - 1)
    hi = (lo + 1).clamp(max=t_in - 1)
    frac = (idx - lo.double()).to(dtype)
    rows = torch.arange(t_out)
    m[rows, lo] += 1.0 - frac
    m[rows, hi] += frac
    return m


def torch_resample(x: torch.Tensor, t_out: int) -> torch.Tensor:
    """[B, T_in, C] -> [B, T_out, C] via the matrix form (host reference)."""
    m = linear_resample_matrix(x.shape[1], t_out).to(x.dtype)
    return torch.einsum("oi,bic->boc", m, x)


class TtInterpolateRegulator:
    """Resample + a small conv stack. Tensors are `[B, T, C]` channels-last."""

    OVERLAP = 20  # token_overlap_len in cosyvoice/cli/model.py

    def __init__(self, device, bag, channels: int = 80, dtype=ttnn.bfloat16):
        from ..hifigan.conv import TtConv1d
        from .estimator import TtGroupNorm

        self.device, self.dtype, self.channels = device, dtype, channels
        # model is [Conv, GroupNorm, Mish] * 4 then a final Conv1d(k=1).
        # Convs sit at the indices that actually carry a weight of rank 3.
        self.convs, self.norms = [], []
        i = 0
        while bag.sub(f"model.{i}").has("weight") or bag.sub(f"model.{i + 1}").has("weight"):
            sub = bag.sub(f"model.{i}")
            if not sub.has("weight"):
                break
            w = sub.tensor("weight")
            if w.dim() == 3:  # Conv1d
                k = w.shape[-1]
                self.convs.append(TtConv1d(device, w, sub.optional("bias"), padding=(k - 1) // 2, dtype=dtype))
                nsub = bag.sub(f"model.{i + 1}")
                if nsub.has("weight") and nsub.tensor("weight").dim() == 1:
                    # groups=1 here, so the statistic spans all 80 channels and the
                    # whole segment jointly. Same TtGroupNorm as the estimator uses --
                    # see its docstring for why that beats ttnn.group_norm.
                    self.norms.append(TtGroupNorm(device, bag, f"model.{i + 1}", num_groups=1, dtype=dtype))
                    i += 3  # Conv, GroupNorm, Mish
                else:
                    self.norms.append(None)
                    i += 1
            else:
                i += 1
            if i > 32:
                break

    def resample(self, x, t_in: int, t_out: int, batch: int = 1):
        """One matmul against a constant [t_out, t_in] interpolation matrix."""
        if t_in == t_out:
            return x
        m = ttnn.from_torch(
            linear_resample_matrix(t_in, t_out).unsqueeze(0),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        out = ttnn.matmul(m, x)  # [1, t_out, t_in] @ [B, t_in, C]
        ttnn.deallocate(m)
        return out

    def resample_split(self, x, t_in: int, t_out: int, frame_rate: int = 50, batch: int = 1):
        """Head/mid/tail resample, matching the reference's streaming seam.

        Segments longer than 2*OVERLAP get their first and last 20 tokens
        resampled independently so a streaming chunk boundary lands cleanly.
        """
        edge = int(self.OVERLAP / frame_rate * 22050 / 256)
        if t_in <= 2 * self.OVERLAP:
            return self.resample(x, t_in, t_out, batch)

        c = x.shape[-1]
        head = ttnn.slice(x, [0, 0, 0], [batch, self.OVERLAP, c])
        mid = ttnn.slice(x, [0, self.OVERLAP, 0], [batch, t_in - self.OVERLAP, c])
        tail = ttnn.slice(x, [0, t_in - self.OVERLAP, 0], [batch, t_in, c])

        rh = self.resample(head, self.OVERLAP, edge, batch)
        rm = self.resample(mid, t_in - 2 * self.OVERLAP, t_out - 2 * edge, batch)
        rt = self.resample(tail, self.OVERLAP, edge, batch)
        for t in (head, mid, tail):
            ttnn.deallocate(t)
        out = ttnn.concat([rh, rm, rt], dim=1)
        for t in (rh, rm, rt):
            ttnn.deallocate(t)
        return out

    def __call__(self, x, length: int, batch: int = 1):
        """Run the conv stack over an already-resampled `[B, T, C]` tensor."""
        h = x
        for conv, norm in zip(self.convs, self.norms):
            out, _ = conv(h, length, batch)
            if h is not x:  # the caller's tensor is not ours to free
                ttnn.deallocate(h)
            h = out
            if norm is not None:
                n = norm(h)
                ttnn.deallocate(h)
                h = ttnn.mish(n)
                ttnn.deallocate(n)
        return h

    @staticmethod
    def torch_reference_resample(x1, x2, mel_len1: int, mel_len2: int, frame_rate: int = 50):
        """cosyvoice.flow.length_regulator.InterpolateRegulator.inference, resample
        part only, in [B, T, C]."""
        overlap = TtInterpolateRegulator.OVERLAP
        edge = int(overlap / frame_rate * 22050 / 256)
        if x2.shape[1] > 2 * overlap:
            parts = [
                torch_resample(x2[:, :overlap], edge),
                torch_resample(x2[:, overlap:-overlap], mel_len2 - 2 * edge),
                torch_resample(x2[:, -overlap:], edge),
            ]
            r2 = torch.cat(parts, dim=1)
        else:
            r2 = torch_resample(x2, mel_len2)
        if x1.shape[1] != 0:
            return torch.cat([torch_resample(x1, mel_len1), r2], dim=1)
        return r2
