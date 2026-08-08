# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
2D RoPE for MoonViT.

This implements the interleaved 2D rotary-positional-embedding scheme
from MoonViT's `Rope2DPosEmb` (modeling_kimi_k25.py). For each token
position with grid coords (y, x), the per-head head_dim is rotated
pair-wise:

    head_dim is split into pairs of 4 real values [a, b, c, d].
    The (a, b) pair is rotated by angle x * theta_k.
    The (c, d) pair is rotated by angle y * theta_k.
    The next group of 4 uses k+1, and so on.

    With theta_k = theta_base ** (-4k / head_dim) for k in [0, head_dim/4).

The HF reference returns a complex tensor of shape (L, head_dim/2)
that gets multiplied by per-head Q/K reshaped as complex pairs.
Step 6 (this file) verifies that our host-side `get_freqs_cis`
matches HF. Step 7 (attention) handles the on-device application.

Comparison-point precedents in the repo:
  - Pixtral 2D RoPE (`tt_transformers/tt/multimodal/mistral_24b/vision_rope.py`)
    uses a SPLIT-HALF scheme (first half of head_dim for one axis, second
    half for the other). MoonViT INTERLEAVES x and y at the pair-of-pairs
    granularity, so we cannot reuse precompute_mistral_vision_freqs directly.
  - Qwen2.5-VL similarly interleaves (different from MoonViT in detail
    but same complex-multiplication idiom).
"""
from __future__ import annotations

from typing import Optional

import torch

from models.demos.deepseek_v3.tt.moonvit.pos_emb import GridHws, _grid_hws_to_list


class Rope2DSetup:
    """Host-side precompute of the 2D RoPE freqs.

    Mirrors HF Rope2DPosEmb: precomputes `freqs_cis` for the maximum
    (max_height, max_width) grid once, then slices per-image on each
    `get_freqs_cis(grid_hws)` call.

    Attribute names match the HF class for clarity.
    """

    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base: float = 10000.0,
    ):
        if dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE; got {dim}")
        self.dim = int(dim)
        self.max_height = int(max_height)
        self.max_width = int(max_width)
        self.theta_base = float(theta_base)
        self._freqs_cis: Optional[torch.Tensor] = None

    @classmethod
    def from_torch(cls, ref: torch.nn.Module) -> "Rope2DSetup":
        """Construct from the HF Rope2DPosEmb module."""
        return cls(
            dim=int(ref.dim),
            max_height=int(ref.max_height),
            max_width=int(ref.max_width),
            theta_base=float(ref.theta_base),
        )

    # ------------------------------------------------------------------
    # Base precompute, identical to HF._precompute_freqs_cis but always on CPU.

    def _precompute_freqs_cis(self) -> torch.Tensor:
        """Build the (max_h, max_w, dim/2) complex freqs tensor.

        Layout (matches HF):
          For position (y, x), pair index j in [0, dim/4):
            freqs_cis[y, x, 2j]   = cis(x * theta_base ** (-4j / dim))
            freqs_cis[y, x, 2j+1] = cis(y * theta_base ** (-4j / dim))
        where cis(t) = cos(t) + i sin(t).
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N, dtype=torch.float32)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width

        # dim_range: 0, 4, 8, ... up to dim - 4. Length dim/4.
        dim_range = torch.arange(0, self.dim, 4, dtype=torch.float32)[: self.dim // 4]
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))  # (dim/4,)

        x_freqs = torch.outer(x_pos, freqs)  # (N, dim/4)
        y_freqs = torch.outer(y_pos, freqs)  # (N, dim/4)
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # (N, dim/4) complex
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)

        # Interleave x and y in the last dim: (N, dim/4, 2) -> (N, dim/2).
        interleaved = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
        freqs_cis = interleaved.reshape(self.max_height, self.max_width, -1)
        return freqs_cis  # (max_h, max_w, dim/2)

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._freqs_cis is None:
            self._freqs_cis = self._precompute_freqs_cis()
        return self._freqs_cis

    # ------------------------------------------------------------------
    # Per-image slicing.

    def get_freqs_cis(self, grid_hws: GridHws) -> torch.Tensor:
        """Slice the precomputed table to match grid_hws.

        Args:
            grid_hws: per-image (H, W) sizes — torch tensor [N, 2] or iterable.

        Returns:
            complex64 tensor of shape (sum(H_i * W_i), dim/2).
        """
        shapes = _grid_hws_to_list(grid_hws)
        for h, w in shapes:
            if not (1 <= h <= self.max_height and 1 <= w <= self.max_width):
                raise ValueError(
                    f"grid_hws entry ({h}, {w}) exceeds Rope2DSetup max " f"({self.max_height}, {self.max_width})"
                )
        table = self.freqs_cis  # (max_h, max_w, dim/2)
        chunks = [table[:h, :w].reshape(-1, self.dim // 2) for h, w in shapes]
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    # Real-form cos/sin for on-device rotary apply (used by step 7 attention).

    def get_cos_sin(self, grid_hws: GridHws, dtype: torch.dtype = torch.bfloat16):
        """Return (cos, sin) real tensors derived from the complex freqs.

        Layout: cos/sin have shape (L, dim). For each head_dim position d
        (0-indexed):
            pair_idx = d // 2        in [0, dim/2)
            cos[l, d] = freqs_cis[l, pair_idx].real
            sin[l, d] = freqs_cis[l, pair_idx].imag
        i.e., each consecutive pair (d, d+1) shares the same cos/sin
        value — the standard "consecutive-pair RoPE" layout that
        `ttnn.experimental.rotary_embedding_llama` consumes.

        The step-7 attention test will verify that applying this with
        ttnn rotary_embedding_llama matches HF's apply_rope numerically.
        """
        complex_freqs = self.get_freqs_cis(grid_hws)  # (L, dim/2) complex
        cos = complex_freqs.real.repeat_interleave(2, dim=-1).to(dtype)  # (L, dim)
        sin = complex_freqs.imag.repeat_interleave(2, dim=-1).to(dtype)  # (L, dim)
        return cos, sin
