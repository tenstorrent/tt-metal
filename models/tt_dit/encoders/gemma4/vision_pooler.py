# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 vision pooler.

Mirrors ``Gemma4VisionPooler`` from ``transformers.models.gemma4.modeling_gemma4``:

    1. Zero out padding positions.
    2. If output_length < input_length: 2-D spatial average pool with kernel ``k×k``
       where ``k = sqrt(input_len / output_len)``. Patches grouped by their 2-D
       positions; the average is computed via a one-hot weights matmul.
    3. Multiply by ``sqrt(hidden_size)`` *in fp32* (the scale can push activations
       past fp16 range).
    4. Return ``(hidden_states_fp32, pooler_mask)``. Caller standardizes + casts.

For minimum implementation risk this round, the pooling is computed on host
(torch) since it requires gathered position-based grouping. The compute volume
is modest: per-image, kernel-size 3 → 9-patch averages of 1152-dim vectors.
We round-trip via ``ttnn.to_torch`` / ``ttnn.from_torch``. Optimization
opportunity: move on-device via a one-hot ``ttnn.matmul`` when this becomes a
bottleneck.
"""

from __future__ import annotations

import torch

import ttnn

from ...layers.module import Module


class Gemma4VisionPooler(Module):
    """2-D spatial pooler with sqrt(hidden_size) scaling."""

    def __init__(self, *, hidden_size: int, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.root_hidden_size = hidden_size**0.5
        self.mesh_device = mesh_device

    @staticmethod
    def _avg_pool_by_positions(
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same math as HF's _avg_pool_by_positions (run on host, fp32)."""
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // output_length) ** 0.5)
        k_squared = k**2
        if k_squared * output_length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {output_length}: k²·output_length must equal input length."
            )

        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = torch.nn.functional.one_hot(kernel_idxs.long(), output_length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:       ttnn ``[B, num_patches, hidden_size]`` (replicated, bf16).
            pixel_position_ids:  torch ``[B, num_patches, 2]`` long.
            padding_positions:   torch ``[B, num_patches]`` bool.
            output_length:       target number of soft tokens (e.g. 280).

        Returns:
            (pooled_fp32, pooler_mask): pooled features in fp32 (caller standardizes + casts back),
                                         pooler_mask as bool ``[B, output_length]``.
        """
        # Pull to host for the pooling math.
        h = ttnn.to_torch(hidden_states).to(torch.float32).squeeze(0)  # (B, P, hidden) if mesh dim was unsqueezed
        if h.ndim == 2:
            h = h.unsqueeze(0)

        # Zero padding patches.
        h = h.masked_fill(padding_positions.unsqueeze(-1), 0.0)

        if output_length > h.shape[1]:
            raise ValueError(f"Cannot output more soft tokens ({output_length}) than patches ({h.shape[1]}).")

        if h.shape[1] != output_length:
            h, pooler_mask = self._avg_pool_by_positions(h, pixel_position_ids, output_length)
        else:
            pooler_mask = torch.logical_not(padding_positions)

        # fp32 sqrt(hidden_size) scaling.
        h = h.float() * self.root_hidden_size
        return h, pooler_mask
