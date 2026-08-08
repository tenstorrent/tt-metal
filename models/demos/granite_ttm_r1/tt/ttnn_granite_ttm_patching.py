# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class TtnnGraniteTTMPatching:
    """Implements TinyTimeMixerPatchify with TTNN.

    Input shape:  [B, T, C]  where T = num_patches * patch_length
    Output shape: [B, C, num_patches, patch_length]

    TinyTimeMixerPatchify.forward for stride == patch_length (non-overlapping)
    is equivalent to:
        output = past_values[:, sequence_start:, :]
        output = output.unfold(dim=-2, size=patch_length, step=patch_stride)
        # → [B, num_patches, C, patch_length]
        output = output.transpose(-2, -3)
        # → [B, C, num_patches, patch_length]

    For stride == patch_length this reduces to a reshape + permute:
        [B, T, C] → reshape → [B, P, pl, C] → permute(0,3,1,2) → [B, C, P, pl]
    """

    def __init__(self, *, num_patches: int, patch_length: int, config=None):
        self._num_patches = num_patches
        self._patch_length = patch_length

    def __call__(self, history, *, device=None, **kwargs):
        return self._forward_ttnn(history, device=device)

    def _forward_ttnn(self, history, *, device):
        import ttnn

        B, T, C = history.shape
        P = self._num_patches
        pl = self._patch_length

        # [B, T, C] → [B, P, pl, C] using reshape (requires ROW_MAJOR)
        history = ttnn.to_layout(history, ttnn.ROW_MAJOR_LAYOUT)
        history = ttnn.reshape(history, [B, P, pl, C])

        # [B, P, pl, C] → [B, C, P, pl] via permute
        history = ttnn.to_layout(history, ttnn.TILE_LAYOUT)
        history = ttnn.permute(history, (0, 3, 1, 2))

        return history
