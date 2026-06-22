# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 Gemma FeatureExtractorV2 (on device): per-token RMS-norm of the 49 Gemma
hidden states, concat, rescale, and dual aggregate_embed projection into the
video/audio feature dims the connectors consume.

Reference: ltx_core.text_encoders.gemma.feature_extractor.FeatureExtractorV2
"""

from __future__ import annotations

import math

import torch

import ttnn

from ...layers.linear import Linear
from ...layers.module import Module


class GemmaFeatureExtractor(Module):
    """Per-token RMS norm + rescale + dual aggregate_embed (video/audio)."""

    def __init__(
        self,
        *,
        input_dim: int,  # 188160 = gemma_num_layers * gemma_hidden_size
        embedding_dim: int,  # 3840 — rescale source dim (Gemma hidden size)
        video_dim: int,  # 4096
        audio_dim: int | None,  # 2048 (av) or None
        mesh_device=None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.mesh_device = mesh_device
        # HiFi4 + fp32 dest-acc for the per-token RMS norms (native default fidelity costs ~1e-3 PCC).
        self.rmsnorm_cc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.video_aggregate_embed = Linear(input_dim, video_dim, bias=True, mesh_device=mesh_device)
        self.audio_aggregate_embed = (
            Linear(input_dim, audio_dim, bias=True, mesh_device=mesh_device) if audio_dim is not None else None
        )

    @staticmethod
    def _weight_to_layer_major(weight: torch.Tensor, hidden_size: int, num_layers: int) -> torch.Tensor:
        """Reorder aggregate_embed input columns D-major → layer-major.

        The checkpoint weight expects features flattened D-major ([B,T,D,L] → column d*L+l).
        The on-device path concatenates the per-layer-normed states layer-major (column l*D+d)
        to avoid a non-tile-aligned L permute, so the weight is permuted once here (column
        l*D+d ← d*L+l) to consume that order directly.
        """
        out_dim = weight.shape[0]
        return (
            weight.reshape(out_dim, hidden_size, num_layers).permute(0, 2, 1).reshape(out_dim, hidden_size * num_layers)
        )

    def _normed_concat(self, hidden_states: list[ttnn.Tensor], attention_mask: torch.Tensor) -> ttnn.Tensor:
        """Per-token RMS norm of each hidden state over its hidden dim, concatenated layer-major,
        with padded tokens zeroed — matches FeatureExtractorV2.norm_and_concat_per_token_rms."""
        normed = [
            ttnn.experimental.dit_rms_norm_unary_fused(
                hs, weight=None, epsilon=1e-6, compute_kernel_config=self.rmsnorm_cc
            )
            for hs in hidden_states
        ]
        tt_normed = ttnn.concat(normed, dim=-1)
        for hs in normed:
            ttnn.deallocate(hs)

        tt_mask = ttnn.from_torch(
            attention_mask.to(torch.float32).unsqueeze(-1),  # (B, T, 1)
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        out = ttnn.multiply(tt_normed, tt_mask)
        ttnn.deallocate(tt_normed)
        ttnn.deallocate(tt_mask)
        return out

    def _aggregate(self, aggregate_embed: Linear, tt_normed: ttnn.Tensor, out_dim: int) -> ttnn.Tensor:
        """Rescale by sqrt(out_dim / embedding_dim) then project. tt_normed is shared across
        the video/audio calls, so multiply produces a fresh tensor and leaves it intact."""
        rescaled = ttnn.multiply(tt_normed, math.sqrt(out_dim / self.embedding_dim))
        out = aggregate_embed(rescaled)
        ttnn.deallocate(rescaled)
        return out

    def forward(
        self, hidden_states: list[ttnn.Tensor], attention_mask: torch.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """Returns (video_features, audio_features) at the connector input dims."""
        tt_normed = self._normed_concat(hidden_states, attention_mask)
        video = self._aggregate(self.video_aggregate_embed, tt_normed, self.video_dim)
        audio = (
            self._aggregate(self.audio_aggregate_embed, tt_normed, self.audio_dim)
            if self.audio_aggregate_embed is not None
            else None
        )
        ttnn.deallocate(tt_normed)
        return video, audio
