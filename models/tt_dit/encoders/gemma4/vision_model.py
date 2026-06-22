# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full Gemma 4 vision tower.

Mirrors ``Gemma4VisionModel`` from ``transformers.models.gemma4.modeling_gemma4``:

    1. patch_embedder(pixels, position_ids, padding_positions) → patch embeddings
    2. encoder: 27 × Gemma4VisionEncoderLayer with multidim RoPE per layer
    3. pooler: 2-D spatial pool to ``output_length`` soft tokens (fp32 scaling)
    4. standardize (subtract std_bias, multiply std_scale) in fp32, cast back to bf16
"""

from __future__ import annotations

import torch

import ttnn

from ...layers.module import Module, ModuleList, Parameter
from ...parallel.config import DiTParallelConfig
from .vision_layer import Gemma4VisionEncoderLayer
from .vision_patch_embedder import Gemma4VisionPatchEmbedder
from .vision_pooler import Gemma4VisionPooler
from .vision_rope import Gemma4VisionRotaryEmbedding


class Gemma4VisionModel(Module):
    """Vision encoder: patch embed → 27 encoder layers → pool → standardize.

    Outputs the pooled & standardized soft tokens (still in vision ``hidden_size`` —
    the multimodal embedder projects them to the text hidden size separately).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        head_dim: int,
        head_dim_padded: int,
        patch_size: int,
        position_embedding_size: int,
        pooling_kernel_size: int,
        default_output_length: int,
        rms_norm_eps: float,
        rope_theta: float,
        standardize: bool,
        mesh_device: ttnn.MeshDevice,
        ccl_manager,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.default_output_length = default_output_length
        self.pooling_kernel_size = pooling_kernel_size
        self.standardize = standardize
        self.mesh_device = mesh_device

        self.patch_embedder = Gemma4VisionPatchEmbedder(
            hidden_size=hidden_size,
            patch_size=patch_size,
            position_embedding_size=position_embedding_size,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        self.rope = Gemma4VisionRotaryEmbedding(
            head_dim=head_dim,
            head_dim_padded=head_dim_padded,
            position_embedding_size=position_embedding_size,
            rope_theta=rope_theta,
            mesh_device=mesh_device,
        )

        self.encoder = ModuleList(
            Gemma4VisionEncoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                head_dim=head_dim,
                head_dim_padded=head_dim_padded,
                rms_norm_eps=rms_norm_eps,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
            )
            for _ in range(num_hidden_layers)
        )

        self.pooler = Gemma4VisionPooler(hidden_size=hidden_size, mesh_device=mesh_device)

        if standardize:
            self.std_bias = Parameter(
                total_shape=[1, hidden_size], device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            self.std_scale = Parameter(
                total_shape=[1, hidden_size], device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # HF stores std_bias / std_scale as 1-D buffers; reshape to [1, hidden_size] for ttnn.
        for name in ("std_bias", "std_scale"):
            if name in state and state[name].ndim == 1:
                state[name] = state[name].reshape(1, -1)

        # HF encoder is stored as `encoder.layers.{i}.*`; our ModuleList is `encoder.{i}.*`.
        # Re-key the substate to match.
        for k in list(state.keys()):
            if k.startswith("encoder.layers."):
                rest = k[len("encoder.layers.") :]
                state[f"encoder.{rest}"] = state.pop(k)

    def forward(
        self,
        pixel_values: ttnn.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values:        ttnn ``[B, num_patches, 3*patch_size**2]`` bf16, already
                                  pixel-normalized to [-1, 1] (host preprocessing).
            pixel_position_ids:  torch ``[B, num_patches, 2]`` long.
            padding_positions:   torch ``[B, num_patches]`` bool.
            output_length:       target soft-token count; defaults to ``default_output_length``.

        Returns:
            torch ``[total_valid_soft_tokens, hidden_size]`` bf16 (HF model also strips padding).
        """
        if output_length is None:
            output_length = self.default_output_length

        # Patch embed (replicated [B, P, hidden_size]).
        h = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)

        # RoPE tables for this batch of positions.
        cos, sin = self.rope.get_cos_sin(pixel_position_ids)

        # Encoder stack. Bidirectional attention (no causal mask).
        for layer in self.encoder:
            h = layer(h, cos, sin, attention_mask=None)

        # Pool (host-side fp32).
        pooled_fp32, pooler_mask = self.pooler(h, pixel_position_ids, padding_positions, output_length)

        # Standardize in fp32, then cast back.
        if self.standardize:
            std_bias = ttnn.to_torch(self.std_bias.data).float().reshape(-1)
            std_scale = ttnn.to_torch(self.std_scale.data).float().reshape(-1)
            pooled_fp32 = (pooled_fp32 - std_bias) * std_scale

        # Strip padding (matches HF) and cast to bf16.
        result = pooled_fp32[pooler_mask].to(torch.bfloat16)
        return result
