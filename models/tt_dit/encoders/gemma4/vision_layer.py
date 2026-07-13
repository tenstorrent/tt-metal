# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 vision encoder layer.

Mirrors ``Gemma4VisionEncoderLayer`` from ``transformers.models.gemma4.modeling_gemma4``:
the Gemma-3-style norm sandwich, no MoE, no per-layer scalar.

    residual = x
    h = input_layernorm(x)
    h = self_attn(h, ...)
    h = post_attention_layernorm(h)
    h = residual + h

    residual = h
    h = pre_feedforward_layernorm(h)
    h = mlp(h)
    h = post_feedforward_layernorm(h)
    h = residual + h
"""

from __future__ import annotations

import ttnn

from ...layers.feedforward import GatedMLP
from ...layers.module import Module
from ...layers.normalization import RMSNorm
from ...parallel.config import DiTParallelConfig
from .vision_attention import Gemma4VisionAttention


class Gemma4VisionEncoderLayer(Module):
    """One vision encoder layer (norm-attn-norm + residual, norm-mlp-norm + residual)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        head_dim: int,
        head_dim_padded: int,
        rms_norm_eps: float,
        mesh_device: ttnn.MeshDevice,
        ccl_manager,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.self_attn = Gemma4VisionAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            head_dim_padded=head_dim_padded,
            rms_norm_eps=rms_norm_eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )
        self.mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        norm_kwargs = dict(
            embedding_dim=hidden_size,
            norm_eps=rms_norm_eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.input_layernorm = RMSNorm(**norm_kwargs)
        self.post_attention_layernorm = RMSNorm(**norm_kwargs)
        self.pre_feedforward_layernorm = RMSNorm(**norm_kwargs)
        self.post_feedforward_layernorm = RMSNorm(**norm_kwargs)

    def _prepare_torch_state(self, state: dict) -> None:
        """Rewrite HF's ``mlp.<proj>.linear.<name>`` keys to our flat ``mlp.<proj>.<name>``.

        HF's Gemma4 vision MLP wraps each Linear in a small container whose actual
        weight lives under ``.linear`` (same pattern as Gemma4VisionAttention). The MLP is
        our tt_dit ``GatedMLP`` which expects flat ``gate_proj.weight`` / ``up_proj.weight`` /
        ``down_proj.weight``, so strip the extra ``.linear.`` segment before descending.
        """
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "bias"):
                src = f"mlp.{proj}.linear.{suffix}"
                if src in state:
                    state[f"mlp.{proj}.{suffix}"] = state.pop(src)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Args:
            hidden_states: replicated [B, num_patches, hidden_size]
            cos, sin:      [B, 1, num_patches, head_dim_padded/2] from vision rope
            attention_mask: optional bf16 additive mask (None → unmasked).
        """
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, cos, sin, attention_mask=attention_mask)
        h = self.post_attention_layernorm(h)
        hidden_states = ttnn.add(residual, h)
        ttnn.deallocate(h)

        residual = hidden_states
        h = self.pre_feedforward_layernorm(hidden_states)
        h = self.mlp(h)
        # GatedMLP output is TP-fractured on the hidden axis (RowParallel's reduce_scatter);
        # gather back to replicated so the post-FF norm reduction sees the full hidden dim
        # and the residual add matches ``residual``'s replicated layout.
        if self.parallel_config.tensor_parallel.factor > 1:
            h = self.ccl_manager.all_gather_persistent_buffer(
                h, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        h = self.post_feedforward_layernorm(h)
        hidden_states = ttnn.add(residual, h)
        ttnn.deallocate(h)
        return hidden_states
