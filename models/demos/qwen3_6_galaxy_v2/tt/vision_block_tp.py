# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 (5/N): qwen3.6 vision Block composite.

Structure (mirrors HF `Qwen3VLVisionBlock`):
  x_norm = LayerNorm(x, eps=1e-6)
  x = x + Attention(x_norm, cos, sin)
  x_norm = LayerNorm(x, eps=1e-6)
  x = x + MLP(x_norm)
  return x

Uses the proven TP=8 blocks:
  - `Qwen36VisionAttentionTP` (commit ee639ec3c69, PCC 0.987)
  - `Qwen36VisionMlpTP` (commit 6c460be138a, PCC 0.9999)
  - tt_dit `LayerNorm` with eps=1e-6, bias=True (weights replicated)
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.vision_attention_tp import Qwen36VisionAttentionTP
from models.demos.qwen3_6_galaxy_v2.tt.vision_mlp_tp import Qwen36VisionMlpTP
from models.tt_dit.layers.module import Module
from models.tt_dit.layers.normalization import LayerNorm
from models.tt_dit.parallel.manager import CCLManager


class Qwen36VisionBlockTP(Module):
    """A single qwen3.6 vision block (norm1 → attn → residual → norm2 → mlp → residual)."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        state_dict: dict[str, torch.Tensor],
        *,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_heads: int = 16,
        head_dim: int = 72,
        norm_eps: float = 1e-6,
        tp_mesh_axis: int = 0,
        dtype: ttnn.DataType = ttnn.bfloat16,
        state_dict_prefix: str = "",
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager

        # Pre-strip the user-provided prefix once.
        stripped: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            k2 = k[len(state_dict_prefix) :] if state_dict_prefix and k.startswith(state_dict_prefix) else k
            stripped[k2] = v

        # norm1, norm2: pure LayerNorm with learnable affine
        self.norm1 = LayerNorm(
            embedding_dim=hidden_size,
            norm_eps=norm_eps,
            norm_elementwise_affine=True,
            bias=True,
            mesh_device=mesh_device,
        )
        self.norm2 = LayerNorm(
            embedding_dim=hidden_size,
            norm_eps=norm_eps,
            norm_elementwise_affine=True,
            bias=True,
            mesh_device=mesh_device,
        )

        # Attention block — pass only its sub-state-dict (attn.*) without re-stripping
        attn_state = {k[len("attn.") :]: v for k, v in stripped.items() if k.startswith("attn.")}
        self.attn = Qwen36VisionAttentionTP(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            state_dict=attn_state,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            tp_mesh_axis=tp_mesh_axis,
            dtype=dtype,
        )

        # MLP block — pass only its sub-state-dict (mlp.*) without re-stripping
        mlp_state = {k[len("mlp.") :]: v for k, v in stripped.items() if k.startswith("mlp.")}
        self.mlp = Qwen36VisionMlpTP(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            state_dict=mlp_state,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_mesh_axis=tp_mesh_axis,
            dtype=dtype,
        )

        # Load only the norm1/norm2 weights — attn and mlp already loaded
        # their own state during their constructors. Use strict=False so the
        # outer block's load_state_dict doesn't complain about the already-
        # loaded sub-module params.
        norm_state = {f"norm1.{k[len('norm1.') :]}": v for k, v in stripped.items() if k.startswith("norm1.")}
        norm_state.update({f"norm2.{k[len('norm2.') :]}": v for k, v in stripped.items() if k.startswith("norm2.")})
        self.load_torch_state_dict(norm_state, strict=False)

    def forward(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass: norm + attn + residual + norm + mlp + residual."""
        # x: [B, 1, S, H] or [B, S, H] replicated input
        # Ensure 4D for tt_dit LayerNorm
        needs_unsqueeze_x = len(x.shape) == 3
        if needs_unsqueeze_x:
            x = ttnn.unsqueeze(x, 0)

        # Attention sub-block
        x_norm = self.norm1.forward(x)
        attn_out = self.attn.forward(x_norm, cos, sin)
        ttnn.deallocate(x_norm)
        # attn returns 3D; restore 4D for residual add
        if len(attn_out.shape) == 3:
            attn_out = ttnn.unsqueeze(attn_out, 0)
        x = ttnn.add(x, attn_out)
        ttnn.deallocate(attn_out)

        # MLP sub-block
        x_norm = self.norm2.forward(x)
        mlp_out = self.mlp.forward(x_norm)
        ttnn.deallocate(x_norm)
        if len(mlp_out.shape) == 3:
            mlp_out = ttnn.unsqueeze(mlp_out, 0)
        x = ttnn.add(x, mlp_out)
        ttnn.deallocate(mlp_out)

        if needs_unsqueeze_x:
            x = ttnn.squeeze(x, 0)
        return x
