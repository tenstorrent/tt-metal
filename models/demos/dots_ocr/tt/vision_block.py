# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Vision Block for Dots OCR Vision Transformer.

Combines RMSNorm + VisionAttention + VisionMLP for each of the 42 layers.
Uses post-norm architecture as specified in DotsVisionConfig.
"""

from __future__ import annotations

import torch

from models.common.lightweightmodule import LightweightModule
from models.demos.dots_ocr.tt._ttnn_import import get_ttnn
from models.demos.dots_ocr.tt.vision_attention import VisionAttentionTT
from models.demos.dots_ocr.tt.vision_mlp import VisionMLPTT
from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs
from models.demos.dots_ocr.tt.vision_rmsnorm import RMSNorm


class VisionBlockTT(LightweightModule):
    """
    TTNN Vision Block for Dots OCR.

    Architecture: Post-Norm (RMSNorm -> Attention -> RMSNorm -> MLP)
    This matches the DotsVisionConfig.post_norm = True setting.
    """

    def __init__(
        self,
        mesh_device,
        model_args: DotsVisionModelArgs,
        state_dict: dict,
        layer_num: int,
        weight_cache_path=None,
        dtype=None,
    ):
        super().__init__()
        ttnn = get_ttnn()
        if dtype is None:
            dtype = ttnn.bfloat16 if ttnn is not None else torch.bfloat16
        self.mesh_device = mesh_device
        self.model_args = model_args
        self.layer_num = layer_num
        self.dtype = dtype

        # Post-norm architecture: norm -> attention/mlp -> residual
        self.norm1 = RMSNorm(
            device=mesh_device,
            dim=model_args.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=model_args.get_state_dict_prefix("VisionBlock", layer_num),
            weight_key="norm1",
            weight_dtype=dtype,
            eps=model_args.rms_norm_eps,
        )

        self.norm2 = RMSNorm(
            device=mesh_device,
            dim=model_args.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=model_args.get_state_dict_prefix("VisionBlock", layer_num),
            weight_key="norm2",
            weight_dtype=dtype,
            eps=model_args.rms_norm_eps,
        )

        self.attention = VisionAttentionTT(
            mesh_device=mesh_device,
            model_args=model_args,
            state_dict=state_dict,
            layer_num=layer_num,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

        self.mlp = VisionMLPTT(
            mesh_device=mesh_device,
            model_args=model_args,
            state_dict=state_dict,
            layer_num=layer_num,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

    def forward(
        self,
        x,
        rot_mats: tuple[object, object] | None = None,
        **kwargs,
    ):
        """
        Vision block forward with post-norm architecture.

        x -> norm1 -> attention -> residual -> norm2 -> mlp -> residual
        """
        ttnn = get_ttnn()
        if ttnn is None:
            raise RuntimeError("VisionBlockTT requires ttnn")
        if not isinstance(x, ttnn.Tensor):
            raise TypeError(f"Expected ttnn.Tensor, got {type(x)}")

        residual = x
        x_norm = self.norm1(x)
        x_attn = self.attention(x_norm, rot_mats=rot_mats, cu_seqlens=kwargs.get("cu_seqlens"))
        x = ttnn.add(residual, x_attn)
        x_norm2 = self.norm2(x)
        x_mlp = self.mlp(x_norm2)
        return ttnn.add(x, x_mlp)


# Convenience function
def create_vision_block(mesh_device, model_args, state_dict, layer_num, weight_cache_path=None, dtype=None):
    """Create VisionBlockTT instance."""
    return VisionBlockTT(
        mesh_device=mesh_device,
        model_args=model_args,
        state_dict=state_dict,
        layer_num=layer_num,
        weight_cache_path=weight_cache_path,
        dtype=dtype,
    )
