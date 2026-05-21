# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Deformable Transformer Encoder Layer in ttnn.

Each layer: MSDeformAttn(self-attention) → residual + LayerNorm → FFN → residual + LayerNorm
FFN: Linear(256→1024) → ReLU → Linear(1024→256)
Dropout is skipped for inference.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_ms_deform_attn import TTMSDeformAttn


class TTDeformableEncoderLayer(LightweightModule):

    def __init__(self, device, state_dict, prefix, d_model=256, d_ffn=1024, n_levels=5, n_heads=8, n_points=4):
        super().__init__()
        self.device = device
        self.d_model = d_model
        p = f"{prefix}." if prefix else ""

        self.self_attn = TTMSDeformAttn(
            device, state_dict, f"{p}self_attn", d_model, n_heads, n_levels, n_points
        )

        self.norm1_w = self._param(state_dict[f"{p}norm1.weight"])
        self.norm1_b = self._param(state_dict[f"{p}norm1.bias"])
        self.norm2_w = self._param(state_dict[f"{p}norm2.weight"])
        self.norm2_b = self._param(state_dict[f"{p}norm2.bias"])

        self.ffn1_w = self._weight(state_dict[f"{p}linear1.weight"])
        self.ffn1_b = self._bias(state_dict[f"{p}linear1.bias"])
        self.ffn2_w = self._weight(state_dict[f"{p}linear2.weight"])
        self.ffn2_b = self._bias(state_dict[f"{p}linear2.bias"])

    def _weight(self, w):
        return ttnn.from_torch(
            w.T.contiguous().to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _bias(self, b):
        return ttnn.from_torch(
            b.unsqueeze(0).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _param(self, t):
        return ttnn.from_torch(
            t.unsqueeze(0).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index,
                key_padding_mask=None, normalizer=None, ref_expanded=None):
        """
        Args:
            src: ttnn (N, Len_in, d_model)
            pos: ttnn (N, Len_in, d_model) — positional embedding
            reference_points: torch (N, Len_in, n_levels, 2)
            spatial_shapes: torch (n_levels, 2)
            level_start_index: torch (n_levels,)
            key_padding_mask: optional torch (N, Len_in)
            normalizer: optional precomputed torch (n_levels, 2) float
            ref_expanded: optional precomputed torch (N, Lq, 1, n_levels, 1, 2) float
        Returns:
            ttnn (N, Len_in, d_model)
        """
        query = ttnn.add(src, pos)
        src2 = self.self_attn(
            query, reference_points, src, spatial_shapes, level_start_index,
            key_padding_mask, normalizer=normalizer, ref_expanded=ref_expanded,
        )
        ttnn.deallocate(query)

        src = ttnn.add(src, src2)
        ttnn.deallocate(src2)
        src = ttnn.layer_norm(src, weight=self.norm1_w, bias=self.norm1_b)

        ffn = ttnn.linear(src, self.ffn1_w, bias=self.ffn1_b)
        ffn = ttnn.relu(ffn)
        ffn = ttnn.linear(ffn, self.ffn2_w, bias=self.ffn2_b)
        src = ttnn.add(src, ffn)
        ttnn.deallocate(ffn)
        src = ttnn.layer_norm(src, weight=self.norm2_w, bias=self.norm2_b)

        return src


class TTDeformableEncoder(LightweightModule):

    def __init__(self, device, state_dict, prefix, n_layers=6, d_model=256, d_ffn=1024,
                 n_levels=5, n_heads=8, n_points=4):
        super().__init__()
        p = f"{prefix}." if prefix else ""
        self.layers = []
        for i in range(n_layers):
            self.layers.append(
                TTDeformableEncoderLayer(
                    device, state_dict, f"{p}{i}",
                    d_model, d_ffn, n_levels, n_heads, n_points
                )
            )

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        import torch as _torch

        normalizer = _torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
        ).float()
        ref_expanded = reference_points.float()[:, :, None, :, None, :]

        output = src
        for layer in self.layers:
            output = layer(
                output, pos, reference_points, spatial_shapes, level_start_index,
                key_padding_mask, normalizer=normalizer, ref_expanded=ref_expanded,
            )
        return output
