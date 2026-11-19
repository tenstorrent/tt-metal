# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure-torch reference of the Alpamayo 2 Edge SigLIP2 vision tower and patch-merger projector.

Ported from the Cosmos3 Edge modeling code so that it carries no transformers dependency; it is the golden
the TT vision tower is compared against.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.alpamayo2_edge.reference.config import ProjectorConfig, VisionConfig


def vision_cu_seqlens(grid_thw: torch.Tensor) -> list[int]:
    """Cumulative patch counts delimiting each image or video frame in the packed sequence."""
    cu = [0]
    for t, h, w in grid_thw.tolist():
        for _ in range(t):
            cu.append(cu[-1] + h * w)
    return cu


def vision_position_embeddings(pos_weight: torch.Tensor, grid_thw: torch.Tensor, merge: int) -> torch.Tensor:
    """Resize the learned square position grid to each image and pack it in block-major 2x2 order."""
    side = int(round(pos_weight.shape[0] ** 0.5))
    source_dtype = pos_weight.dtype
    # antialiased bilinear is only implemented for fp32 on CPU
    grid = pos_weight.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0).float()

    chunks = []
    for t, h, w in grid_thw.tolist():
        resized = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        resized = resized.squeeze(0).permute(1, 2, 0).to(source_dtype)
        resized = resized.reshape(h // merge, merge, w // merge, merge, -1)
        resized = resized.transpose(1, 2).reshape(h * w, -1)
        chunks.append(resized.repeat(t, 1))
    return torch.cat(chunks, dim=0)


def _linear(weights: dict, name: str, dtype: torch.dtype) -> nn.Linear:
    w = weights[f"{name}.weight"]
    layer = nn.Linear(w.shape[1], w.shape[0])
    layer.weight.data = w.to(dtype)
    layer.bias.data = weights[f"{name}.bias"].to(dtype)
    return layer


def _layer_norm(weights: dict, name: str, eps: float, dtype: torch.dtype) -> nn.LayerNorm:
    w = weights[f"{name}.weight"]
    norm = nn.LayerNorm(w.shape[0], eps=eps)
    norm.weight.data = w.to(dtype)
    norm.bias.data = weights[f"{name}.bias"].to(dtype)
    return norm


class VisionBlock(nn.Module):
    def __init__(self, vcfg: VisionConfig, weights: dict, prefix: str, dtype: torch.dtype):
        super().__init__()
        self.heads = vcfg.heads
        self.head_dim = vcfg.hidden // vcfg.heads
        self.layer_norm1 = _layer_norm(weights, f"{prefix}layer_norm1", vcfg.eps, dtype)
        self.layer_norm2 = _layer_norm(weights, f"{prefix}layer_norm2", vcfg.eps, dtype)
        self.q_proj = _linear(weights, f"{prefix}self_attn.q_proj", dtype)
        self.k_proj = _linear(weights, f"{prefix}self_attn.k_proj", dtype)
        self.v_proj = _linear(weights, f"{prefix}self_attn.v_proj", dtype)
        self.out_proj = _linear(weights, f"{prefix}self_attn.out_proj", dtype)
        self.fc1 = _linear(weights, f"{prefix}mlp.fc1", dtype)
        self.fc2 = _linear(weights, f"{prefix}mlp.fc2", dtype)

    def attention(self, x: torch.Tensor, cu_seqlens: list[int]) -> torch.Tensor:
        n = x.shape[0]
        q, k, v = (
            proj(x).reshape(n, self.heads, self.head_dim).transpose(0, 1)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        outs = [
            F.scaled_dot_product_attention(q[:, s:e], k[:, s:e], v[:, s:e], scale=self.head_dim**-0.5)
            for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:])
        ]
        return self.out_proj(torch.cat(outs, dim=1).transpose(0, 1).reshape(n, -1))

    def forward(self, x: torch.Tensor, cu_seqlens: list[int]) -> torch.Tensor:
        x = x + self.attention(self.layer_norm1(x), cu_seqlens)
        h = self.fc2(F.gelu(self.fc1(self.layer_norm2(x)), approximate="tanh"))
        return x + h


class VisionTower(nn.Module):
    """Packed variable-resolution SigLIP2 tower followed by the 2x2 patch-merger projector.

    Rows are consumed in the processor's block-major 2x2 order and are never reordered.
    """

    def __init__(
        self,
        vcfg: VisionConfig,
        pcfg: ProjectorConfig,
        weights_visual: dict,
        weights_projector: dict,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vcfg = vcfg
        self.pcfg = pcfg
        self.dtype = dtype
        self.patch_embedding = _linear(weights_visual, "embeddings.patch_embedding", dtype)
        self.position_weight = weights_visual["embeddings.position_embedding.weight"].to(dtype)
        self.blocks = nn.ModuleList(
            VisionBlock(vcfg, weights_visual, f"encoder.layers.{i}.", dtype) for i in range(vcfg.layers)
        )
        self.post_layernorm = _layer_norm(weights_visual, "post_layernorm", vcfg.eps, dtype)
        self.proj_norm = _layer_norm(weights_projector, "norm", 1e-6, dtype)
        self.linear_fc1 = _linear(weights_projector, "linear_fc1", dtype)
        self.linear_fc2 = _linear(weights_projector, "linear_fc2", dtype)

    def encode(self, patches: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Patch embedding through `post_layernorm`: [N, 768] -> [N, 1152]."""
        x = self.patch_embedding(patches.to(self.dtype))
        pos = vision_position_embeddings(self.position_weight, grid_thw, self.vcfg.merge)
        assert x.shape[0] == pos.shape[0], "patch count does not match grid_thw"
        x = x + pos
        cu_seqlens = vision_cu_seqlens(grid_thw)
        for block in self.blocks:
            x = block(x, cu_seqlens)
        return self.post_layernorm(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Merge each group of merge**2 adjacent rows: [N, 1152] -> [N // 4, 2048]."""
        x = self.proj_norm(x).reshape(-1, self.vcfg.merge**2 * self.vcfg.hidden)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))

    def forward(self, patches: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        return self.project(self.encode(patches, grid_thw))
