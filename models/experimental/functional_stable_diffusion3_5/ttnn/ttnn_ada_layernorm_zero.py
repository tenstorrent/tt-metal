# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional, Tuple


class ttnn_AdaLayerNormZero:
    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        self.emb = None

        self.silu = ttnn.silu
        self.linear = ttnn.linear
        if norm_type == "layer_norm":
            self.norm = ttnn.layer_norm
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def __call__(
        self,
        x: torch.Tensor,
        timestep: Optional[ttnn.Tensor] = None,
        class_labels: Optional[ttnn.Tensor] = None,
        hidden_dtype=None,
        emb: Optional[ttnn.Tensor] = None,
        parameters=None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb), parameters["linear"]["weight"], bias=parameters["linear"]["bias"])
        emb = ttnn.to_torch(emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, 1)

        shift_msa = ttnn.from_torch(shift_msa, layout=ttnn.TILE_LAYOUT, device=x.device())
        scale_msa = ttnn.from_torch(scale_msa, layout=ttnn.TILE_LAYOUT, device=x.device())
        gate_msa = ttnn.from_torch(gate_msa, layout=ttnn.TILE_LAYOUT, device=x.device())
        shift_mlp = ttnn.from_torch(shift_mlp, layout=ttnn.TILE_LAYOUT, device=x.device())
        scale_mlp = ttnn.from_torch(scale_mlp, layout=ttnn.TILE_LAYOUT, device=x.device())
        gate_mlp = ttnn.from_torch(gate_mlp, layout=ttnn.TILE_LAYOUT, device=x.device())

        x = self.norm(x) * (1 + ttnn.reshape(scale_msa, (scale_msa.shape[0], 1, scale_msa.shape[1]))) + ttnn.reshape(
            shift_msa, (shift_msa.shape[0], 1, shift_msa.shape[1])
        )  # shift_msa[:, None] replaced with ttnn.reshape(shift_msa,(shift_msa.shape[0],1,shift_msa.shape[1])) same for scale_msa[:,None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
