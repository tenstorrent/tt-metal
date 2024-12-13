# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class ttnn_AdaLayerNormContinuous:
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        self.eps = eps
        self.silu = ttnn.silu
        self.linear = ttnn.linear
        if norm_type == "layer_norm":
            self.norm = ttnn.layer_norm
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def __call__(self, x: ttnn.Tensor, conditioning_embedding: ttnn.Tensor, parameters=None) -> ttnn.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(
            self.silu(conditioning_embedding), parameters["linear"]["weight"], bias=parameters["linear"]["bias"]
        )
        emb = ttnn.to_layout(emb, layout=ttnn.ROW_MAJOR_LAYOUT)
        scale, shift = ttnn.split(emb, 2, dim=1)
        scale = ttnn.to_layout(scale, layout=ttnn.TILE_LAYOUT)
        shift = ttnn.to_layout(shift, layout=ttnn.TILE_LAYOUT)
        x = self.norm(x, epsilon=self.eps) * ttnn.reshape(
            (1 + scale), (scale.shape[0], 1, scale.shape[1])
        ) + ttnn.reshape(
            shift, (shift.shape[0], 1, shift.shape[1])
        )  # (1+scale[:, None,:]) replaced with ttnn.reshape((1 + scale),(scale.shape[0],1,scale.shape[1])) same for shift[:,None,:]
        return x
