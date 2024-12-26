# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class ttnn_AdaLayerNormContinuous:
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-6,
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

    def __call__(self, hidden_states: ttnn.Tensor, emb: ttnn.Tensor, parameters=None) -> ttnn.Tensor:
        hifi2_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
        )

        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)

        mm_a_y = 8
        mm_a_x = 8
        mm_a_x_strategy = ttnn.ShardStrategy.WIDTH
        mm_a_x_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

        emb = self.linear(
            self.silu(emb),
            parameters["linear"]["weight"],
            bias=parameters["linear"]["bias"],
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
            compute_kernel_config=hifi2_kernel_config,
        )

        emb = ttnn.to_memory_config(emb, ttnn.L1_MEMORY_CONFIG)
        one_chunk = emb.shape[-1] // 2

        # TODO: double-check in reference model that scale comes first, oppostie to the other 2 modules
        i_beg = 0
        i_end = one_chunk
        scale = ttnn.slice(emb, [0, 0, 0, i_beg], [2, 1, 1, i_end])
        i_beg += one_chunk
        i_end += one_chunk
        shift = ttnn.slice(emb, [0, 0, 0, i_beg], [2, 1, 1, i_end])

        ttnn.deallocate(emb)

        # x = self.norm(
        #     x,
        #     epsilon=self.eps,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        #     compute_kernel_config=hifi2_kernel_config,
        # )
        # x = x * (1 + scale) + shift

        norm_hidden_states = self.norm(hidden_states, compute_kernel_config=hifi2_kernel_config)
        scale = scale + 1
        norm_hidden_states = norm_hidden_states * scale
        norm_hidden_states = norm_hidden_states + shift

        return norm_hidden_states
