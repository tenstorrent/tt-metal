# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.segformer.tt.common import Conv


class TtSegformerOverlapPatchEmbeddings:
    """Construct the overlapping patch embeddings."""

    def __init__(self, parameters, stride, patch_size):
        super().__init__()
        self.proj = Conv([stride, stride, patch_size // 2, patch_size // 2], parameters=parameters["proj"])

    def __call__(
        self,
        device,
        pixel_values: ttnn.Tensor,
        parameters,
    ):
        embeddings, input_height, input_width = self.proj(device, pixel_values)
        embeddings = ttnn.to_memory_config(embeddings, memory_config=ttnn.L1_MEMORY_CONFIG)

        ttnn.deallocate(pixel_values)
        embeddings = ttnn.reallocate(embeddings)

        embeddings = ttnn.layer_norm(
            embeddings,
            weight=parameters.layer_norm.weight,
            bias=parameters.layer_norm.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
        )

        return embeddings, input_height, input_width
