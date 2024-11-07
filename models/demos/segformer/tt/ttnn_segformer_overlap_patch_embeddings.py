# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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
        pixel_values: ttnn.Tensor,
        parameters,
    ):
        device = pixel_values.device()

        if pixel_values.shape[-1] == 3:
            pixel_values_rm = ttnn.from_device(pixel_values)
            pixel_values_rm = ttnn.to_layout(pixel_values_rm, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            pixel_values_rm = pixel_values

        embeddings, input_height, input_width = self.proj(device, pixel_values_rm)
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
