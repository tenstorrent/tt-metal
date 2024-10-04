# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_segformer.tt.common import Conv


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
        # print("ov0", pixel_values.shape)
        device = pixel_values.device()

        pixel_values = ttnn.permute(pixel_values, (0, 2, 3, 1))

        if pixel_values.shape[3] == 3:
            pixel_values = ttnn.from_device(pixel_values)
            pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.ROW_MAJOR_LAYOUT)
        embeddings, input_height, input_width = self.proj(device, pixel_values)
        embeddings = ttnn.to_memory_config(embeddings, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pixel_values)
        # print("ov1", embeddings.shape)
        """
        #embeddings = ttnn.to_layout(embeddings, layout=ttnn.TILE_LAYOUT)
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.ROW_MAJOR_LAYOUT)
        embeddings = ttnn.to_device(embeddings, device=device)

        embeddings = ttnn.permute(embeddings, (0, 3, 1, 2))
        batch_size, _, input_height, input_width = embeddings.shape

        ttnn.deallocate(pixel_values)
        embeddings = ttnn.from_device(embeddings)
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.ROW_MAJOR_LAYOUT)

        print("ov2", embeddings.shape)
        embeddings = ttnn.reshape(
            embeddings, (embeddings.shape[0], embeddings.shape[1], embeddings.shape[2] * embeddings.shape[3])
        )
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.TILE_LAYOUT)
        embeddings = ttnn.to_device(embeddings, device)

        print("ov3", embeddings.shape)

        embeddings = ttnn.permute(embeddings, (0, 2, 1))
        if len(embeddings.shape) == 2:
            embeddings = ttnn.reshape(embeddings, (1, embeddings.shape[0], embeddings.shape[1]))

        print("ov4", embeddings.shape)
        """

        embeddings = ttnn.layer_norm(
            embeddings,
            weight=parameters.layer_norm.weight,
            bias=parameters.layer_norm.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
        )

        # print("ov5", embeddings.shape)

        return embeddings, input_height, input_width
