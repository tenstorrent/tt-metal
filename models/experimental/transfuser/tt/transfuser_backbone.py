# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.transfuser.tt.utils import TTConv2D


class TtTransfuserBackbone:
    def __init__(
        self,
        parameters,
        stride,
        model_config,
        # layer_optimisations=neck_optimisations,
    ) -> None:
        self.conv1 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            parameters=parameters.image_encoder.features.conv1,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=2),
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
            # is_reshape=True,
        )

    def normalize_imagenet_ttnn(self, x):
        """Normalize input images according to ImageNet standards using TTNN operations."""
        # ImageNet normalization constants
        # Channel 0 (R): (x/255.0 - 0.485) / 0.229
        # Channel 1 (G): (x/255.0 - 0.456) / 0.224
        # Channel 2 (B): (x/255.0 - 0.406) / 0.225

        # First divide by 255.0 to convert from [0,255] to [0,1]
        x = ttnn.multiply(x, 1.0 / 255.0)

        # Split channels for per-channel normalization
        # Note: x is in NCHW format at this point
        x_r = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], 1, x.shape[2], x.shape[3]])  # Red channel
        x_g = ttnn.slice(x, [0, 1, 0, 0], [x.shape[0], 2, x.shape[2], x.shape[3]])  # Green channel
        x_b = ttnn.slice(x, [0, 2, 0, 0], [x.shape[0], 3, x.shape[2], x.shape[3]])  # Blue channel

        # Normalize each channel: (x - mean) / std
        x_r = ttnn.subtract(x_r, 0.485)
        x_r = ttnn.multiply(x_r, 1.0 / 0.229)

        x_g = ttnn.subtract(x_g, 0.456)
        x_g = ttnn.multiply(x_g, 1.0 / 0.224)

        x_b = ttnn.subtract(x_b, 0.406)
        x_b = ttnn.multiply(x_b, 1.0 / 0.225)

        # Concatenate channels back together
        x = ttnn.concat([x_r, x_g, x_b], dim=1)

        return x

    def __call__(
        self,
        x,
        device,
    ):
        x = self.normalize_imagenet_ttnn(x)
        # conv1 is stride 2 conv 3x3
        x = ttnn.permute(x, (0, 2, 3, 1))
        out, shape = self.conv1(device, x, x.shape)
        # Reshape to spatial dimensions: 80 * 352 = 28160
        # out = ttnn.reshape(out, (1, 80, 352, 32))
        # out = ttnn.permute(out, (0, 3, 1, 2))
        return out
