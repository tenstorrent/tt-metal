# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


class VitPosePatchEmbeddings:
    def __init__(self, parameters, device, *, batch_size=1):
        self.device = device
        self.weight = parameters["projection.weight"]
        self.bias = parameters["projection.bias"]
        self.batch_size = batch_size
        self._weights_prepared = False

    def __call__(self, pixel_values):
        """
        Args:
            pixel_values: ttnn tensor (batch, 256, 192, 3) NHWC ROW_MAJOR on device

        Returns:
            ttnn tensor (batch, 192, 768) in TILE_LAYOUT
        """
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            output_layout=ttnn.TILE_LAYOUT,
            reshard_if_not_optimal=True,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        conv_kwargs = {
            "in_channels": 3,
            "out_channels": 768,
            "batch_size": self.batch_size,
            "input_height": 256,
            "input_width": 192,
            "kernel_size": (16, 16),
            "stride": (16, 16),
            "padding": (2, 2),
            "dilation": (1, 1),
            "groups": 1,
            "device": self.device,
            "conv_config": conv_config,
        }

        if not self._weights_prepared:
            self.weight = ttnn.prepare_conv_weights(
                weight_tensor=self.weight,
                weights_format="OIHW",
                input_memory_config=pixel_values.memory_config(),
                input_layout=pixel_values.get_layout(),
                has_bias=True,
                **conv_kwargs,
                input_dtype=ttnn.bfloat16,
            )
            self.bias = ttnn.prepare_conv_bias(
                bias_tensor=self.bias,
                input_memory_config=pixel_values.memory_config(),
                input_layout=pixel_values.get_layout(),
                **conv_kwargs,
                input_dtype=ttnn.bfloat16,
            )
            self.weight = ttnn.to_device(self.weight, self.device)
            self.bias = ttnn.to_device(self.bias, self.device)
            self._weights_prepared = True

        [output, [out_h, out_w]] = ttnn.conv2d(
            input_tensor=pixel_values,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            **conv_kwargs,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=False,
        )

        batch_size = self.batch_size
        num_patches = out_h * out_w
        output = ttnn.reshape(output, (batch_size, num_patches, 768))
        return output


def vitpose_embeddings(patch_embeddings_output, *, pos_patches, pos_cls):
    """
    Add position encoding to patch embeddings.

    Args:
        patch_embeddings_output: (batch, 192, 768) TILE_LAYOUT
        pos_patches: (1, 192, 768) TILE_LAYOUT — pre-sliced position_embeddings[:, 1:]
        pos_cls: (1, 1, 768) TILE_LAYOUT — pre-sliced position_embeddings[:, :1]

    Returns:
        (batch, 192, 768) TILE_LAYOUT
    """
    output = ttnn.add(patch_embeddings_output, pos_patches)
    output = ttnn.add(output, pos_cls)
    return output


def preprocess_embedding_parameters(state_dict, *, dtype=ttnn.bfloat16):
    """
    Preprocess embedding parameters from HuggingFace state dict for TTNN.

    Returns:
        dict with:
          - projection.weight: torch tensor [768,3,16,16] for conv2d (OIHW)
          - projection.bias: torch tensor [768] for conv2d
          - pos_patches: ttnn tensor [1,192,768] TILE_LAYOUT
          - pos_cls: ttnn tensor [1,1,768] TILE_LAYOUT
    """
    params = {}

    weight = state_dict["backbone.embeddings.patch_embeddings.projection.weight"]
    bias = state_dict["backbone.embeddings.patch_embeddings.projection.bias"]
    params["projection.weight"] = ttnn.from_torch(weight, dtype=dtype)
    params["projection.bias"] = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=dtype)

    pos_emb = state_dict["backbone.embeddings.position_embeddings"]
    pos_patches = pos_emb[:, 1:, :]
    pos_cls = pos_emb[:, :1, :]
    params["pos_patches"] = ttnn.from_torch(pos_patches, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    params["pos_cls"] = ttnn.from_torch(pos_cls, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    return params
