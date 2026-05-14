# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class VitPoseSimpleDecoder:
    """
    SimpleDecoder: ReLU → Bilinear Upsample 4x → Conv2d(768→17, k=3, p=1).
    """

    def __init__(self, parameters, device, *, batch_size=1, scale_factor=4, patch_height=16, patch_width=12):
        self.device = device
        self.weight = parameters["conv.weight"]
        self.bias = parameters["conv.bias"]
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.out_height = patch_height * scale_factor
        self.out_width = patch_width * scale_factor
        self._weights_prepared = False

    def __call__(self, hidden_states):
        """
        Args:
            hidden_states: (batch, 192, 768) TILE_LAYOUT from backbone

        Returns:
            ttnn tensor with heatmaps (batch, out_h, out_w, 17) in device memory
        """
        batch_size = self.batch_size

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, (batch_size, self.patch_height, self.patch_width, 768))

        hidden_states = ttnn.relu(hidden_states)

        hidden_states = ttnn.upsample(hidden_states, scale_factor=self.scale_factor, mode="bilinear")

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
            "in_channels": 768,
            "out_channels": 17,
            "batch_size": batch_size,
            "input_height": self.out_height,
            "input_width": self.out_width,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": self.device,
            "conv_config": conv_config,
        }

        if not self._weights_prepared:
            self.weight = ttnn.prepare_conv_weights(
                weight_tensor=self.weight,
                weights_format="OIHW",
                input_memory_config=hidden_states.memory_config(),
                input_layout=hidden_states.get_layout(),
                has_bias=True,
                **conv_kwargs,
                input_dtype=ttnn.bfloat16,
            )
            self.bias = ttnn.prepare_conv_bias(
                bias_tensor=self.bias,
                input_memory_config=hidden_states.memory_config(),
                input_layout=hidden_states.get_layout(),
                **conv_kwargs,
                input_dtype=ttnn.bfloat16,
            )
            self.weight = ttnn.to_device(self.weight, self.device)
            self.bias = ttnn.to_device(self.bias, self.device)
            self._weights_prepared = True

        [output, [out_h, out_w]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            **conv_kwargs,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=False,
        )

        return output


def preprocess_decoder_parameters(state_dict, *, dtype=ttnn.bfloat16):
    """
    Preprocess SimpleDecoder parameters from HuggingFace state dict.
    """
    params = {}
    params["conv.weight"] = ttnn.from_torch(state_dict["head.conv.weight"], dtype=dtype)
    params["conv.bias"] = ttnn.from_torch(state_dict["head.conv.bias"].reshape(1, 1, 1, -1), dtype=dtype)
    return params
