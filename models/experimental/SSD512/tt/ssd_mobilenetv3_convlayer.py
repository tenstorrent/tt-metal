# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Union
import torch
import torch.nn as nn
import ttnn


ACT_FN_1 = ttnn.relu
ACT_FN_2 = ttnn.hardswish


class TtMobileNetV3ConvLayer(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_activation: Union[bool, str] = False,
        activation: str = "",
        state_dict=None,
        base_address="",
        device=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        # Load conv weights
        weight_tensor = state_dict[f"{base_address}.0.weight"]

        # Load batch norm parameters
        bn_weight = state_dict[f"{base_address}.1.weight"]
        bn_bias = state_dict[f"{base_address}.1.bias"]
        bn_running_mean = state_dict[f"{base_address}.1.running_mean"]
        bn_running_var = state_dict[f"{base_address}.1.running_var"]
        bn_eps = 0.001  # Standard epsilon for MobileNetV3

        # Fuse batch norm into conv weights
        # Formula: w_fused = w * (gamma / sqrt(var + eps))
        bn_var_rsqrt = torch.rsqrt(bn_running_var + bn_eps)
        weight_scale = (bn_weight * bn_var_rsqrt).reshape(-1, 1, 1, 1)
        fused_weight = weight_tensor * weight_scale

        # Fuse batch norm into bias
        # Formula: b_fused = gamma * (b - mean) / sqrt(var + eps) + beta
        if bias and f"{base_address}.0.bias" in state_dict:
            conv_bias = state_dict[f"{base_address}.0.bias"]
            fused_bias = (conv_bias - bn_running_mean) * bn_var_rsqrt * bn_weight + bn_bias
        else:
            # No conv bias, so just use batch norm's transformation of -mean
            fused_bias = bn_bias - bn_running_mean * bn_var_rsqrt * bn_weight

        # Store fused weights as PyTorch tensors (will be converted on first forward pass)
        self.weight_torch = fused_weight
        self.bias_torch = fused_bias

        # Placeholders for TTNN tensors
        self.weight_ttnn = None
        self.bias_ttnn = None

        # Set activation
        if use_activation:
            if activation == "HS":
                self.activation = ACT_FN_2
            else:
                self.activation = ACT_FN_1
        else:
            self.activation = None

    def forward(self, features: ttnn.Tensor) -> ttnn.Tensor:
        # Get input dimensions
        batch_size = features.shape[0]
        input_height = features.shape[2]
        input_width = features.shape[3]

        # Configure convolution
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=True,
            reallocate_halo_output=False,
        )

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Convert weights to TTNN on first forward pass
        # This allows conv2d to preprocess them properly
        if self.weight_ttnn is None:
            self.weight_ttnn = ttnn.from_torch(
                self.weight_torch,
                dtype=ttnn.bfloat16,
                device=self.device,
            )
            self.bias_ttnn = ttnn.from_torch(
                self.bias_torch,
                dtype=ttnn.bfloat16,
                device=self.device,
            )

        # Perform convolution (batch norm is already fused)
        result = ttnn.conv2d(
            input_tensor=features,
            weight_tensor=self.weight_ttnn,
            bias_tensor=self.bias_ttnn,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=[self.kernel_size, self.kernel_size],
            stride=[self.stride, self.stride],
            padding=[self.padding, self.padding],
            dilation=[self.dilation, self.dilation],
            groups=self.groups,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        # Unpack results
        features = result[0]
        out_h, out_w = result[1]
        self.weight_ttnn, self.bias_ttnn = result[2]  # Update with preprocessed weights

        # Reshape from conv2d output format [B, 1, H*W, C] to [B, C, H, W]
        features = ttnn.reshape(features, (batch_size, out_h, out_w, self.out_channels))
        features = ttnn.permute(features, (0, 3, 1, 2))

        # Apply activation if specified
        if self.activation is not None:
            features = self.activation(features)

        return features
