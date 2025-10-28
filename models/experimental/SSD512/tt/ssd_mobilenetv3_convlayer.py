# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Union
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
        # self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        # Convert weights to TTNN tensor
        weight_tensor = state_dict[f"{base_address}.0.weight"]
        print(f"Weight tensor shape: {weight_tensor.shape}")
        print(f"Weight tensor total elements: {weight_tensor.numel()}")
        self.weight = ttnn.from_torch(
            weight_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Handle bias
        self.has_bias = bias
        if bias and f"{base_address}.0.bias" in state_dict:
            bias_tensor = state_dict[f"{base_address}.0.bias"]
            self.bias = ttnn.from_torch(
                bias_tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.bias = None

        # Batch normalization parameters
        # self.bn_weight = state_dict[f"{base_address}.1.weight"]
        # self.bn_bias = state_dict[f"{base_address}.1.bias"]
        # self.bn_running_mean = state_dict[f"{base_address}.1.running_mean"]
        # self.bn_running_var = state_dict[f"{base_address}.1.running_var"]
        bn_weight = state_dict[f"{base_address}.1.weight"]
        bn_bias = state_dict[f"{base_address}.1.bias"]
        bn_running_mean = state_dict[f"{base_address}.1.running_mean"]
        bn_running_var = state_dict[f"{base_address}.1.running_var"]
        # bn_weight = bn_weight.unsqueeze(0)  # [channels] -> [1, channels]
        # bn_bias = bn_bias.unsqueeze(0)
        # bn_running_mean = bn_running_mean.unsqueeze(0)
        # bn_running_var = bn_running_var.unsqueeze(0)
        bn_weight = bn_weight.reshape(1, -1, 1, 1)
        bn_bias = bn_bias.reshape(1, -1, 1, 1)
        bn_running_mean = bn_running_mean.reshape(1, -1, 1, 1)
        bn_running_var = bn_running_var.reshape(1, -1, 1, 1)

        self.bn_weight = ttnn.from_torch(
            bn_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bn_bias = ttnn.from_torch(
            bn_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bn_running_mean = ttnn.from_torch(
            bn_running_mean,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bn_running_var = ttnn.from_torch(
            bn_running_var,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bn_num_batches_tracked = state_dict[f"{base_address}.1.num_batches_tracked"]
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
        # features = ttnn.to_memory_config(features, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        print(f"Input features shape: {features.shape}")
        print(f"Input height: {input_height}, width: {input_width}")
        # Configure convolution
        # conv_config = ttnn.Conv2dConfig(
        #     weights_dtype=ttnn.bfloat16,
        #     output_layout=ttnn.TILE_LAYOUT,
        # )

        # compute_config = ttnn.init_device_compute_kernel_config(
        #     self.device.arch(),
        #     math_fidelity=ttnn.MathFidelity.HiFi4,
        #     fp32_dest_acc_en=False,
        #     packer_l1_acc=False,
        # )
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=True,  # Free activation memory
            reallocate_halo_output=False,  # Reduce memory usage
            config_tensors_in_dram=True,  # Store config tensors in DRAM
        )

        # Perform convolution
        [features, [out_h, out_w], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=features,
            weight_tensor=self.weight,
            bias_tensor=self.bias if self.has_bias else None,
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
            # compute_config=compute_config,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        # features = ttnn.reshape(features, (batch_size, self.out_channels, out_h, out_w))
        print(f"After conv2d - features shape: {features.shape}")
        print(f"out_h: {out_h}, out_w: {out_w}, out_channels: {self.out_channels}")

        # Try explicit reshape with the correct dimensions
        features = ttnn.reshape(features, (batch_size, out_h, out_w, self.out_channels))

        # Then permute to [batch, out_channels, out_h, out_w]
        features = ttnn.permute(features, (0, 3, 1, 2))
        print(f"After reshape - features shape: {features.shape}")

        # Apply batch normalization
        features = ttnn.batch_norm(
            features,
            running_mean=self.bn_running_mean,
            running_var=self.bn_running_var,
            training=False,
            eps=0.001,
            momentum=0.03,
            weight=self.bn_weight,
            bias=self.bn_bias,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Apply activation
        if self.activation is not None:
            features = self.activation(features)

        return features
