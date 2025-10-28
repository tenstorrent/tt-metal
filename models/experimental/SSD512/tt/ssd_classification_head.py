# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from typing import (
    List,
)
import ttnn

from models.experimental.SSD512.tt.ssd_mobilenetv3_convlayer import (
    TtMobileNetV3ConvLayer,
)

from models.common.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor


class TtSSDclassificationhead(nn.Module):
    def __init__(
        self,
        config,
        in_channels: List[int],
        num_classes: int,
        state_dict=None,
        base_address="",
        device=None,
    ):
        super().__init__()
        self.channels = in_channels
        self.num_classes = num_classes
        self.device = device

        # Use a plain list for depthwise conv modules and a separate list for
        # 1x1 projection configs. Register each depthwise conv module so
        # parameters are tracked by PyTorch.
        self.classification_head = []
        self.projections = []
        for i in range(len(self.channels)):
            # 3x3 depthwise convolution
            conv = TtMobileNetV3ConvLayer(
                config,
                in_channels=self.channels[i],
                out_channels=self.channels[i],
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.channels[i],
                use_activation=True,
                activation="ReLU6",
                state_dict=state_dict,
                base_address=f"{base_address}.{i}.{'0'}",
                device=device,
            )
            # register and store
            self.add_module(f"dw_conv_{i}", conv)
            self.classification_head.append(conv)

            # 1x1 projection convolution - store weights for TTNN operations
            weight = state_dict[f"{base_address}.{i}.1.weight"]
            bias = state_dict[f"{base_address}.{i}.1.bias"] if f"{base_address}.{i}.1.bias" in state_dict else None

            # Convert to TTNN tensors
            tt_weight = ttnn.from_torch(
                weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            tt_bias = (
                ttnn.from_torch(
                    bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                )
                if bias is not None
                else None
            )

            # Store projection config for use in forward pass. Derive the
            # projection's out_channels from the weight tensor to match the
            # reference PyTorch head (num_classes * anchors_per_location).
            out_ch = weight.shape[0]
            self.projections.append(
                {
                    "type": "conv1x1",
                    "weight": tt_weight,
                    "bias": tt_bias,
                    "in_channels": self.channels[i],
                    "out_channels": out_ch,
                }
            )

    def _apply_conv1x1(self, x: ttnn.Tensor, conv_config: dict) -> ttnn.Tensor:
        """Apply 1x1 convolution using TTNN operations"""
        batch_size = x.shape[0]
        input_height = x.shape[1]
        input_width = x.shape[2]

        # Configure convolution
        conv_config_ttnn = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            output_layout=ttnn.TILE_LAYOUT,
        )

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Perform convolution
        [output, [out_h, out_w], [weight, bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=conv_config["weight"],
            bias_tensor=conv_config["bias"],
            in_channels=conv_config["in_channels"],
            out_channels=conv_config["out_channels"],
            device=self.device,
            kernel_size=1,
            stride=1,
            padding=0,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=conv_config_ttnn,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        return output

    def forward(self, x: List[ttnn.Tensor]) -> ttnn.Tensor:
        all_results = []
        count = min(len(x), len(self.classification_head))
        for i in range(count):
            # Apply depthwise convolution
            result = self.classification_head[i](x[i])

            # Apply 1x1 projection convolution
            conv_config = self.projections[i]
            result = self._apply_conv1x1(result, conv_config)

            # Convert to torch for reshaping
            result = tt_to_torch_tensor(result)
            N, _, H, W = result.shape
            # Reshape from (N, A * K, H, W) to (N, HWA, K) where A=6 anchors, K=num_classes
            result = result.view(N, -1, self.num_classes, H, W)
            result = result.permute(0, 3, 4, 1, 2)
            result = result.reshape(N, -1, self.num_classes)

            result = torch_to_tt_tensor_rm(result, device=self.device, put_on_device=False)
            all_results.append(result)

        return ttnn.concat(all_results, 1)  # Concatenate along the spatial dimension
