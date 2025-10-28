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


class TtSSDregressionhead(nn.Module):
    def __init__(
        self,
        config,
        in_channels: List[int],
        num_anchors: List[int],
        num_columns: int,
        state_dict=None,
        base_address="",
        device=None,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.device = device
        self.num_columns = num_columns
        self.in_channels = in_channels
        # store depthwise conv modules and projection configs separately
        self.regression_head = []
        self.projections = []
        for i in range(len(self.in_channels)):
            # 3x3 depthwise convolution
            conv = TtMobileNetV3ConvLayer(
                config,
                in_channels=self.in_channels[i],
                out_channels=self.in_channels[i],
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.in_channels[i],
                use_activation=True,
                activation="ReLU6",
                state_dict=state_dict,
                base_address=f"{base_address}.{i}.{'0'}",
                device=device,
            )
            # register module and store
            self.add_module(f"dw_conv_{i}", conv)
            self.regression_head.append(conv)

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

            # Derive out channels from weight shape (typically 4 * anchors_per_loc)
            out_ch = weight.shape[0]
            self.projections.append(
                {
                    "type": "conv1x1",
                    "weight": tt_weight,
                    "bias": tt_bias,
                    "in_channels": self.in_channels[i],
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
        count = min(len(x), len(self.regression_head))
        for i in range(count):
            # Apply depthwise convolution
            result = self.regression_head[i](x[i])

            # Apply 1x1 projection convolution
            conv_config = self.projections[i]
            result = self._apply_conv1x1(result, conv_config)

            # Convert to torch for reshaping
            result = tt_to_torch_tensor(result)
            N, _, H, W = result.shape
            # Reshape from (N, A * 4, H, W) to (N, HWA, 4) where A=6 anchors, 4=coordinates
            result = result.view(N, -1, self.num_columns, H, W)
            result = result.permute(0, 3, 4, 1, 2)
            result = result.reshape(N, -1, self.num_columns)

            result = torch_to_tt_tensor_rm(result, self.device)
            all_results.append(result)

        return ttnn.concat(all_results, 1)  # Concatenate along the spatial dimension
