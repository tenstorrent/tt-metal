# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from models.experimental.swin.tt.swin_layer import TtSwinLayer
import ttnn
from tt_lib.fallback_ops import fallback_ops


class TtSwinStage(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        depth,
        num_heads,
        downsample,
        state_dict,
        base_address,
        device,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.device = device
        self.blocks = nn.ModuleList(
            [
                TtSwinLayer(
                    config=self.config,
                    dim=self.dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    state_dict=state_dict,
                    base_address=f"{base_address}.blocks.{i}",
                    device=self.device,
                    shift_size=0 if i % 2 == 0 else self.config.window_size // 2,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                self.config,
                input_resolution,
                self.dim,
                state_dict,
                base_address,
                self.device,
            )
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[ttnn.Tensor]:
        height, width = input_dimensions
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
                always_partition,
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states,
            hidden_states_before_downsampling,
            output_dimensions,
        )

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs
