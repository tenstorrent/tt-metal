# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from models.experimental.swin.tt.swin_stage import TtSwinStage
from models.experimental.swin.tt.swin_patch_merging import TtSwinPatchMerging

import ttnn
from tt_lib.fallback_ops import fallback_ops

from dataclasses import dataclass


@dataclass
class TtSwinEncoderOutput:
    last_hidden_state: ttnn.Tensor = None
    hidden_states: Optional[Tuple[ttnn.Tensor]] = None
    attentions: Optional[Tuple[ttnn.Tensor]] = None
    reshaped_hidden_states: Optional[Tuple[ttnn.Tensor]] = None


class TtSwinEncoder(nn.Module):
    def __init__(
        self,
        config,
        grid_size,
        state_dict,
        base_address,
        device,
    ):
        super().__init__()
        self.config = config
        self.num_layers = len(config.depths)
        self.device = device
        self.layers = nn.ModuleList(
            [
                TtSwinStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer),
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    downsample=TtSwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                    state_dict=state_dict,
                    base_address=f"{base_address}.layers.{i_layer}",
                    device=self.device,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, TtSwinEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            _, batch_size, _, hidden_size = hidden_states.shape.with_tile_padding()

            reshaped_hidden_state = fallback_ops.reshape(hidden_states, batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = ttnn.permute(reshaped_hidden_state, (0, 3, 1, 2))
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                assert False, "We do not support training yet"

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                )

            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                (
                    _,
                    batch_size,
                    _,
                    hidden_size,
                ) = hidden_states_before_downsampling.shape.with_tile_padding()
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = fallback_ops.reshape(
                    reshaped_hidden_state,
                    batch_size,
                    *(output_dimensions[0], output_dimensions[1]),
                    hidden_size,
                )
                reshaped_hidden_state = ttnn.permute(reshaped_hidden_state, (0, 3, 1, 2))
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                _, batch_size, _, hidden_size = hidden_states.shape.with_tile_padding()
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = fallback_ops.reshape(
                    reshaped_hidden_state, batch_size, *input_dimensions, hidden_size
                )
                reshaped_hidden_state = ttnn.permute(reshaped_hidden_state, (0, 3, 1, 2))
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return TtSwinEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )
