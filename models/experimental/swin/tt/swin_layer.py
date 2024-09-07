# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

import ttnn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from models.experimental.swin.tt.swin_attention import TtSwinAttention
from models.experimental.swin.tt.swin_intermediate import TtSwinIntermediate
from models.experimental.swin.tt.swin_output import TtSwinOutput
from models.experimental.swin.swin_utils import (
    window_partition,
    window_reverse,
)

import ttnn
from tt_lib.fallback_ops import fallback_ops


class TtSwinLayer(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        num_heads,
        state_dict,
        base_address,
        device,
        shift_size,
    ):
        super().__init__()
        self.device = device
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution

        gamma_before = torch_to_tt_tensor_rm(state_dict[f"{base_address}.layernorm_before.weight"], self.device)
        beta_before = torch_to_tt_tensor_rm(state_dict[f"{base_address}.layernorm_before.bias"], self.device)
        self.LayerNorm_before = fallback_ops.LayerNorm(
            gamma_before, beta_before, normalized_shape=dim, eps=config.layer_norm_eps
        )

        self.attention = TtSwinAttention(
            config,
            dim,
            num_heads,
            self.window_size,
            state_dict,
            f"{base_address}.attention",
            device,
        )

        gamma_after = torch_to_tt_tensor_rm(state_dict[f"{base_address}.layernorm_after.weight"], self.device)
        beta_after = torch_to_tt_tensor_rm(state_dict[f"{base_address}.layernorm_after.bias"], self.device)

        self.LayerNorm_after = fallback_ops.LayerNorm(
            gamma_after,
            beta_after,
            normalized_shape=dim,
            eps=config.layer_norm_eps,
        )

        self.intermediate = TtSwinIntermediate(
            config,
            dim,
            state_dict,
            f"{base_address}.intermediate",
            device,
        )
        self.output = TtSwinOutput(
            config,
            dim,
            state_dict,
            f"{base_address}.output",
            device,
        )

    def set_shift_and_window_size(self, input_resolution):
        if min(input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(input_resolution)

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )

            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            img_mask = torch_to_tt_tensor_rm(img_mask, self.device, put_on_device=False)
            mask_windows = window_partition(img_mask, self.window_size, self.device, False)

            mask_windows = fallback_ops.reshape(mask_windows, -1, self.window_size * self.window_size, 1, 1)

            mask_windows = tt_to_torch_tensor(mask_windows).squeeze()
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            attn_mask = torch_to_tt_tensor_rm(attn_mask, self.device, put_on_device=False)
        else:
            attn_mask = None

        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size

        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = fallback_ops.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            pass
        height, width = input_dimensions
        _, batch_size, _, channels = hidden_states.get_legacy_shape()
        shortcut = hidden_states

        hidden_states = self.LayerNorm_before(hidden_states)

        hidden_states = fallback_ops.reshape(hidden_states, batch_size, height, width, channels)

        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.get_legacy_shape()
        hidden_states = tt_to_torch_tensor(hidden_states)
        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        shifted_hidden_states = torch_to_tt_tensor_rm(shifted_hidden_states, self.device)
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size, self.device, False).to(
            self.device
        )

        hidden_states_windows = fallback_ops.reshape(
            hidden_states_windows, 1, -1, self.window_size * self.window_size, channels
        )
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)

        attention_outputs = self.attention(
            hidden_states_windows,
            attn_mask,
            head_mask,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

        attention_windows = fallback_ops.reshape(attention_output, -1, self.window_size, self.window_size, channels)

        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad, self.device, False
        ).to(self.device)

        shifted_windows = tt_to_torch_tensor(shifted_windows)
        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows
        attention_windows = torch_to_tt_tensor_rm(attention_windows, self.device)

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :]
        attention_windows = fallback_ops.reshape(attention_windows, 1, batch_size, height * width, channels)
        hidden_states = ttnn.add(shortcut, attention_windows)

        layer_output = self.LayerNorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = ttnn.add(hidden_states, self.output(layer_output))

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
