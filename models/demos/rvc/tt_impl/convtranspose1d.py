# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

dims_to_num_slices = {
    # ConvTranspose1d: batch_size=1, input_length=35600, output_length=213600, in_channels=256, out_channels: 128, kernel_size=16, stride=6, padding=5, dilation=1
    (213600, 256, 16): 4,
    # ConvTranspose1d: batch_size=1, input_length=213600, output_length=427200, in_channels=128, out_channels: 64, kernel_size=4, stride=2, padding=1, dilation=1
    (427200, 128, 4): 3,
    # ConvTranspose1d: batch_size=1, input_length=427200, output_length=854400, in_channels=64, out_channels: 32, kernel_size=4, stride=2, padding=1, dilation=1
    (854400, 64, 4): 3,
    # ConvTranspose1d: batch_size=1, input_length=854400, output_length=1708800, in_channels=32, out_channels: 16, kernel_size=4, stride=2, padding=1, dilation=1
    (1708800, 32, 4): 3,
    # ConvTranspose1d: batch_size=1, input_length=3560, output_length=35600, in_channels=512, out_channels: 256, kernel_size=16, stride=10, padding=3, dilation=1
    (35600, 512, 16): 4,
}

dims_to_act_block_h_override = {
    # ConvTranspose1d: batch_size=1, input_length=3560, output_length=35600, in_channels=512, out_channels: 256, kernel_size=16, stride=10, padding=3, dilation=1
    (35600, 512, 16): 32,
    # ConvTranspose1d: batch_size=1, input_length=35600, output_length=213600, in_channels=256, out_channels: 128, kernel_size=16, stride=6, padding=5, dilation=1
    (213600, 256, 16): 32,
    # ConvTranspose1d: batch_size=1, input_length=213600, output_length=427200, in_channels=128, out_channels: 64, kernel_size=4, stride=2, padding=1, dilation=1
    (427200, 128, 4): 32,
    # ConvTranspose1d: batch_size=1, input_length=427200, output_length=854400, in_channels=64, out_channels: 32, kernel_size=4, stride=2, padding=1, dilation=1
    (854400, 64, 4): 32,
    # ConvTranspose1d: batch_size=1, input_length=854400, output_length=1708800, in_channels=32, out_channels: 16, kernel_size=4, stride=2, padding=1, dilation=1
    (1708800, 32, 4): 32,
}


def determine_slice_strategy(
    batch_size: int, ouput_length: int, in_channels: int, kernel_size: int
) -> Optional[SliceStrategy]:
    if (ouput_length, in_channels, kernel_size) in dims_to_num_slices:
        num_slices = dims_to_num_slices[(ouput_length, in_channels, kernel_size)]
        return ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth)
    else:
        return ttnn.Op2DSliceConfig(num_slices=1, slice_type=ttnn.Op2DDRAMSliceWidth)
    l1_free_th = 1_300_000 * 60  # in bytes
    memory_cost = batch_size * ouput_length * in_channels * kernel_size * 2  # assuming bfloat16, so 2 bytes per element
    if memory_cost > l1_free_th:
        num_slices = (memory_cost + l1_free_th - 1) // l1_free_th + 2
        return ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth)
    return None


def determine_conv2d_config(
    batch_size: int, ouput_length: int, in_channels: int, kernel_size: int, out_channels: int
) -> ttnn.Conv2dConfig:
    TILE_WIDTH = 32
    if out_channels % TILE_WIDTH == 0:
        output_layout = ttnn.TILE_LAYOUT
    elif (TILE_WIDTH - out_channels % TILE_WIDTH) / out_channels < 1.2:
        output_layout = ttnn.TILE_LAYOUT
    else:
        output_layout = ttnn.ROW_MAJOR_LAYOUT
    if (ouput_length, in_channels, kernel_size) in dims_to_act_block_h_override:
        act_block_h_override = dims_to_act_block_h_override[(ouput_length, in_channels, kernel_size)]
        return ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            deallocate_activation=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            config_tensors_in_dram=True,
            act_block_h_override=act_block_h_override,
            output_layout=output_layout,
        )
    else:
        return ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            deallocate_activation=True,
            enable_act_double_buffer=False,
            config_tensors_in_dram=True,
        )


def _normalize_input(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    batch_size = input_tensor.shape[0]
    input_length = input_tensor.shape[1]
    input_channel = input_tensor.shape[2]
    input_t = ttnn.to_memory_config(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(input_t, (batch_size, 1, input_length, input_channel))


class ConvTranspose1d:
    """Stateful ConvTranspose1d wrapper built on top of `ttnn.conv_transpose2d`."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        weight_tensor: ttnn.Tensor | None = None,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias_tensor: ttnn.Tensor | None = None,
        dtype: ttnn.DataType | None = None,
        conv_config: ttnn.Conv2dConfig | None = None,
        compute_config: ttnn.DeviceComputeKernelConfig | None = None,
        memory_config: ttnn.MemoryConfig | None = None,
    ) -> None:
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_tensor = weight_tensor
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias_tensor = bias_tensor
        self.dtype = dtype
        self.conv_config = conv_config
        self.compute_config = compute_config
        self.memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        self._effective_conv_config = conv_config
        if self._effective_conv_config is None:
            self._effective_conv_config = ttnn.Conv2dConfig(
                deallocate_activation=True,
                enable_act_double_buffer=False,
                config_tensors_in_dram=True,
            )
        self.config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)

    def load_parameters(self, parameters: dict[str, torch.Tensor], key: str, prefix: str = "") -> None:
        base_key = f"{prefix}{key}" if prefix else key
        bias_key = f"{base_key}.bias"
        wt_torch = parameters[f"{base_key}.weight"]
        wt = wt_torch.reshape(self.in_channels, self.out_channels // self.groups, 1, self.kernel_size)
        self.weight_tensor = ttnn.from_torch(
            wt,
            dtype=ttnn.bfloat16,
            # device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.bias_tensor = None
        if bias_key in parameters and parameters[bias_key] is not None:
            self.bias_tensor = ttnn.from_torch(
                parameters[bias_key].reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                # device=self.device,
            )

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if self.weight_tensor is None:
            raise ValueError("weight_tensor is not set. Provide it in __init__ or call load_parameters().")

        # conv_config = ttnn.Conv2dConfig(
        #     weights_dtype=ttnn.bfloat16,
        #     # shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        #     deallocate_activation=True,
        #     enable_act_double_buffer=False,
        #     enable_weights_double_buffer=False,
        #     config_tensors_in_dram=True,
        #     # output_layout=ttnn.TILE_LAYOUT,
        #     # act_block_h_override=96,
        # )
        compute_config = ttnn.init_device_compute_kernel_config(
            input_tensor.device().arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        normalized_input = _normalize_input(input_tensor)
        batch_size = normalized_input.shape[0]
        input_length = normalized_input.shape[2]
        output_length = (
            (input_length - 1) * self.stride
            - 2 * self.padding
            + self.dilation * (self.kernel_size - 1)
            + self.output_padding
            + 1
        )
        slice_config = determine_slice_strategy(batch_size, output_length, self.in_channels, self.kernel_size)
        conv_config = determine_conv2d_config(
            batch_size, output_length, self.in_channels, self.kernel_size, self.out_channels
        )
        output, [self.weight_tensor, self.bias_tensor] = ttnn.conv_transpose2d(
            input_tensor=normalized_input,
            weight_tensor=self.weight_tensor,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            bias_tensor=self.bias_tensor,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, self.padding),
            # output_padding=(0, self.output_padding),
            dilation=(1, self.dilation),
            groups=self.groups,
            batch_size=batch_size,
            input_height=1,
            input_width=input_length,
            conv_config=conv_config,
            compute_config=compute_config,
            device=self.device,
            # memory_config=self.memory_config,
            return_output_dim=False,
            return_weights_and_bias=True,
            mirror_kernel=True,
            dtype=self.dtype,
            dram_slice_config=slice_config,
        )
        output_shape = output.shape
        x = ttnn.reshape(output, (batch_size, output_shape[2], output_shape[3]))
        return x
