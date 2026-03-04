# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn


def _normalize_input(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    batch_size = input_tensor.shape[0]
    input_length = input_tensor.shape[1]
    input_channel = input_tensor.shape[2]
    input_t = ttnn.to_memory_config(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(input_t, (batch_size, 1, input_length, input_channel))


class TTConvTranspose1d:
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

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            # shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            config_tensors_in_dram=True,
            # output_layout=ttnn.TILE_LAYOUT,
            act_block_h_override=96,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            input_tensor.device().arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        normalized_input = _normalize_input(input_tensor)
        batch_size = normalized_input.shape[0]
        input_length = normalized_input.shape[2]
        dram_slice_config = ttnn.Op2DSliceConfig(num_slices=16, slice_type=ttnn.Op2DDRAMSliceWidth)
        # normalized_input0 = ttnn.to_layout(normalized_input, ttnn.TILE_LAYOUT)
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
            dram_slice_config=dram_slice_config,
        )
        return output
