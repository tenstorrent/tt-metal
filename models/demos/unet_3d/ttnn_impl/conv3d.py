# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import ttnn


def pad_dimension_t(x: ttnn.Tensor, padding_z) -> ttnn.Tensor:
    if padding_z == 0:
        return x
    x_old_shape = x.shape
    x0 = ttnn.reshape(x, (x_old_shape[0], x_old_shape[1], x_old_shape[2] * x_old_shape[3], x_old_shape[4]))
    x1 = ttnn.pad(
        x0, [(0, 0), (padding_z, padding_z), (0, 0), (0, 0)], 0, use_multicore=True, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn.deallocate(x0)
    return ttnn.reshape(
        x1, (x_old_shape[0], x_old_shape[1] + 2 * padding_z, x_old_shape[2], x_old_shape[3], x_old_shape[4])
    )


def get_conv3d_config(in_channels, out_channels, kernel_size, grid_size):
    config_to_blocking = {
        # (in_channels, out_channels, kernel_size) -> (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)
        (32, 32, 3): (32, 32, 1, 1, 16),
        (32, 64, 3): (32, 32, 2, 2, 16),
        (64, 64, 3): (32, 32, 1, 1, 16),
        (64, 128, 3): (32, 64, 1, 1, 16),
        (128, 128, 3): (64, 64, 1, 1, 16),
        (128, 256, 3): (64, 64, 1, 1, 8),
        (384, 128, 3): (128, 64, 1, 1, 4),
        (192, 64, 3): (64, 32, 4, 4, 8),
        (96, 32, 3): (32, 32, 2, 2, 8),
        (32, 32, 1): (32, 32, 4, 4, 16),
        # does not occur in model run for testing
        (64, 32, 3): (32, 32, 2, 8, 8),
    }

    blocking = config_to_blocking.get((in_channels, out_channels, kernel_size), (32, 32, 1, 1, 1))
    # default blocking if not found
    C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = blocking
    return ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=T_out_block,
        W_out_block=W_out_block,
        H_out_block=H_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        compute_with_storage_grid_size=grid_size,
    )


def prepare_conv3d_weights(mesh_device, weight, bias, C_in_block, ALIGNMENT=32):
    """Prepare weights and bias for TTNN."""
    C_in = weight.shape[1]
    w = weight.permute(2, 3, 4, 1, 0)  # kD, kH, kW, C, out_chan
    ALIGN_PAD = (ALIGNMENT - C_in % ALIGNMENT) % ALIGNMENT
    if C_in % ALIGNMENT != 0:
        w = torch.nn.functional.pad(w, (0, 0, 0, ALIGN_PAD), "constant", 0)

    # Reshape weights so that num_C_in_blocks is the first dimension
    kD, kH, kW, C_in_aligned, out_channels = w.shape
    assert C_in_aligned % C_in_block == 0
    num_C_in_blocks = C_in_aligned // C_in_block

    # Kernel expects num_C_in_blocks to be the first dimension to stride over it
    w = w.reshape(kD, kH, kW, num_C_in_blocks, C_in_block, out_channels)
    w = w.permute(3, 0, 1, 2, 4, 5)
    w = w.reshape(-1, out_channels)

    tt_weight = ttnn.from_torch(
        w,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
        mesh_mapper=None,
        pad_value=0,
    )

    if bias is not None:
        tt_bias = ttnn.from_torch(
            bias.reshape(1, -1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh_device,
            mesh_mapper=None,
            pad_value=0,
        )
    else:
        tt_bias = None
    return tt_weight, tt_bias


class Conv3D:
    def __init__(self, device, in_channels: int, out_channels: int, kernel_size: int = 3):
        assert kernel_size % 2 == 1, "Only odd kernel sizes are supported"
        self.in_channels = in_channels
        self.in_channels_padding = (32 - in_channels % 32) % 32
        self.out_channels = out_channels
        self.out_channels_padding = (32 - out_channels % 32) % 32
        padding = (kernel_size - 1) // 2
        self.padding = [0, padding, padding]
        self.padding_z = padding
        self.grid_size = device.compute_with_storage_grid_size()
        self.kernel_size = [kernel_size, kernel_size, kernel_size]

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.conv3d_config = get_conv3d_config(
            self.in_channels + self.in_channels_padding,
            self.out_channels + self.out_channels_padding,
            kernel_size,
            self.grid_size,
        )

    def load_state_dict(self, device, params_dict: dict[str, torch.Tensor], module_prefix: Optional[str] = None):
        weight_tensor = params_dict[f"{module_prefix}.weight"] if module_prefix else params_dict["weight"]
        weight_tensor = torch.nn.functional.pad(
            weight_tensor, (0, 0, 0, 0, 0, 0, 0, 0, 0, self.out_channels_padding), "constant", 0
        )

        if f"{module_prefix}.bias" in params_dict or (not module_prefix and "bias" in params_dict):
            bias_tensor = params_dict[f"{module_prefix}.bias"] if module_prefix else params_dict["bias"]
            bias_tensor = torch.nn.functional.pad(bias_tensor, (0, self.out_channels_padding), "constant", 0)
        else:
            bias_tensor = None
        self.weight, self.bias = prepare_conv3d_weights(
            mesh_device=device,
            weight=weight_tensor,
            bias=bias_tensor,
            C_in_block=self.conv3d_config.C_in_block,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x0 = pad_dimension_t(x, self.padding_z)
        x1 = ttnn.experimental.conv3d(
            input_tensor=x0,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            output_channels=self.out_channels + self.out_channels_padding,
            stride=[1, 1, 1],
            dilation=[1, 1, 1],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            groups=1,
            padding=self.padding,
            config=self.conv3d_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(x0)
        return x1
