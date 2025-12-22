# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import ttnn


def pad_dimension_t(x0: ttnn.Tensor) -> ttnn.Tensor:
    x_old_shape = x0.shape
    x0 = ttnn.reshape(x0, (x_old_shape[0], x_old_shape[1], x_old_shape[2] * x_old_shape[3], x_old_shape[4]))
    x1 = ttnn.to_layout(x0, ttnn.ROW_MAJOR_LAYOUT)
    x2 = ttnn.to_memory_config(x1, ttnn.DRAM_MEMORY_CONFIG)
    x3 = ttnn.pad(x2, [(0, 0), (1, 1), (0, 0), (0, 0)], 0, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x2)
    x3 = ttnn.reshape(x3, (x_old_shape[0], x_old_shape[1] + 2, x_old_shape[2], x_old_shape[3], x_old_shape[4]))
    return x3


def get_block_size_conv3d(C_in, C_out, K, T, H, W):
    return T * H * W + (K**3) * (T + H + W) * C_in


def get_conv3d_config(C_in, C_out, K, T, H, W, grid_size):
    C_in_block = max((C_in + 31) // 32 * 32, 128)
    C_in_block = min(C_in_block, C_in)
    C_out_block = max((C_out + 31) // 32 * 32, 128)
    C_out_block = min(C_out_block, C_out)

    t_out_o = 32
    h_out_o = 32
    w_out_o = 32

    L1_FREE = 0.01 * 1024 * 1024  # 1.3 MB
    is_free = True
    while is_free:
        is_free = False
        if get_block_size_conv3d(C_in_block, C_out_block, K, t_out_o * 2, h_out_o, w_out_o) < L1_FREE:
            t_out_o *= 2
            is_free = True
        if get_block_size_conv3d(C_in_block, C_out_block, K, t_out_o, h_out_o * 2, w_out_o) < L1_FREE:
            h_out_o *= 2
            is_free = True
        if get_block_size_conv3d(C_in_block, C_out_block, K, t_out_o, h_out_o, w_out_o * 2) < L1_FREE:
            w_out_o *= 2
            is_free = True

    C_in_block = 32
    C_out_block = 32
    t_out_o = 1
    h_out_o = 2
    w_out_o = 16
    return ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=t_out_o,
        W_out_block=w_out_o,
        H_out_block=h_out_o,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        compute_with_storage_grid_size=grid_size,
    )


def prepare_conv3d_weights(mesh_device, weight, bias, ALIGNMENT=16):
    """Prepare weights and bias for TTNN."""
    C_in = weight.shape[1]
    w = weight.permute(2, 3, 4, 1, 0)  # kD, kH, kW, C, out_chan
    ALIGN_PAD = ALIGNMENT - C_in % ALIGNMENT
    if C_in % ALIGNMENT != 0:
        w = torch.nn.functional.pad(w, (0, 0, 0, ALIGN_PAD))

    # Reshape weights so that num_C_in_blocks is the first dimension
    kD, kH, kW, C_in_aligned, out_channels = w.shape

    C_in_block = 32
    C_in_block = C_in_aligned if C_in_block == 0 else C_in_block
    num_C_in_blocks = C_in_aligned // C_in_block
    assert num_C_in_blocks * C_in_block == C_in_aligned

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = 1
        # self.padding = [0, 1, 1]
        if kernel_size == 3:
            self.padding = [0, 1, 1]
        elif kernel_size == 1:
            self.padding = [0, 0, 0]
        else:
            raise ValueError("Unsupported kernel size")
        self.grid_size = device.compute_with_storage_grid_size()
        self.kernel_size = [kernel_size, kernel_size, kernel_size]

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def init_params(self, device, params_dict: dict[str, torch.Tensor], module_prefix: Optional[str] = None):
        weight_tensor = params_dict[f"{module_prefix}.weight"] if module_prefix else params_dict["weight"]
        bias_tensor = None

        if f"{module_prefix}.bias" in params_dict or (not module_prefix and "bias" in params_dict):
            bias_tensor = params_dict[f"{module_prefix}.bias"] if module_prefix else params_dict["bias"]
        self.weight, self.bias = prepare_conv3d_weights(
            mesh_device=device,
            weight=weight_tensor,
            bias=bias_tensor,
            # conv_config=self.conv_config,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # format N, D, H, W, C
        conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            self.grid_size,
        )

        if self.kernel_size[0] == 3:
            x = pad_dimension_t(x)
        out = ttnn.experimental.conv3d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            output_channels=self.out_channels,
            stride=[1, 1, 1],
            dilation=[1, 1, 1],
            dtype=ttnn.bfloat16,
            # output_layout=ttnn.ROW_MAJOR_LAYOUT,
            compute_kernel_config=self.compute_kernel_config,
            groups=self.groups,
            padding=self.padding,
            config=conv_config,
            memory_config=None,
        )
        return out
