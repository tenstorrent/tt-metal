# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn
from tt_py_test_utils_common.ttnn_utility_functions import get_shard_grid_from_num_cores

# ---------------------------
# Head conv settings, shared by both heads and their weight preparation
# ---------------------------

# The bbox_reg conv's DRAM slicing. Its weights and bias are prepared for this same slicing
# (custom_preprocessor): weights prepared for another slicing are laid out for another program.
BBOX_REG_SLICE_CONFIG = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=4)


# ---------------------------
# TTNN utility modules
# ---------------------------


class TTConv2D:
    def __init__(
        self,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        parameters: dict | None = None,
        kernel_fidelity: dict | None = None,
        *,
        memory_config=None,
        act_block_h=None,
        act_block_w=None,
        deallocate_activation=False,
        reallocate_halo_output=False,
        shard_layout=None,
        activation=None,
        groups=1,
        num_cores_nhw=None,
        is_reshape=True,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        math_approx_mode=False,
        input_channels_alignment=32,
        reshard_if_not_optimal=False,
        slice_config=None,
        dtype=None,
        weights_dtype=None,
        math_fidelity=None,
    ) -> None:
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise ValueError("Invalid config: kernel_size must be int or tuple")
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            raise ValueError("Invalid config: stride must be int or tuple")
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            raise ValueError("Invalid config: padding must be int or tuple")
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        elif isinstance(dilation, tuple):
            self.dilation = dilation
        else:
            raise ValueError("Invalid config: dilation must be int or tuple")

        self.kernel_fidelity = kernel_fidelity
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        self.deallocate_activation = deallocate_activation
        self.reallocate_halo_output = reallocate_halo_output
        self.fp32_dest_acc_en = fp32_dest_acc_en
        self.packer_l1_acc = packer_l1_acc
        self.math_approx_mode = math_approx_mode
        self.input_channels_alignment = input_channels_alignment
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.act_block_w = act_block_w
        self.groups = groups
        self.activation = activation
        self.memory_config = memory_config
        self.shard_layout = shard_layout
        self.slice_config = slice_config
        self.num_cores_nhw = num_cores_nhw
        self.is_reshape = is_reshape
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_weights_double_buffer = enable_weights_double_buffer
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.kernel_fidelity["ACTIVATIONS_DTYPE"]
        if weights_dtype is not None:
            self.weights_dtype = weights_dtype
        else:
            self.weights_dtype = self.kernel_fidelity["WEIGHTS_DTYPE"]
        if math_fidelity is not None:
            self.math_fidelity = math_fidelity
        else:
            self.math_fidelity = self.kernel_fidelity["MATH_FIDELITY"]

    def __call__(self, device, input_tensor, input_shape):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            activation=self.activation,
            deallocate_activation=self.deallocate_activation,
            reallocate_halo_output=self.reallocate_halo_output,
            reshard_if_not_optimal=self.reshard_if_not_optimal,
            shard_layout=self.shard_layout,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_weights_double_buffer=self.enable_weights_double_buffer,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.kernel_fidelity["MATH_FIDELITY"],
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
            math_approx_mode=self.math_approx_mode,
        )
        if self.num_cores_nhw is not None:
            shard_grid = get_shard_grid_from_num_cores(self.num_cores_nhw, device)
            conv_config.core_grid = shard_grid
            conv_config.override_sharding_config = True

        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h
        if self.act_block_w is not None:
            conv_config.act_block_w_div = self.act_block_w

        [output_tensor, [_out_height, _out_width], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=self.weights.shape[1],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=input_shape[-4],
            input_height=input_shape[-3],
            input_width=input_shape[-2],
            conv_config=conv_config,
            compute_config=compute_config,
            slice_config=self.slice_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
            dtype=self.dtype,
            memory_config=self.memory_config,
        )

        if self.is_reshape:
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
            output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
            output_tensor = ttnn.reshape(
                output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[-1])
            )
        return output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[-1])


class TTUpsample:
    def __init__(
        self,
        scale_factor: int = 1,
        mode: str = "nearest",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
    ) -> None:
        self.scale_factor = scale_factor
        self.mode = mode
        self.memory_config = memory_config

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=fp32_dest_acc_en,
        )

    def __call__(
        self,
        device,
        input_tensor,
        input_shape=None,
        reshape_output=False,
        pad_ch_to_32=False,
        sent_to_dram=False,
        dtype=ttnn.bfloat8_b,
    ):
        if sent_to_dram:
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        else:
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        # Pad channels to a multiple of 32 to match TT tile/channel alignment if specified.
        if pad_ch_to_32:
            input_tensor = ttnn.pad(input_tensor, [(0, 0), (0, 0), (0, 0), (0, 32 - input_tensor.shape[-1] % 32)], 0)

        output_tensor = ttnn.upsample(
            input_tensor,
            scale_factor=self.scale_factor,
            mode=self.mode,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Remove channel padding if added.
        if pad_ch_to_32:
            output_tensor = ttnn.slice(
                output_tensor,
                [0, 0, 0, 0],
                [output_tensor.shape[0], output_tensor.shape[1], output_tensor.shape[2], input_shape[-1]],
            )

        if reshape_output:
            host = ttnn.from_device(output_tensor)
            host = ttnn.to_dtype(host, dtype)
            B, H, W, C = host.shape
            host = ttnn.reshape(host, [1, 1, B * H * W, C])
            output_tensor = ttnn.to_device(host, device)

        return output_tensor


class Conv2dNormActivation:
    """
    TTNN implementation of Conv2d + GroupNorm + ReLU block.
    """

    def __init__(
        self,
        parameters: dict,
        device: ttnn.Device,
        in_channels: int = 256,
        out_channels: int = 256,
        kernel_size: tuple = (3, 3),
        stride: tuple = (1, 1),
        padding: tuple = (1, 1),
        num_groups: int = 32,
        grid_size: Optional[ttnn.CoreGrid] = None,
        input_mask: Optional[ttnn.Tensor] = None,
        model_config: dict = None,
        compute_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
        conv_config: Optional[ttnn.Conv2dConfig] = None,
    ):
        """
        Args:
            parameters: Dict with keys 'weight', 'norm_weight', 'norm_bias'
            device: TTNN device
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            num_groups: Number of groups for GroupNorm
            grid_size: CoreGrid for GroupNorm (defaults to 8x8)
            input_mask: Pre-created input mask for GroupNorm
        """
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        self.model_config = model_config
        self.compute_config = compute_config
        self.conv_config = conv_config
        self.conv_weight = parameters["weight"]
        self.conv_bias = parameters["bias"]
        self.norm_weight = parameters["norm_weight"]
        self.norm_bias = parameters["norm_bias"]
        self.grid_size = grid_size if grid_size is not None else ttnn.CoreGrid(y=8, x=8)

        self.input_mask = input_mask

        self.slice_config = ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceHeight,
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        batch_size: int,
        input_height: int,
        input_width: int,
    ) -> ttnn.Tensor:
        """
        Forward pass: Conv2d -> GroupNorm -> ReLU

        Args:
            x: Input tensor in NHWC format
            batch_size: Batch size
            input_height: Input height
            input_width: Input width

        Returns:
            Output tensor after Conv2d + GroupNorm + ReLU
        """

        x, [H_out, W_out], [prepared_weight, prepared_bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv_weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.conv_bias,
            kernel_size=list(self.kernel_size),
            stride=list(self.stride),
            padding=list(self.padding),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            slice_config=self.slice_config,
            compute_config=self.compute_config,
            conv_config=self.conv_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        N, H_out, W_out, C = x.shape

        spatial_size = H_out * W_out
        x_flat = ttnn.reshape(x, (N, 1, spatial_size, C))

        # group_norm counts every row it is given in the group statistics, so the level is not
        # padded; its grid rows instead divide the level's tile rows. The grid keeps its width,
        # which the norm weight and bias are prepared for (create_group_norm_weight_bias_rm).
        tile_rows = -(-spatial_size // 32)
        grid_rows = max(y for y in range(1, self.grid_size.y + 1) if tile_rows % y == 0)
        x_normalized = ttnn.group_norm(
            x_flat,
            num_groups=self.num_groups,
            input_mask=self.input_mask,
            weight=self.norm_weight,
            bias=self.norm_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=grid_rows, x=self.grid_size.x),
            inplace=False,
            # A few grid rows leave each core many tile rows; group_norm splits them to fit L1.
            num_out_blocks=-1,
            compute_kernel_config=self.compute_config,
        )

        x = ttnn.reshape(x_normalized, (N, input_height, input_width, C))

        # ReLU activation
        x = ttnn.relu(x)

        return x
