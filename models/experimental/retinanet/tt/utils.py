# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn
from ttnn.operations.normalization import dram_group_norm_virtual_columns
from tt_py_test_utils_common.ttnn_utility_functions import get_shard_grid_from_num_cores

# The L1 plan (each layer's shard layout, L1 activations and unsliced convs) is sized for a full
# Wormhole or Blackhole grid; a grid with fewer cores runs every conv and pool from DRAM instead.
FULL_PLAN_MIN_CORES = 64


def reduced_grid(device):
    """Whether ``device``'s grid has fewer cores than the L1 plan is sized for. Its convs and pools
    then also keep their config tensors in DRAM: on a few cores L1_SMALL holds only a few banks."""
    grid = device.compute_with_storage_grid_size()
    return grid.x * grid.y < FULL_PLAN_MIN_CORES


def activation_memory_config(device):
    """Where layer outputs live: L1 on the full grid, DRAM on a reduced one."""
    return ttnn.DRAM_MEMORY_CONFIG if reduced_grid(device) else ttnn.L1_MEMORY_CONFIG


# ---------------------------
# Head conv and GroupNorm settings, shared by both heads and their weight preparation
# ---------------------------

# The bbox_reg conv's DRAM slicing on the full grid.
BBOX_REG_SLICE_CONFIG = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=4)
# On a reduced grid the heads' final convs run in the height slices conv2d chooses.
REDUCED_GRID_SLICE_CONFIG = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=0)


def bbox_reg_slice_config(device):
    """The bbox_reg conv's slicing on ``device``."""
    return REDUCED_GRID_SLICE_CONFIG if reduced_grid(device) else BBOX_REG_SLICE_CONFIG


def cls_logits_slice_config(device):
    """The cls_logits conv's slicing on ``device``: conv2d's own height slices on a reduced grid;
    unsliced in L1 on the full grid."""
    return REDUCED_GRID_SLICE_CONFIG if reduced_grid(device) else ttnn.Conv2dL1FullSliceConfig


def head_conv_config(device, config_dict):
    """A head conv's Conv2dConfig from its optimization ``config_dict``; on a reduced grid its config
    tensors go to DRAM, as L1_SMALL holds only a few banks there."""
    conv_config = ttnn.Conv2dConfig(**config_dict) if config_dict else ttnn.Conv2dConfig()
    if reduced_grid(device):
        conv_config.config_tensors_in_dram = True
    return conv_config


def group_norm_grid(device):
    """The heads' GroupNorm grid: the device's. Each level uses as many of its rows as divide the
    level's tile rows (Conv2dNormActivation)."""
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


def conv2d_with_level_weights(parameters, level, **conv2d_args):
    """ttnn.conv2d of a head conv on pyramid level ``level``, with the conv's weights prepared for
    that level; returns the output and its height and width. A head runs the same weights on every
    level, but conv2d lays prepared weights out for one program, and the program depends on the
    level's size. So ``parameters`` keeps the host weights and, under "prepared", the copy conv2d
    prepared on each level's first run."""
    prepared = parameters.setdefault("prepared", {})
    weight, bias = prepared.get(level, (parameters["weight"], parameters["bias"]))
    output, [height, width], [weight, bias] = ttnn.conv2d(
        weight_tensor=weight,
        bias_tensor=bias,
        return_output_dim=True,
        return_weights_and_bias=True,
        **conv2d_args,
    )
    prepared[level] = (weight, bias)
    return output, height, width


def group_norm_channel_chunks(device, num_channels, num_groups):
    """The channel chunks group_norm splits ``num_channels`` into on group_norm_grid, which its
    weight and bias are prepared for (create_group_norm_weight_bias_rm)."""
    return dram_group_norm_virtual_columns(group_norm_grid(device), num_channels, num_groups)


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
        fp32_dest_acc_en=None,
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
        # The convs accumulate in fp32 unless the model config or the layer says otherwise: the
        # backbone's sums run over thousands of products, and bfloat16 accumulation of them costs
        # its feature levels most of their margin to the reference.
        if fp32_dest_acc_en is None:
            fp32_dest_acc_en = self.kernel_fidelity.get("FP32_DEST_ACC_EN", True)
        self.fp32_dest_acc_en = fp32_dest_acc_en
        self.packer_l1_acc = packer_l1_acc
        self.math_approx_mode = math_approx_mode
        self.input_channels_alignment = input_channels_alignment
        self.reshard_if_not_optimal = reshard_if_not_optimal
        # conv2d replaces self.weights with its prepared tensor after the first call, which no
        # longer has the OIHW shape.
        self.in_channels = self.weights.shape[1]
        self.out_channels = self.weights.shape[0]
        # The DRAM matmul weights of a pointwise conv on a reduced grid (_pointwise_in_dram).
        self.linear_weight = None
        self.linear_bias = None
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

    # A spatial conv's weight block may take at most 1/WEIGHT_L1_DIVISOR of a core's L1 on a
    # reduced grid; the rest holds its activation, halo and output blocks.
    WEIGHT_L1_DIVISOR = 4

    def _weight_split(self, device):
        """How many blocks a reduced grid splits the conv's input channels into so that a block
        of its bfloat16 weights takes at most 1/WEIGHT_L1_DIVISOR of a core's L1; 1 if whole.
        Width sharding first splits the channels over the grid's cores, and each block must stay
        whole tiles of a core's share; conv2d does not reject a split that breaks a tile."""
        in_channels = self.in_channels
        weight_bytes = in_channels * self.kernel_size[0] * self.kernel_size[1] * self.out_channels * 2
        budget = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank // self.WEIGHT_L1_DIVISOR
        grid = device.compute_with_storage_grid_size()
        core_channels = in_channels // (grid.x * grid.y)
        split = 1
        while weight_bytes // split > budget and core_channels % (split * 2 * ttnn.TILE_SIZE) == 0:
            split *= 2
        return split

    def _is_pointwise(self):
        return (
            self.kernel_size == (1, 1)
            and all(p == 0 for p in self.padding)
            and self.dilation == (1, 1)
            and self.groups == 1
        )

    def _pointwise_in_dram(self, device, input_tensor, input_shape, compute_config):
        """A 1x1 conv as a DRAM matmul: conv2d runs one as a matmul whose output is L1 sharded, or
        with its whole weight in L1, neither of which a reduced grid holds. A strided one takes
        every stride-th pixel first. Its weights stay the host OIHW tensors conv2d would prepare,
        from which the matmul's are made once."""
        if self.linear_weight is None:
            weight = ttnn.to_torch(self.weights).reshape(self.out_channels, -1).transpose(0, 1).contiguous()
            self.linear_weight = ttnn.from_torch(
                weight, dtype=self.weights_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )
            bias = ttnn.to_torch(self.bias).reshape(1, self.out_channels)
            self.linear_bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        n, h, w = input_shape[-4], input_shape[-3], input_shape[-2]
        channels = input_tensor.shape[-1]
        activation = input_tensor
        if self.stride != (1, 1):
            sh, sw = self.stride
            # The input is tiled (__call__), so each step below makes a new tensor.
            row_major = ttnn.to_layout(
                ttnn.reshape(activation, (n, h, w, channels)),
                ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            pixels = ttnn.slice(
                row_major, [0, 0, 0, 0], [n, h, w, channels], [1, sh, sw, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(row_major)
            h, w = pixels.shape[1], pixels.shape[2]
            input_tensor = ttnn.to_layout(pixels, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(pixels)
        rows = ttnn.reshape(input_tensor, (1, 1, n * h * w, channels))
        relu = self.activation is not None and self.activation.op_type == ttnn.UnaryOpType.RELU
        assert self.activation is None or relu, f"a DRAM pointwise conv fuses only RELU, got {self.activation}"
        output_tensor = ttnn.linear(
            rows,
            self.linear_weight,
            bias=self.linear_bias,
            activation="relu" if relu else None,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        if self.stride != (1, 1):
            ttnn.deallocate(input_tensor)
        if self.deallocate_activation:
            ttnn.deallocate(activation)
        if self.is_reshape:
            output_tensor = ttnn.reshape(output_tensor, (n, h, w, self.out_channels))
        return output_tensor, (n, h, w, self.out_channels)

    def __call__(self, device, input_tensor, input_shape):
        reduced = reduced_grid(device)
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
        memory_config, slice_config = self.memory_config, self.slice_config
        if reduced:
            # A reduced grid's L1 holds no whole activation: the conv reads and writes DRAM in the
            # height slices conv2d chooses, with conv2d's own block sizes.
            conv_config.config_tensors_in_dram = True
            conv_config.act_block_h_override = 0
            conv_config.act_block_w_div = 1
            memory_config = ttnn.DRAM_MEMORY_CONFIG
            slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=0)
            weight_split = self._weight_split(device)
            if weight_split > 1:
                # Slices shrink the activation, not the weights; a conv whose weights outgrow a
                # reduced grid's L1 runs width sharded, its input channels in weight_split blocks.
                conv_config.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
                conv_config.act_block_w_div = weight_split
            if input_tensor.layout != ttnn.TILE_LAYOUT:
                # conv2d tilizes a row-major input whole into L1, which a reduced grid cannot hold.
                input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if self._is_pointwise():
                return self._pointwise_in_dram(device, input_tensor, input_shape, compute_config)

        [output_tensor, [_out_height, _out_width], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=self.in_channels,
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
            slice_config=slice_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
            dtype=self.dtype,
            memory_config=memory_config,
        )

        if self.is_reshape:
            output_tensor = ttnn.to_memory_config(output_tensor, activation_memory_config(device))
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
        reduced = reduced_grid(device)
        if sent_to_dram or reduced:
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG if reduced else self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        if reduced:
            # The add that consumes it would tilize a row-major output whole into L1.
            output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

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
        compute_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
        conv_config: Optional[ttnn.Conv2dConfig] = None,
    ):
        """
        Args:
            parameters: Dict with the conv's host 'weight' and 'bias' (conv2d_with_level_weights),
                and the GroupNorm 'norm_weight' and 'norm_bias' (create_group_norm_weight_bias_rm)
            device: TTNN device
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            num_groups: Number of groups for GroupNorm
            grid_size: CoreGrid for GroupNorm (defaults to group_norm_grid)
            input_mask: Pre-created input mask for GroupNorm
            compute_config: Compute kernel config of the conv and the GroupNorm
            conv_config: Conv2dConfig of the conv (head_conv_config)
        """
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        self.compute_config = compute_config
        self.conv_config = conv_config
        self.parameters = parameters
        self.norm_weight = parameters["norm_weight"]
        self.norm_bias = parameters["norm_bias"]
        self.grid_size = grid_size if grid_size is not None else group_norm_grid(device)

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
        level: int,
    ) -> ttnn.Tensor:
        """
        Forward pass: Conv2d -> GroupNorm -> ReLU

        Args:
            x: Input tensor in NHWC format
            batch_size: Batch size
            input_height: Input height
            input_width: Input width
            level: Pyramid level of ``x``, whose prepared conv weights to use

        Returns:
            Output tensor after Conv2d + GroupNorm + ReLU
        """

        x, H_out, W_out = conv2d_with_level_weights(
            self.parameters,
            level,
            input_tensor=x,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=list(self.kernel_size),
            stride=list(self.stride),
            padding=list(self.padding),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            slice_config=self.slice_config,
            compute_config=self.compute_config,
            conv_config=self.conv_config,
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
