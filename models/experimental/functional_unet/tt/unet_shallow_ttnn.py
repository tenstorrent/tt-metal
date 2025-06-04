# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
import torch

from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, ParameterDict

from ttnn.device import is_wormhole_b0


def nearest_16(x):
    return math.ceil(x / 16) * 16


def determine_num_cores_for_upsample(nhw: int, width: int, max_cores: int) -> int:
    gcd_nhw_width = math.gcd(nhw, width)
    cores = nhw // gcd_nhw_width
    if cores > max_cores:
        for divisor in range(max_cores, 0, -1):
            if nhw % divisor == 0 and (nhw // divisor) % width == 0:
                cores = divisor
                break
    return cores


def get_core_grid_from_num_cores(num_cores: int, grid_rows: int, grid_cols: int):
    rows = num_cores // grid_cols
    assert rows <= grid_rows, "Not enough cores for specified core grid"
    ranges = []
    if rows != 0:
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_rows - 1, rows - 1),
            )
        )
    remainder = num_cores % grid_rows
    if remainder != 0:
        assert rows + 1 <= grid_rows, "Not enough cores for specified core grid"
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, rows),
                ttnn.CoreCoord(remainder - 1, rows),
            )
        )
    return ttnn.CoreRangeSet({*ranges})


def is_valid_device_for_unet(device):
    """Check that each device is an 8x8 grid."""
    return (
        device.core_grid.x == 8 and device.core_grid.y == 8
        if is_wormhole_b0(device)
        else device.core_grid.x >= 11 and device.core_grid.y >= 10
    )


def preprocess_unet_input_tensor(input_tensor, output_shard_height=2688):
    """
    Pad (if needed) and reshape to [1,1,N*H*W,C] for downstream convolution
    """
    assert len(input_tensor.shape), "Expected input tensor to be rank 4 (was {len(input_tensor.shape)})"

    _, _, C, HW = input_tensor.shape
    output_core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
        }
    )
    output_shard_shape = (output_shard_height, C)
    output_shard_spec = ttnn.ShardSpec(output_core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )
    return ttnn.experimental.convert_to_hwc(input_tensor, memory_config=output_memory_config, dtype=ttnn.bfloat16)


def concatenate(activation, residual, dim=-1, groups=1, final_block=False):
    """
    Concatenate along the final dimension. The `final_block` flag is used for
    the final upblock where L1 memory pressure is highest.
    """
    assert dim < 0, "Concatenation must happen along inner-dimension"
    assert activation.is_sharded() and residual.is_sharded(), "Both inputs must be sharded"

    if final_block:
        residual_tile = ttnn.to_layout(residual, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(residual)

        output_memory_config = residual_tile.memory_config()
        activation_shard_shape = activation.memory_config().shard_spec.shape
        output_shard_shape = output_memory_config.shard_spec.shape
        output_shard_shape[-1] += activation_shard_shape[-1]
        output_memory_config.shard_spec.shape = output_shard_shape

        memory_config = activation.memory_config()
        memory_config.shard_spec.shape = [output_shard_shape[0], activation_shard_shape[1]]
        memory_config.shard_spec.grid = output_memory_config.shard_spec.grid
        memory_config.shard_spec.orientation = output_memory_config.shard_spec.orientation
        x = ttnn.reshard(activation, memory_config)
        ttnn.deallocate(activation)

        x_tile = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(x)

        return ttnn.concat([x_tile, residual_tile], dim=dim, memory_config=output_memory_config, groups=groups)

    else:
        output_memory_config = residual.memory_config()
        activation_shard_shape = activation.memory_config().shard_spec.shape
        output_shard_shape = output_memory_config.shard_spec.shape
        output_shard_shape[-1] += activation_shard_shape[-1]
        output_memory_config.shard_spec.shape = output_shard_shape

        memory_config = activation.memory_config()
        memory_config.shard_spec.shape = [output_shard_shape[0], activation_shard_shape[1]]
        memory_config.shard_spec.grid = output_memory_config.shard_spec.grid
        memory_config.shard_spec.orientation = output_memory_config.shard_spec.orientation
        x = ttnn.reshard(activation, memory_config)
        ttnn.deallocate(activation)

        return ttnn.concat([x, residual], dim=dim, memory_config=output_memory_config, groups=groups)


class UNetConv2D:
    def __init__(
        self,
        conv,
        bn=None,
        device=None,
        activation="relu",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        output_layout=ttnn.TILE_LAYOUT,
        reshard_if_not_optimal=False,
        reallocate_halo_output=False,
        override_core_grid=None,
        mesh_mapper=None,
    ):
        assert is_valid_device_for_unet(device), "UNet Shallow requires an 8x8 grid on all devices"

        self.device = device
        self.batch_size = conv.batch_size
        self.input_height = conv.input_height
        self.input_width = conv.input_width
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.use_1d_systolic_array = conv.use_1d_systolic_array
        self.mesh_mapper = mesh_mapper

        shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            if self.use_1d_systolic_array
            else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )

        assert (not reshard_if_not_optimal) or (
            reshard_if_not_optimal or override_core_grid
        ), f"Cannot enable `reshard_if_not_optimal` (was {reshard_if_not_optimal}) and `override_core_grid` (was {override_core_grid}) at the same time "

        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=True,
            enable_act_double_buffer=(
                conv.use_activation_double_buffer if "use_activation_double_buffer" in conv else False
            ),
            enable_split_reader=(conv.use_split_reader if "use_split_reader" in conv else False),
            enable_subblock_padding=False,
            activation=activation,
            output_layout=output_layout,
            reshard_if_not_optimal=reshard_if_not_optimal,
            reallocate_halo_output=reallocate_halo_output,
            enable_weights_double_buffer=True,
        )

        if override_core_grid is not None:
            self.conv_config.core_grid = get_core_grid_from_num_cores(
                override_core_grid,
                grid_rows=8 if is_wormhole_b0(self.device) else 11,
                grid_cols=8 if is_wormhole_b0(self.device) else 10,
            )
            self.conv_config.override_sharding_config = True

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        config_override = conv.conv_blocking_and_parallelization_config_override
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if bn is not None:
            weight, bias = fold_batch_norm2d_into_conv2d(conv.module, bn.module)
        else:
            weight, bias = conv.module.weight, conv.module.bias

        bias = torch.reshape(bias, (1, 1, 1, -1))

        self.weight = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        self.bias = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

    def get_conv2d_kwargs(self):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "batch_size": self.batch_size,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": [1, 1],
            "groups": self.groups,
            "device": self.device,
            "conv_config": self.conv_config,
        }

    def __call__(self, x):
        x, [self.weight, self.bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            compute_config=self.compute_config,
            return_output_dim=False,
            return_weights_and_bias=True,
            **self.get_conv2d_kwargs(),
        )
        return x


class UNetMaxPool2D:
    def __init__(self, pool, channels, device=None):
        self.pool = pool
        self.channels = channels
        self.device = device

    def __call__(self, x):
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.pool.batch_size,
            input_h=self.pool.input_height,
            input_w=self.pool.input_width,
            channels=self.channels,
            kernel_size=[self.pool.kernel_size, self.pool.kernel_size],
            stride=[self.pool.stride, self.pool.stride],
            padding=[self.pool.padding, self.pool.padding],
            dilation=[self.pool.dilation, self.pool.dilation],
        )
        return x


class UNetDownblock:
    def __init__(
        self,
        conv1,
        bn1,
        conv2,
        bn2,
        pool,
        device,
        mesh_mapper=None,
        reshard_if_not_optimal=True,
        override_core_grid=None,
    ):
        self.conv1 = UNetConv2D(
            conv1,
            bn=bn1,
            device=device,
            reshard_if_not_optimal=reshard_if_not_optimal,
            mesh_mapper=mesh_mapper,
            override_core_grid=override_core_grid,
        )
        self.conv2 = UNetConv2D(conv2, bn=bn2, device=device, mesh_mapper=mesh_mapper)
        self.pool1 = UNetMaxPool2D(pool, conv2.out_channels, device=device)

    def __call__(self, x):
        expected_HW = self.conv1.input_height * self.conv1.input_width
        assert list(x.shape) == [
            1,
            1,
            expected_HW * self.conv1.batch_size,
            x.shape[-1],
        ], f"Downblock input is shape {list(x.shape)}, expected [1,1,BHW,C]"
        x = self.conv1(x)
        x = self.conv2(x)
        x = ttnn.move(x)
        residual = x
        x = self.pool1(x)
        return x, residual


class UNetUpblock:
    def __init__(
        self,
        conv1,
        bn1,
        conv2,
        bn2,
        conv3,
        bn3,
        device,
        mesh_mapper=None,
        reshard_if_not_optimal=True,
        final_block=False,
        override_core_grid=None,
    ):
        self.final_block = final_block
        self.device = device

        self.conv1 = UNetConv2D(
            conv1,
            bn1,
            device,
            reshard_if_not_optimal=reshard_if_not_optimal,
            mesh_mapper=mesh_mapper,
            reallocate_halo_output=True,
            override_core_grid=override_core_grid,
        )
        self.conv2 = UNetConv2D(conv2, bn2, device, mesh_mapper=mesh_mapper)
        self.conv3 = UNetConv2D(conv3, bn3, device, mesh_mapper=mesh_mapper)

        self.batch_size = conv1.batch_size
        self.input_height = conv1.input_height
        self.input_width = conv1.input_width

    def upsample(self, x):
        # Need to reshape into (B, H, W, C) to get correct output from ttnn.upsample
        x = ttnn.reshape(x, (self.batch_size, self.input_height // 2, self.input_width // 2, x.shape[-1]))
        nhw = x.shape[0] * x.shape[1] * x.shape[2]
        num_cores = determine_num_cores_for_upsample(
            nhw, x.shape[2], max_cores=64 if is_wormhole_b0(self.device) else 110
        )
        core_grid = get_core_grid_from_num_cores(
            num_cores,
            grid_rows=8 if is_wormhole_b0(self.device) else 11,
            grid_cols=8 if is_wormhole_b0(self.device) else 10,
        )
        shardspec = ttnn.create_sharded_memory_config_(
            x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )

        if not x.is_sharded():
            x = ttnn.interleaved_to_sharded(x, shardspec)

        upsampled = ttnn.upsample(x, (2, 2))  # , memory_config=x.memory_config())
        ttnn.deallocate(x)
        return ttnn.reshape(
            upsampled,
            [1, 1, self.batch_size * self.input_height * self.input_width, upsampled.shape[-1]],
        )

    def __call__(self, x, residual):
        x_rm = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x)
        residual_rm = ttnn.to_layout(residual, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(residual)

        x_upsampled = self.upsample(x_rm)
        ttnn.deallocate(x_rm)

        if not residual_rm.is_sharded():
            core_grid = get_core_grid_from_num_cores(
                x_upsampled.memory_config().shard_spec.num_cores(),
                grid_rows=8 if is_wormhole_b0(self.device) else 11,
                grid_cols=8 if is_wormhole_b0(self.device) else 10,
            )
            mem_cfg = ttnn.create_sharded_memory_config_(
                residual_rm.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
            )
            new_resid = ttnn.to_memory_config(residual_rm, mem_cfg)
            ttnn.deallocate(residual_rm)
            residual_rm = new_resid

        y = concatenate(x_upsampled, residual_rm, dim=-1, groups=self.conv1.groups, final_block=self.final_block)
        ttnn.deallocate(x_upsampled)
        ttnn.deallocate(residual_rm)

        if self.final_block:
            y = ttnn.move(y)

        out = self.conv1(y)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class UNet:
    def __init__(self, parameters: ParameterDict, device, mesh_mapper=None):
        assert is_valid_device_for_unet(device), "UNet Shallow requires an 8x8 grid on all devices"

        self.device = device
        self.downblock1 = UNetDownblock(
            parameters.c1,
            parameters.b1,
            parameters.c1_2,
            parameters.b1_2,
            parameters.p1,
            device,
            override_core_grid=63 if is_wormhole_b0(self.device) else None,
            reshard_if_not_optimal=not is_wormhole_b0(self.device),
            mesh_mapper=mesh_mapper,
        )
        self.downblock2 = UNetDownblock(
            parameters.c2,
            parameters.b2,
            parameters.c2_2,
            parameters.b2_2,
            parameters.p2,
            device,
            reshard_if_not_optimal=False,
            mesh_mapper=mesh_mapper,
        )
        self.downblock3 = UNetDownblock(
            parameters.c3,
            parameters.b3,
            parameters.c3_2,
            parameters.b3_2,
            parameters.p3,
            device,
            mesh_mapper=mesh_mapper,
        )
        self.downblock4 = UNetDownblock(
            parameters.c4,
            parameters.b4,
            parameters.c4_2,
            parameters.b4_2,
            parameters.p4,
            device,
            mesh_mapper=mesh_mapper,
        )

        self.bnc = UNetConv2D(
            parameters.bnc,
            parameters.bnb,
            device,
            reshard_if_not_optimal=True,
            mesh_mapper=mesh_mapper,
        )
        self.bnc2 = UNetConv2D(
            parameters.bnc_2,
            parameters.bnb_2,
            device,
            mesh_mapper=mesh_mapper,
        )

        self.upblock1 = UNetUpblock(
            parameters.c5,
            parameters.b5,
            parameters.c5_2,
            parameters.b5_2,
            parameters.c5_3,
            parameters.b5_3,
            device,
            final_block=False,
            mesh_mapper=mesh_mapper,
        )
        self.upblock2 = UNetUpblock(
            parameters.c6,
            parameters.b6,
            parameters.c6_2,
            parameters.b6_2,
            parameters.c6_3,
            parameters.b6_3,
            device,
            final_block=False,
            mesh_mapper=mesh_mapper,
        )
        self.upblock3 = UNetUpblock(
            parameters.c7,
            parameters.b7,
            parameters.c7_2,
            parameters.b7_2,
            parameters.c7_3,
            parameters.b7_3,
            device,
            override_core_grid=63 if is_wormhole_b0(self.device) else None,
            reshard_if_not_optimal=not is_wormhole_b0(self.device),
            final_block=False,
            mesh_mapper=mesh_mapper,
        )
        self.upblock4 = UNetUpblock(
            parameters.c8,
            parameters.b8,
            parameters.c8_2,
            parameters.b8_2,
            parameters.c8_3,
            parameters.b8_3,
            device,
            reshard_if_not_optimal=False,
            final_block=True,  # Special case due to high memory pressure in final upblock
            mesh_mapper=mesh_mapper,
        )

        self.output_layer = UNetConv2D(
            parameters.output_layer,
            device=device,
            activation="",
            mesh_mapper=mesh_mapper,
            activation_dtype=ttnn.bfloat16,
        )

    input_core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
        }
    )
    input_shard_shape = (16, 2688)
    input_shard_spec = ttnn.ShardSpec(input_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_sharded_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )

    def bottleneck(self, x):
        x = self.bnc(x)
        x = self.bnc2(x)
        return x

    def preprocess_input_tensor(self, x, deallocate_input_activation):
        out = preprocess_unet_input_tensor(x)
        if deallocate_input_activation:
            ttnn.deallocate(x)  # Some use-cases have a persistent input tensor that we don't want to delete
        return out

    def postprocess_output_tensor(self, x):
        # Convert the output tensor (in TILE layout) to RM to prevent transferring padding back to host.
        assert x.is_sharded(), "Expected output to be sharded"
        input_shard_spec = x.memory_config().shard_spec
        output_shard_shape = (x.shape[-1], input_shard_spec.shape[0])
        output_shard_spec = ttnn.ShardSpec(
            input_shard_spec.grid,
            output_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec
        )
        return ttnn.experimental.convert_to_chw(x, memory_config=output_memory_config, dtype=ttnn.bfloat16)

    def __call__(self, x, move_input_tensor_to_device=True, deallocate_input_activation=True):
        assert len(x.shape) == 4, f"Expected UNet input tensors to be rank 4 (was {len(x.shape)})"

        if move_input_tensor_to_device:
            assert (
                x.storage_type() == ttnn.StorageType.HOST
            ), "Expected UNet input tensor to be on host if move_input_tensor_to_device=True"
            B, C, H, W = x.shape
            x = ttnn.reshape(x, [1, 1, C, H * W])  # Reshape so we can width-shard along inner-dim
            x = ttnn.to_device(x, device=self.device, memory_config=self.input_sharded_memory_config)
        else:
            assert (
                x.storage_type() == ttnn.StorageType.DEVICE
            ), "Expected UNet input tensor to be on device if move_input_tensor_to_device=False"

        x = self.preprocess_input_tensor(
            x, deallocate_input_activation=deallocate_input_activation or move_input_tensor_to_device
        )

        x, c1_res = self.downblock1(x)
        x, c2_res = self.downblock2(x)
        x, c3_res = self.downblock3(x)
        x, c4_res = self.downblock4(x)

        x = self.bottleneck(x)

        x = self.upblock1(x, c4_res)
        ttnn.deallocate(c4_res)
        x = self.upblock2(x, c3_res)
        ttnn.deallocate(c3_res)
        x = self.upblock3(x, c2_res)
        ttnn.deallocate(c2_res)
        x = self.upblock4(x, c1_res)
        ttnn.deallocate(c1_res)

        x = self.output_layer(x)

        return self.postprocess_output_tensor(x)
