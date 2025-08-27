import ttnn
import math


def _nearest_32_per_core(x, core):
    return math.ceil(x / core / 32) * 32 * core


def _nearest_32(x):
    return math.ceil(x / 32) * 32


class Conv:
    def __init__(
        self,
        parameters,
        conv_pt,
        *,
        stride=1,
        padding=1,
        has_bias=False,
        act_block_h=32,
        reshard=False,
        deallocate=False,
        height_sharding=None,
        activation="",
        width_sharding=False,
        block_sharding=False,
        dtype=ttnn.bfloat8_b,
        output_layout=ttnn.TILE_LAYOUT,
        is_sliced=False,
    ) -> None:
        self.weights = parameters.weight
        # print(f"Conv weights: {self.weights.shape}")
        # print(f"Conv parameters: {self.weights}")
        self.conv_pt = conv_pt
        # print(f"Conv: {self.conv_pt}")
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = parameters.bias

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.stride = conv_pt.stride  # stride
        self.padding = conv_pt.padding  # padding
        self.out_channels = conv_pt.out_channels
        # print(f"Conv out channels: {self.out_channels}")
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.dtype = dtype
        self.output_layout = output_layout

        if width_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        elif height_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif block_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        else:
            self.shard_layout = None

        self.deallocate = deallocate
        self.activation = activation
        self.is_sliced = is_sliced
        self.slice_config = None
        if is_sliced:
            self.slice_config = ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,
                num_slices=2,
            )

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape if self.has_bias else ''} {self.kernel_size}"

    def __call__(self, device, input_tensor):
        # print(f"ULAZ: Conv output layout: {self.output_layout}")
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=self.shard_layout,
            deallocate_activation=self.deallocate,
            activation=self.activation,
            # reshard_if_not_optimal=True,
            output_layout=self.output_layout,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        # print(
        #     f"inpit_tensor shape: {input_tensor.shape}, conv_pt: {self.conv_pt} stride: {self.stride}, padding: {self.padding}"
        # )
        [output_tensor, [out_h, out_w]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias if self.has_bias else None,
            in_channels=self.conv_pt.in_channels,
            out_channels=self.conv_pt.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=self.conv_pt.batch_size,
            input_height=self.conv_pt.input_height,
            input_width=self.conv_pt.input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            slice_config=self.slice_config,
        )
        # print(f"Output tensor shape: {output_tensor.shape}, out_h: {out_h}, out_w: {out_w}")
        # print(
        #     f"Output tensor dtype: {output_tensor.dtype}, layout: {output_tensor.layout}, memory config: {output_tensor.memory_config}"
        # )
        return output_tensor, out_h, out_w


class GroupNorm:
    def __init__(self, parameters, num_groups, channels, eps=1e-5, dtype=ttnn.bfloat8_b, is_sliced=False):
        self.weight = parameters.weight
        self.bias = parameters.bias
        self.num_groups = num_groups
        self.channels = channels
        self.eps = eps
        self.dtype = dtype
        self.is_sliced = is_sliced
        self.num_splited_groups = num_groups
        self.num_splited_channels = channels

    def __call__(self, device, input_tensor, H, W, shard="HS", num_splits=1):
        compute_grid = device.compute_with_storage_grid_size()
        grid_size = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
        grid_y = grid_size.y
        grid_x = grid_size.x
        print(f"{grid_x=}, {grid_y=}, {shard=}, {num_splits=} {self.is_sliced=}")
        # spliting tensor into multiple splits for very large tensors

        if shard == "HS" and not self.is_sliced:
            grid_x *= grid_y
            grid_y = 1

        if num_splits > 1:
            unpadded_shape = input_tensor.shape
            print(f"Input tensor shape before tilize: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            out_shape = [
                unpadded_shape[0],
                unpadded_shape[1],
                _nearest_32_per_core(unpadded_shape[2], grid_x),
                # _nearest_32_per_core(unpadded_shape[3], grid_y),
                _nearest_32_per_core(unpadded_shape[3], grid_y),
            ]
            print(f"Output tensor shape after tilize: {out_shape}")
            # input_tensor = ttnn.tilize_with_val_padding(
            #    input_tensor, output_tensor_shape=out_shape, pad_value=0, use_multicore=True
            # )
            print(f"Input tensor shape after tilize: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        # Generate input mask
        input_mask_tensor = ttnn.create_group_norm_input_mask(self.channels, self.num_groups, grid_y)
        input_mask_tensor = ttnn.from_torch(
            input_mask_tensor,
            dtype=ttnn.DataType.BFLOAT8_B,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Generate gamma/beta tensors
        gamma = ttnn.create_group_norm_weight_bias_rm(self.weight, self.channels, grid_y)
        beta = ttnn.create_group_norm_weight_bias_rm(self.bias, self.channels, grid_y)

        gamma_t = ttnn.from_torch(
            gamma,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        beta_t = ttnn.from_torch(
            beta,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Generate shard config
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        if shard == "HS":
            # print(f"Shard height: {H}, width: {W}, grid_size: {grid_size}")
            shard_shape = (H * W) // grid_size.x // grid_size.y, self.channels
            # print(f"Shard shape: {shard_shape}")
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
            sharded_mem_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )
        elif shard == "BS":
            shard_shape = (H * W) // grid_size.x, self.channels // grid_size.y
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
            sharded_mem_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )
        # print(
        #     f"input tensor shape: {input_tensor.shape}, layout: {input_tensor.layout} memory config: {input_tensor.memory_config}"
        # )
        if not num_splits > 1:
            input_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)
            tt_output_tensor = ttnn.group_norm(
                input_tensor,
                num_groups=self.num_groups,
                input_mask=input_mask_tensor,
                weight=gamma_t,
                bias=beta_t,
                memory_config=sharded_mem_config,
                core_grid=grid_size,
                epsilon=1e-5,
                # inplace=False,
            )
        else:
            print(f"SLICED GN {input_tensor.shape=}, {input_tensor.memory_config()=}")
            tt_output_tensor = ttnn.group_norm(
                input_tensor,
                num_groups=self.num_groups,
                input_mask=input_mask_tensor,
                weight=gamma_t,
                bias=beta_t,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_layout=ttnn.TILE_LAYOUT,
                core_grid=grid_size,
                inplace=False,
                num_out_blocks=num_splits,
                epsilon=1e-5,
            )
            # ttnn.synchronize_device(device)

        return tt_output_tensor


class GroupNormDRAM:
    def __init__(self, parameters, num_groups, channels, eps=1e-5, dtype=ttnn.bfloat8_b, is_sliced=False):
        self.weight = parameters.weight
        self.bias = parameters.bias
        self.num_groups = num_groups
        self.channels = channels
        self.eps = eps
        self.dtype = dtype
        self.is_sliced = is_sliced
        self.num_splited_groups = num_groups
        self.num_splited_channels = channels

    def __call__(self, device, input_tensor, H, W, shard="HS", num_splits=1):
        compute_grid = device.compute_with_storage_grid_size()
        grid_x, grid_y = compute_grid.x, compute_grid.y
        if num_splits > 4:
            grid_y = 2
        grid_size = ttnn.CoreGrid(y=grid_y, x=grid_x)

        # torch input tensor
        unpadded_shape = input_tensor.shape
        out_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32_per_core(unpadded_shape[2], grid_x),
            _nearest_32_per_core(unpadded_shape[3], grid_y),
        ]
        print(f"unpadded_shape: {unpadded_shape} out_shape: {out_shape}")
        input_tensor_tilized = ttnn.tilize_with_val_padding(
            input_tensor, output_tensor_shape=out_shape, pad_value=0, use_multicore=True
        )
        print(
            f"input_tensor_tilized shape: {input_tensor_tilized.shape} padded shape: {input_tensor_tilized.padded_shape}"
        )
        input_mask_tensor = ttnn.create_group_norm_input_mask(self.channels, self.num_groups, grid_size.y)
        input_mask_tensor = ttnn.from_torch(
            input_mask_tensor,
            dtype=ttnn.DataType.BFLOAT8_B,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # gamma/beta
        gamma = ttnn.create_group_norm_weight_bias_rm(self.weight, self.channels, grid_size.y)
        beta = ttnn.create_group_norm_weight_bias_rm(self.bias, self.channels, grid_size.y)
        gamma_t = ttnn.from_torch(
            gamma,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        beta_t = ttnn.from_torch(
            beta,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # groupnorm
        output_tensor = ttnn.group_norm(
            input_tensor_tilized,
            num_groups=self.num_groups,
            input_mask=input_mask_tensor,
            weight=gamma_t,
            bias=beta_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            core_grid=grid_size,
            inplace=False,
            num_out_blocks=num_splits,
            epsilon=1e-5,
        )

        # ttnn.synchronize_device(device)
        return output_tensor
