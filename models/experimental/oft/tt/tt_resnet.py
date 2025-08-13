import ttnn


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
        height_sharding=False,
        activation="",
        width_sharding=False,
        block_sharding=False,
        dtype=ttnn.bfloat8_b,
    ) -> None:
        self.weights = parameters.weight

        self.conv_pt = conv_pt
        print(f"Conv: {self.conv_pt}")
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = parameters.bias

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.stride = stride
        self.padding = padding
        self.out_channels = conv_pt["out_channels"]
        print(f"Conv out channels: {self.out_channels}")
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.dtype = dtype

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

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape if self.has_bias else ''} {self.kernel_size}"

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=self.shard_layout,
            deallocate_activation=self.deallocate,
            activation=self.activation,
            # reshard_if_not_optimal=True,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        print(
            f"inpit_tensor shape: {input_tensor.shape}, conv_pt: {self.conv_pt} stride: {self.stride}, padding: {self.padding}"
        )
        [output_tensor, [out_h, out_w]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias if self.has_bias else None,
            in_channels=self.conv_pt["in_channels"],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            batch_size=self.conv_pt["batch_size"],
            input_height=self.conv_pt["input_height"],
            input_width=self.conv_pt["input_width"],
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
        )
        return output_tensor, out_h, out_w


class GroupNorm:
    def __init__(self, parameters, num_groups, channels, eps=1e-5, dtype=ttnn.bfloat8_b):
        self.weight = parameters.weight
        self.bias = parameters.bias
        self.num_groups = num_groups
        self.channels = channels
        self.eps = eps
        self.dtype = dtype

    def __call__(self, device, input_tensor, H, W, shard="HS"):
        compute_grid = device.compute_with_storage_grid_size()
        grid_size = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
        # Generate input mask
        input_mask_tensor = ttnn.create_group_norm_input_mask(self.channels, self.num_groups, grid_size.y)
        input_mask_tensor = ttnn.from_torch(
            input_mask_tensor,
            dtype=ttnn.DataType.BFLOAT8_B,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Generate gamma/beta tensors
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
        # Generate shard config
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        if shard == "HS":
            print(f"Shard height: {H}, width: {W}, grid_size: {grid_size}")
            shard_shape = (H * W) // grid_size.x // grid_size.y, self.channels
            print(f"Shard shape: {shard_shape}")
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
        input_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)

        out = ttnn.group_norm(
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

        return out


class TTBasicBlock:
    def __init__(
        self, device, parameters, conv_pt, inplanes, planes, channels, cell_size, grid_height, stride=1, scale=1
    ):
        super().__init__()

        self.conv1 = Conv(parameters["conv1"], conv_pt["conv1"], stride=stride)
        self.bn1 = GroupNorm(parameters.bn1, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat8_b)

        self.conv2 = Conv(parameters["conv2"], conv_pt["conv2"])
        self.bn2 = GroupNorm(parameters.bn2, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat8_b)

        if stride != 1 or inplanes != planes:
            self.downsample = True
            self.downsample_conv = Conv(parameters.downsample[0], conv_pt["conv_downsample"], stride=stride, padding=0)
            self.downsample_bn = GroupNorm(
                parameters.downsample[1], num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat8_b
            )
        else:
            self.downsample = None

    def forward(self, device, x):
        # identity = x
        out, out_h, out_w = self.conv1(device, x)
        print(f"Conv1 output shape: {out.shape}, out_h: {out_h}, out_w: {out_w}")
        print(f"conv1 output dtype: {out.dtype} layout: {out.layout} shard layout: {out.memory_config}")
        out = self.bn1(device, out, out_h, out_w, shard="HS")
        print(f"BN1 output shape: {out.shape}")
        out = ttnn.relu(out)
        # ttnn.deallocate(out)
        # ttnn.move(out1)
        print(f"ReLU output shape: {out.shape}, dtype: {out.dtype}")
        out, out_h, out_w = self.conv2(device, out)

        print(f"Conv2 output shape: {out.shape}")
        print(f"conv2 output dtype: {out.dtype} layout: {out.layout} shard layout: {out.memory_config}")
        out = self.bn2(device, out, out_h, out_w, shard="HS")
        print(f"BN2 output shape: {out.shape}")

        if self.downsample is not None:
            print(f"Downsample output shape: {x.shape} self.downsample: {self.downsample}")
            x, out_h, out_w = self.downsample_conv(device, x)
            print(f"Downsample conv output shape: {x.shape}, out_h: {out_h}, out_w: {out_w}")
            x = self.downsample_bn(device, x, out_h, out_w, shard="HS")
        else:
            print(f"reshape x shape: {x.shape} self.downsample: {self.downsample}")
            x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))

        print(f"{out.shape=}, {x.shape=}")
        print(f"memory config: {out.memory_config=},\n {x.memory_config=}")

        out += x
        # out = ttnn.add(out, x)
        print(f"Add output shape: {out.shape}")
        print(f"Add output dtype: {out.dtype} layout: {out.layout} shard layout: {out.memory_config}")
        out = ttnn.relu(out)
        return out
