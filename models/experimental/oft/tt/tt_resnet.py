import ttnn
import torch.nn as nn


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
        self.stride = stride
        self.padding = padding
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

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape if self.has_bias else ''} {self.kernel_size}"

    def __call__(self, device, input_tensor):
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

        print(
            f"inpit_tensor shape: {input_tensor.shape}, conv_pt: {self.conv_pt} stride: {self.stride}, padding: {self.padding}"
        )
        [output_tensor, [out_h, out_w]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias if self.has_bias else None,
            in_channels=self.conv_pt.in_channels,
            out_channels=self.conv_pt.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            batch_size=self.conv_pt.batch_size,
            input_height=self.conv_pt.input_height,
            input_width=self.conv_pt.input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
        )
        print(f"Output tensor shape: {output_tensor.shape}, out_h: {out_h}, out_w: {out_w}")
        print(
            f"Output tensor dtype: {output_tensor.dtype}, layout: {output_tensor.layout}, memory config: {output_tensor.memory_config}"
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
        grid_y = grid_size.y
        grid_x = grid_size.x
        if shard == "HS":
            grid_x *= grid_y
            grid_y = 1
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
        print(
            f"input tensor shape: {input_tensor.shape}, layout: {input_tensor.layout} memory config: {input_tensor.memory_config}"
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
    expansion = 1

    def __init__(
        self, device, parameters, conv_pt, inplanes, planes, channels, cell_size, grid_height, stride=1, scale=1
    ):
        # super().__init__()
        print("------------BOJANJE TTBasicBlock------------")
        # print(f"TTBasicBlock: {parameters=},\n {conv_pt=},\n {inplanes=},\n {planes=},\n {channels=},\n {cell_size=},\n {grid_height=},\n {stride=}")
        self.conv1 = Conv(parameters.conv1, conv_pt.conv1, stride=stride, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bn1 = GroupNorm(parameters.bn1, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat8_b)

        self.conv2 = Conv(parameters.conv2, conv_pt.conv2, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bn2 = GroupNorm(parameters.bn2, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat8_b)

        if stride != 1 or inplanes != planes:
            self.downsample = True
            self.downsample_conv = Conv(
                parameters.downsample[0],
                conv_pt.downsample[0],
                stride=stride,
                padding=0,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self.downsample_bn = GroupNorm(
                parameters.downsample[1], num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat8_b
            )
        else:
            self.downsample = None

    def forward(self, device, x, gn_shard="HS"):
        # identity = x
        # x1 = x
        print("TTBasicBlock forward")
        out, out_h, out_w = self.conv1(device, x)
        print(f"FORWARD X Input shape: {x.shape}, dtype: {x.dtype}, layout: {x.layout}")
        print(f"Conv1 output shape: {out.shape}, out_h: {out_h}, out_w: {out_w}")
        # print(f"conv1 output dtype: {out.dtype} layout: {out.layout} shard layout: {out.memory_config}")
        out = ttnn.move(out)
        print(f"SSHARDING {gn_shard=}")
        out = self.bn1(device, out, out_h, out_w, shard=gn_shard)
        print(f"BN1 output shape: {out.shape}")
        out = ttnn.relu(out)
        # ttnn.deallocate(out)
        # ttnn.move(out1)
        print(f"ReLU output shape: {out.shape}, dtype: {out.dtype}")
        out, out_h, out_w = self.conv2(device, out)

        print(f"Conv2 output shape: {out.shape}")
        # print(f"conv2 output dtype: {out.dtype} layout: {out.layout} shard layout: {out.memory_config}")
        out = ttnn.move(out)
        out = self.bn2(device, out, out_h, out_w, shard=gn_shard)
        print(f"BN2 output shape: {out.shape}")

        if self.downsample is not None:
            print(f"Downsample output shape: {x.shape} self.downsample: {self.downsample}")
            x, out_h_ds, out_w_ds = self.downsample_conv(device, x)
            print(f"Downsample conv output shape: {x.shape}, out_h: {out_h}, out_w: {out_w}")
            x = self.downsample_bn(device, x, out_h_ds, out_w_ds, shard=gn_shard)
        else:
            print(f"reshape x shape: {x.shape} self.downsample: {self.downsample}")
            # x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))

        # print(f"{out.shape=}, {x.shape=}")
        # print(f"memory config: {out.memory_config=},\n {x.memory_config=}")
        print(
            f"FORWARD X Output shape: {x.shape}, dtype: {x.dtype}, layout: {x.layout} memory config: {x.memory_config()}"
        )
        print(
            f"FORWARD OUT Output shape: {out.shape}, dtype: {out.dtype}, layout: {out.layout} memory config: {out.memory_config()}"
        )

        print(f"------------------Memory layout------------------")
        print(f"X memory layout: {x.layout}, dtype: {x.dtype}, memory config: {x.memory_config()}")
        print(f"OUT memory layout: {out.layout}, dtype: {out.dtype}, memory config: {out.memory_config()}")
        print("--------------------------------------------------")

        #         print(f"X shape: {x.shape}, OUT shape: {out.shape}")
        #         if out.is_sharded() and out.memory_config().memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        # #             # Reshardujte tensor u HEIGHT_SHARDED
        #             # Create height-sharded memory config
        #             height_sharded_config = ttnn.create_sharded_memory_config(
        #                 shape=[out.shape[2]//20, out.shape[3]],  # e.g., [12, 128] for 8 cores
        #                 core_grid=ttnn.CoreGrid(y=4, x=5),
        #                 strategy=ttnn.ShardStrategy.HEIGHT,
        #                 orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #                 )
        #             out_hs = ttnn.to_memory_config(out, height_sharded_config)
        #             print(f"Resharded output tensor {out.shape} to HEIGHT_SHARDED with config: {height_sharded_config}")
        #         if x.is_sharded() and x.memory_config().memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        #             # Reshardujte tensor u HEIGHT_SHARDED
        #             # Create height-sharded memory config
        #             height_sharded_config = ttnn.create_sharded_memory_config(
        #                 shape=[x.shape[2]//20, x.shape[3]],  # e.g., [12, 128] for 8 cores
        #                 core_grid=ttnn.CoreGrid(y=4, x=5),
        #                 strategy=ttnn.ShardStrategy.HEIGHT,
        #                 orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #             )
        #             x_hs = ttnn.to_memory_config(x, height_sharded_config)
        #             print(f"Resharded input tensor {x.shape} to HEIGHT_SHARDED with config: {height_sharded_config}")

        #             print(f"Resharding output tensor {out.shape} to HEIGHT_SHARDED"
        #             print(f"Resharding output tensor {out.shape} to HEIGHT_SHARDED"
        #             height_sharded_config = ttnn.create_sharded_memory_config(
        #             shard_shape=[, tensor_width],  # Height will be divided across cores
        #             core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(19, 0))}),  # 20 cores in a line
        #             strategy=ttnn.ShardStrategy.HEIGHT,
        #             orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #             use_height_and_width_as_shard_shape=True
        # )
        #             out = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)
        #         if x.is_sharded() and x.memory_config().memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        #         #     # Reshardujte tensor u HEIGHT_SHARDED
        #         #     height_sharded_config = ttnn.MemoryConfig(
        #         #     memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        #         #     buffer_type=x.memory_config().buffer_type
        #         #     )
        #         #     x = ttnn.to_memory_config(x, height_sharded_config)
        #             x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        # out_tt = ttnn.add(out, x, use_legasy=False)
        if gn_shard == "HS":
            out += x
        else:
            block_sharded_config = ttnn.create_sharded_memory_config(
                shape=[out.shape[2] // 5 // 3, out.shape[3]],  # e.g., [12, 128] for 8 cores
                core_grid=ttnn.CoreGrid(y=3, x=5),  # 20 cores in a line
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            out = ttnn.to_memory_config(out, block_sharded_config)
            x = ttnn.to_memory_config(x, block_sharded_config)
            out = ttnn.add(out, x, memory_config=block_sharded_config)
        print(f"Add output shape: {out.shape}")
        # print(f"Add output dtype: {out.dtype} layout: {out.layout} shard layout: {out.memory_config}")
        out = ttnn.relu(out)
        return out


class TTResNetFeatures:
    def __init__(self, device, parameters, conv_pt, block, layers):
        self.inplanes = 64

        # print(f"TTResNetFeatures: {parameters=},\n {conv_pt=},\n {block=},\n {layers=}")
        # with open("models/experimental/oft/tt/params.txt", "a") as f:
        #     f.write(f"TTResNetFeatures: {parameters=},\n {conv_pt=},\n {block=},\n {layers=}\n")
        self.conv1 = Conv(parameters.conv1, conv_pt.conv1, stride=2, padding=3)
        # self.bn1 = GroupNorm(parameters.bn1, num_groups=16, channels=64, eps=1e-5)
        self.gn1 = nn.GroupNorm(16, 64)
        self.gn1.weight = nn.Parameter(parameters.bn1.weight)
        self.gn1.bias = nn.Parameter(parameters.bn1.bias)

        self.layer1 = self._make_layer(device, parameters.layer1, conv_pt.layer1, block, 64, layers[0])
        print(f"Layer1: {len(self.layer1)} blocks")
        print(f"Layer1: {self.layer1}")
        self.layer2 = self._make_layer(device, parameters.layer2, conv_pt.layer2, block, 128, layers[1], stride=2)
        print(f"Layer2: {len(self.layer2)} blocks")
        print(f"Layer2: {self.layer2}")
        self.layer3 = self._make_layer(
            device,
            parameters.layer3,
            conv_pt.layer3,
            block,
            256,
            layers[2],
            stride=2,
        )
        print(f"Layer3: {len(self.layer3)} blocks")
        print(f"Layer3: {self.layer3}")
        self.layer4 = self._make_layer(
            device,
            parameters.layer4,
            conv_pt.layer4,
            block,
            512,
            layers[3],
            stride=2,
        )
        print(f"Layer4: {len(self.layer4)} blocks")
        print(f"Layer4: {self.layer4}")

    def _make_layer(self, device, parameters, conv_pt, block, planes, blocks, stride=1):
        layers = []
        layers.append(
            block(
                device,
                parameters[0],
                conv_pt[0],
                self.inplanes,
                planes,
                planes,
                32,
                32,
                stride,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    device,
                    parameters[i],
                    conv_pt[i],
                    inplanes=self.inplanes,
                    planes=planes,
                    channels=planes,
                    cell_size=32,
                    grid_height=32,
                )
            )
        return layers

    def _run_layer(self, device, x, layer, gn_shard="HS"):
        print(f"Running layer with gn_shard: {gn_shard}")
        for block in layer:
            x = block.forward(device, x, gn_shard)
        return x

    def forward(self, device, x):
        print(f"Input shape: {x.shape}, dtype: {x.dtype}, layout: {x.layout}")
        conv1, out_h, out_w = self.conv1(device, x)
        print(f"Conv1 output shape: {conv1.shape}, out_h: {out_h}, out_w: {out_w}")
        # print(f"Conv1 output shape: {conv1.shape}, out_h: {out_h}, out_w: {out_w}")
        # conv1 = ttnn.untilize(conv1, memory_config=ttnn.DRAM_MEMORY_CONFIG,  use_multicore=True)
        # conv1 = self.bn1(device, conv1, out_h, out_w, shard="HS")
        conv1 = ttnn.to_torch(conv1).reshape(conv1.shape[0], out_h, out_w, conv1.shape[-1]).permute((0, 3, 1, 2))

        conv1 = self.gn1(conv1)

        print(f"Conv1 output shape after GN: {conv1.shape}, dtype: {conv1.dtype}, layout: {conv1.layout}")
        N, C, H, W = conv1.shape
        conv1 = conv1.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)  # [1, 1, N*H*W, C]
        conv1 = ttnn.from_torch(
            conv1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
        )
        print(f"Conv1 output shape after GN: {conv1.shape}, dtype: {conv1.dtype}, layout: {conv1.layout}")
        # memory_config = ttnn.MemoryConfig(
        #     ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        #     ttnn.BufferType.L1,
        #     ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}), [(H * W) // 20, C], ttnn.ShardOrientation.ROW_MAJOR)
        # )
        # conv1 = ttnn.to_memory_config(conv1, memory_config=memory_config)
        print(f"Conv1 output shape after memory config: {conv1.shape}, dtype: {conv1.dtype}, layout: {conv1.layout}")
        conv1 = ttnn.relu(conv1)
        print(f"ReLU output shape: {conv1.shape}, dtype: {conv1.dtype}, layout: {conv1.layout}")
        print(f"relu output shape: {conv1.shape} {out_h=} {out_w=}")

        cv1 = conv1[:, :, :, :32]  # Assuming conv1 has shape [N, C, H, W] and we want to keep the first 32 channels
        print(f"cv1 shape: {cv1.shape}, dtype: {cv1.dtype}, layout: {cv1.layout}")
        cv2 = conv1[:, :, :, 32:]  # The rest of the channels
        print(f"cv2 shape: {cv2.shape}, dtype: {cv2.dtype}, layout: {cv2.layout}")
        ttnn.deallocate(conv1)
        print(f"cv1 shape: {cv1.shape}, cv2 shape: {cv2.shape}")
        conv1 = ttnn.max_pool2d(
            input_tensor=cv1,
            batch_size=cv1.shape[0],
            input_h=out_h,  # conv1.shape[1],
            input_w=out_w,  # conv1.shape[2],
            channels=cv1.shape[3],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(cv1)
        conv2 = ttnn.max_pool2d(
            input_tensor=cv2,
            batch_size=cv2.shape[0],
            input_h=out_h,  # conv1.shape[1],
            input_w=out_w,  # conv1.shape[2],
            channels=cv1.shape[3],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(cv2)
        print(f"Max pool output shape: {conv1.shape=}, {conv2.shape=}")
        conv_c = ttnn.concat([conv1, conv2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"Concat output shape: {conv1.shape}, dtype: {conv1.dtype}, layout: {conv1.layout}")
        ttnn.deallocate(conv1)
        ttnn.deallocate(conv2)
        conv_c = ttnn.move(conv_c)
        # print(f"Concat output shape after deallocate: {conv_c.shape}, dtype: {conv_c.dtype}, layout: {conv_c.layout}")
        feats4 = self._run_layer(device, conv_c, self.layer1)
        # print(f"Feats4 output shape: {feats4.shape}, dtype: {feats4.dtype}, layout: {feats4.layout}")
        ttnn.deallocate(conv_c)
        feats8 = self._run_layer(device, feats4, self.layer2)
        # feats8_clone = ttnn.clone(feats8, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(feats4)
        feats16 = self._run_layer(device, feats8, self.layer3)
        ttnn.deallocate(feats8)
        feats32 = self._run_layer(device, feats16, self.layer4, gn_shard="BS")

        # print(f"Feats4 output shape: {feats4.shape}, dtype: {feats4.dtype}, layout: {feats4.layout}")
        # print(f"Feats8 output shape: {feats8.shape}, dtype: {feats8.dtype}, layout: {feats8.layout}")
        return feats32
