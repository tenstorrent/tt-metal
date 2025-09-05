import ttnn
from models.experimental.oft.tt.common import Conv, GroupNorm, GroupNormDRAM

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TTBasicBlock:
    expansion = 1

    def __init__(self, device, parameters, conv_pt, inplanes, planes, stride=1, scale=1, is_sliced=False):
        self.is_sliced = is_sliced
        print(f"TTBasicBlock: {inplanes=},\n {planes=},\n {stride=},\n {is_sliced=}")
        self.conv1 = Conv(
            parameters.conv1, conv_pt.conv1, stride=stride, output_layout=ttnn.ROW_MAJOR_LAYOUT, is_sliced=is_sliced
        )
        if not is_sliced:
            self.bn1 = GroupNorm(parameters.bn1, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat8_b)
        else:
            self.bn1 = GroupNormDRAM(parameters.bn1, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat8_b)
        self.conv2 = Conv(parameters.conv2, conv_pt.conv2, output_layout=ttnn.ROW_MAJOR_LAYOUT, is_sliced=is_sliced)
        if not is_sliced:
            self.bn2 = GroupNorm(parameters.bn2, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat8_b)
        else:
            self.bn2 = GroupNormDRAM(parameters.bn2, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat8_b)
        self.downsample = None
        if not is_sliced:
            if stride != 1 or inplanes != planes:
                self.downsample = True
                self.downsample_conv = Conv(
                    parameters.downsample[0],
                    conv_pt.downsample[0],
                    stride=stride,
                    padding=0,
                    output_layout=ttnn.ROW_MAJOR_LAYOUT,
                    is_sliced=is_sliced,
                )
                self.downsample_bn = GroupNorm(
                    parameters.downsample[1], num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat8_b
                )

    def forward(self, device, x, gn_shard="HS", num_splits=1):
        # identity = x
        # x1 = x
        # print("TTBasicBlock forward")
        if use_signpost:
            signpost(header="TTBasicBlock forward started")
        out, out_h, out_w = self.conv1(device, x)
        print(f"FORWARD X Input shape: {x.shape}, dtype: {x.dtype}, layout: {x.layout}")
        # print(f"Conv1 output shape: {out.shape}, out_h: {out_h}, out_w: {out_w}")
        # print(f"conv1 output dtype: {out.dtype} layout: {out.layout} shard layout: {out.memory_config}")
        out = ttnn.move(out)
        # print(f"SSHARDING {gn_shard=}")
        out = self.bn1(device, out, out_h, out_w, shard=gn_shard, num_splits=num_splits)
        print(f"BN1 output shape: {out.shape}")
        # if not self.is_sliced:
        out1 = ttnn.relu(out)
        ttnn.deallocate(out)  # added for tracy pass
        out = ttnn.move(out1)  # added for tracy pass
        # print(f"ReLU output shape: {out.shape}, dtype: {out.dtype}")
        # out=ttnn.move(out) # added for tracy, dont pass without this
        out, out_h, out_w = self.conv2(device, out)

        print(f"Conv2 output shape: {out.shape}")
        # print(f"conv2 output dtype: {out.dtype} layout: {out.layout} shard layout: {out.memory_config}")
        # if not self.is_sliced:
        out = ttnn.move(out)
        out = self.bn2(device, out, out_h, out_w, shard=gn_shard, num_splits=num_splits)
        print(f"BN2 output shape: {out.shape}")

        if self.downsample is not None:
            print(f"Downsample output shape: {x.shape} self.downsample: {self.downsample}")
            # print(
            #     f"Input to downsample conv shape: {x.shape}, dtype: {x.dtype}, layout: {x.layout} memory config: {x.memory_config()}"
            # )
            x, out_h_ds, out_w_ds = self.downsample_conv(device, x)
            # print(f"Downsample conv output shape: {x.shape}, out_h: {out_h}, out_w: {out_w}")
            # print(f"downsample conv output dtype: {x.dtype} layout: {x.layout} shard layout: {x.memory_config()}")
            x = self.downsample_bn(device, x, out_h_ds, out_w_ds, shard=gn_shard)
        else:
            print(f"reshape x shape: {x.shape} self.downsample: {self.downsample}")
            # x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))

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
        # print(f"Add output shape: {out.shape}")
        # print(f"Add output dtype: {out.dtype} layout: {out.layout} shard layout: {out.memory_config}")
        out = ttnn.relu(out)
        if use_signpost:
            signpost(header="TTBasicBlock forward finished")
        return out


class TTResNetFeatures:
    def __init__(self, device, parameters, conv_pt, block, layers):
        self.inplanes = 64

        # print(f"TTResNetFeatures: {parameters=},\n {conv_pt=},\n {block=},\n {layers=}")
        # with open("models/experimental/oft/tt/params.txt", "a") as f:
        #     f.write(f"TTResNetFeatures: {parameters=},\n {conv_pt=},\n {block=},\n {layers=}\n")
        self.conv1 = Conv(parameters.conv1, conv_pt.conv1, stride=2, padding=3)
        # self.bn1 = GroupNorm(parameters.bn1, num_groups=16, channels=64, eps=1e-5)
        # self.gn1 = nn.GroupNorm(16, 64)
        self.bn1 = GroupNormDRAM(parameters.bn1, num_groups=16, channels=64, eps=1e-5, dtype=ttnn.bfloat8_b)
        # self.gn1.weight = nn.Parameter(parameters.bn1.weight)
        # self.gn1.bias = nn.Parameter(parameters.bn1.bias)

        self.layer1 = self._make_layer(device, parameters.layer1, conv_pt.layer1, block, 64, layers[0])
        # print(f"Layer1: {len(self.layer1)} blocks")
        # print(f"Layer1: {self.layer1}")
        self.layer2 = self._make_layer(device, parameters.layer2, conv_pt.layer2, block, 128, layers[1], stride=2)
        # print(f"Layer2: {len(self.layer2)} blocks")
        # print(f"Layer2: {self.layer2}")
        self.layer3 = self._make_layer(
            device,
            parameters.layer3,
            conv_pt.layer3,
            block,
            256,
            layers[2],
            stride=2,
        )
        # print(f"Layer3: {len(self.layer3)} blocks")
        # print(f"Layer3: {self.layer3}")
        self.layer4 = self._make_layer(
            device,
            parameters.layer4,
            conv_pt.layer4,
            block,
            512,
            layers[3],
            stride=2,
        )
        # print(f"Layer4: {len(self.layer4)} blocks")
        # print(f"Layer4: {self.layer4}")

    def _make_layer(self, device, parameters, conv_pt, block, planes, blocks, stride=1):
        layers = []
        layers.append(
            block(
                device,
                parameters[0],
                conv_pt[0],
                self.inplanes,
                planes,
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
                )
            )
        return layers

    def _run_layer(self, device, x, layer, gn_shard="HS"):
        # print(f"Running layer with gn_shard: {gn_shard}")
        for block in layer:
            x = block.forward(device, x, gn_shard)
        return x

    def forward(self, device, x):
        # print(f"Input shape: {x.shape}, dtype: {x.dtype}, layout: {x.layout}")
        if use_signpost:
            signpost(header="ResNet module started")
        conv1, out_h, out_w = self.conv1(device, x)
        # print(f"Conv1 output shape: {conv1.shape}, out_h: {out_h}, out_w: {out_w}")
        # print(f"Conv1 output shape: {conv1.shape}, out_h: {out_h}, out_w: {out_w}")
        # conv1 = ttnn.untilize(conv1, memory_config=ttnn.DRAM_MEMORY_CONFIG,  use_multicore=True)
        conv1 = ttnn.to_layout(conv1, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        conv1 = self.bn1(device, conv1, out_h, out_w, num_splits=10)  # mbezulj: to investigate, was 8
        # conv1 = self.bn1(device, conv1, out_h, out_w, shard="HS") #this is for later
        #
        # conv1 = ttnn.to_torch(conv1).reshape(conv1.shape[0], out_h, out_w, conv1.shape[-1]).permute((0, 3, 1, 2))

        # conv1 = self.gn1(conv1)

        # print(f"Conv1 output shape after GN: {conv1.shape}, dtype: {conv1.dtype}, layout: {conv1.layout}")
        # N, C, H, W = conv1.shape
        # conv1 = conv1.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)  # [1, 1, N*H*W, C]
        # conv1 = ttnn.from_torch(
        #    conv1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
        # )

        # print(f"Conv1 output shape after GN: {conv1.shape}, dtype: {conv1.dtype}, layout: {conv1.layout}")
        # memory_config = ttnn.MemoryConfig(
        #     ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        #     ttnn.BufferType.L1,
        #     ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}), [(H * W) // 20, C], ttnn.ShardOrientation.ROW_MAJOR)
        # )
        # conv1 = ttnn.to_memory_config(conv1, memory_config=memory_config)
        # print(f"Conv1 output shape after memory config: {conv1.shape}, dtype: {conv1.dtype}, layout: {conv1.layout}")
        conv1 = ttnn.relu(conv1)
        # print(f"ReLU output shape: {conv1.shape}, dtype: {conv1.dtype}, layout: {conv1.layout}")
        # print(f"relu output shape: {conv1.shape} {out_h=} {out_w=}")

        cv1 = conv1[:, :, :, :32]  # Assuming conv1 has shape [N, C, H, W] and we want to keep the first 32 channels
        # print(f"cv1 shape: {cv1.shape}, dtype: {cv1.dtype}, layout: {cv1.layout}")
        cv2 = conv1[:, :, :, 32:]  # The rest of the channels
        # print(f"cv2 shape: {cv2.shape}, dtype: {cv2.dtype}, layout: {cv2.layout}")
        ttnn.deallocate(conv1)
        # print(f"cv1 shape: {cv1.shape}, cv2 shape: {cv2.shape}")
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
        # print(f"Max pool output shape: {conv1.shape=}, {conv2.shape=}")
        conv_c = ttnn.concat([conv1, conv2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        # print(f"Concat output shape: {conv1.shape}, dtype: {conv1.dtype}, layout: {conv1.layout}")
        ttnn.deallocate(conv1)
        ttnn.deallocate(conv2)
        conv_c = ttnn.move(conv_c)
        # print(f"Concat output shape after deallocate: {conv_c.shape}, dtype: {conv_c.dtype}, layout: {conv_c.layout}")
        feats4 = self._run_layer(device, conv_c, self.layer1)

        # print(f"Feats4 output shape: {feats4.shape}, dtype: {feats4.dtype}, layout: {feats4.layout}")
        ttnn.deallocate(conv_c)
        feats8 = self._run_layer(device, feats4, self.layer2)
        feats8_interleaved = ttnn.sharded_to_interleaved(feats8, ttnn.DRAM_MEMORY_CONFIG)

        # feats8_clone = ttnn.clone(feats8, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(feats4)
        feats16 = self._run_layer(device, feats8, self.layer3)
        feats16_interleaved = ttnn.sharded_to_interleaved(feats16, ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(feats8)
        feats32 = self._run_layer(device, feats16, self.layer4, gn_shard="BS")
        feats32_interleaved = ttnn.sharded_to_interleaved(feats32, ttnn.DRAM_MEMORY_CONFIG)

        # print(f"Feats4 output shape: {feats4.shape}, dtype: {feats4.dtype}, layout: {feats4.layout}")
        # print(f"Feats8 output shape: {feats8.shape}, dtype: {feats8.dtype}, layout: {feats8.layout}")
        if use_signpost:
            signpost(header="ResNet module finished")
        return feats8_interleaved, feats16_interleaved, feats32_interleaved
