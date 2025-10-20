# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.oft.tt.common import GroupNorm, GroupNormDRAM
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d, MaxPool2dConfiguration

# from models.experimental.oft.tt.common import Conv
# from models.experimental.oft.tt.common import GroupNorm_fallback as GroupNorm
# from models.experimental.oft.tt.common import GroupNorm_fallback as GroupNormDRAM
from loguru import logger

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TTBasicBlock:
    expansion = 1
    block_counter = 0

    def __init__(self, device, state_dict, layer_args, is_sliced=False):
        self.block_id = TTBasicBlock.block_counter
        TTBasicBlock.block_counter += 1
        self.is_sliced = is_sliced
        # logger.debug(f"TTBasicBlock: {inplanes=}, {planes=}, {stride=}, {is_sliced=}")
        self.conv1 = TtConv2d(layer_args.conv1["optimized_configuration"], device)
        if not is_sliced:
            self.bn1 = GroupNorm(
                state_dict.bn1,
                layer_args=layer_args.bn1,
                dtype=ttnn.bfloat16,
            )
        else:
            self.bn1 = GroupNormDRAM(
                state_dict.bn1,
                layer_args=layer_args.bn1,
                dtype=ttnn.bfloat16,
            )
        self.conv2 = TtConv2d(layer_args.conv2["optimized_configuration"], device)
        if not is_sliced:
            self.bn2 = GroupNorm(
                state_dict.bn2,
                layer_args=layer_args.bn2,
                dtype=ttnn.bfloat16,
            )
        else:
            self.bn2 = GroupNormDRAM(
                state_dict.bn2,
                layer_args=layer_args.bn2,
                dtype=ttnn.bfloat16,
            )

        if "downsample" in state_dict.keys():
            self.downsample = True
            self.downsample_conv = TtConv2d(layer_args.downsample[0]["optimized_configuration"], device)
            self.downsample_bn = GroupNorm(
                state_dict.downsample[1],
                layer_args=layer_args.downsample[1],
                dtype=ttnn.bfloat16,
            )
        else:
            self.downsample = None

    def forward(self, device, x, gn_shard="HS", num_splits=1):
        if use_signpost:
            signpost(header=f"TTBasicBlock {self.block_id} forward started")
        if x.layout != ttnn.ROW_MAJOR_LAYOUT and self.is_sliced:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        out = self.conv1(x)
        logger.debug(
            f"FORWARD X Input shape: {x.shape}, dtype: {x.dtype}, layout: {x.layout} memory_config: {x.memory_config()}"
        )
        out = ttnn.move(out)
        # logger.debug(f"SSHARDING {gn_shard=}")
        out = self.bn1(device, out, shard=gn_shard, num_splits=num_splits)
        logger.debug(f"BN1 output shape: {out.shape}")
        ttnn.relu(out, output_tensor=out)

        if out.layout != ttnn.ROW_MAJOR_LAYOUT and self.is_sliced:
            out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = self.conv2(out)
        logger.debug(f"Conv2 output shape: {out.shape}")
        out = ttnn.move(out)
        out = self.bn2(device, out, shard=gn_shard, num_splits=num_splits)
        logger.debug(f"BN2 output shape: {out.shape}")

        if self.downsample is not None:
            x = self.downsample_conv(x)
            x = self.downsample_bn(device, x, shard=gn_shard)
        else:
            logger.debug(f"reshape x shape: {x.shape} self.downsample: {self.downsample}")
            # x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))

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

        out = ttnn.relu(out)
        if use_signpost:
            signpost(header=f"TTBasicBlock {self.block_id} forward finished")
        return out


class TTResNetFeatures:
    def __init__(self, device, parameters, conv_pt, block, layers, return_intermediates=False):
        self.conv1 = TtConv2d(conv_pt.conv1["optimized_configuration"], device)
        self.bn1 = GroupNorm(parameters.bn1, conv_pt.bn1)
        self.maxpool = TtMaxPool2d(
            configuration=MaxPool2dConfiguration(
                input_height=conv_pt.maxpool.input_height,
                input_width=conv_pt.maxpool.input_width,
                channels=conv_pt.maxpool.input_channels,
                batch_size=conv_pt.maxpool.batch_size,
                kernel_size=(conv_pt.maxpool.kernel_size, conv_pt.maxpool.kernel_size),
                stride=(conv_pt.maxpool.stride, conv_pt.maxpool.stride),
                padding=(conv_pt.maxpool.padding, conv_pt.maxpool.padding),
                dilation=(conv_pt.maxpool.dilation, conv_pt.maxpool.dilation),
                deallocate_input=True,
                in_place=True,
            ),
            device=device,
        )

        self.layer1 = self._make_layer(device, parameters.layer1, conv_pt.layer1, block, layers[0])
        self.layer2 = self._make_layer(device, parameters.layer2, conv_pt.layer2, block, layers[1])
        self.layer3 = self._make_layer(device, parameters.layer3, conv_pt.layer3, block, layers[2])
        self.layer4 = self._make_layer(
            device,
            parameters.layer4,
            conv_pt.layer4,
            block,
            layers[3],
        )
        self.return_intermediates = return_intermediates

        self.num_splits_gn = 2  # Number of splits for GroupNorm to fit into L1
        self.num_slices = 2  # Number of slices used for partial sharding during concatenation of GN outputs

    def _make_layer(self, device, parameters, conv_pt, block, blocks):
        layers = []
        layers.append(
            block(
                device,
                parameters[0],
                conv_pt[0],
            )
        )
        for i in range(1, blocks):
            layers.append(
                block(
                    device,
                    parameters[i],
                    conv_pt[i],
                )
            )
        return layers

    def _run_layer(self, device, x, layer, gn_shard="HS", return_intermediates=False):
        """Run a layer with optional intermediate activation capture."""
        intermediates = []

        if return_intermediates:
            intermediates.append(ttnn.to_torch(x).permute(0, 3, 1, 2))

        for block in layer:
            x = block.forward(device, x, gn_shard)

            if return_intermediates:
                # Clone/copy each intermediate activation
                intermediates.append(ttnn.to_torch(x).permute(0, 3, 1, 2))

        if return_intermediates:
            return x, intermediates
        else:
            return x

    def forward(self, device, x):
        if use_signpost:
            signpost(header="ResNet module started")

        host_x = ttnn.to_torch(x).permute(0, 3, 1, 2)
        conv1 = self.conv1(x)
        host_conv1f = ttnn.to_torch(conv1).permute(0, 3, 1, 2)

        # Split the tensor into multiple slices to fit into L1 for GN
        splits = ttnn.split(conv1, conv1.shape[-1] // self.num_splits_gn, dim=3)
        compute_grid = device.compute_with_storage_grid_size()
        core_grid = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)

        # GN on each split
        processed_splits = []
        for split in splits:
            split = self.bn1(device, split, shard="HS", num_splits=self.num_splits_gn, negative_mask=True)
            split = ttnn.to_layout(split, ttnn.TILE_LAYOUT)
            split = ttnn.to_memory_config(split, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            processed_splits.append(split)

        # Concat back the splits and ReLU
        shard_height = conv1.shape[2] // (core_grid.x * core_grid.y * self.num_slices)
        shard_width = conv1.shape[3]

        sharded_mem_config = ttnn.create_sharded_memory_config(
            shape=[shard_height, shard_width],
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        for i in range(self.num_slices):
            slice_0 = ttnn.interleaved_to_sharded_partial(
                processed_splits[0],
                (core_grid.x, core_grid.y),
                [shard_height, shard_width // self.num_splits_gn],
                self.num_slices,
                i,
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            slice_1 = ttnn.interleaved_to_sharded_partial(
                processed_splits[1],
                (core_grid.x, core_grid.y),
                [shard_height, shard_width // self.num_splits_gn],
                self.num_slices,
                i,
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )

            slice_1 = ttnn.concat([slice_0, slice_1], dim=3, memory_config=sharded_mem_config)
            slice_1 = ttnn.relu(slice_1, output_tensor=slice_1)

            ttnn.sharded_to_interleaved_partial(slice_1, conv1, self.num_slices, i, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(slice_1)
        ttnn.deallocate(slice_0)

        host_gn = ttnn.to_torch(conv1).permute(0, 3, 1, 2)
        host_relu = ttnn.to_torch(conv1).permute(0, 3, 1, 2).reshape(1, 64, 192, 640)

        conv_1 = self.maxpool(conv1)

        conv_1 = ttnn.to_memory_config(
            conv_1, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # this resolves the underlying issue that residual needs to be reused inside the block; to investigated better approach
        host_mp = ttnn.to_torch(conv_1).permute(0, 3, 1, 2)
        feats4, i4 = self._run_layer(device, conv_1, self.layer1, return_intermediates=True)

        ttnn.deallocate(conv_1)
        feats8, i8 = self._run_layer(device, feats4, self.layer2, return_intermediates=True)
        feats8_interleaved = ttnn.sharded_to_interleaved(feats8, ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(feats4)
        feats16, i16 = self._run_layer(device, feats8, self.layer3, return_intermediates=True)
        feats16_interleaved = ttnn.sharded_to_interleaved(feats16, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(feats8)

        feats32, i32 = self._run_layer(device, feats16, self.layer4, gn_shard="BS", return_intermediates=True)
        feats32_interleaved = ttnn.sharded_to_interleaved(feats32, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(feats16)

        if use_signpost:
            signpost(header="ResNet module finished")

        if self.return_intermediates:
            return (
                [host_x, i4, i8, i16, i32, host_conv1f, host_gn, host_relu, host_mp],
                feats8_interleaved,
                feats16_interleaved,
                feats32_interleaved,
            )
        else:
            return feats8_interleaved, feats16_interleaved, feats32_interleaved
