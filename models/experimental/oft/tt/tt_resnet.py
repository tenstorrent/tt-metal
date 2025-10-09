# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.oft.tt.common import Conv, GroupNorm, GroupNormDRAM

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

    def __init__(self, device, parameters, conv_pt, inplanes, planes, stride=1, scale=1, is_sliced=False):
        self.is_sliced = is_sliced
        logger.debug(f"TTBasicBlock: {inplanes=}, {planes=}, {stride=}, {is_sliced=}")
        self.conv1 = Conv(
            parameters.conv1, conv_pt.conv1, stride=stride, output_layout=ttnn.ROW_MAJOR_LAYOUT, is_sliced=is_sliced
        )
        if not is_sliced:
            self.bn1 = GroupNorm(parameters.bn1, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat16)
        else:
            self.bn1 = GroupNormDRAM(parameters.bn1, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat16)
        self.conv2 = Conv(parameters.conv2, conv_pt.conv2, output_layout=ttnn.ROW_MAJOR_LAYOUT, is_sliced=is_sliced)
        if not is_sliced:
            self.bn2 = GroupNorm(parameters.bn2, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat16)
        else:
            self.bn2 = GroupNormDRAM(parameters.bn2, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat16)
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
        if use_signpost:
            signpost(header="TTBasicBlock forward started")
        out, out_h, out_w = self.conv1(device, x)
        logger.debug(f"FORWARD X Input shape: {x.shape}, dtype: {x.dtype}, layout: {x.layout}")
        out = ttnn.move(out)
        # logger.debug(f"SSHARDING {gn_shard=}")
        out = self.bn1(device, out, out_h, out_w, shard=gn_shard, num_splits=num_splits)
        logger.debug(f"BN1 output shape: {out.shape}")
        ttnn.relu(out, output_tensor=out)

        out, out_h, out_w = self.conv2(device, out)
        logger.debug(f"Conv2 output shape: {out.shape}")
        out = ttnn.move(out)
        out = self.bn2(device, out, out_h, out_w, shard=gn_shard, num_splits=num_splits)
        logger.debug(f"BN2 output shape: {out.shape}")

        if self.downsample is not None:
            x, out_h_ds, out_w_ds = self.downsample_conv(device, x)
            x = self.downsample_bn(device, x, out_h_ds, out_w_ds, shard=gn_shard)
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
            signpost(header="TTBasicBlock forward finished")
        return out


class TTResNetFeatures:
    def __init__(self, device, parameters, conv_pt, block, layers, return_intermediates=False):
        self.inplanes = 64

        self.conv1 = Conv(parameters.conv1, conv_pt.conv1, stride=2, padding=3)
        self.bn1 = GroupNormDRAM(parameters.bn1, num_groups=16, channels=64, eps=1e-5, dtype=ttnn.bfloat8_b)

        self.layer1 = self._make_layer(device, parameters.layer1, conv_pt.layer1, block, 64, layers[0])
        self.layer2 = self._make_layer(device, parameters.layer2, conv_pt.layer2, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            device,
            parameters.layer3,
            conv_pt.layer3,
            block,
            256,
            layers[2],
            stride=2,
        )
        self.layer4 = self._make_layer(
            device,
            parameters.layer4,
            conv_pt.layer4,
            block,
            512,
            layers[3],
            stride=2,
        )
        self.return_intermediates = return_intermediates

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
        conv1, out_h, out_w = self.conv1(device, x)
        host_conv1f = ttnn.to_torch(conv1).permute(0, 3, 1, 2)

        conv1 = ttnn.to_layout(conv1, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        conv1 = self.bn1(device, conv1, out_h, out_w, num_splits=10)
        conv1 = ttnn.to_layout(conv1, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        host_gn = ttnn.to_torch(conv1).permute(0, 3, 1, 2)
        conv1 = ttnn.relu(conv1)
        host_relu = ttnn.to_torch(conv1).permute(0, 3, 1, 2).reshape(1, 64, 192, 640)

        # import torch.nn.functional as F
        # h_mp = F.max_pool2d(host_relu, 3, stride=2, padding=1).permute(0, 2, 3, 1).reshape(1, 1, 96 * 320, 64)

        cv1 = conv1[:, :, :, :32]  # Assuming conv1 has shape [N, H, W, C] and we want to keep the first 32 channels
        cv2 = conv1[:, :, :, 32:]  # The rest of the channels
        ttnn.deallocate(conv1)
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
            ceil_mode=False,
        )
        ttnn.deallocate(cv1)
        conv2 = ttnn.max_pool2d(
            input_tensor=cv2,
            batch_size=cv2.shape[0],
            input_h=out_h,  # conv1.shape[1],
            input_w=out_w,  # conv1.shape[2],
            channels=cv2.shape[3],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            ceil_mode=False,
        )
        ttnn.deallocate(cv2)
        conv_c = ttnn.concat([conv1, conv2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        ttnn.deallocate(conv1)
        ttnn.deallocate(conv2)
        # conv_c = ttnn.move(conv_c)
        conv_c = ttnn.to_memory_config(conv_c, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # conv_c = ttnn.from_torch(h_mp, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        host_mp = ttnn.to_torch(conv_c).permute(0, 3, 1, 2)
        feats4, i4 = self._run_layer(device, conv_c, self.layer1, return_intermediates=True)

        ttnn.deallocate(conv_c)
        feats8, i8 = self._run_layer(device, feats4, self.layer2, return_intermediates=True)
        feats8_interleaved = ttnn.sharded_to_interleaved(feats8, ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(feats4)
        feats16, i16 = self._run_layer(device, feats8, self.layer3, return_intermediates=True)
        feats16_interleaved = ttnn.sharded_to_interleaved(feats16, ttnn.DRAM_MEMORY_CONFIG)

        # if feats8.memory_config().buffer_type == ttnn.BufferType.DRAM:
        # ttnn.deallocate(feats8)
        feats32, i32 = self._run_layer(device, feats16, self.layer4, gn_shard="BS", return_intermediates=True)
        feats32_interleaved = ttnn.sharded_to_interleaved(feats32, ttnn.DRAM_MEMORY_CONFIG)

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
