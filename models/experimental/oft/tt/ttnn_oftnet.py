import ttnn
import torch.nn as nn

from models.experimental.oft.tt.ttnn_resnet import ResNetFeatures, Conv, BasicBlock
from models.experimental.oft.tt.ttnn_oft import OFT


class OftNet:
    def __init__(
        self,
        device,
        parameters,
        conv_pt,
        block,
        layers,
        mean,
        std,
        y_corners,
        topdown_layers=8,
        grid_res=1,
        grid_height=4,
    ):
        self.frontend = ResNetFeatures(device, parameters.frontend, conv_pt.frontend, block, layers)

        self.lat8 = Conv(parameters.lat8, conv_pt.lat8, stride=1, padding=0)
        self.lat16 = Conv(parameters.lat16, conv_pt.lat16, stride=1, padding=0)
        self.lat32 = Conv(parameters.lat32, conv_pt.lat32, stride=1, padding=0)
        self.gn8 = nn.GroupNorm(16, 256)
        self.gn16 = nn.GroupNorm(16, 256)
        self.gn32 = nn.GroupNorm(16, 256)
        self.gn8.weight = nn.Parameter(parameters.bn8.weight)
        self.gn8.bias = nn.Parameter(parameters.bn8.bias)
        self.gn16.weight = nn.Parameter(parameters.bn16.weight)
        self.gn16.bias = nn.Parameter(parameters.bn16.bias)
        self.gn32.weight = nn.Parameter(parameters.bn32.weight)
        self.gn32.bias = nn.Parameter(parameters.bn32.bias)

        self.oft8 = OFT(device, parameters.oft8, y_corners, 1 / 8.0)
        self.oft16 = OFT(device, parameters.oft16, y_corners, 1 / 16.0)
        self.oft32 = OFT(device, parameters.oft32, y_corners, 1 / 32.0)

        self.topdown = [
            BasicBlock(
                device,
                parameters.topdown[i],
                conv_pt.topdown[i],
                256,
                256,
                stride=1,
                height_sharding=False,
                # act_block_h=32,
                # layer="topdown",
            )
            for i in range(topdown_layers)
        ]

        self.head = Conv(parameters.head, conv_pt.head, stride=1, padding=1)

        self.mean = mean
        self.std = std

    def __call__(self, device, image, calib, grid):
        self.mean = ttnn.reshape(self.mean, (3, 1, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        self.std = ttnn.reshape(self.std, (3, 1, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        image = ttnn.subtract(image, self.mean, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

        image = ttnn.div(image, self.std, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        if image.get_layout() == ttnn.TILE_LAYOUT:
            image = ttnn.to_layout(
                image, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
            )
        image = ttnn.permute(image, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)

        ttnn.deallocate(self.mean)
        ttnn.deallocate(self.std)
        calib = ttnn.to_memory_config(calib, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        grid = ttnn.to_memory_config(grid, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        feats8, feats16, feats32 = self.frontend(device, image)
        ttnn.deallocate(image)
        lat8, out_h, out_w = self.lat8(device, feats8)
        ttnn.deallocate(feats8)
        if lat8.is_sharded():
            lat8 = ttnn.sharded_to_interleaved(lat8, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat8_b)
        lat8 = ttnn.reshape(lat8, (lat8.shape[0], out_h, out_w, lat8.shape[-1]), memory_config=ttnn.L1_MEMORY_CONFIG)
        lat8 = ttnn.permute(lat8, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        lat8 = ttnn.to_torch(lat8)

        lat8 = self.gn8(lat8)
        lat8 = ttnn.from_torch(
            lat8, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
        )
        lat8 = ttnn.relu(lat8, memory_config=ttnn.L1_MEMORY_CONFIG)

        lat16, out_h, out_w = self.lat16(device, feats16)
        ttnn.deallocate(feats16)
        if lat16.is_sharded():
            lat16 = ttnn.sharded_to_interleaved(lat16, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat8_b)
        lat16 = ttnn.reshape(
            lat16, (lat16.shape[0], out_h, out_w, lat16.shape[-1]), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        lat16 = ttnn.permute(lat16, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        lat16 = ttnn.to_torch(lat16)

        lat16 = self.gn16(lat16)
        lat16 = ttnn.from_torch(
            lat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
        )
        lat16 = ttnn.relu(lat16, memory_config=ttnn.L1_MEMORY_CONFIG)

        lat32, out_h, out_w = self.lat32(device, feats32)
        ttnn.deallocate(feats32)
        if lat32.is_sharded():
            lat32 = ttnn.sharded_to_interleaved(lat32, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat8_b)
        lat32 = ttnn.reshape(
            lat32, (lat32.shape[0], out_h, out_w, lat32.shape[-1]), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        lat32 = ttnn.permute(lat32, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        lat32 = ttnn.to_torch(lat32)

        lat32 = self.gn32(lat32)
        lat32 = ttnn.from_torch(
            lat32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
        )
        lat32 = ttnn.relu(lat32, memory_config=ttnn.L1_MEMORY_CONFIG)

        calib = ttnn.to_memory_config(calib, memory_config=ttnn.L1_MEMORY_CONFIG)
        grid = ttnn.to_memory_config(grid, memory_config=ttnn.L1_MEMORY_CONFIG)

        ortho8 = self.oft8(device, lat8, calib, grid)
        ttnn.deallocate(lat8)
        ortho16 = self.oft16(device, lat16, calib, grid)
        ttnn.deallocate(lat16)

        ortho32 = self.oft32(device, lat32, calib, grid)
        ttnn.deallocate(lat32)
        ttnn.deallocate(calib)
        ttnn.deallocate(grid)
        ortho = ttnn.add(ortho8, ortho16, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(ortho8)
        ttnn.deallocate(ortho16)

        ortho = ttnn.add(ortho, ortho32, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(ortho32)

        topdown = ortho
        topdown = ttnn.permute(topdown, [0, 2, 3, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

        ttnn.deallocate(ortho)

        for layer in self.topdown:
            topdown = layer(device, topdown)

        batch, depth, width, _ = topdown.shape

        outputs, out_h, out_w = self.head(device, topdown)
        if outputs.is_sharded():
            outputs = ttnn.sharded_to_interleaved(outputs, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat8_b)
        outputs = ttnn.permute(outputs, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        outputs = ttnn.reshape(outputs, (batch, -1, 9, depth, width), memory_config=ttnn.L1_MEMORY_CONFIG)

        slices = [1, 3, 3, 2]
        start = 0
        parts = []

        for s in slices:
            parts.append(outputs[:, :, start : start + s, :, :])
            start += s
        parts[0] = ttnn.squeeze(parts[0], 2)
        return parts
