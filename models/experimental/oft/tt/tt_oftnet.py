import torch
import torch.nn as nn
from models.experimental.oft.reference.oftnet import OFT as ReferenceOFT
import ttnn
from models.experimental.oft.tt.common import Conv, GroupNorm
from models.experimental.oft.tt.tt_resnet import TTResNetFeatures


class TTOftNet:
    def __init__(
        self,
        device,
        parameters,
        conv_pt,
        block,
        layers,
        mean,
        std,
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4,
    ):
        # print(f"Creating TTOftNet with parameters: {parameters}")
        # print(f"param keys {parameters.keys()}")
        # print(f"frontend values {parameters.frontend}")
        # print(f"conv_pt values {conv_pt}")
        # resnet_feats = parameters.frontend
        self.frontend = TTResNetFeatures(device, parameters.frontend, conv_pt.frontend, block, layers)
        print(f"-------------------boja-------------------")
        print(f"{conv_pt.frontend.conv1.stride=}")
        print(f"--------------------boja-------------------")
        self.lat8 = Conv(parameters.lat8, conv_pt.lat8, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bn8 = GroupNorm(parameters.bn8, num_groups=16, channels=256, eps=1e-5, dtype=ttnn.bfloat8_b)

        self.lat16 = Conv(parameters.lat16, conv_pt.lat16, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bn16 = GroupNorm(parameters.bn16, num_groups=16, channels=256, eps=1e-5, dtype=ttnn.bfloat8_b)

        self.lat32 = Conv(parameters.lat32, conv_pt.lat32, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bn32 = GroupNorm(parameters.bn32, num_groups=16, channels=256, eps=1e-5, dtype=ttnn.bfloat8_b)

        # self.oft8 = OFT(device, parameters.oft8, 256, grid_res, grid_height, features, calib, grid, 1 / 8.0)
        # self.oft16 = OFT(device, parameters.oft16, 256, grid_res, grid_height, features, calib, grid, 1 / 16.0)
        # self.oft32 = OFT(device, parameters.oft32, 256, grid_res, grid_height, features, calib, grid, 1 / 32.0)
        self.oft8 = ReferenceOFT(256, grid_res, grid_height, 1 / 8.0)
        # print(f"parameters.oft8.conv3d.weight shape: {parameters.oft8.conv3d.weight}")
        self.oft8.conv3d.weight = nn.Parameter(parameters.oft8.conv3d.weight)
        self.oft8.conv3d.bias = nn.Parameter(parameters.oft8.conv3d.bias)

        self.oft16 = ReferenceOFT(256, grid_res, grid_height, 1 / 16.0)
        self.oft16.conv3d.weight = nn.Parameter(parameters.oft16.conv3d.weight)
        self.oft16.conv3d.bias = nn.Parameter(parameters.oft16.conv3d.bias)

        self.oft32 = ReferenceOFT(256, grid_res, grid_height, 1 / 32.0)
        self.oft32.conv3d.weight = nn.Parameter(parameters.oft32.conv3d.weight)
        self.oft32.conv3d.bias = nn.Parameter(parameters.oft32.conv3d.bias)

        self.topdown = [
            block(
                device,
                parameters.topdown[i],
                conv_pt.topdown[i],
                256,
                256,
                stride=1,
                is_sliced=True,
                # height_sharding=False,
                # act_block_h=32,
                # layer="topdown",
            )
            for i in range(topdown_layers)
        ]
        self.mean = mean
        self.std = std

    def forward(self, device, input_tensor, calib, grid):
        print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        # print(f"Calib shape: {calib.shape}, dtype: {calib.dtype}")
        # print(f"Grid shape: {grid.shape}, dtype: {grid.dtype}")
        print(f"self.mean shape: {self.mean.shape}, dtype: {self.mean.dtype}")
        print(f"self.std shape: {self.std.shape}, dtype: {self.std.dtype}")
        input_tensor = input_tensor - self.mean
        input_tensor = ttnn.div(input_tensor, self.std)
        if input_tensor.layout == ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"Input tensor after normalization: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"input_tensor layout: {input_tensor.layout}, memory_config: {input_tensor.memory_config()}")
        # input_tensor = ttnn.div(input_tensor, self.std)
        # ttnn.deallocate(self.mean)
        # ttnn.deallocate(self.std)
        # input_tensor = ttnn.move(input_tensor)
        print(f"Input tensor after normalization: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"input_tensor layout: {input_tensor.layout}, memory_config: {input_tensor.memory_config()}")
        # return input_tensor
        feats8, feats16, feats32 = self.frontend.forward(device, input_tensor)
        # print(f"feats8 layout: {feats8.layout}, memory_config: {feats8.memory_config()}")
        # print(f"feats16 layout: {feats16.layout}, memory_config: {feats16.memory_config()}")
        # print(f"feats32 layout: {feats32.layout}, memory_config: {feats32.memory_config()}")
        # return feats8, feats16, feats32 #input_tensor

        lat8, lat8_h, lat8_w = self.lat8(device, feats8)
        # print(f"Lat8 shape: {lat8.shape}, dtype: {lat8.dtype}")
        # print(f"lat8 layout: {lat8.layout}, memory_config: {lat8.memory_config()}")
        if lat8.layout == ttnn.TILE_LAYOUT:
            lat8 = ttnn.to_layout(lat8, ttnn.ROW_MAJOR_LAYOUT)
        # print(f"Lat8 after sharded_to_interleaved shape: {lat8.shape}, dtype: {lat8.dtype}")
        # print(f"lat8 layout: {lat8.layout}, memory_config: {lat8.memory_config()}")
        lat8 = self.bn8(device, lat8, lat8_h, lat8_w, shard="HS")
        lat8 = ttnn.relu(lat8)

        lat16, lat16_h, lat16_w = self.lat16(device, feats16)
        if lat16.layout == ttnn.TILE_LAYOUT:
            lat16 = ttnn.to_layout(lat16, ttnn.ROW_MAJOR_LAYOUT)
        lat16 = self.bn16(device, lat16, lat16_h, lat16_w, shard="HS")
        lat16 = ttnn.relu(lat16)

        lat32, lat32_h, lat32_w = self.lat32(device, feats32)
        if lat32.layout == ttnn.TILE_LAYOUT:
            lat32 = ttnn.to_layout(lat32, ttnn.ROW_MAJOR_LAYOUT)
        lat32 = self.bn32(device, lat32, lat32_h, lat32_w, shard="BS")
        lat32 = ttnn.relu(lat32)
        # return lat8, lat16, lat32
        print("Convertting to torch for OFT")
        lat8_torch = ttnn.to_torch(lat8).detach()
        lat8_torch = lat8_torch.permute(0, 3, 1, 2)
        n, c, h, w = lat8_torch.shape
        lat8_torch = lat8_torch.reshape(n, c, lat8_h, lat8_w).to(torch.float32)
        # print (f"Lat8 torch shape: {lat8_torch.shape}, dtype: {lat8_torch.dtype}")
        lat16_torch = ttnn.to_torch(lat16).detach()
        lat16_torch = lat16_torch.permute(0, 3, 1, 2)
        n, c, h, w = lat16_torch.shape
        lat16_torch = lat16_torch.reshape(n, c, lat16_h, lat16_w).to(torch.float32)
        # print (f"Lat16 torch shape: {lat16_torch.shape}, dtype: {lat16_torch.dtype}")
        lat32_torch = ttnn.to_torch(lat32).detach()
        lat32_torch = lat32_torch.permute(0, 3, 1, 2)
        n, c, h, w = lat32_torch.shape
        lat32_torch = lat32_torch.reshape(n, c, lat32_h, lat32_w).to(torch.float32)
        # print (f"Lat32 torch shape: {lat32_torch.shape}, dtype: {lat32_torch.dtype}")
        print("Calling OFT")
        ortho8 = self.oft8(lat8_torch, calib, grid)  # ortho8
        ortho16 = self.oft16(lat16_torch, calib, grid)
        ortho32 = self.oft32(lat32_torch, calib, grid)
        ortho = ortho8 + ortho16 + ortho32
        print("TTNN OFT finished")
        n, c, h, w = ortho.shape
        ortho = ortho.permute(0, 2, 3, 1).view(1, 1, n * h * w, c)
        print(f"Ortho shape: {ortho.shape}, dtype: {ortho.dtype}")
        ortho = ttnn.from_torch(ortho, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        print(f"Ortho shape: {ortho.shape}, dtype: {ortho.dtype}")
        ttnn.deallocate(lat8)
        ttnn.deallocate(lat16)
        ttnn.deallocate(lat32)
        td = ortho
        for layer in self.topdown:
            td = layer.forward(device, td)
        # return lat8_torch, lat16_torch, lat32_torch
        return td  # ortho, ortho16, ortho32 #input_tensor

    def normalize(self, input_tensor):
        """
        Normalize the input tensor using the mean and std.
        """
        input_tensor = ttnn.subtract(input_tensor, self.mean, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor = ttnn.div(input_tensor, self.std, memory_config=ttnn.DRAM_MEMORY_CONFIG)
