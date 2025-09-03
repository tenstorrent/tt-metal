import ttnn
from models.experimental.oft.tt.common import Conv, GroupNorm
from models.experimental.oft.tt.tt_resnet import TTResNetFeatures
from models.experimental.oft.tt.tt_oft import OFT as TtOFT

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


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
        features,
        calib,
        grid,
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
        # print(f"-------------------boja-------------------")
        # print(f"{conv_pt.frontend.conv1.stride=}")
        # print(f"--------------------boja-------------------")
        self.lat8 = Conv(parameters.lat8, conv_pt.lat8, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bn8 = GroupNorm(parameters.bn8, num_groups=16, channels=256, eps=1e-5, dtype=ttnn.bfloat8_b)

        self.lat16 = Conv(parameters.lat16, conv_pt.lat16, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bn16 = GroupNorm(parameters.bn16, num_groups=16, channels=256, eps=1e-5, dtype=ttnn.bfloat8_b)

        self.lat32 = Conv(parameters.lat32, conv_pt.lat32, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bn32 = GroupNorm(parameters.bn32, num_groups=16, channels=256, eps=1e-5, dtype=ttnn.bfloat8_b)

        self.oft8 = TtOFT(device, parameters.oft8, 256, grid_res, grid_height, features, calib, grid)
        self.oft16 = TtOFT(device, parameters.oft16, 256, grid_res, grid_height, features, calib, grid)
        self.oft32 = TtOFT(device, parameters.oft32, 256, grid_res, grid_height, features, calib, grid)

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

        self.head = Conv(parameters.head, conv_pt.head, output_layout=ttnn.ROW_MAJOR_LAYOUT, is_sliced=True)
        self.mean = mean
        self.std = std

    def forward(self, device, input_tensor, calib, grid):
        if use_signpost:
            signpost(header="OftNet module started")

        input_tensor = input_tensor - self.mean
        input_tensor = ttnn.div(input_tensor, self.std)
        if input_tensor.layout == ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        feats8, feats16, feats32 = self.frontend.forward(device, input_tensor)

        lat8, lat8_h, lat8_w = self.lat8(device, feats8)
        if lat8.layout == ttnn.TILE_LAYOUT:
            lat8 = ttnn.to_layout(lat8, ttnn.ROW_MAJOR_LAYOUT)

        lat8 = self.bn8(device, lat8, lat8_h, lat8_w, shard="HS")
        lat8 = ttnn.relu(lat8)
        lat8 = ttnn.sharded_to_interleaved(lat8, ttnn.DRAM_MEMORY_CONFIG)
        lat8 = ttnn.to_layout(lat8, ttnn.TILE_LAYOUT)

        lat16, lat16_h, lat16_w = self.lat16(device, feats16)
        if lat16.layout == ttnn.TILE_LAYOUT:
            lat16 = ttnn.to_layout(lat16, ttnn.ROW_MAJOR_LAYOUT)
        lat16 = self.bn16(device, lat16, lat16_h, lat16_w, shard="HS")
        lat16 = ttnn.relu(lat16)
        lat16 = ttnn.sharded_to_interleaved(lat16, ttnn.DRAM_MEMORY_CONFIG)
        lat16 = ttnn.to_layout(lat16, ttnn.TILE_LAYOUT)

        lat32, lat32_h, lat32_w = self.lat32(device, feats32)
        if lat32.layout == ttnn.TILE_LAYOUT:
            lat32 = ttnn.to_layout(lat32, ttnn.ROW_MAJOR_LAYOUT)
        lat32 = self.bn32(device, lat32, lat32_h, lat32_w, shard="BS")
        lat32 = ttnn.relu(lat32)
        lat32 = ttnn.sharded_to_interleaved(lat32, ttnn.DRAM_MEMORY_CONFIG)
        lat32 = ttnn.to_layout(lat32, ttnn.TILE_LAYOUT)

        if use_signpost:
            signpost(header="Oft module")
        # return lat8, lat16, lat32
        # print("Convertting to torch for OFT")
        # lat8_torch = ttnn.to_torch(lat8).detach()
        # lat8_torch = lat8_torch.permute(0, 3, 1, 2)
        # n, c, h, w = lat8_torch.shape
        # lat8_torch = lat8_torch.reshape(n, c, lat8_h, lat8_w).to(torch.float32)
        # # torch.save(lat8_torch, "lat8_torch.pt")
        # # print (f"Lat8 torch shape: {lat8_torch.shape}, dtype: {lat8_torch.dtype}")
        # lat16_torch = ttnn.to_torch(lat16).detach()
        # lat16_torch = lat16_torch.permute(0, 3, 1, 2)
        # n, c, h, w = lat16_torch.shape
        # lat16_torch = lat16_torch.reshape(n, c, lat16_h, lat16_w).to(torch.float32)
        # # torch.save(lat16_torch, "lat16_torch.pt")

        # # print (f"Lat16 torch shape: {lat16_torch.shape}, dtype: {lat16_torch.dtype}")
        # lat32_torch = ttnn.to_torch(lat32).detach()
        # lat32_torch = lat32_torch.permute(0, 3, 1, 2)
        # n, c, h, w = lat32_torch.shape
        # lat32_torch = lat32_torch.reshape(n, c, lat32_h, lat32_w).to(torch.float32)
        # torch.save(lat32_torch, "lat32_torch.pt")
        # print (f"Lat32 torch shape: {lat32_torch.shape}, dtype: {lat32_torch.dtype}")
        print("Calling OFT")
        # ortho8 = self.oft8(lat8_torch, calib, grid)  # ortho8
        # ortho16 = self.oft16(lat16_torch, calib, grid)
        # ortho32 = self.oft32(lat32_torch, calib, grid)
        ortho8 = self.oft8.forward(device, lat8, calib, grid)  # ortho8
        ttnn.deallocate(lat8)
        ortho16 = self.oft16.forward(device, lat16, calib, grid)
        ttnn.deallocate(lat16)
        ortho32 = self.oft32.forward(device, lat32, calib, grid)
        ttnn.deallocate(lat32)
        ortho = ortho8 + ortho16 + ortho32
        print(
            f"OFT 8 output shape: {ortho8.shape}, dtype: {ortho8.dtype} layout: {ortho8.layout} memory_config: {ortho8.memory_config()}"
        )
        print(
            f"OFT 16 output shape: {ortho16.shape}, dtype: {ortho16.dtype} layout: {ortho16.layout} memory_config: {ortho16.memory_config()}"
        )
        print(
            f"OFT 32 output shape: {ortho32.shape}, dtype: {ortho32.dtype} layout: {ortho32.layout} memory_config: {ortho32.memory_config()}"
        )
        print(
            f"OFT output shape: {ortho.shape}, dtype: {ortho.dtype} layout: {ortho.layout} memory_config: {ortho.memory_config()}"
        )
        print("TTNN OFT finished")
        # n, c, h, w = ortho.shape
        # print(f"ORTHO::: {n=}, {c=}, {h=}, {w=}")
        # ortho = ortho.permute(0, 2, 3, 1).view(1, 1, n * h * w, c)
        print(f"Ortho shape: {ortho.shape}, dtype: {ortho.dtype}")
        if use_signpost:
            signpost(header="Oft module from host")
        # ortho = ttnn.from_torch(ortho, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        print(f"Ortho shape: {ortho.shape}, dtype: {ortho.dtype}")
        if ortho.layout == ttnn.TILE_LAYOUT:
            ortho = ttnn.to_layout(ortho, ttnn.ROW_MAJOR_LAYOUT)
        print(
            f"Ortho shape: {ortho.shape}, dtype: {ortho.dtype} layout: {ortho.layout} memory_config: {ortho.memory_config()}"
        )
        # ttnn.deallocate(lat32)
        td = ortho
        # return ortho
        if use_signpost:
            signpost(header="Topdown started")
        for layer in self.topdown:
            print(f"Topdown layer {layer=}")
            td = layer.forward(device, td, num_splits=2)
        print(f"Topdown 2 output shape: {td.shape}, dtype: {td.dtype}")
        # td = self.topdown[2].forward(device, td, num_splits=2)
        # print(f"Topdown 3 output shape: {td.shape}, dtype: {td.dtype}")
        # print()
        # result = ttnn.reshape(td, (1, 1, n, h, w, c))
        if use_signpost:
            signpost(header="Topdown finished")
            signpost(header="Head started")
        outputs, out_h, out_w = self.head(device, td)
        print(f"Head output shape: {outputs.shape}, dtype: {outputs.dtype} {out_h=} {out_w=}")
        outputs = ttnn.permute(outputs, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        outputs = ttnn.reshape(outputs, (1, -1, 9, out_h, out_w))
        slices = [1, 3, 3, 2]
        start = 0
        parts = []
        if use_signpost:
            signpost(header="Head finished")
            signpost(header="Slicing started")
        for i in range(len(slices)):
            parts.append(outputs[:, :, start : start + slices[i], :, :])
            start += slices[i]
        parts[0] = ttnn.squeeze(parts[0], dim=2)  # remove the 1 slice dimension
        for part in parts:
            print(f"Part shape: {part.shape}, dtype: {part.dtype}")
            # print(f"Part: {part}")
        if use_signpost:
            signpost(header="Slicing finished")
            signpost(header="OftNet finished")
        return parts  # ortho, ortho16, ortho32 #input_tensor
