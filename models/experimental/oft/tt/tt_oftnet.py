import ttnn
from models.experimental.oft.tt.common import Conv, GroupNorm
from models.experimental.oft.tt.tt_resnet import TTResNetFeatures
from models.experimental.oft.tt.tt_oft import OFT as TtOFT
from loguru import logger

try:
    from tracy import signpost

except ModuleNotFoundError:

    def signpost(*args, **kwargs):
        pass


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
        input_shape_hw,
        calib,
        grid,
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4,
        host_fallback_model=None,
        OFT_fallback=False,
    ):
        self.frontend = TTResNetFeatures(device, parameters.frontend, conv_pt.frontend, block, layers)
        self.lat8 = Conv(parameters.lat8, conv_pt.lat8, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bn8 = GroupNorm(parameters.bn8, num_groups=16, channels=256, eps=1e-5, dtype=ttnn.bfloat8_b)

        self.lat16 = Conv(parameters.lat16, conv_pt.lat16, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bn16 = GroupNorm(parameters.bn16, num_groups=16, channels=256, eps=1e-5, dtype=ttnn.bfloat8_b)

        self.lat32 = Conv(parameters.lat32, conv_pt.lat32, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bn32 = GroupNorm(parameters.bn32, num_groups=16, channels=256, eps=1e-5, dtype=ttnn.bfloat8_b)

        self.oft8 = TtOFT(
            device,
            parameters.oft8,
            256,
            grid_res,
            grid_height,
            [int(x * 1 / 8) for x in input_shape_hw],
            calib,
            grid,
            scale=1 / 8,
            use_precomputed_grid=False,
        )
        self.oft16 = TtOFT(
            device,
            parameters.oft16,
            256,
            grid_res,
            grid_height,
            [int(x * 1 / 16) for x in input_shape_hw],
            calib,
            grid,
            scale=1 / 16,
            use_precomputed_grid=False,
        )
        self.oft32 = TtOFT(
            device,
            parameters.oft32,
            256,
            grid_res,
            grid_height,
            [int(x * 1 / 32) for x in input_shape_hw],
            calib,
            grid,
            scale=1 / 32,
            use_precomputed_grid=False,
        )

        self.topdown = [
            block(
                device,
                parameters.topdown[i],
                conv_pt.topdown[i],
                256,
                256,
                stride=1,
                is_sliced=True,
            )
            for i in range(topdown_layers)
        ]

        self.head = Conv(parameters.head, conv_pt.head, output_layout=ttnn.ROW_MAJOR_LAYOUT, is_sliced=True)
        self.mean = ttnn.from_torch(
            mean, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.std = ttnn.from_torch(
            std, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.host_fallback_model = host_fallback_model
        self.OFT_fallback = OFT_fallback

    def forward_normalization(self, device, input_tensor):
        """Normalize input tensor by mean and std-dev"""
        # Normalize by mean and std-dev
        input_tensor = input_tensor - self.mean
        input_tensor = ttnn.div(input_tensor, self.std)
        if input_tensor.layout == ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return input_tensor

    def forward_lateral_layers(self, device, feats8, feats16, feats32):
        """Apply lateral layers to convert image features to common feature size"""
        # Apply lateral layers to convert image features to common feature size
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

        return lat8, lat16, lat32

    def forward_oft(self, device, lat8, lat16, lat32, calib, grid):
        """Apply orthographic feature transform (OFT) and sum the results"""
        signpost(header="Oft module")
        if self.host_fallback_model is not None and self.OFT_fallback:
            logger.warning("Using host fallback OFT model")
            import torch

            lat8_torch = ttnn.to_torch(lat8, dtype=torch.float32).permute((0, 3, 1, 2)).contiguous()
            lat16_torch = ttnn.to_torch(lat16, dtype=torch.float32).permute((0, 3, 1, 2)).contiguous()
            lat32_torch = ttnn.to_torch(lat32, dtype=torch.float32).permute((0, 3, 1, 2)).contiguous()
            ref_torch_ortho8 = self.host_fallback_model.oft8(
                lat8_torch,
                calib=ttnn.to_torch(calib, dtype=torch.float32),
                grid=ttnn.to_torch(grid, dtype=torch.float32),
            )
            ref_torch_ortho16 = self.host_fallback_model.oft16(
                lat16_torch,
                calib=ttnn.to_torch(calib, dtype=torch.float32),
                grid=ttnn.to_torch(grid, dtype=torch.float32),
            )
            ref_torch_ortho32 = self.host_fallback_model.oft32(
                lat32_torch,
                calib=ttnn.to_torch(calib, dtype=torch.float32),
                grid=ttnn.to_torch(grid, dtype=torch.float32),
            )

        # Apply OFT and sum
        ortho8 = self.oft8.forward(device, lat8, calib, grid)  # ortho8
        ttnn.deallocate(lat8)
        ortho16 = self.oft16.forward(device, lat16, calib, grid)
        ttnn.deallocate(lat16)
        ortho32 = self.oft32.forward(device, lat32, calib, grid)
        ttnn.deallocate(lat32)

        if self.OFT_fallback:
            host_ortho8 = ttnn.to_torch(ortho8).permute((0, 3, 1, 2)).reshape(ref_torch_ortho8.shape)
            host_ortho16 = ttnn.to_torch(ortho16).permute((0, 3, 1, 2)).reshape(ref_torch_ortho16.shape)
            host_ortho32 = ttnn.to_torch(ortho32).permute((0, 3, 1, 2)).reshape(ref_torch_ortho32.shape)
            from tests.ttnn.utils_for_testing import check_with_pcc

            orth8_pcc_passed, orth8_pcc = check_with_pcc(host_ortho8, ref_torch_ortho8, 0.99)
            orth16_pcc_passed, orth16_pcc = check_with_pcc(host_ortho16, ref_torch_ortho16, 0.99)
            orth32_pcc_passed, orth32_pcc = check_with_pcc(host_ortho32, ref_torch_ortho32, 0.99)
            logger.warning(
                f"{orth8_pcc_passed=}, {orth8_pcc=}, {orth16_pcc_passed=}, {orth16_pcc=}, {orth32_pcc_passed=}, {orth32_pcc=}"
            )
            logger.warning("OFT_fallback")
            ref_ortho8 = ref_torch_ortho8.permute((0, 2, 3, 1))
            ref_ortho8 = ref_ortho8.reshape(
                ref_ortho8.shape[0], 1, ref_ortho8.shape[1] * ref_ortho8.shape[2], ref_ortho8.shape[3]
            )
            ref_ortho16 = ref_torch_ortho16.permute((0, 2, 3, 1))
            ref_ortho16 = ref_ortho16.reshape(
                ref_ortho16.shape[0], 1, ref_ortho16.shape[1] * ref_ortho16.shape[2], ref_ortho16.shape[3]
            )
            ref_ortho32 = ref_torch_ortho32.permute((0, 2, 3, 1))
            ref_ortho32 = ref_ortho32.reshape(
                ref_ortho32.shape[0], 1, ref_ortho32.shape[1] * ref_ortho32.shape[2], ref_ortho32.shape[3]
            )
            ref_ortho8 = ttnn.from_torch(ref_ortho8, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            ref_ortho16 = ttnn.from_torch(ref_ortho16, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            ref_ortho32 = ttnn.from_torch(ref_ortho32, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            ref_ortho8 = ttnn.to_layout(ref_ortho8, ttnn.TILE_LAYOUT)
            ref_ortho16 = ttnn.to_layout(ref_ortho16, ttnn.TILE_LAYOUT)
            ref_ortho32 = ttnn.to_layout(ref_ortho32, ttnn.TILE_LAYOUT)
            ortho8 = ref_ortho8
            ortho16 = ref_ortho16
            ortho32 = ref_ortho32

        ortho = ortho8 + ortho16 + ortho32
        signpost(header="Oft module finished")
        logger.debug(f"Ortho shape: {ortho.shape}, dtype: {ortho.dtype}")

        if ortho.layout == ttnn.TILE_LAYOUT:
            ortho = ttnn.to_layout(ortho, ttnn.ROW_MAJOR_LAYOUT)
        logger.debug(
            f"Ortho shape: {ortho.shape}, dtype: {ortho.dtype} layout: {ortho.layout} memory_config: {ortho.memory_config()}"
        )

        return ortho

    def forward_topdown_network(self, device, ortho):
        """Apply topdown network"""
        signpost(header="Topdown started")
        td = ortho
        for layer in self.topdown:
            logger.debug(f"Topdown layer {layer=}")
            td = layer.forward(device, td, num_splits=3)
        signpost(header="Topdown finished")
        return td

    def forward_predict_encoded_outputs(self, device, td):
        """Predict encoded outputs and slice them"""
        signpost(header="Head started")
        outputs, out_h, out_w = self.head(device, td)
        logger.debug(f"Head output shape: {outputs.shape}, dtype: {outputs.dtype} {out_h=} {out_w=}")
        outputs = ttnn.permute(outputs, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        outputs = ttnn.reshape(outputs, (1, -1, 9, out_h, out_w))
        signpost(header="Head finished")
        signpost(header="Slicing started")
        slices = [1, 3, 3, 2]
        start = 0
        parts = []
        for i in range(len(slices)):
            parts.append(outputs[:, :, start : start + slices[i], :, :])
            start += slices[i]
        parts[0] = ttnn.squeeze(parts[0], dim=2)  # remove the 1 slice dimension
        for part in parts:
            logger.debug(f"Part shape: {part.shape}, dtype: {part.dtype}")
        signpost(header="Slicing finished")
        return parts

    def forward(self, device, input_tensor, calib, grid):
        signpost(header="OftNet module started")

        # Normalize input tensor
        normalized_input = self.forward_normalization(device, input_tensor)

        # Run frontend network
        feats8, feats16, feats32 = self.frontend.forward(device, normalized_input)

        # Apply lateral layers
        lat8, lat16, lat32 = self.forward_lateral_layers(device, feats8, feats16, feats32)

        # Apply OFT transformation
        ortho = self.forward_oft(device, lat8, lat16, lat32, calib, grid)

        # Apply topdown network
        td = self.forward_topdown_network(device, ortho)

        # Predict encoded outputs
        parts = self.forward_predict_encoded_outputs(device, td)

        signpost(header="OftNet finished")
        return parts
