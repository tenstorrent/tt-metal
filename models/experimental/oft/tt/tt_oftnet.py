# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

# from models.experimental.oft.tt.common import Conv
from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.oft.tt.common import GroupNorm
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
        state_dict,
        layer_args,
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
        fallback_feedforward=False,
        fallback_lateral=False,
        fallback_oft=False,
    ):
        self.frontend = TTResNetFeatures(device, state_dict.frontend, layer_args.frontend, block, layers)
        self.lat8 = TtConv2d(layer_args.lat8["optimized_configuration"], device)
        self.bn8 = GroupNorm(state_dict.bn8, layer_args.bn8)

        self.lat16 = TtConv2d(layer_args.lat16["optimized_configuration"], device)
        self.bn16 = GroupNorm(state_dict.bn16, layer_args.bn16)

        self.lat32 = TtConv2d(layer_args.lat32["optimized_configuration"], device)
        self.bn32 = GroupNorm(state_dict.bn32, layer_args.bn32)

        self.oft8 = TtOFT(
            device,
            state_dict.oft8,
            256,
            grid_res,
            grid_height,
            [int(x * 1 / 8) for x in input_shape_hw],
            calib,
            grid,
            scale=1 / 8,
            use_precomputed_grid=True,
        )
        self.oft16 = TtOFT(
            device,
            state_dict.oft16,
            256,
            grid_res,
            grid_height,
            [int(x * 1 / 16) for x in input_shape_hw],
            calib,
            grid,
            scale=1 / 16,
            use_precomputed_grid=True,
        )
        self.oft32 = TtOFT(
            device,
            state_dict.oft32,
            256,
            grid_res,
            grid_height,
            [int(x * 1 / 32) for x in input_shape_hw],
            calib,
            grid,
            scale=1 / 32,
            use_precomputed_grid=True,
        )

        self.topdown = [
            block(
                device,
                state_dict.topdown[i],
                state_dict.layer_args.topdown[i],
                is_sliced=True,
            )
            for i in range(topdown_layers)
        ]

        self.head = TtConv2d(layer_args.head["optimized_configuration"], device)
        self.mean = ttnn.from_torch(
            mean, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.std = ttnn.from_torch(
            std, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.host_fallback_model = host_fallback_model
        self.OFT_fallback = fallback_oft
        self.FeedForward_fallback = fallback_feedforward
        self.Lateral_fallback = fallback_lateral
        assert not (
            (host_fallback_model is None) and (fallback_oft or fallback_feedforward or fallback_lateral)
        ), "If host_fallback_model is None, all fallbacks must be False"

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
        lat8 = self.lat8(feats8)
        if lat8.layout == ttnn.TILE_LAYOUT:
            lat8 = ttnn.to_layout(lat8, ttnn.ROW_MAJOR_LAYOUT)
        lat8 = self.bn8(device, lat8, shard="HS")
        lat8 = ttnn.relu(lat8)
        lat8 = ttnn.sharded_to_interleaved(lat8, ttnn.DRAM_MEMORY_CONFIG)
        lat8 = ttnn.to_layout(lat8, ttnn.TILE_LAYOUT)

        lat16 = self.lat16(feats16)
        if lat16.layout == ttnn.TILE_LAYOUT:
            lat16 = ttnn.to_layout(lat16, ttnn.ROW_MAJOR_LAYOUT)
        lat16 = self.bn16(device, lat16, shard="HS")
        lat16 = ttnn.relu(lat16)
        lat16 = ttnn.sharded_to_interleaved(lat16, ttnn.DRAM_MEMORY_CONFIG)
        lat16 = ttnn.to_layout(lat16, ttnn.TILE_LAYOUT)

        lat32 = self.lat32(feats32)
        if lat32.layout == ttnn.TILE_LAYOUT:
            lat32 = ttnn.to_layout(lat32, ttnn.ROW_MAJOR_LAYOUT)
        lat32 = self.bn32(device, lat32, shard="BS")
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
            ref_torch_ortho = ref_torch_ortho8 + ref_torch_ortho16 + ref_torch_ortho32

        # Apply OFT and sum
        ortho8, integral_img8, bbox_top_left8, bbox_btm_right8, bbox_top_right8, bbox_btm_left8 = self.oft8.forward(
            device, lat8, calib, grid
        )  # ortho8
        # ttnn.deallocate(lat8)
        lat8 = ttnn.to_memory_config(lat8, ttnn.DRAM_MEMORY_CONFIG)
        (
            ortho16,
            integral_img16,
            bbox_top_left16,
            bbox_btm_right16,
            bbox_top_right16,
            bbox_btm_left16,
        ) = self.oft16.forward(device, lat16, calib, grid)
        # ttnn.deallocate(lat16)
        lat16 = ttnn.to_memory_config(lat16, ttnn.DRAM_MEMORY_CONFIG)
        (
            ortho32,
            integral_img32,
            bbox_top_left32,
            bbox_btm_right32,
            bbox_top_right32,
            bbox_btm_left32,
        ) = self.oft32.forward(device, lat32, calib, grid)
        # ttnn.deallocate(lat32)
        lat32 = ttnn.to_memory_config(lat32, ttnn.DRAM_MEMORY_CONFIG)

        ortho = ortho8 + ortho16 + ortho32

        if self.OFT_fallback:
            host_ortho8 = ttnn.to_torch(ortho8).permute((0, 3, 1, 2)).reshape(ref_torch_ortho8.shape)
            host_ortho16 = ttnn.to_torch(ortho16).permute((0, 3, 1, 2)).reshape(ref_torch_ortho16.shape)
            host_ortho32 = ttnn.to_torch(ortho32).permute((0, 3, 1, 2)).reshape(ref_torch_ortho32.shape)
            host_ortho = ttnn.to_torch(ortho).permute((0, 3, 1, 2)).reshape(ref_torch_ortho.shape)

            from tests.ttnn.utils_for_testing import check_with_pcc

            orth8_pcc_passed, orth8_pcc = check_with_pcc(host_ortho8, ref_torch_ortho8, 0.99)
            orth16_pcc_passed, orth16_pcc = check_with_pcc(host_ortho16, ref_torch_ortho16, 0.99)
            orth32_pcc_passed, orth32_pcc = check_with_pcc(host_ortho32, ref_torch_ortho32, 0.99)
            ortho_pcc_passed, ortho_pcc = check_with_pcc(host_ortho, ref_torch_ortho, 0.99)

            logger.warning(
                f"{orth8_pcc_passed=}, {orth8_pcc=}, {orth16_pcc_passed=}, {orth16_pcc=}, {orth32_pcc_passed=}, {orth32_pcc=} {ortho_pcc_passed=}, {ortho_pcc=}"
            )
            logger.warning("OFT_fallback")

            host_to_device_with_permute_and_reshape = lambda tensor: ttnn.from_torch(
                tensor.permute((0, 2, 3, 1)).reshape(
                    tensor.shape[0], 1, tensor.shape[2] * tensor.shape[3], tensor.shape[1]
                ),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )

            ref_ortho8 = host_to_device_with_permute_and_reshape(ref_torch_ortho8)
            ref_ortho16 = host_to_device_with_permute_and_reshape(ref_torch_ortho16)
            ref_ortho32 = host_to_device_with_permute_and_reshape(ref_torch_ortho32)
            ref_torch_ortho = ref_torch_ortho.permute((0, 2, 3, 1))
            host_to_device_with_permute_and_reshape = lambda tensor: ttnn.to_layout(
                ttnn.from_torch(
                    tensor.permute((0, 2, 3, 1)).reshape(
                        tensor.shape[0], 1, tensor.shape[2] * tensor.shape[3], tensor.shape[1]
                    ),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                ),
                ttnn.TILE_LAYOUT,
            )

            ref_ortho8 = host_to_device_with_permute_and_reshape(ref_torch_ortho8)
            ref_ortho16 = host_to_device_with_permute_and_reshape(ref_torch_ortho16)
            ref_ortho32 = host_to_device_with_permute_and_reshape(ref_torch_ortho32)
            ref_ortho = host_to_device_with_permute_and_reshape(ref_torch_ortho)

            ortho8 = ref_ortho8
            ortho16 = ref_ortho16
            ortho32 = ref_ortho32
            ortho = ref_ortho

        signpost(header="Oft module finished")
        logger.debug(f"Ortho shape: {ortho.shape}, dtype: {ortho.dtype}")

        if ortho.layout == ttnn.TILE_LAYOUT:
            ortho = ttnn.to_layout(ortho, ttnn.ROW_MAJOR_LAYOUT)
        logger.debug(
            f"Ortho shape: {ortho.shape}, dtype: {ortho.dtype} layout: {ortho.layout} memory_config: {ortho.memory_config()}"
        )

        return (
            lat8,
            lat16,
            lat32,
            integral_img8,
            integral_img16,
            integral_img32,
            bbox_top_left8,
            bbox_btm_right8,
            bbox_top_right8,
            bbox_btm_left8,
            bbox_top_left16,
            bbox_btm_right16,
            bbox_top_right16,
            bbox_btm_left16,
            bbox_top_left32,
            bbox_btm_right32,
            bbox_top_right32,
            bbox_btm_left32,
            ortho8,
            ortho16,
            ortho32,
            ortho,
        )

    def forward_topdown_network(self, device, ortho):
        """Apply topdown network"""
        signpost(header="Topdown started")
        td = ortho
        for layer in self.topdown:
            logger.debug(f"Topdown layer {layer=}")
            td = layer.forward(device, td, gn_shard="HS", num_splits=2)  # hangs on top down with these settings;
            # td = layer.forward(device, td)
        signpost(header="Topdown finished")
        return td

    def forward_predict_encoded_outputs(self, device, td):
        """Predict encoded outputs and slice them"""
        signpost(header="Head started")
        out_h, out_w = 159, 159  # todo extract magic numbers from a state dict
        outputs = self.head(td)
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
        if self.FeedForward_fallback and self.host_fallback_model is not None:
            import torch

            logger.warning("Using host fallback FeedForward model")
            normalized_input_torch = (
                ttnn.to_torch(normalized_input, dtype=torch.float32).permute((0, 3, 1, 2)).contiguous()
            )
            feats8_torch, feats16_torch, feats32_torch = self.host_fallback_model.frontend(normalized_input_torch)
            feats8 = ttnn.from_torch(
                feats8_torch.permute((0, 2, 3, 1)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            feats16 = ttnn.from_torch(
                feats16_torch.permute((0, 2, 3, 1)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            feats32 = ttnn.from_torch(
                feats32_torch.permute((0, 2, 3, 1)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
        else:
            # feats8, feats16, feats32 = self.frontend.forward(device, normalized_input)
            feats8, feats16, feats32 = self.frontend.forward(device, normalized_input)

        # Apply lateral layers
        if self.Lateral_fallback and self.host_fallback_model is not None:
            logger.warning("Using host fallback Lateral model")
            import torch
            import torch.nn.functional as F

            feats8_torch = (
                ttnn.to_torch(feats8, dtype=self.host_fallback_model.dtype).permute((0, 3, 1, 2)).contiguous()
            )
            feats16_torch = (
                ttnn.to_torch(feats16, dtype=self.host_fallback_model.dtype).permute((0, 3, 1, 2)).contiguous()
            )
            feats32_torch = (
                ttnn.to_torch(feats32, dtype=self.host_fallback_model.dtype).permute((0, 3, 1, 2)).contiguous()
            )
            lat8_torch = F.relu(self.host_fallback_model.bn8(self.host_fallback_model.lat8(feats8_torch)))
            lat16_torch = F.relu(self.host_fallback_model.bn16(self.host_fallback_model.lat16(feats16_torch)))
            lat32_torch = F.relu(self.host_fallback_model.bn32(self.host_fallback_model.lat32(feats32_torch)))
            lat8 = ttnn.from_torch(
                lat8_torch.permute((0, 2, 3, 1)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            lat16 = ttnn.from_torch(
                lat16_torch.permute((0, 2, 3, 1)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            lat32 = ttnn.from_torch(
                lat32_torch.permute((0, 2, 3, 1)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
        else:
            lat8, lat16, lat32 = self.forward_lateral_layers(device, feats8, feats16, feats32)

        # Apply OFT transformation
        import torch  # HACK

        calib_torch = ttnn.to_torch(calib, dtype=torch.float32)
        grid_torch = ttnn.to_torch(grid, dtype=torch.float32)
        if self.OFT_fallback:
            import torch

            lat8_torch = (
                ttnn.to_torch(lat8, dtype=torch.float32)
                .permute((0, 3, 1, 2))
                .contiguous()
                .reshape((1, 256, input_tensor.shape[1] // 8, input_tensor.shape[2] // 8))
            )
            lat16_torch = (
                ttnn.to_torch(lat16, dtype=torch.float32)
                .permute((0, 3, 1, 2))
                .contiguous()
                .reshape((1, 256, input_tensor.shape[1] // 16, input_tensor.shape[2] // 16))
            )
            lat32_torch = (
                ttnn.to_torch(lat32, dtype=torch.float32)
                .permute((0, 3, 1, 2))
                .contiguous()
                .reshape((1, 256, input_tensor.shape[1] // 32, input_tensor.shape[2] // 32))
            )
            (
                ortho8,
                integral_img8,
                bbox_top_left8,
                bbox_btm_right8,
                bbox_top_right8,
                bbox_btm_left8,
            ) = self.host_fallback_model.oft8(lat8_torch, calib_torch, grid_torch)
            (
                ortho16,
                integral_img16,
                bbox_top_left16,
                bbox_btm_right16,
                bbox_top_right16,
                bbox_btm_left16,
            ) = self.host_fallback_model.oft16(lat16_torch, calib_torch, grid_torch)
            (
                ortho32,
                integral_img32,
                bbox_top_left32,
                bbox_btm_right32,
                bbox_top_right32,
                bbox_btm_left32,
            ) = self.host_fallback_model.oft32(lat32_torch, calib_torch, grid_torch)
            ortho = ortho8 + ortho16 + ortho32
            ortho = ttnn.from_torch(
                ortho.permute((0, 2, 3, 1)).reshape((1, 1, ortho.shape[2] * ortho.shape[3], ortho.shape[1])),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            ortho = ttnn.to_layout(ortho, ttnn.TILE_LAYOUT)
        else:
            (
                lat8,
                lat16,
                lat32,
                integral_img8,
                integral_img16,
                integral_img32,
                bbox_top_left8,
                bbox_btm_right8,
                bbox_top_right8,
                bbox_btm_left8,
                bbox_top_left16,
                bbox_btm_right16,
                bbox_top_right16,
                bbox_btm_left16,
                bbox_top_left32,
                bbox_btm_right32,
                bbox_top_right32,
                bbox_btm_left32,
                ortho8,
                ortho16,
                ortho32,
                ortho,
            ) = self.forward_oft(device, lat8, lat16, lat32, calib, grid)
        # return (feats8, feats16, feats32, lat8, lat16, lat32, ortho8, ortho16, ortho32, ortho, calib_torch, grid_torch), ("feats8", "feats16", "feats32", "lat8", "lat16", "lat32", "ortho8", "ortho16", "ortho32", "ortho", "calib", "grid")

        # Apply topdown network
        td = self.forward_topdown_network(device, ortho)

        # Predict encoded outputs
        tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = self.forward_predict_encoded_outputs(device, td)

        signpost(header="OftNet finished")

        return (
            [
                (
                    normalized_input,
                    feats8,
                    feats16,
                    feats32,
                    lat8,
                    lat16,
                    lat32,
                    integral_img8,
                    integral_img16,
                    integral_img32,
                    ttnn.to_torch(bbox_top_left8) if self.OFT_fallback == False else bbox_top_left8,
                    ttnn.to_torch(bbox_btm_right8) if self.OFT_fallback == False else bbox_btm_right8,
                    ttnn.to_torch(bbox_top_right8) if self.OFT_fallback == False else bbox_top_right8,
                    ttnn.to_torch(bbox_btm_left8) if self.OFT_fallback == False else bbox_btm_left8,
                    ttnn.to_torch(bbox_top_left16) if self.OFT_fallback == False else bbox_top_left16,
                    ttnn.to_torch(bbox_btm_right16) if self.OFT_fallback == False else bbox_btm_right16,
                    ttnn.to_torch(bbox_top_right16) if self.OFT_fallback == False else bbox_top_right16,
                    ttnn.to_torch(bbox_btm_left16) if self.OFT_fallback == False else bbox_btm_left16,
                    ttnn.to_torch(bbox_top_left32) if self.OFT_fallback == False else bbox_top_left32,
                    ttnn.to_torch(bbox_btm_right32) if self.OFT_fallback == False else bbox_btm_right32,
                    ttnn.to_torch(bbox_top_right32) if self.OFT_fallback == False else bbox_top_right32,
                    ttnn.to_torch(bbox_btm_left32) if self.OFT_fallback == False else bbox_btm_left32,
                    ortho8,
                    ortho16,
                    ortho32,
                    ortho,
                    calib_torch,
                    grid_torch,
                    td,
                ),
                (
                    "image",
                    "feats8",
                    "feats16",
                    "feats32",
                    "lat8",
                    "lat16",
                    "lat32",
                    "integral_img8",
                    "integral_img16",
                    "integral_img32",
                    "bbox_top_left8",
                    "bbox_btm_right8",
                    "bbox_top_right8",
                    "bbox_btm_left8",
                    "bbox_top_left16",
                    "bbox_btm_right16",
                    "bbox_top_right16",
                    "bbox_btm_left16",
                    "bbox_top_left32",
                    "bbox_btm_right32",
                    "bbox_top_right32",
                    "bbox_btm_left32",
                    "ortho8",
                    "ortho16",
                    "ortho32",
                    "ortho",
                    "calib",
                    "grid",
                    "td",
                ),
            ],
            tt_scores,
            tt_pos_offsets,
            tt_dim_offsets,
            tt_ang_offsets,
        )
