import ttnn
import torch
import torch.nn.functional as F
from loguru import logger

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

from models.experimental.oft.reference.oft import EPSILON
from models.experimental.oft.reference.utils import perspective


def calculate_initialization_parameters(
    device, channels, cell_size, grid_height, feature_shape_hw, calib, grid, scale, use_precomputed_grid
):
    y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
    y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
    # Expand the grid in the y dimension
    corners = grid.unsqueeze(1) + y_corners.view(-1, 1, 1, 3)

    # Project grid corners to image plane and normalize to [-1, 1]
    img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners, dtype=torch.float32)
    feature_height, feature_width = feature_shape_hw
    # Normalize to [-1, 1]
    img_size = corners.new([feature_width, feature_height]) / scale
    norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)
    # Get top-left and bottom-right coordinates of voxel bounding boxes
    bbox_corners = torch.cat(
        [
            torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
            torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
        ],
        dim=-1,
    )
    batch, _, depth, width, _ = bbox_corners.size()
    bbox_corners = bbox_corners.flatten(2, 3)
    area = (
        (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * feature_height * feature_width * 0.25 + EPSILON
    ).unsqueeze(1)
    visible = area > EPSILON

    area = 1 / area
    area_nhwc = area.permute(0, 2, 3, 1)  # Convert to NHWC format
    visible_nhwc = visible.permute(0, 2, 3, 1)  # Convert to NHWC format
    top_left_bc = bbox_corners[..., [0, 1]]
    btm_right_bc = bbox_corners[..., [2, 3]]
    top_right_bc = bbox_corners[..., [2, 1]]
    btm_left_bc = bbox_corners[..., [0, 3]]

    batch_size, grid_h, grid_w, _ = top_left_bc.shape
    input_shape_nhwc = [batch_size, feature_height, feature_width, channels]

    # bbox_corners_tt = ttnn.from_torch(bbox_corners, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    if use_precomputed_grid:
        prepare_grid_lambda = lambda torch_grid_bf16, input_shape_nhwc: ttnn.to_device(
            ttnn.prepare_grid_sample_grid(
                ttnn.from_torch(torch_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32),
                input_shape_nhwc,
                padding_mode="zeros",
                output_dtype=ttnn.bfloat16,
            ),
            device,
        )
        top_left_bc_tt = prepare_grid_lambda(top_left_bc, input_shape_nhwc)
        btm_right_bc_tt = prepare_grid_lambda(btm_right_bc, input_shape_nhwc)
        top_right_bc_tt = prepare_grid_lambda(top_right_bc, input_shape_nhwc)
        btm_left_bc_tt = prepare_grid_lambda(btm_left_bc, input_shape_nhwc)

    else:
        top_left_bc_tt = ttnn.from_torch(top_left_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        btm_right_bc_tt = ttnn.from_torch(
            btm_right_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        top_right_bc_tt = ttnn.from_torch(
            top_right_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        btm_left_bc_tt = ttnn.from_torch(btm_left_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    visible_tt = ttnn.from_torch(visible_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    area_tt = ttnn.from_torch(area_nhwc, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return (
        [top_left_bc_tt, btm_right_bc_tt, top_right_bc_tt, btm_left_bc_tt],
        visible_tt,
        area_tt,
        [batch, depth, width],
    )
    # return area_tt


class OFT:
    def __init__(
        self,
        device,
        parameters,
        channels,
        cell_size,
        grid_height,
        features_shape_hw,
        calib,
        grid,
        scale,
        use_precomputed_grid,
    ):
        # params for conv3d
        self.linear_weight = parameters.conv3d.weight
        self.linear_bias = parameters.conv3d.bias
        self.scale = scale
        self.use_precomputed_grid = use_precomputed_grid
        self.features_shape_hw = features_shape_hw

        self.bbox_corners, self.visible, self.area, self.shape = calculate_initialization_parameters(
            device, channels, cell_size, grid_height, features_shape_hw, calib, grid, self.scale, use_precomputed_grid
        )

        # integral_image_quantization_strategy
        # None - no quantization
        # "to_uint32" - quantize to uint32 before integral image, dequantize after
        # "to_float32" - quantize to float32 before integral image, dequantize after
        self.integral_image_quantization_strategy = "to_uint32"
        logger.info(f"Integral image quantization strategy: {self.integral_image_quantization_strategy}")
        if self.integral_image_quantization_strategy == "to_uint32":
            self.prescaler = ttnn.from_torch(torch.tensor(1024 * 1024), device=device, dtype=ttnn.bfloat16)
            self.postscaler = ttnn.from_torch(torch.tensor(1 / 1024 / 1024), device=device, dtype=ttnn.bfloat16)

    def forward(self, device, features, calib, grid):
        if use_signpost:
            signpost(header="OFT block started")

        features = ttnn.reshape(features, [1, self.features_shape_hw[0], self.features_shape_hw[1], -1])
        if features.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
            features = ttnn.to_layout(features, ttnn.TILE_LAYOUT)

        if self.integral_image_quantization_strategy == None:
            integral_image = ttnn_integral_image_channel_last(features)
        elif self.integral_image_quantization_strategy == "to_uint32":
            features = ttnn.mul(features, self.prescaler, dtype=ttnn.bfloat16)
            features = ttnn.typecast(features, ttnn.uint32)
            integral_image = ttnn_integral_image_channel_last(features)
            integral_image = ttnn.typecast(integral_image, ttnn.bfloat16)
            integral_image = ttnn.mul(integral_image, self.postscaler, dtype=ttnn.bfloat16)
        elif self.integral_image_quantization_strategy == "to_float32":
            features = ttnn.typecast(features, ttnn.float32)
            integral_image = ttnn_integral_image_channel_last(features)
            integral_image = ttnn.typecast(integral_image, ttnn.bfloat16)

        if integral_image.get_layout() == ttnn.TILE_LAYOUT:
            integral_image = ttnn.to_layout(integral_image, ttnn.ROW_MAJOR_LAYOUT)

        top_left = ttnn.grid_sample(
            integral_image, self.bbox_corners[0], use_precomputed_grid=self.use_precomputed_grid
        )
        btm_right = ttnn.grid_sample(
            integral_image, self.bbox_corners[1], use_precomputed_grid=self.use_precomputed_grid
        )
        top_right = ttnn.grid_sample(
            integral_image, self.bbox_corners[2], use_precomputed_grid=self.use_precomputed_grid
        )
        btm_left = ttnn.grid_sample(
            integral_image, self.bbox_corners[3], use_precomputed_grid=self.use_precomputed_grid
        )

        vox_feats = ttnn.subtract(top_left, top_right)
        vox_feats = ttnn.add(vox_feats, btm_right)
        vox_feats = ttnn.subtract(vox_feats, btm_left)

        vox_feats = ttnn.mul(vox_feats, self.area)  # mull because area is 1/area
        vox_feats = ttnn.mul(vox_feats, self.visible)

        n, h, w, c = vox_feats.shape
        logger.debug(f"TTNN: {n=}, {h=}, {w=}, {c=}")
        vox_feats = ttnn.permute(vox_feats, (0, 2, 3, 1))
        vox_feats = ttnn.reshape(vox_feats, (1, 1, w, h * c))  # PCC 0.0019216915015543698

        if vox_feats.get_layout() != ttnn.TILE_LAYOUT:
            vox_feats = ttnn.to_layout(vox_feats, ttnn.TILE_LAYOUT)
        ortho_feats = ttnn.linear(
            vox_feats,
            self.linear_weight,
            bias=self.linear_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(vox_feats)
        ortho_feats = ttnn.relu(ortho_feats, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if use_signpost:
            signpost(header="OFT block ended")
        # return ortho_feats
        return (
            ortho_feats,
            integral_image,
            self.bbox_corners[0],
            self.bbox_corners[1],
            self.bbox_corners[2],
            self.bbox_corners[3],
        )


def ttnn_integral_image_channel_last(features_nhwc):
    assert len(features_nhwc.shape) == 4, "Input tensor must be 4D"
    assert features_nhwc.shape[0] == 1, "Batch size must be 1"
    tmp = ttnn.cumsum(features_nhwc, dim=1, dtype=features_nhwc.dtype)
    # ttnn.deallocate(features_nhwc) remove if needed, for now it work without move
    # tmp = ttnn.move(tmp)
    return ttnn.cumsum(tmp, dim=2, dtype=features_nhwc.dtype)
