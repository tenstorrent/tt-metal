import ttnn
import torch
import torch.nn.functional as F

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

EPSILON = 1e-6


def plot_tensor(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


def perspective(matrix, vector):
    vector = vector.unsqueeze(-1)
    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]


def calculate_initialization_parameters(device, channels, cell_size, grid_height, features, calib, grid, scale):
    y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
    y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
    # Expand the grid in the y dimension
    corners = grid.unsqueeze(1) + y_corners.view(-1, 1, 1, 3)

    # Project grid corners to image plane and normalize to [-1, 1]
    img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners)
    img_height, img_width = features.size()[2:]
    # Normalize to [-1, 1]
    img_size = corners.new([img_width, img_height]) / scale

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
        (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * img_height * img_width * 0.25 + EPSILON
    ).unsqueeze(1)
    visible = area > EPSILON
    # print(f"visible shape: {visible.shape}, dtype: {visible.dtype}")

    area = 1 / area
    area_nhwc = area.permute(0, 2, 3, 1)  # Convert to NHWC format
    visible_nhwc = visible.permute(0, 2, 3, 1)  # Convert to NHWC format
    top_left_bc = bbox_corners[..., [0, 1]]
    # print(f"TTNN: top_left_bc shape: {top_left_bc.shape}, dtype: {top_left_bc.dtype}")
    btm_right_bc = bbox_corners[..., [2, 3]]
    top_right_bc = bbox_corners[..., [2, 1]]
    btm_left_bc = bbox_corners[..., [0, 3]]

    # bbox_corners_tt = ttnn.from_torch(bbox_corners, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    top_left_bc_tt = ttnn.from_torch(top_left_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    btm_right_bc_tt = ttnn.from_torch(btm_right_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    top_right_bc_tt = ttnn.from_torch(top_right_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
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
    def __init__(self, device, parameters, channels, cell_size, grid_height, features, calib, grid, scale=1):
        # params for conv3d
        self.linear_weight = parameters.conv3d.weight
        self.linear_bias = parameters.conv3d.bias
        self.scale = scale

        self.bbox_corners, self.visible, self.area, self.shape = calculate_initialization_parameters(
            device, channels, cell_size, grid_height, features, calib, grid, self.scale
        )

    def forward(self, device, features, calib, grid):
        if use_signpost:
            signpost(header="OFT block started")

        integral_image = ttnn_integral_image_channel_last(features)
        if integral_image.get_layout() == ttnn.TILE_LAYOUT:
            integral_image = ttnn.to_layout(integral_image, ttnn.ROW_MAJOR_LAYOUT)

        top_left = ttnn.grid_sample(integral_image, self.bbox_corners[0])
        btm_right = ttnn.grid_sample(integral_image, self.bbox_corners[1])
        top_right = ttnn.grid_sample(integral_image, self.bbox_corners[2])
        btm_left = ttnn.grid_sample(integral_image, self.bbox_corners[3])

        vox_feats = ttnn.subtract(top_left, top_right)
        vox_feats = ttnn.add(vox_feats, btm_right)
        vox_feats = ttnn.subtract(vox_feats, btm_left)

        vox_feats = ttnn.mul(vox_feats, self.area)  # mull because area is 1/area
        vox_feats = ttnn.mul(vox_feats, self.visible)

        n, h, w, c = vox_feats.shape
        print(f"TTNN: {n=}, {h=}, {w=}, {c=}")
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
        return ortho_feats


def ttnn_integral_image_channel_last(features_nhwc):
    assert len(features_nhwc.shape) == 4, "Input tensor must be 4D"
    assert features_nhwc.shape[0] == 1, "Batch size must be 1"
    tmp = ttnn.cumsum(features_nhwc, dim=1, dtype=features_nhwc.dtype)
    # ttnn.deallocate(features_nhwc) remove if needed, for now it work without move
    # tmp = ttnn.move(tmp)
    return ttnn.cumsum(tmp, dim=2, dtype=features_nhwc.dtype)
