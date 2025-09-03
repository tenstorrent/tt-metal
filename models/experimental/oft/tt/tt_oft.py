import ttnn
import torch
import torch.nn.functional as F

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

EPSILON = 1e-6  # 0.0078125  #1e-6


def plot_tensor(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


def perspective_torch(matrix, vector):
    print(f"TT: matrix shape: {matrix.shape}, dtype: {matrix.dtype}")
    print(f"TT: vector shape: {vector.shape}, dtype: {vector.dtype}")
    vector = vector.unsqueeze(-1)
    print(f"TT: after unsqueeze vector shape: {vector.shape}, dtype: {vector.dtype}")
    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    print(f"TT: homogenous shape: {homogenous.shape}, dtype: {homogenous.dtype}")
    homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]


def calculate_initialization_parameters(device, channels, cell_size, grid_height, features, calib, grid, scale):
    y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
    y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
    # Expand the grid in the y dimension
    corners = grid.unsqueeze(1) + y_corners.view(-1, 1, 1, 3)

    # Project grid corners to image plane and normalize to [-1, 1]
    img_corners = perspective_torch(calib.view(-1, 1, 1, 1, 3, 4), corners)

    img_height, img_width = features.size()[2:]
    # print(f"img_height: {img_height}, img_width: {img_width}")
    # Normalize to [-1, 1]
    img_size = corners.new([img_width, img_height]) / scale
    # print(f"img_size shape: {img_size.shape}, dtype: {img_size.dtype}")
    # print(f"scale: {scale}")
    # print(f"img_size: {img_size}")
    norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)
    # print(f"TTNN:: norm_corners shape: {norm_corners.shape}, dtype: {norm_corners.dtype}")
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
        # self.conv3d = ttnn.Linear()
        self.scale = scale

        self.bbox_corners, self.visible, self.area, self.shape = calculate_initialization_parameters(
            device, channels, cell_size, grid_height, features, calib, grid, self.scale
        )
        # self.area = calculate_initialization_parameters(
        #     device, channels, cell_size, grid_height, features, calib, grid, self.scale
        # )
        # print("area shape:", self.area.shape)

    def forward(self, device, features, calib, grid):
        if use_signpost:
            signpost(header="OFT block started")
        # print(f"TTNN: features shape: {features.shape}, dtype: {features.dtype}")
        integral_image = ttnn_integral_image_channel_last(features)
        # print(
        #     f"TTNN: integral_image shape: {integral_image.shape}, dtype: {integral_image.dtype}, layout: {integral_image.layout}, memory_config: {integral_image.memory_config()}"
        # )
        if integral_image.get_layout() == ttnn.TILE_LAYOUT:
            integral_image = ttnn.to_layout(integral_image, ttnn.ROW_MAJOR_LAYOUT)
        # print(
        #     f"TTNN: integral_image shape: {integral_image.shape}, dtype: {integral_image.dtype}, layout: {integral_image.layout}, memory_config: {integral_image.memory_config()}"
        # )
        # print(f"TTNN: bbox_corners shape: {self.bbox_corners[0].shape}, dtype: {self.bbox_corners[0].dtype}")

        # return integral_image
        # torch grid sample
        # box_corners_tl = ttnn.to_torch(self.bbox_corners[0], dtype=torch.float32)
        # box_corners_br = ttnn.to_torch(self.bbox_corners[1], dtype=torch.float32)
        # box_corners_tr = ttnn.to_torch(self.bbox_corners[2], dtype=torch.float32)
        # box_corners_bl = ttnn.to_torch(self.bbox_corners[3], dtype=torch.float32)

        # integral_image_torch = ttnn.to_torch(integral_image, dtype=torch.float32).permute(0,3,1,2)  # Convert to NCHW for torch
        # print(f"TORCH: integral_image_torch shape: {integral_image_torch.shape}, dtype: {integral_image_torch.dtype}")
        # # return box_corners_bl
        # top_left = F.grid_sample(integral_image_torch, box_corners_tl)
        # btm_right = F.grid_sample(integral_image_torch, box_corners_br)
        # top_right = F.grid_sample(integral_image_torch, box_corners_tr)
        # btm_left = F.grid_sample(integral_image_torch, box_corners_bl)
        # print(f"btm_left shape: {btm_left.shape}, dtype: {btm_left.dtype}")
        # # return btm_left

        # vox_feats_torch = top_left + btm_right - top_right - btm_left
        # # return vox_feats_torch
        # vox_feats_torch = vox_feats_torch.permute(0, 2, 3, 1)  # Convert to NHWC for TTNN
        # vox_feats = ttnn.from_torch(vox_feats_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        # end torch grid sample

        top_left = ttnn.grid_sample(integral_image, self.bbox_corners[0])
        # # return top_left
        # # print(f"TTNN: top_left after grid_sample shape: {top_left.shape}, dtype: {top_left.dtype}")

        btm_right = ttnn.grid_sample(integral_image, self.bbox_corners[1])
        top_right = ttnn.grid_sample(integral_image, self.bbox_corners[2])
        btm_left = ttnn.grid_sample(integral_image, self.bbox_corners[3])

        vox_feats = ttnn.subtract(top_left, top_right)
        vox_feats = ttnn.add(vox_feats, btm_right)
        vox_feats = ttnn.subtract(vox_feats, btm_left)
        # return vox_feats
        print(
            f"vox_feats shape: {vox_feats.shape}, dtype: {vox_feats.dtype}, layout: {vox_feats.layout}, memory_config: {vox_feats.memory_config()}"
        )
        # return vox_feats
        # vox_feats = ttnn.subtract(vox_feats, 0.005, dtype=ttnn.bfloat16)
        # return vox_feats #pcc na vox_feats 0.9831520098479232
        # if vox_feats.get_layout() != ttnn.TILE_LAYOUT:
        #     vox_feats = ttnn.to_layout(vox_feats, ttnn.TILE_LAYOUT)
        # if self.area.get_layout() != ttnn.TILE_LAYOUT:
        #     self.area = ttnn.to_layout(self.area, ttnn.TILE_LAYOUT)
        # vox_feats = ttnn.div(vox_feats, self.area, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # print(f"TTNN: visible shape: {self.visible.shape}, dtype: {self.visible.dtype}")
        # print(f"TTNN: visible layout: {self.visible.layout}, memory_config: {self.visible.memory_config()}")
        # print(f"TTNN: area shape: {self.area.shape}, dtype: {self.area.dtype}")
        # print(f"TTNN: area layout: {self.area.layout}, memory_config: {self.area.memory_config()}")
        # print(f"TTNN: vox_feats before visibility mask shape: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        # print(f"TTNN: vox_feats before visibility mask layout: {vox_feats.layout}, memory_config: {vox_feats.memory_config()}")
        # return vox_feats
        # vox_feats = vox_feats #- EPSILON
        vox_feats = ttnn.mul(vox_feats, self.area)

        # return vox_feats # -> pcc 0.008397531667143535
        # print(f"TTNN: vox_feats shape before division: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        # print(f"TTNN: area shape: {self.area.shape}, dtype: {self.area.dtype}")
        # return vox_feats # -> pcc 0.008397531667143535
        # if vox_feats.get_layout() != ttnn.TILE_LAYOUT:
        #     vox_feats = ttnn.to_layout(vox_feats, ttnn.TILE_LAYOUT)
        # if self.area.get_layout() != ttnn.TILE_LAYOUT:
        #     self.area = ttnn.to_layout(self.area, ttnn.TILE_LAYOUT)
        # vox_feats = ttnn.div(vox_feats, self.area, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # vox_feats_torch = ttnn.to_torch(vox_feats)
        # print(f"max vox_feats_torch: {torch.max(vox_feats_torch)}")
        # print(f"min vox_feats_torch: {torch.min(vox_feats_torch)}")

        # area_torch = ttnn.to_torch(self.area)
        # print(f"max area_torch: {torch.max(area_torch)}")
        # print(f"min area_torch: {torch.min(area_torch)}")
        # vox_feats_torch = vox_feats_torch * area_torch
        # vox_feats = ttnn.from_torch(vox_feats_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        # return vox_feats

        # print(f"TTNN: vox_feats shape after division: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        # print(f"TTNN: visible shape: {self.visible.shape}, dtype: {self.visible.dtype}")
        vox_feats = ttnn.mul(vox_feats, self.visible)
        # print(f"TTNN: vox_feats shape after division: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        # return vox_feats
        n, h, w, c = vox_feats.shape
        print(f"TTNN: {n=}, {h=}, {w=}, {c=}")
        vox_feats = ttnn.permute(vox_feats, (0, 2, 3, 1))
        # print(f"TTNN: vox_feats shape after permute: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        vox_feats = ttnn.reshape(vox_feats, (1, 1, w, h * c))  # PCC 0.0019216915015543698
        # print(f"TTNN: vox_feats shape after reshape: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        # return vox_feats

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
        print(f"TTNN: ortho_feats shape after linear: {ortho_feats.shape}, dtype: {ortho_feats.dtype}")
        # ortho_feats = ttnn.reshape(ortho_feats, (self.shape[0], self.shape[1], self.shape[2], -1), memory_config=ttnn.L1_MEMORY_CONFIG)
        # ortho_feats = ttnn.permute(ortho_feats, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        ortho_feats = ttnn.relu(ortho_feats, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"TTNN: ortho_feats shape after relu: {ortho_feats.shape}, dtype: {ortho_feats.dtype}")
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
