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


def perspective(matrix, vector, device):
    print(f"perspective matrix shape: {matrix.shape}, dtype: {matrix.dtype}")
    print(f"perspective vector shape: {vector.shape}, dtype: {vector.dtype}")
    vector = ttnn.unsqueeze(vector, -1)
    print(f"perspective vector after unsqueeze shape: {vector.shape}, dtype: {vector.dtype}")

    tile_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttnn.BufferType.L1
    )
    if matrix.get_layout() != ttnn.TILE_LAYOUT:
        print(f"Converting matrix to TILE_LAYOUT")
        matrix = ttnn.to_layout(matrix, layout=ttnn.TILE_LAYOUT)
    if vector.get_layout() != ttnn.TILE_LAYOUT:
        print(f"Converting vector to TILE_LAYOUT")
        vector = ttnn.to_memory_config(vector, layout=ttnn.TILE_LAYOUT, memory_config=tile_mem_config)
    homogenous = ttnn.matmul(matrix[..., :-1], vector)
    homogenous += matrix[..., [-1]]
    homogenous = ttnn.squeeze(homogenous, -1)
    # homogenous = ttnn.to_memory_config(homogenous, memory_config=ttnn.L1_MEMORY_CONFIG)
    homogenous = ttnn.div(homogenous[..., :-1], homogenous[..., -1:], memory_config=ttnn.L1_MEMORY_CONFIG)
    return homogenous


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
    print(f"img_height: {img_height}, img_width: {img_width}")
    # Normalize to [-1, 1]
    img_size = corners.new([img_width, img_height]) / scale
    print(f"img_size shape: {img_size.shape}, dtype: {img_size.dtype}")
    print(f"scale: {scale}")
    print(f"img_size: {img_size}")
    norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)
    print(f"TTNN:: norm_corners shape: {norm_corners.shape}, dtype: {norm_corners.dtype}")
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
    print(f"bbox_corners shape: {bbox_corners.shape}, dtype: {bbox_corners.dtype}")
    # Compute the area of each bounding box
    area = (
        (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * img_height * img_width * 0.25 + EPSILON
    ).unsqueeze(1)
    area = area.to(torch.bfloat16)  # Avoid division by zero
    step = 0.01  # or any value you want
    # area[area == 0] += step
    print("Area values in (-EPSILON, EPSILON):")
    mask = (area > -EPSILON) & (area < EPSILON)
    print("broj elemenata ", torch.sum(mask))
    print("min element ", torch.min(area))
    print("max element ", torch.max(area))

    num_zeros = (area == 0).sum().item()
    print("Broj nula:", num_zeros)
    visible = area > EPSILON
    print(f"visible shape: {visible.shape}, dtype: {visible.dtype}")

    area = 1 / area
    area_nhwc = area.permute(0, 2, 3, 1)  # Convert to NHWC format
    visible_nhwc = visible.permute(0, 2, 3, 1)  # Convert to NHWC format
    top_left_bc = bbox_corners[..., [0, 1]]
    btm_right_bc = bbox_corners[..., [2, 3]]
    top_right_bc = bbox_corners[..., [2, 1]]
    btm_left_bc = bbox_corners[..., [0, 3]]

    # bbox_corners_tt = ttnn.from_torch(bbox_corners, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    top_left_bc_tt = ttnn.from_torch(top_left_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    btm_right_bc_tt = ttnn.from_torch(btm_right_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    top_right_bc_tt = ttnn.from_torch(top_right_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    btm_left_bc_tt = ttnn.from_torch(btm_left_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    visible_tt = ttnn.from_torch(visible_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    area_tt = ttnn.from_torch(area_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
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
        print(f"TTNN: features shape: {features.shape}, dtype: {features.dtype}")
        integral_image = ttnn_integral_image_channel_last(features)
        print(
            f"TTNN: integral_image shape: {integral_image.shape}, dtype: {integral_image.dtype}, layout: {integral_image.layout}, memory_config: {integral_image.memory_config()}"
        )
        if integral_image.get_layout() == ttnn.TILE_LAYOUT:
            integral_image = ttnn.to_layout(integral_image, ttnn.ROW_MAJOR_LAYOUT)
        print(
            f"TTNN: integral_image shape: {integral_image.shape}, dtype: {integral_image.dtype}, layout: {integral_image.layout}, memory_config: {integral_image.memory_config()}"
        )
        print(f"TTNN: bbox_corners shape: {self.bbox_corners[0].shape}, dtype: {self.bbox_corners[0].dtype}")

        # return integral_image
        top_left = ttnn.grid_sample(integral_image, self.bbox_corners[0])
        # return top_left
        print(f"TTNN: top_left after grid_sample shape: {top_left.shape}, dtype: {top_left.dtype}")

        btm_right = ttnn.grid_sample(integral_image, self.bbox_corners[1])
        top_right = ttnn.grid_sample(integral_image, self.bbox_corners[2])
        btm_left = ttnn.grid_sample(integral_image, self.bbox_corners[3])
        # return top_right
        # vox_feats = ttnn.add(top_left, btm_right, dtype=ttnn.bfloat16)
        # vox_feats = ttnn.subtract(vox_feats, top_right, dtype=ttnn.bfloat16)
        # vox_feats = ttnn.subtract(vox_feats, btm_left, dtype=ttnn.bfloat16)
        vox_feats = top_left + btm_right - top_right - btm_left
        # return vox_feats #pcc na vox_feats 0.9831520098479232
        # if vox_feats.get_layout() != ttnn.TILE_LAYOUT:
        #     vox_feats = ttnn.to_layout(vox_feats, ttnn.TILE_LAYOUT)
        # if self.area.get_layout() != ttnn.TILE_LAYOUT:
        #     self.area = ttnn.to_layout(self.area, ttnn.TILE_LAYOUT)
        # vox_feats = ttnn.div(vox_feats, self.area, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        vox_feats = ttnn.multiply(vox_feats, self.area, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        print(f"TTNN: vox_feats shape before division: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        print(f"TTNN: area shape: {self.area.shape}, dtype: {self.area.dtype}")
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

        print(f"TTNN: vox_feats shape after division: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        print(f"TTNN: visible shape: {self.visible.shape}, dtype: {self.visible.dtype}")
        vox_feats = ttnn.multiply(vox_feats, self.visible, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"TTNN: vox_feats shape after division: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        # return vox_feats
        vox_feats = ttnn.permute(vox_feats, (0, 2, 1, 3))
        print(f"TTNN: vox_feats shape after permute: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        vox_feats = ttnn.reshape(
            vox_feats, (vox_feats.shape[0], 1, vox_feats.shape[1], vox_feats.shape[2] * vox_feats.shape[3])
        )  # PCC 0.0019216915015543698
        print(f"TTNN: vox_feats shape after reshape: {vox_feats.shape}, dtype: {vox_feats.dtype}")
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
        print(f"TTNN: ortho_feats shape after linear: {ortho_feats.shape}, dtype: {ortho_feats.dtype}")
        # ortho_feats = ttnn.reshape(ortho_feats, (self.shape[0], self.shape[1], self.shape[2], -1), memory_config=ttnn.L1_MEMORY_CONFIG)
        # ortho_feats = ttnn.permute(ortho_feats, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        ortho_feats = ttnn.relu(ortho_feats, memory_config=ttnn.L1_MEMORY_CONFIG)
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


class OLD_OFT:
    def __init__(self, device, parameters, y_corners, img_corners, scale=1):
        self.y_corners = y_corners
        print(
            f"TT_OFT:: y_corners shape: {self.y_corners.shape}, dtype: {self.y_corners.dtype}, memory_config: {self.y_corners.memory_config()}"
        )
        self.linear_weight = parameters.conv3d.weight
        self.linear_bias = parameters.conv3d.bias

        self.img_corners = img_corners
        print(
            f"TT_OFT:: img_corners shape: {self.img_corners.shape}, dtype: {self.img_corners.dtype} (memory_config: {self.img_corners.memory_config()})"
        )

        self.scale = scale

    def forward(self, device, features, calib, grid):
        # grid = ttnn.unsqueeze(grid, 1)
        # print(f"grid shape: {grid.shape}, dtype: {grid.dtype}")
        # self.y_corners = ttnn.reshape(self.y_corners, (-1, 1, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
        # print(f"y_corners shape: {self.y_corners.shape}, dtype: {self.y_corners.dtype}")

        # corners = grid + self.y_corners
        # print(f"corners shape: {corners.shape}, dtype: {corners.dtype}")

        # img_corners = perspective(ttnn.reshape(calib, (-1, 1, 1, 1, 3, 4)), corners, device)

        # print(f"TTNN: img_corners shape: {img_corners.shape}, dtype: {img_corners.dtype}")

        ### this is moved to torch implementation under init for TT_OFT
        print(f"TTNN: features shape: {features.shape}, dtype: {features.dtype}")
        img_height, img_width = features.shape[2], features.shape[3]
        print(f"TTNN: img_height: {img_height}, img_width: {img_width}")
        img_size = ttnn.Tensor(
            [img_width, img_height], [1, 1, 1, 2], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        print(f"TTNN: img_size shape: {img_size.shape}, dtype: {img_size.dtype}")
        img_size = ttnn.reshape(img_size, (1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"TTNN: img_size after reshape shape: {img_size.shape}, dtype: {img_size.dtype}")
        img_size = img_size[0]
        print(f"TTNN: img_size after indexing shape: {img_size.shape}, dtype: {img_size.dtype}")
        img_size = ttnn.div(img_size, self.scale, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(
            f"TTNN: img_size after scaling shape: {img_size.shape}, dtype: {img_size.dtype}, memory_config: {img_size.memory_config()}"
        )
        norm_corners = ttnn.multiply(self.img_corners, 2.0, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        print(
            f"TTNN: norm_corners after multiply shape: {norm_corners.shape}, dtype: {norm_corners.dtype}, memory_config: {norm_corners.memory_config()}"
        )
        norm_corners = ttnn.div(norm_corners, img_size)
        norm_corners = ttnn.subtract(norm_corners, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        norm_corners = ttnn.clamp(norm_corners, -1, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(
            f"TTNN: norm_corners shape: {norm_corners.shape}, dtype: {norm_corners.dtype}, memory_config: {norm_corners.memory_config()}"
        )
        ttnn.deallocate(img_size)
        # ttnn.deallocate(self.img_corners)

        bbox_corners = ttnn.concat(
            [
                ttnn.minimum(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
                ttnn.maximum(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
            ],
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        batch, _, depth, width, _ = bbox_corners.shape
        print(f"bbox_corners shape: {bbox_corners.shape}, dtype: {bbox_corners.dtype}")
        ttnn.deallocate(norm_corners)
        bbox_corners = ttnn.reshape(
            bbox_corners,
            (
                bbox_corners.shape[0],
                bbox_corners.shape[1],
                bbox_corners.shape[2] * bbox_corners.shape[3],
                bbox_corners.shape[4],
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        area = ttnn.subtract(
            bbox_corners[..., 2:], bbox_corners[..., :2], memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
        )
        print(f"TTNN: area shape: {area.shape}, dtype: {area.dtype}, memory_config: {area.memory_config()}")
        coef = img_height * img_width * 0.25
        area = ttnn.prod(area, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"TTNN: area after prod shape: {area.shape}, dtype: {area.dtype}, memory_config: {area.memory_config()}")
        area = ttnn.multiply(area, coef, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        print(f"TTNN: area shape: {area.shape}, dtype: {area.dtype}, memory_config: {area.memory_config()}")
        area = ttnn.add(area, EPSILON, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        area = ttnn.unsqueeze(area, -1)
        print(
            f"TTNN: area after unsqueeze shape: {area.shape}, dtype: {area.dtype}, memory_config: {area.memory_config()}"
        )

        visible = ttnn.gt(area, EPSILON, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        print(f"TTNN: visible shape: {visible.shape}, dtype: {visible.dtype}, memory_config: {visible.memory_config()}")

        return bbox_corners
