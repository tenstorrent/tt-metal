import ttnn
import torch
import torch.nn.functional as F

EPSILON = 1e-6


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


def perspective(matrix, vector, device):
    vector = ttnn.unsqueeze(vector, -1)
    matrix = ttnn.to_torch(matrix).float()
    vector = ttnn.to_torch(vector).float()
    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    homogenous = ttnn.from_torch(homogenous, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # OOM - L1
    homogenous = ttnn.squeeze(homogenous, -1)
    homogenous = ttnn.to_memory_config(homogenous, memory_config=ttnn.L1_MEMORY_CONFIG)
    # return homogenous
    homogenous = ttnn.div(homogenous[..., :-1], homogenous[..., -1:], memory_config=ttnn.L1_MEMORY_CONFIG)

    return homogenous


class OFT:
    def __init__(self, device, parameters, y_corners, scale=1):
        self.y_corners = y_corners

        self.linear_weight = parameters.conv3d.weight
        self.linear_bias = parameters.conv3d.bias

        self.scale = scale

    def __call__(self, device, features, calib, grid):
        grid = ttnn.unsqueeze(grid, 1)
        self.y_corners = ttnn.reshape(self.y_corners, (-1, 1, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
        grid = ttnn.to_torch(grid).float()
        self.y_corners = ttnn.to_torch(self.y_corners).float()
        corners = grid + self.y_corners

        self.y_corners = ttnn.from_torch(
            self.y_corners,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        corners = ttnn.from_torch(corners, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # OOM - L1

        img_corners = perspective(ttnn.reshape(calib, (-1, 1, 1, 1, 3, 4)), corners, device)
        # ttnn.deallocate(calib)
        img_height, img_width = features.shape[2], features.shape[3]

        img_size = ttnn.Tensor(
            [img_width, img_height], [1, 1, 1, 2], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )

        img_size = ttnn.reshape(img_size, (1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        img_size = img_size[0]
        img_size = ttnn.div(img_size, self.scale, memory_config=ttnn.L1_MEMORY_CONFIG)
        norm_corners = ttnn.multiply(img_corners, 2, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        norm_corners = ttnn.div(norm_corners, img_size, memory_config=ttnn.L1_MEMORY_CONFIG)
        norm_corners = ttnn.subtract(norm_corners, 1, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        norm_corners = ttnn.clamp(norm_corners, -1, 1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(img_size)
        ttnn.deallocate(img_corners)
        bbox_corners = ttnn.concat(
            [
                ttnn.minimum(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
                ttnn.maximum(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
            ],
            dim=-1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        batch, _, depth, width, _ = bbox_corners.shape
        ttnn.deallocate(norm_corners)
        bbox_corners = ttnn.reshape(
            bbox_corners,
            (
                bbox_corners.shape[0],
                bbox_corners.shape[1],
                bbox_corners.shape[2] * bbox_corners.shape[3],
                bbox_corners.shape[4],
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        area = ttnn.subtract(
            bbox_corners[..., 2:], bbox_corners[..., :2], memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
        )
        area = ttnn.prod(area, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        area = ttnn.multiply(area, img_height, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        area = ttnn.multiply(area, img_width, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        area = ttnn.multiply(area, 0.25, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        area = ttnn.add(area, EPSILON, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        area = ttnn.unsqueeze(area, 1)

        visible = ttnn.gt(area, EPSILON, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

        bbox_corners = ttnn.to_torch(bbox_corners).float()
        features = ttnn.to_memory_config(features, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        integral_img = integral_image(features)
        integral_img = ttnn.to_torch(integral_img).float()
        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]])
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]])
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]])
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]])

        top_left = ttnn.from_torch(top_left, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        btm_right = ttnn.from_torch(btm_right, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        top_right = ttnn.from_torch(top_right, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        btm_left = ttnn.from_torch(btm_left, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

        vox_feats = ttnn.add(top_left, btm_right, dtype=ttnn.bfloat8_b)
        vox_feats = ttnn.subtract(vox_feats, top_right, dtype=ttnn.bfloat8_b)
        vox_feats = ttnn.subtract(vox_feats, btm_left, dtype=ttnn.bfloat8_b)
        area = ttnn.to_memory_config(area, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        visible = ttnn.to_memory_config(visible, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        vox_feats = ttnn.div(vox_feats, area, dtype=ttnn.bfloat8_b)
        vox_feats = ttnn.multiply(vox_feats, visible, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(area)
        ttnn.deallocate(visible)
        vox_feats = ttnn.permute(vox_feats, (0, 3, 1, 2))
        vox_feats = ttnn.reshape(
            vox_feats,
            (vox_feats.shape[0], vox_feats.shape[1], vox_feats.shape[2] * vox_feats.shape[3]),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # grid_size = (8, 8)
        # shard_grid = ttnn.CoreRangeSet(
        #     {
        #         ttnn.CoreRange(
        #             ttnn.CoreCoord(0, 0),
        #             ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
        #         )
        #     }
        # )
        # shard_spec = ttnn.ShardSpec(shard_grid, [791, 768], ttnn.ShardOrientation.ROW_MAJOR)
        # height_sharded_mem_config = ttnn.MemoryConfig(
        #     ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec
        # )
        # vox_feats = ttnn.to_layout(vox_feats, layout=ttnn.ROW_MAJOR_LAYOUT)
        # vox_feats = ttnn.to_memory_config(vox_feats, height_sharded_mem_config)
        # p(self.linear_weight)
        # p(self.linear_bias)
        # ortho_feats = ttnn.linear(vox_feats, self.linear_weight, bias=self.linear_bias, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        ortho_feats = ttnn.linear(
            vox_feats,
            self.linear_weight,
            bias=self.linear_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        ttnn.deallocate(vox_feats)

        ortho_feats = ttnn.reshape(ortho_feats, (batch, depth, width, -1), memory_config=ttnn.L1_MEMORY_CONFIG)
        ortho_feats = ttnn.permute(ortho_feats, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        ortho_feats = ttnn.relu(ortho_feats, memory_config=ttnn.L1_MEMORY_CONFIG)

        return ortho_feats


def integral_image(features):
    return ttnn.cumsum(ttnn.cumsum(features, dim=-1), dim=-2)
