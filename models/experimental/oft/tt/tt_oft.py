import ttnn
import torch

EPSILON = 1e-6


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


class OFT:
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
        print(f"TTNN: features shape: {features.shape}, dtype: {features.dtype}")
        img_height, img_width = features.shape[2], features.shape[3]
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
