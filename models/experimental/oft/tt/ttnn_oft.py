import ttnn
import torch
import torch.nn.functional as F

EPSILON = 1e-6


def perspective(matrix, vector, device):
    vector = ttnn.unsqueeze(vector, -1)
    matrix = ttnn.to_torch(matrix).float()
    vector = ttnn.to_torch(vector).float()
    homogenous = torch.matmul(matrix[..., :-1], vector)
    homogenous = homogenous + matrix[..., -1:]
    homogenous = ttnn.from_torch(homogenous, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    homogenous = ttnn.squeeze(homogenous, -1)
    homogenous = ttnn.div(homogenous[..., :-1], homogenous[..., -1:])

    return homogenous


class OFT:
    def __init__(self, device, cell_size, grid_height, model, path, y_corners, scale=1):
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model

        self.y_corners = y_corners

        self.linear_weight = ttnn.from_torch(torch_model[path + ".conv3d.weight"])
        self.linear_bias = ttnn.from_torch(torch_model[path + ".conv3d.bias"])

        self.scale = scale

    def __call__(self, device, features, calib, grid):
        grid = ttnn.unsqueeze(grid, 1)
        self.y_corners = ttnn.reshape(self.y_corners, (-1, 1, 1, 3))
        grid = ttnn.to_torch(grid).float()
        self.y_corners = ttnn.to_torch(self.y_corners).float()
        corners = grid + self.y_corners

        corners = ttnn.from_torch(corners, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        img_corners = perspective(ttnn.reshape(calib, (-1, 1, 1, 1, 3, 4)), corners, device)

        return img_corners
        img_height, img_width = features.shape[2], features.shape[3]

        img_size = ttnn.Tensor(
            [float(img_width), float(img_height)],
            [1, 1, 1, 2],
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        img_size = ttnn.pad(img_size, ((0, 0), (0, 0), (0, 31), (0, 30)), 0)
        img_size = ttnn.to_layout(img_size, layout=ttnn.TILE_LAYOUT)
        img_size = img_size.to(device)
        img_size = ttnn.reshape(img_size, (1, 2))
        img_size = img_size[0]
        img_size = ttnn.div(img_size, self.scale)

        norm_corners = ttnn.multiply(img_corners, 2)
        norm_corners = ttnn.div(norm_corners, img_size)
        norm_corners = ttnn.subtract(norm_corners, 1)
        norm_corners = ttnn.clamp(norm_corners, -1, 1)

        return norm_corners

        bbox_corners = ttnn.concat(
            [
                ttnn.minimum(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
                ttnn.maximum(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
            ],
            dim=-1,
        )

        batch, _, depth, width, _ = bbox_corners.shape
        bbox_corners = ttnn.reshape(
            bbox_corners,
            (
                bbox_corners.shape[0],
                bbox_corners.shape[1],
                bbox_corners.shape[2] * bbox_corners.shape[3],
                bbox_corners.shape[4],
            ),
        )
        return bbox_corners
        area = ttnn.subtract(bbox_corners[..., 2:], bbox_corners[..., :2])
        area = ttnn.prod(area, dim=-1)
        area = ttnn.multiply(area, img_height)
        area = ttnn.multiply(area, img_width)
        area = ttnn.multiply(area, 0.25)
        area = ttnn.add(area, EPSILON)
        area = ttnn.unsqueeze(area, 1)
        return area
        visible = ttnn.gt(area, EPSILON)
        return visible
        features = ttnn.to_torch(features).float()
        bbox_corners = ttnn.to_torch(bbox_corners).float()

        integral_img = integral_image(features)
        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]])
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]])
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]])
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]])

        top_left = ttnn.from_torch(top_left, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        btm_right = ttnn.from_torch(btm_right, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        top_right = ttnn.from_torch(top_right, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        btm_left = ttnn.from_torch(btm_left, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        vox_feats = ttnn.add(top_left, btm_right)
        vox_feats = ttnn.subtract(vox_feats, top_right)
        vox_feats = ttnn.subtract(vox_feats, btm_left)
        vox_feats = ttnn.div(vox_feats, area)
        vox_feats = ttnn.multiply(vox_feats, visible)
        vox_feats = ttnn.permute(vox_feats, (0, 3, 1, 2))
        vox_feats = ttnn.reshape(
            vox_feats, (vox_feats.shape[0] * vox_feats.shape[1], vox_feats.shape[2], vox_feats.shape[3])
        )
        vox_feats = ttnn.reshape(vox_feats, (vox_feats.shape[0], vox_feats.shape[1] * vox_feats.shape[2]))
        return vox_feats
        ortho_feats = ttnn.linear(vox_feats, self.linear_weight, self.linear_bias)
        ortho_feats = ttnn.reshape(ortho_feats, (batch, depth, width, -1))
        ortho_feats = ttnn.permute(ortho_feats, (0, 3, 1, 2))
        ortho_feats = ttnn.relu(ortho_feats)

        return ortho_feats


def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)
