import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


def make_grid(grid_size, grid_offset, grid_res, dtype=torch.float32):
    """
    Constructs an array representing the corners of an orthographic grid
    """
    depth, width = grid_size
    xoff, yoff, zoff = grid_offset

    xcoords = torch.arange(0.0, width, grid_res, dtype=dtype) + xoff
    zcoords = torch.arange(0.0, depth, grid_res, dtype=dtype) + zoff

    zz, xx = torch.meshgrid(zcoords, xcoords)
    return torch.stack([xx, torch.full_like(xx, yoff), zz], dim=-1)


def load_image(image_path, pad_hw=(384, 1280), dtype=torch.float32):
    image = Image.open(image_path)
    image = to_tensor(image)
    padded_image = torch.zeros((3, pad_hw[0], pad_hw[1]), dtype=dtype)
    _, h, w = image.shape
    padded_image[:, :h, :w] = image
    return padded_image


def load_calib(filename, dtype=torch.float32):
    with open(filename) as f:
        for line in f:
            data = line.split(" ")
            if data[0] == "P2:":
                calib = torch.tensor([float(x) for x in data[1:13]], dtype=dtype)
                return calib.view(3, 4)

    raise Exception("Could not find entry for P2 in calib file {}".format(filename))


def perspective(matrix, vector):
    """
    Applies perspective projection to a vector using projection matrix
    """
    # # Make sure both inputs are the same dtype
    # if matrix.dtype != vector.dtype:
    #     # Convert to the higher precision dtype
    #     if matrix.dtype.is_floating_point and vector.dtype.is_floating_point:
    #         dtype = torch.promote_types(matrix.dtype, vector.dtype)
    #         matrix = matrix.to(dtype)
    #         vector = vector.to(dtype)

    vector = vector.unsqueeze(-1)
    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]
