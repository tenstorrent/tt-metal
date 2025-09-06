import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


def make_grid(grid_size, grid_offset, grid_res):
    """
    Constructs an array representing the corners of an orthographic grid
    """
    depth, width = grid_size
    xoff, yoff, zoff = grid_offset

    xcoords = torch.arange(0.0, width, grid_res) + xoff
    zcoords = torch.arange(0.0, depth, grid_res) + zoff

    zz, xx = torch.meshgrid(zcoords, xcoords)
    return torch.stack([xx, torch.full_like(xx, yoff), zz], dim=-1)


def load_image(image_path, pad_hw=(384, 1280)):
    image = Image.open(image_path)
    image = to_tensor(image)
    padded_image = torch.zeros((3, pad_hw[0], pad_hw[1]))
    _, h, w = image.shape
    padded_image[:, :h, :w] = image
    return padded_image


def load_calib(filename):
    with open(filename) as f:
        for line in f:
            data = line.split(" ")
            if data[0] == "P2:":
                calib = torch.tensor([float(x) for x in data[1:13]])
                return calib.view(3, 4)

    raise Exception("Could not find entry for P2 in calib file {}".format(filename))
