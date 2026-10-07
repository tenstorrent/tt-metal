"""t222: frame 72 of seed 0 (yuv420p 1920x1088) from the 2-D arm and the reference, side by side, as a jpg."""

import sys

import numpy as np
from PIL import Image

W, H, FR = 1920, 1088, 1920 * 1088 * 3 // 2


def rgb(path, f=72):
    y = np.fromfile(path, np.uint8, FR, offset=f * FR)
    Y = y[: W * H].reshape(H, W)
    U = y[W * H : W * H * 5 // 4].reshape(H // 2, W // 2).repeat(2, 0).repeat(2, 1)
    V = y[W * H * 5 // 4 :].reshape(H // 2, W // 2).repeat(2, 0).repeat(2, 1)
    img = Image.fromarray(np.stack([Y, U, V], -1), "YCbCr").convert("RGB")
    return img


a, b = rgb(sys.argv[1]), rgb(sys.argv[2])
out = Image.new("RGB", (W, H * 2))
out.paste(a, (0, 0))
out.paste(b, (0, H))
out.resize((W // 2, H)).save(sys.argv[3], quality=90)
