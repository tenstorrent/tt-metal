# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import pytest
import matplotlib.pyplot as plt
from loguru import logger


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


def perspective(matrix, vector, dtype):
    """
    Applies perspective projection to a vector using projection matrix
    """
    # Make sure both inputs are the same dtype
    if matrix.dtype != dtype:
        matrix = matrix.to(dtype)
    if vector.dtype != dtype:
        vector = vector.to(dtype)

    vector = vector.unsqueeze(-1)
    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]


def rotate(vector, angle):
    """
    Rotate a vector around the y-axis
    """
    sinA, cosA = torch.sin(angle), torch.cos(angle)
    xvals = cosA * vector[..., 0] + sinA * vector[..., 2]
    yvals = vector[..., 1]
    zvals = -sinA * vector[..., 0] + cosA * vector[..., 2]
    return torch.stack([xvals, yvals, zvals], dim=-1)


def bbox_corners(obj):
    """
    Return the 2D
    """

    # Get corners of bounding box in object space
    offsets = torch.tensor(
        [
            [-0.5, 0.0, -0.5],  # Back-left lower
            [0.5, 0.0, -0.5],  # Front-left lower
            [-0.5, 0.0, 0.5],  # Back-right lower
            [0.5, 0.0, 0.5],  # Front-right lower
            [-0.5, -1.0, -0.5],  # Back-left upper
            [0.5, -1.0, -0.5],  # Front-left upper
            [-0.5, -1.0, 0.5],  # Back-right upper
            [0.5, -1.0, 0.5],  # Front-right upper
        ]
    )
    corners = offsets * torch.tensor(obj.dimensions)
    # corners = corners[:, [2, 0, 1]]

    # Apply y-axis rotation
    corners = rotate(corners, torch.tensor(obj.angle))

    # Apply translation
    corners = corners + torch.tensor(obj.position)
    return corners


def get_abs_and_relative_error(tensor_a, tensor_b):
    abs_error = torch.abs(tensor_a - tensor_b).mean().item()
    # relative_error = (abs_error / (torch.abs(tensor_a) + 1e-8)).mean().item()  # Avoid division by zero

    # Create relative error, using NaN where tensor_a is zero
    rel_err = torch.where(tensor_a != 0, torch.abs(tensor_a - tensor_b) / torch.abs(tensor_a), float("nan"))
    # Compute mean ignoring NaN values
    relative_error = torch.nanmean(rel_err).item() if rel_err.numel() > 0 else float("nan")
    return abs_error, relative_error


def check_monotonic_increase(tensor, mode="first"):
    """
    Check if tensor values are monotonically increasing along H and W dimensions

    Args:
        tensor: 4D tensor in either NCHW (mode="first") or NHWC (mode="last") format
        mode: "first" for channel-first (NCHW) or "last" for channel-last (NHWC)

    Returns:
        bool: True if monotonically increasing along both H and W
    """
    assert tensor.ndim == 4, "Input tensor must be 4D"

    if mode == "first":  # NCHW
        n, c, h, w = tensor.shape
        # Check along H axis (for each position in W)
        for n_idx in range(n):
            for c_idx in range(c):
                for w_idx in range(w):
                    values = tensor[n_idx, c_idx, :, w_idx]
                    diffs = values[1:] - values[:-1]
                    assert torch.all(
                        diffs >= 0
                    ), f"Not monotonically increasing along H axis at n={n_idx}, c={c_idx}, w={w_idx}"

        # Check along W axis (for each position in H)
        for n_idx in range(n):
            for c_idx in range(c):
                for h_idx in range(h):
                    values = tensor[n_idx, c_idx, h_idx, :]
                    diffs = values[1:] - values[:-1]
                    assert torch.all(
                        diffs >= 0
                    ), f"Not monotonically increasing along W axis at n={n_idx}, c={c_idx}, h={h_idx}"

    elif mode == "last":  # NHWC
        n, h, w, c = tensor.shape
        # Check along H axis (for each position in W)
        for n_idx in range(n):
            for c_idx in range(c):
                for w_idx in range(w):
                    values = tensor[n_idx, :, w_idx, c_idx]
                    diffs = values[1:] - values[:-1]
                    assert torch.all(
                        diffs >= 0
                    ), f"Not monotonically increasing along H axis at n={n_idx}, c={c_idx}, w={w_idx}"

        # Check along W axis (for each position in H)
        for n_idx in range(n):
            for c_idx in range(c):
                for h_idx in range(h):
                    values = tensor[n_idx, h_idx, :, c_idx]
                    diffs = values[1:] - values[:-1]
                    assert torch.all(
                        diffs >= 0
                    ), f"Not monotonically increasing along W axis at n={n_idx}, c={c_idx}, h={h_idx}"
    else:
        raise ValueError("Mode must be either 'first' (NCHW) or 'last' (NHWC)")

    return True


def test_monotonically_growing_integral_image():
    """Test that checks if values in an integral image are monotonically increasing along height and width dimensions"""

    # Test with channel-first format (NCHW)
    n, c, h, w = 2, 3, 5, 4
    # Create a monotonically increasing tensor
    x_nchw = torch.cumsum(torch.ones(n, c, h, w), dim=2)  # Cumsum along H
    x_nchw = torch.cumsum(x_nchw, dim=3)  # Cumsum along W
    assert check_monotonic_increase(x_nchw, mode="first")

    # Test with channel-last format (NHWC)
    n, h, w, c = 2, 5, 4, 3
    # Create a monotonically increasing tensor
    x_nhwc = torch.cumsum(torch.ones(n, h, w, c), dim=1)  # Cumsum along H
    x_nhwc = torch.cumsum(x_nhwc, dim=2)  # Cumsum along W
    assert check_monotonic_increase(x_nhwc, mode="last")

    # Test failure case (channel-first)
    x_nchw_bad = x_nchw.clone()
    x_nchw_bad[0, 0, 2, 2] = 0  # Introduce a violation
    with pytest.raises(AssertionError):
        check_monotonic_increase(x_nchw_bad, mode="first")

    # Test failure case (channel-last)
    x_nhwc_bad = x_nhwc.clone()
    x_nhwc_bad[0, 2, 2, 0] = 0  # Introduce a violation
    with pytest.raises(AssertionError):
        check_monotonic_increase(x_nhwc_bad, mode="last")


def vis_score(score, grid, cmap="cividis", ax=None, title=None):
    score = score.cpu().float().detach().numpy()
    grid = grid.cpu().float().detach().numpy()

    # Create a new axis if one is not provided
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    # Plot scores
    ax.clear()
    ax.pcolormesh(grid[..., 0], grid[..., 2], score, cmap=cmap, vmin=0, vmax=1)
    ax.set_aspect("equal")

    # Set title if provided
    if title is not None:
        ax.set_title(title)

    # Format axes
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")

    return ax


def visualize_score(ref_scores, scores, grid):
    # Visualize score
    fig_score = plt.figure(num="score", figsize=(8, 6))
    fig_score.clear()

    vis_score(ref_scores[0, 0], grid[0], ax=plt.subplot(121), title="Torch")
    vis_score(scores[0, 0], grid[0], ax=plt.subplot(122), title="TTNN")

    return fig_score


def print_object_comparison(ref_objects, tt_objects):
    """
    Print comparison between reference and TT objects.

    Args:
        ref_objects: List of reference objects
        tt_objects: List of TT objects
    """
    logger.info(f"Reference objects count: {len(ref_objects)}")
    logger.info(f"TTNN objects count: {len(tt_objects)}")

    logger.info("=== Reference Objects ===")
    for i, obj in enumerate(ref_objects):
        logger.info(f"Ref Object {i}: {obj}")

    logger.info("=== TTNN Objects ===")
    for i, obj in enumerate(tt_objects):
        logger.info(f"TT Object {i}: {obj}")

    # Compare object counts and properties if they match
    if len(ref_objects) == len(tt_objects):
        logger.info("=== Object Comparison ===")
        for i, (ref_obj, tt_obj) in enumerate(zip(ref_objects, tt_objects)):
            logger.info(f"Object {i} comparison:")
            logger.info(f"  Classname: {ref_obj.classname} vs {tt_obj.classname}")
            logger.info(f"  Position: {ref_obj.position} vs {tt_obj.position}")
            logger.info(f"  Dimensions: {ref_obj.dimensions} vs {tt_obj.dimensions}")
            logger.info(f"  Angle: {ref_obj.angle} vs {tt_obj.angle}")
            logger.info(f"  Score: {ref_obj.score} vs {tt_obj.score}")
    else:
        logger.warning(f"Object count mismatch: {len(ref_objects)} ref vs {len(tt_objects)} ttnn")
