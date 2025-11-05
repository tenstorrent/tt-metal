# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib import transforms
from . import utils
import torch


def draw_bbox2d(objects, color="k", ax=None):
    limits = ax.axis()

    for obj in objects:
        x, _, z = obj.position
        l, _, w = obj.dimensions

        # Setup transform
        t = transforms.Affine2D().rotate(obj.angle + math.pi / 2)
        t = t.translate(x, z) + ax.transData

        # Draw 2D object bounding box
        rect = Rectangle((-w / 2, -l / 2), w, l, edgecolor=color, transform=t, fill=False)
        ax.add_patch(rect)

        # Draw dot indicating object center
        center = Circle((x, z), 0.25, facecolor="k")
        ax.add_patch(center)

    ax.axis(limits)
    return ax


def draw_bbox3d(obj, calib, ax, color="b"):
    # Get corners of 3D bounding box
    corners = utils.bbox_corners(obj)

    # Project into image coordinates
    img_corners = utils.perspective(calib.cpu(), corners, dtype=torch.float32).numpy()

    # Draw polygons
    # Front face
    ax.add_patch(Polygon(img_corners[[1, 3, 7, 5]], ec=color, fill=False))
    # Back face
    ax.add_patch(Polygon(img_corners[[0, 2, 6, 4]], ec=color, fill=False))
    ax.add_line(Line2D(*img_corners[[0, 1]].T, c=color))  # Lower left
    ax.add_line(Line2D(*img_corners[[2, 3]].T, c=color))  # Lower right
    ax.add_line(Line2D(*img_corners[[4, 5]].T, c=color))  # Upper left
    ax.add_line(Line2D(*img_corners[[6, 7]].T, c=color))  # Upper right


def visualize_objects(image, calib, objects, cmap="tab20", ax=None):
    # Create a figure if it doesn't already exist
    if ax is None:
        fig, ax = plt.subplots()
    ax.clear()

    # Visualize image
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    extents = ax.axis()

    # Visualize objects
    cmap = cm.get_cmap(cmap, len(objects))
    for i, obj in enumerate(objects):
        draw_bbox3d(obj, calib, ax, cmap(i))

    # Format axis
    ax.axis(extents)
    ax.axis(False)
    ax.grid(False)
    return ax
