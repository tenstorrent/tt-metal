# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = (
    (255, 0, 0, 128),
    (0, 255, 0, 128),
    (0, 0, 255, 128),
    (0, 255, 255, 128),
    (255, 0, 255, 128),
    (255, 255, 0, 128),
)

MEANS = (104, 117, 123)

# SSD512 CONFIGS
voc = {
    "SSD512": {
        "num_classes": 21,
        "lr_steps": (100, 200, 300),
        "max_iter": 120000,
        "feature_maps": [64, 32, 16, 8, 4, 2, 1],
        "min_dim": 512,
        "steps": [8, 16, 32, 64, 100, 300, 512],
        "min_sizes": [30, 60, 111, 162, 213, 264, 315],
        "max_sizes": [60, 111, 162, 213, 264, 315, 366],
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
        "variance": [0.1, 0.2],
        "clip": True,
        "name": "VOC",
    },
}
