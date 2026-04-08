# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""COCO Dataset configuration for SSD512."""

# COCO configuration (used for variance in loss computation)
coco = {
    "num_classes": 81,  # COCO has 80 classes + 1 background
    "lr_steps": (280000, 360000, 400000),
    "max_iter": 400000,
    "feature_maps": [64, 32, 16, 8, 4, 2, 1],
    "min_dim": 512,
    "steps": [8, 16, 32, 64, 128, 256, 512],
    "min_sizes": [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],
    "max_sizes": [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    "variance": [0.1, 0.2],
    "clip": True,
    "name": "COCO",
}
