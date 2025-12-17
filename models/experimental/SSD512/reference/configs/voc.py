# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""VOC Dataset configuration for SSD512."""

# VOC class names (20 classes + background)
VOC_CLASSES = (
    "__background__",  # always index 0
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

# SSD512 VOC configuration
voc = {
    "SSD512": {
        "num_classes": 21,  # VOC has 20 classes + 1 background
        "lr_steps": (80000, 100000, 120000),
        "max_iter": 120000,
        "feature_maps": [64, 32, 16, 8, 4, 2, 1],
        "min_dim": 512,
        "steps": [8, 16, 32, 64, 128, 256, 512],
        "min_sizes": [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],
        "max_sizes": [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        "variance": [0.1, 0.2],
        "clip": True,
        "name": "VOC",
    },
    "SSD300": {
        "num_classes": 21,
        "lr_steps": (80000, 100000, 120000),
        "max_iter": 120000,
        "feature_maps": [38, 19, 10, 5, 3, 1],
        "min_dim": 300,
        "steps": [8, 16, 32, 64, 100, 300],
        "min_sizes": [30, 60, 111, 162, 213, 264],
        "max_sizes": [60, 111, 162, 213, 264, 315],
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        "variance": [0.1, 0.2],
        "clip": True,
        "name": "VOC",
    },
}
