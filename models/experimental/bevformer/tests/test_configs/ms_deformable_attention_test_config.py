# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test configuration parameters for multi-scale deformable attention tests.
This file contains all the parametrize values for the pytest test suite.
"""

# Model parameter combinations for testing
MODEL_PARAMS = [
    # (batch_size, num_query, embed_dims, num_levels, num_points_per_anchor, num_anchors, num_heads)
    (1, 900, 256, 4, 2, 4, 4),
    (1, 1200, 256, 4, 2, 4, 4),
    (1, 100 * 100, 256, 4, 2, 4, 4),
    (1, 200 * 200, 256, 4, 2, 4, 4),
]

# Spatial shape configurations for different datasets/scenarios
SPATIAL_SHAPES = [
    [[200, 113], [100, 57], [50, 29], [25, 15]],  # nuScenes, input size 1600x900
    [[160, 90], [80, 45], [40, 23], [20, 12]],  # nuScenes, input size 1280x720
    [[28, 28], [14, 14], [7, 7], [4, 4]],  # VAD specific, input size 224x224
    [[16, 16], [8, 8], [4, 4], [2, 2]],  # VAD specific, input size 128x128
    [[100, 75], [50, 38], [25, 19], [13, 10]],  # CARLA, input size 1280x960
    [[120, 80], [60, 40], [30, 20], [15, 10]],  # Waymo, input size 960x640
]

# Parameter names for easy reference
MODEL_PARAM_NAMES = "batch_size, num_query, embed_dims, num_levels, num_points_per_anchor, num_anchors, num_heads"
SPATIAL_SHAPES_PARAM_NAME = "spatial_shapes"
DEVICE_PARAMS_PARAM_NAME = "device_params"
SEED_PARAM_NAME = "seed"

# Default tolerance thresholds for check_with_tolerances
DEFAULT_THRESHOLDS = {
    "pcc_threshold": 0.999,
    "abs_error_threshold": 0.02,
    "rel_error_threshold": 0.2,
    "max_error_ratio": 0.15,
}
