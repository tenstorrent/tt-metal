# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Constants of BEVFormer's detection head and box coder shared by the reference and the TTNN port.

Values are those of the BEVFormer tiny and base configs, which share the head.
"""

from models.experimental.bevformer.config.encoder_config.data_config import get_dataset_config

NUM_CLASSES = 10

# (x_min, y_min, z_min, x_max, y_max, z_max) in metres: the box centres the head can emit.
# The nuScenes range the encoder's BEV grid covers, so the head's metres match that grid.
PC_RANGE = tuple(get_dataset_config("nuscenes_v1.0_full_1600x900").pc_range)
# Box centres outside this range are dropped by the coder.
POST_CENTER_RANGE = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)
# Boxes the coder keeps per sample, by score.
MAX_NUM = 300

# The coder's decoded box: (cx, cy, cz, w, l, h, yaw, vx, vy), cz at the box's gravity centre.
BOX_CENTRE = slice(0, 3)
BOX_SIZE = slice(3, 6)
BOX_YAW = slice(6, 7)
BOX_VELOCITY = slice(7, 9)
