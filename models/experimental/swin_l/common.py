# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common config and constants for the Swin-L backbone.
Standalone – no dependency on any downstream detection model.
"""

# Swin-L architecture parameters
SWIN_L_EMBED_DIM = 192
SWIN_L_DEPTHS = [2, 2, 18, 2]
SWIN_L_NUM_HEADS = [6, 12, 24, 48]
SWIN_L_WINDOW_SIZE = 12
SWIN_L_STAGE_CHANNELS = [192, 384, 768, 1536]  # C2, C3, C4, C5
SWIN_L_MLP_RATIO = 4.0

# Default test input size (DINO-5scale uses 800x1333, but any size works)
DEFAULT_INPUT_H = 800
DEFAULT_INPUT_W = 1333
