# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

GRID_RES = 0.5
GRID_SIZE = (80.0, 80.0)
GRID_HEIGHT = 4.0
Y_OFFSET = 1.74

H_PADDED = 384
W_PADDED = 1280

NMS_THRESH = 0.2

import os
import torch
from loguru import logger


def load_checkpoint(checkpoints_path, ref_model):
    if checkpoints_path is not None and os.path.isfile(checkpoints_path):
        logger.info(f"Loading model weights from {checkpoints_path}")
        checkpoint = torch.load(checkpoints_path, map_location="cpu")

        # Load state dict as is
        ref_model.load_state_dict(checkpoint["model"], strict=True)

        # Ensure all weights are converted to the specified dtype after loading
        ref_model.to(ref_model.dtype)
        logger.info(f"Converted all model weights to {ref_model.dtype}")
    else:
        logger.error(f"Checkpoint path {checkpoints_path} does not exist, using random weights")
        assert False, f"Checkpoint path {checkpoints_path} does not exist"

    return ref_model
