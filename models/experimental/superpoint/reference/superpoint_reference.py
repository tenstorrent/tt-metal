# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import SuperPointForKeypointDetection, AutoImageProcessor


MODEL_ID = "magic-leap-community/superpoint"


def load_reference_model():
    model = SuperPointForKeypointDetection.from_pretrained(MODEL_ID)
    model.eval()
    return model


def load_image_processor():
    return AutoImageProcessor.from_pretrained(MODEL_ID)


def get_dummy_input(batch_size: int = 1, height: int = 480, width: int = 640):
    torch.manual_seed(0)
    return torch.rand(batch_size, 3, height, width)
