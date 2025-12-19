# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
from loguru import logger
from PIL import Image

from models.experimental.smolvla.common import get_model_path


def compute_pcc(x: np.ndarray, y: np.ndarray) -> float:
    x_flat = torch.as_tensor(x).flatten().float()
    y_flat = torch.as_tensor(y).flatten().float()
    x_centered = x_flat - x_flat.mean()
    y_centered = y_flat - y_flat.mean()
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
    return (numerator / denominator).item() if denominator != 0 else 0.0


def create_test_image(size: int = 512, seed: int = 42) -> Image.Image:
    np.random.seed(seed)
    img_array = np.zeros((size, size, 3), dtype=np.uint8)
    img_array[:, :, 0] = np.tile(np.linspace(50, 200, size), (size, 1)).astype(np.uint8)
    img_array[:, :, 1] = np.tile(np.linspace(100, 150, size), (size, 1)).T.astype(np.uint8)
    img_array[:, :, 2] = 128
    return Image.fromarray(img_array)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_smolvla_pcc(device, reset_seeds, model_location_generator):
    from models.experimental.smolvla.tt.smol_vla import SmolVLAForActionPrediction

    model_weights_path = get_model_path(model_location_generator)

    model_cpu = SmolVLAForActionPrediction.from_pretrained(model_weights_path, ttnn_device=None, local_files_only=True)
    model_cpu.processor.image_processor.do_image_splitting = False
    model_cpu.eval()

    model_tt = SmolVLAForActionPrediction.from_pretrained(model_weights_path, ttnn_device=device, local_files_only=True)
    model_tt.processor.image_processor.do_image_splitting = False
    model_tt.eval()

    img = create_test_image()
    instruction = "pick up the red block"

    torch.manual_seed(42)
    np.random.seed(42)
    actions_cpu = model_cpu.predict_action(images=[img], instruction=instruction, num_inference_steps=1, action_dim=6)

    torch.manual_seed(42)
    np.random.seed(42)
    actions_tt = model_tt.predict_action(images=[img], instruction=instruction, num_inference_steps=1, action_dim=6)

    actions_cpu_np = np.asarray(actions_cpu)
    actions_tt_np = np.asarray(actions_tt)
    pcc = compute_pcc(actions_cpu_np, actions_tt_np)

    logger.info(f"PCC: {pcc:.4f}")

    PCC_THRESHOLD = 0.90
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.4f} below threshold {PCC_THRESHOLD}"
