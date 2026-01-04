# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json

import torch

from models.demos.unet_3d.demo.utils import configure_logging

logger = configure_logging()


def get_default_device_name() -> str:
    """Get the default device name based on CUDA availability.

    Returns:
        The default device name ("cuda" if available, otherwise "cpu").
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config() -> dict:
    with open("/root/bounty/tt-metal/models/demos/unet_3d/configs/test_confocal_boundary.json", "r") as f:
        config = json.load(f)

    if "output_dataset" not in config:
        config["output_dataset"] = "predictions"

    if "save_segmentation" not in config:
        config["save_segmentation"] = False

    # assert that dataset exists and raw_internal_path exist inside dataset
    assert "dataset" in config, "dataset configuration must be provided"
    assert "raw_internal_path" in config["dataset"], "raw_internal_path must be provided in dataset configuration"
    assert "label_internal_path" in config["dataset"], "label_internal_path must be provided in dataset configuration"

    logger.info("Configuration loaded successfully")
    return config
