# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Encoder configuration for BEVFormer models."""

from .data_config import DatasetConfig, BEVFormerDataConfig, get_dataset_config, list_available_datasets, DEFAULT_CONFIG

from .model_config import (
    BEVFormerModelConfig,
    EncoderConfig,
    get_model_config,
    get_preset_config,
    list_preset_configs,
    list_model_variants,
    create_custom_config,
    MODEL_VARIANTS,
    PRESET_CONFIGS,
)

from .camera_rig import (
    CameraSpec,
    ring_camera_rig,
    build_lidar2img,
    camera_rig_for_dataset,
    lidar2img_for_dataset,
    img_metas_for_dataset,
)

__all__ = [
    # Data config
    "DatasetConfig",
    "BEVFormerDataConfig",
    "get_dataset_config",
    "list_available_datasets",
    "DEFAULT_CONFIG",
    # Model config
    "BEVFormerModelConfig",
    "EncoderConfig",
    "get_model_config",
    "get_preset_config",
    "list_preset_configs",
    "list_model_variants",
    "create_custom_config",
    "MODEL_VARIANTS",
    "PRESET_CONFIGS",
    # Camera rig
    "CameraSpec",
    "ring_camera_rig",
    "build_lidar2img",
    "camera_rig_for_dataset",
    "lidar2img_for_dataset",
    "img_metas_for_dataset",
]
