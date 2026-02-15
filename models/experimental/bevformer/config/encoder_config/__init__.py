# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
]
