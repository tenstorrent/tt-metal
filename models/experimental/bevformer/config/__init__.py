# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Configuration package for BEVFormer models."""

# Import legacy config classes for backward compatibility
import sys
import os
from importlib.util import spec_from_file_location, module_from_spec

# Path to the original config.py file
config_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.py")

try:
    # Import the legacy config.py module directly to avoid circular imports
    spec = spec_from_file_location("legacy_config", config_file_path)
    legacy_config = module_from_spec(spec)
    spec.loader.exec_module(legacy_config)

    AttentionConfig = legacy_config.AttentionConfig
    DeformableAttentionConfig = legacy_config.DeformableAttentionConfig
    SpatialCrossAttentionConfig = legacy_config.SpatialCrossAttentionConfig
    TemporalSelfAttentionConfig = legacy_config.TemporalSelfAttentionConfig

except (ImportError, FileNotFoundError) as e:
    # Fallback: define basic classes if import fails
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class AttentionConfig:
        embed_dims: int = 256
        num_heads: int = 8  # Fixed: Match new system defaults
        num_levels: int = 4
        num_points: int = 4

    @dataclass
    class DeformableAttentionConfig(AttentionConfig):
        query_embed_dims: int = None

    @dataclass
    class SpatialCrossAttentionConfig(AttentionConfig):
        num_cams: int = 6
        pc_range: List[float] = None

    @dataclass
    class TemporalSelfAttentionConfig(AttentionConfig):
        num_frames: int = 2
        memory_len: int = 256


# Import new encoder config classes
from .encoder_config import *

# Define what gets exported
__all__ = [
    # Legacy configs for backward compatibility
    "AttentionConfig",
    "DeformableAttentionConfig",
    "SpatialCrossAttentionConfig",
    "TemporalSelfAttentionConfig",
    # New encoder config classes (imported from encoder_config.__all__)
    "DatasetConfig",
    "BEVFormerDataConfig",
    "get_dataset_config",
    "list_available_datasets",
    "DEFAULT_CONFIG",
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
