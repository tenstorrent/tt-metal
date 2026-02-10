# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Model configuration for BEVFormer encoder supporting different model variants.

This module provides model-specific parameters for different BEVFormer variants
(base, tiny, etc.) that work in conjunction with the data configurations defined
in data_config.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .data_config import DatasetConfig, get_dataset_config


def calculate_level_start_indices(spatial_shapes: List[List[int]]) -> List[int]:
    """
    Calculate start indices for each pyramid level from spatial shapes.

    This utility function can be used by tests and other components that need
    to compute level start indices from spatial shapes.

    Args:
        spatial_shapes: List of [width, height] for each pyramid level

    Returns:
        List of start indices for each level
    """
    level_start_index = [0]
    for spatial_shape in spatial_shapes[:-1]:
        level_start_index.append(level_start_index[-1] + spatial_shape[0] * spatial_shape[1])
    return level_start_index


@dataclass
class BEVFormerModelConfig:
    """
    Model configuration for BEVFormer encoder variants.

    These parameters control the transformer architecture and training behavior.
    They are designed to work with any dataset configuration from data_config.py.
    """

    # Model Variant
    variant: str = "base"  # "base", "tiny", "small", "large"

    # Transformer Architecture
    num_layers: int = 6
    embed_dims: int = 256
    num_heads: int = 8
    num_levels: int = 4
    num_points: int = 4
    num_points_in_pillar: int = 4
    feedforward_channels: int = 1024

    # Training Parameters
    dropout: float = 0.1
    batch_first: bool = True
    return_intermediate: bool = False

    # Attention Configuration
    use_temporal_self_attention: bool = True
    use_spatial_cross_attention: bool = True

    # Memory and Performance
    im2col_step: int = 64
    memory_len: int = 256
    num_frames: int = 2

    def validate_with_dataset(self, dataset_config: DatasetConfig) -> bool:
        """
        Validate that this model config is compatible with a dataset config.

        Args:
            dataset_config: Dataset configuration to validate against

        Returns:
            True if compatible, raises ValueError if not
        """
        # Check if embed_dims is divisible by num_heads
        if self.embed_dims % self.num_heads != 0:
            raise ValueError(f"embed_dims ({self.embed_dims}) must be divisible by " f"num_heads ({self.num_heads})")

        # Check if num_levels is valid for the dataset spatial_shapes
        if self.num_levels > len(dataset_config.spatial_shapes):
            raise ValueError(
                f"num_levels ({self.num_levels}) cannot be greater than available "
                f"spatial_shapes levels ({len(dataset_config.spatial_shapes)}). "
                f"Available levels: {len(dataset_config.spatial_shapes)}"
            )

        return True


# Model Variant Definitions
MODEL_VARIANTS = {
    "tiny": BEVFormerModelConfig(
        variant="tiny",
        num_layers=3,
        embed_dims=128,
        num_heads=4,
        num_levels=1,
        num_points=4,
        num_points_in_pillar=2,
        feedforward_channels=512,
        dropout=0.1,
        memory_len=128,
        im2col_step=32,
    ),
    "small": BEVFormerModelConfig(
        variant="small",
        num_layers=4,
        embed_dims=192,
        num_heads=6,
        num_levels=4,
        num_points=3,
        num_points_in_pillar=3,
        feedforward_channels=768,
        dropout=0.1,
        memory_len=192,
        im2col_step=48,
    ),
    "base": BEVFormerModelConfig(
        variant="base",
        num_layers=6,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_points_in_pillar=4,
        feedforward_channels=1024,
        dropout=0.1,
        memory_len=256,
        im2col_step=64,
    ),
    "large": BEVFormerModelConfig(
        variant="large",
        num_layers=8,
        embed_dims=384,
        num_heads=12,
        num_levels=4,
        num_points=6,
        num_points_in_pillar=6,
        feedforward_channels=1536,
        dropout=0.1,
        memory_len=384,
        im2col_step=96,
    ),
}


def get_model_config(variant: str = "base") -> BEVFormerModelConfig:
    """
    Get a model configuration for a specific variant.

    Args:
        variant: Model variant name ("tiny", "small", "base", "large")

    Returns:
        BEVFormerModelConfig for the specified variant

    Raises:
        ValueError: If variant is not supported
    """
    if variant not in MODEL_VARIANTS:
        available = list(MODEL_VARIANTS.keys())
        raise ValueError(f"Unsupported variant '{variant}'. Available variants: {available}")

    return MODEL_VARIANTS[variant]


@dataclass
class EncoderConfig:
    """
    Combined configuration for BEVFormer encoder including both model and data parameters.

    This class combines a dataset configuration with model parameters to provide
    a complete configuration for initializing a BEVFormer encoder.
    """

    dataset_name: str
    model_variant: str = "base"
    model_config: Optional[BEVFormerModelConfig] = field(default=None, init=False)
    dataset_config: Optional[DatasetConfig] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize model and dataset configurations and validate compatibility."""
        # Load model configuration
        self.model_config = get_model_config(self.model_variant)

        # Load dataset configuration
        self.dataset_config = get_dataset_config(self.dataset_name)
        if self.dataset_config is None:
            raise ValueError(f"Unknown dataset configuration: {self.dataset_name}")

        # Validate compatibility
        self.model_config.validate_with_dataset(self.dataset_config)

    def get_encoder_kwargs(self) -> Dict[str, Any]:
        """
        Get keyword arguments for BEVFormer encoder initialization.

        Returns:
            Dictionary of parameters for BEVFormerEncoder constructor
        """
        if self.dataset_config is None or self.model_config is None:
            raise ValueError("Configuration not loaded")

        return {
            # Dataset-specific parameters
            "pc_range": self.dataset_config.pc_range,
            "num_cams": self.dataset_config.num_cams,
            "z_cfg": self.dataset_config.z_cfg,
            "dataset": self.dataset_name.split("_")[0],  # Extract base dataset name
            # Model architecture parameters
            "num_layers": self.model_config.num_layers,
            "embed_dims": self.model_config.embed_dims,
            "num_heads": self.model_config.num_heads,
            "num_levels": self.model_config.num_levels,
            "num_points": self.model_config.num_points,
            "num_points_in_pillar": self.model_config.num_points_in_pillar,
            "feedforward_channels": self.model_config.feedforward_channels,
            "batch_first": self.model_config.batch_first,
            "return_intermediate": self.model_config.return_intermediate,
        }

    def get_spatial_shapes_tensor(self):
        """
        Get spatial shapes as tensor for use in forward pass.

        Returns:
            Tensor of spatial shapes that can be used directly in encoder forward
        """
        import torch

        return torch.tensor(self.dataset_config.spatial_shapes, dtype=torch.int32)

    def get_level_start_index(self):
        """
        Calculate level start indices for multi-scale features.

        Returns:
            Tensor of start indices for each pyramid level
        """
        import torch

        # Use the utility function and limit to model's num_levels
        spatial_shapes = self.dataset_config.spatial_shapes[: self.model_config.num_levels]
        level_start_index = calculate_level_start_indices(spatial_shapes)

        return torch.tensor(level_start_index, dtype=torch.int32)

    def summary(self) -> str:
        """
        Get a human-readable summary of the configuration.

        Returns:
            Formatted string describing the configuration
        """
        if self.dataset_config is None or self.model_config is None:
            return f"EncoderConfig(dataset='{self.dataset_name}', model='{self.model_variant}', not loaded)"

        return f"""EncoderConfig Summary:
            Dataset: {self.dataset_name}
            Model Variant: {self.model_variant}
            Description: {self.dataset_config.description}
            PC Range: {self.dataset_config.pc_range}
            Cameras: {self.dataset_config.num_cams}
            Input Size: {self.dataset_config.input_size}
            Spatial Shapes: {self.dataset_config.spatial_shapes}
            Model Layers: {self.model_config.num_layers}
            Embed Dims: {self.model_config.embed_dims}
            Attention Heads: {self.model_config.num_heads}
            Parameters: ~{self._estimate_parameters():.1f}M"""

    def _estimate_parameters(self) -> float:
        """Estimate model parameters in millions."""
        if self.model_config is None:
            return 0.0

        # Rough estimation based on embed_dims and num_layers
        embed_dims = self.model_config.embed_dims
        num_layers = self.model_config.num_layers
        feedforward_channels = self.model_config.feedforward_channels

        # Attention parameters per layer
        attention_params = 4 * embed_dims * embed_dims  # Q, K, V, O projections

        # FFN parameters per layer
        ffn_params = embed_dims * feedforward_channels * 2  # Up and down projections

        # Total per layer
        params_per_layer = attention_params + ffn_params + 2 * embed_dims  # Layer norms

        # Total model parameters
        total_params = num_layers * params_per_layer

        return total_params / 1_000_000  # Convert to millions


# Predefined configurations for common use cases
PRESET_CONFIGS = {
    # NuScenes configurations
    "nuscenes_tiny": EncoderConfig("nuscenes_v1.0_full_640x360", "tiny"),
    "nuscenes_small": EncoderConfig("nuscenes_v1.0_full_640x360", "small"),
    "nuscenes_base": EncoderConfig("nuscenes_v1.0_full_1600x900", "base"),
    "nuscenes_base_fast": EncoderConfig("nuscenes_v1.0_full_640x360", "base"),
    "nuscenes_large": EncoderConfig("nuscenes_v1.0_full_1600x900", "large"),
    # CARLA configurations
    "carla_tiny": EncoderConfig("carla_v0.9.10_640x480", "tiny"),
    "carla_small": EncoderConfig("carla_v0.9.10_800x600", "small"),
    "carla_base": EncoderConfig("carla_v0.9.10_1280x960", "base"),
    "carla_base_fast": EncoderConfig("carla_v0.9.10_640x480", "base"),
    "carla_large": EncoderConfig("carla_v0.9.10_1920x1080", "large"),
    # Other datasets with base model
    "kitti_base": EncoderConfig("kitti360_1408x376", "base"),
    "kitti_tiny": EncoderConfig("kitti360_1242x375", "tiny"),
    "waymo_base": EncoderConfig("waymo_v1.0_1920x1280", "base"),
    "waymo_small": EncoderConfig("waymo_v1.0_960x640", "small"),
    "lyft_base": EncoderConfig("lyft_v1.0_1920x1080", "base"),
    "lyft_small": EncoderConfig("lyft_v1.0_1280x720", "small"),
}


def get_preset_config(preset_name: str) -> Optional[EncoderConfig]:
    """
    Get a predefined configuration by name.

    Args:
        preset_name: Name of the preset configuration

    Returns:
        EncoderConfig if preset exists, None otherwise
    """
    return PRESET_CONFIGS.get(preset_name)


def list_preset_configs() -> List[str]:
    """
    List available preset configuration names.

    Returns:
        List of preset configuration names
    """
    return list(PRESET_CONFIGS.keys())


def list_model_variants() -> List[str]:
    """
    List available model variants.

    Returns:
        List of model variant names
    """
    return list(MODEL_VARIANTS.keys())


def create_custom_config(dataset_name: str, model_variant: str = "base", **model_overrides) -> EncoderConfig:
    """
    Create a custom encoder configuration with optional parameter overrides.

    Args:
        dataset_name: Name of the dataset configuration to use
        model_variant: Base model variant to start from
        **model_overrides: Parameters to override in the model config

    Returns:
        Custom EncoderConfig instance
    """
    # Get base model config
    base_model_config = get_model_config(model_variant)

    # Apply overrides
    for key, value in model_overrides.items():
        if hasattr(base_model_config, key):
            setattr(base_model_config, key, value)
        else:
            raise ValueError(f"Unknown model parameter: {key}")

    # Create encoder config with custom model
    config = EncoderConfig(dataset_name=dataset_name, model_variant=model_variant)
    config.model_config = base_model_config

    # Re-validate
    config.model_config.validate_with_dataset(config.dataset_config)

    return config
