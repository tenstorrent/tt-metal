# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Data configuration for BEVFormer encoder supporting multiple autonomous driving datasets.

This module provides comprehensive dataset configurations for BEVFormer encoder including:
- NuScenes (multiple versions and resolutions)
- CARLA (simulation datasets)
- KITTI (stereo vision datasets)
- Waymo (open dataset)
- Lyft Level 5 (prediction dataset)

Each configuration includes dataset-specific parameters like point cloud range, camera count,
input image sizes, spatial shapes for multi-scale features, and z-axis sampling configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional


def calculate_spatial_shapes(input_size: Tuple[int, int], num_levels: int = 4, base_stride: int = 8) -> List[List[int]]:
    """
    Calculate spatial shapes for multi-scale feature pyramid.

    Args:
        input_size: Input image size (width, height)
        num_levels: Number of pyramid levels (default: 4)
        base_stride: Base stride for first level (default: 8)

    Returns:
        List of [width, height] for each pyramid level (following BEVFormer convention)

    Formula: spatial_shape_level_i = [ceil(input_width / (base_stride * 2^i)), ceil(input_height / (base_stride * 2^i))]
    Uses ceiling division to handle non-exact divisions.
    """
    width, height = input_size
    spatial_shapes = []

    for i in range(num_levels):
        stride = base_stride * (2**i)
        # Use ceiling division: (a + b - 1) // b is equivalent to ceil(a / b)
        level_width = (width + stride - 1) // stride
        level_height = (height + stride - 1) // stride
        spatial_shapes.append([level_width, level_height])

    return spatial_shapes


def create_z_config(pc_range: List[float], num_points: int = 4) -> Dict[str, Any]:
    """
    Create z-axis sampling configuration from point cloud range.

    Args:
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        num_points: Number of sampling points in z-axis (default: 4)

    Returns:
        Z-axis configuration dictionary
    """
    return {
        "num_points": num_points,
        "start": pc_range[2],  # z_min
        "end": pc_range[5],  # z_max
    }


@dataclass
class DatasetConfig:
    """
    Configuration for a specific dataset variant.

    Attributes:
        name: Dataset identifier (e.g., "nuscenes_v1.0_1600x900")
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        num_cams: Number of cameras in the dataset
        input_size: Input image size (width, height)
        spatial_shapes: Multi-scale feature pyramid shapes [[H1,W1], [H2,W2], ...]
        z_cfg: Z-axis sampling configuration
        description: Human-readable description of the dataset
    """

    name: str
    pc_range: List[float]
    num_cams: int
    input_size: Tuple[int, int]
    spatial_shapes: List[List[int]] = field(default_factory=list)
    z_cfg: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self):
        """Validate and auto-calculate derived parameters."""
        # Validate pc_range
        if len(self.pc_range) != 6:
            raise ValueError(f"pc_range must have 6 elements, got {len(self.pc_range)}")

        # Validate input_size
        if len(self.input_size) != 2 or self.input_size[0] <= 0 or self.input_size[1] <= 0:
            raise ValueError(f"input_size must be (width, height) with positive values, got {self.input_size}")

        # Auto-calculate spatial_shapes if not provided
        if not self.spatial_shapes:
            self.spatial_shapes = calculate_spatial_shapes(self.input_size)

        # Auto-create z_cfg if not provided
        if not self.z_cfg:
            self.z_cfg = create_z_config(self.pc_range)


@dataclass
class BEVFormerDataConfig:
    """
    Main configuration container for all supported datasets.

    Provides easy access to dataset configurations through the datasets dictionary.
    Priority datasets (NuScenes and CARLA) are implemented with full resolution variants.
    """

    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize all dataset configurations."""
        self._init_nuscenes_configs()
        self._init_carla_configs()
        self._init_kitti_configs()
        self._init_waymo_configs()
        self._init_lyft_configs()

    def _init_nuscenes_configs(self):
        """Initialize NuScenes dataset configurations (HIGH PRIORITY)."""
        nuscenes_base = {
            "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "num_cams": 6,
        }

        # NuScenes v1.0-full with multiple resolutions
        resolutions = [
            (1600, 900, "Full HD resolution for maximum detail"),
            (1280, 720, "Balanced resolution for good performance"),
            (640, 360, "Fast resolution for real-time applications"),
        ]

        for width, height, desc in resolutions:
            config_name = f"nuscenes_v1.0_full_{width}x{height}"
            self.datasets[config_name] = DatasetConfig(
                name=config_name,
                input_size=(width, height),
                description=f"NuScenes v1.0-full dataset - {desc}",
                **nuscenes_base,
            )

        # NuScenes v1.0-mini (same parameters, smaller dataset)
        for width, height, desc in resolutions:
            config_name = f"nuscenes_v1.0_mini_{width}x{height}"
            self.datasets[config_name] = DatasetConfig(
                name=config_name,
                input_size=(width, height),
                description=f"NuScenes v1.0-mini dataset - {desc}",
                **nuscenes_base,
            )

        # NuScenes v1.0-test (same parameters, test split)
        for width, height, desc in resolutions:
            config_name = f"nuscenes_v1.0_test_{width}x{height}"
            self.datasets[config_name] = DatasetConfig(
                name=config_name,
                input_size=(width, height),
                description=f"NuScenes v1.0-test dataset - {desc}",
                **nuscenes_base,
            )

    def _init_carla_configs(self):
        """Initialize CARLA dataset configurations (HIGH PRIORITY)."""
        carla_base = {
            "pc_range": [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
            "num_cams": 6,
        }

        # CARLA v0.9.10+ with multiple resolutions
        resolutions = [
            (1920, 1080, "Full HD simulation for high-fidelity research"),
            (1280, 960, "Balanced resolution for simulation experiments"),
            (800, 600, "Medium resolution for faster simulation"),
            (640, 480, "Fast simulation for real-time testing"),
        ]

        for width, height, desc in resolutions:
            config_name = f"carla_v0.9.10_{width}x{height}"
            self.datasets[config_name] = DatasetConfig(
                name=config_name,
                input_size=(width, height),
                description=f"CARLA v0.9.10+ simulation - {desc}",
                **carla_base,
            )

        # CARLA v0.9.13 (enhanced version)
        for width, height, desc in resolutions:
            config_name = f"carla_v0.9.13_{width}x{height}"
            self.datasets[config_name] = DatasetConfig(
                name=config_name,
                input_size=(width, height),
                description=f"CARLA v0.9.13 enhanced simulation - {desc}",
                **carla_base,
            )

        # CARLA Custom (research-configurable)
        self.datasets["carla_custom"] = DatasetConfig(
            name="carla_custom",
            pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
            num_cams=6,
            input_size=(1280, 960),
            description="CARLA custom configuration for research (configurable parameters)",
        )

    def _init_kitti_configs(self):
        """Initialize KITTI dataset configurations."""
        kitti_base = {
            "pc_range": [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0],
            "num_cams": 2,
        }

        # KITTI-360
        resolutions = [
            (1408, 376, "Standard KITTI-360 resolution"),
            (1242, 375, "Classic KITTI stereo resolution"),
        ]

        for width, height, desc in resolutions:
            config_name = f"kitti360_{width}x{height}"
            self.datasets[config_name] = DatasetConfig(
                name=config_name, input_size=(width, height), description=f"KITTI-360 dataset - {desc}", **kitti_base
            )

        # KITTI-raw
        self.datasets["kitti_raw_1242x375"] = DatasetConfig(
            name="kitti_raw_1242x375",
            input_size=(1242, 375),
            description="KITTI raw sensor data - classic stereo resolution",
            **kitti_base,
        )

        # KITTI-odometry
        self.datasets["kitti_odometry_1242x375"] = DatasetConfig(
            name="kitti_odometry_1242x375",
            input_size=(1242, 375),
            description="KITTI odometry dataset - stereo visual odometry",
            **kitti_base,
        )

    def _init_waymo_configs(self):
        """Initialize Waymo Open Dataset configurations."""
        waymo_base = {
            "pc_range": [-75.2, -75.2, -2.0, 75.2, 75.2, 4.0],
            "num_cams": 5,
        }

        resolutions = [
            (1920, 1280, "Full resolution Waymo camera data"),
            (960, 640, "Half resolution for faster processing"),
        ]

        for width, height, desc in resolutions:
            config_name = f"waymo_v1.0_{width}x{height}"
            self.datasets[config_name] = DatasetConfig(
                name=config_name,
                input_size=(width, height),
                description=f"Waymo Open Dataset v1.0 - {desc}",
                **waymo_base,
            )

    def _init_lyft_configs(self):
        """Initialize Lyft Level 5 dataset configurations."""
        lyft_base = {
            "pc_range": [-80.0, -80.0, -10.0, 80.0, 80.0, 10.0],
            "num_cams": 7,
        }

        resolutions = [
            (1920, 1080, "Full HD Lyft camera resolution"),
            (1280, 720, "HD resolution for balanced performance"),
        ]

        for width, height, desc in resolutions:
            config_name = f"lyft_v1.0_{width}x{height}"
            self.datasets[config_name] = DatasetConfig(
                name=config_name, input_size=(width, height), description=f"Lyft Level 5 v1.0 - {desc}", **lyft_base
            )

    def get_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        """
        Get configuration for a specific dataset.

        Args:
            dataset_name: Name of the dataset configuration

        Returns:
            DatasetConfig if found, None otherwise
        """
        return self.datasets.get(dataset_name)

    def list_datasets(self, filter_priority: bool = False) -> List[str]:
        """
        List available dataset configurations.

        Args:
            filter_priority: If True, only return priority datasets (NuScenes, CARLA)

        Returns:
            List of dataset configuration names
        """
        if filter_priority:
            priority_prefixes = ["nuscenes", "carla"]
            return [
                name for name in self.datasets.keys() if any(name.startswith(prefix) for prefix in priority_prefixes)
            ]
        return list(self.datasets.keys())

    def get_datasets_by_type(self, dataset_type: str) -> Dict[str, DatasetConfig]:
        """
        Get all configurations for a specific dataset type.

        Args:
            dataset_type: Dataset type (e.g., "nuscenes", "carla", "kitti", etc.)

        Returns:
            Dictionary of configurations matching the type
        """
        return {name: config for name, config in self.datasets.items() if name.startswith(dataset_type)}


# Global instance for easy access
DEFAULT_CONFIG = BEVFormerDataConfig()


def get_dataset_config(dataset_name: str) -> Optional[DatasetConfig]:
    """
    Convenience function to get a dataset configuration.

    Args:
        dataset_name: Name of the dataset configuration

    Returns:
        DatasetConfig if found, None otherwise
    """
    return DEFAULT_CONFIG.get_config(dataset_name)


def list_available_datasets(filter_priority: bool = False) -> List[str]:
    """
    Convenience function to list available dataset configurations.

    Args:
        filter_priority: If True, only return priority datasets (NuScenes, CARLA)

    Returns:
        List of dataset configuration names
    """
    return DEFAULT_CONFIG.list_datasets(filter_priority)
