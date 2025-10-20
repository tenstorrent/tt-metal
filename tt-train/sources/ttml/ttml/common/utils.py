# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Utility functions for transformer training."""
import random
import numpy as np
import ttml


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    ttml.autograd.AutoContext.get_instance().set_seed(seed)


def round_up_to_tile(value: int, tile: int = 32) -> int:
    """Round up value to nearest multiple of tile size.

    Args:
        value: Value to round up
        tile: Tile size (default: 32)

    Returns:
        Value rounded up to nearest tile multiple
    """
    return ((int(value) + int(tile) - 1) // int(tile)) * int(tile)


def initialize_device(yaml_config: dict):
    """Initialize device mesh from configuration.

    Args:
        yaml_config: Dictionary containing device configuration
    """
    from ttml.common.config import DeviceConfig

    device_config = DeviceConfig(yaml_config)
    print('Initializing device mesh with shape', device_config.mesh_shape, 'and device ids', device_config.device_ids, 'total devices', device_config.total_devices())
    ttml.core.distributed.enable_fabric(device_config.total_devices())
    ttml.autograd.AutoContext.get_instance().open_device(device_config.mesh_shape, device_config.device_ids)


def create_optimizer(model, yaml_config: dict):
    """Create AdamW optimizer from configuration.

    Args:
        model: Model to optimize
        yaml_config: Dictionary containing optimizer configuration

    Returns:
        AdamW optimizer instance
    """
    lr = yaml_config.get("learning_rate", 0.0003)
    beta1 = yaml_config.get("beta1", 0.9)
    beta2 = yaml_config.get("beta2", 0.999)
    eps = yaml_config.get("eps", 1e-8)
    weight_decay = yaml_config.get("weight_decay", 0.01)

    adamw_cfg = ttml.optimizers.AdamWConfig.make(
        float(lr),
        float(beta1),
        float(beta2),
        float(eps),
        float(weight_decay),
    )
    return ttml.optimizers.AdamW(model.parameters(), adamw_cfg)
