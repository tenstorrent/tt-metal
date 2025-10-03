# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for transformer training."""
import os
import yaml


def get_config(path: str):
    """Load configuration from YAML file.

    Args:
        path: Path to config file relative to configs directory

    Returns:
        Dictionary containing configuration
    """
    path = f'{os.environ["TT_METAL_HOME"]}/tt-train/configs/{path}'
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


class DeviceConfig:
    """Configuration for device mesh and distributed training."""

    def __init__(self, yaml_config: dict):
        """Initialize device configuration from YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """
        device_config = yaml_config.get("device_config", {})
        self.mesh_shape = device_config.get("mesh_shape", [1, 1])
        self.device_ids = device_config.get("device_ids", [])
        self.enable_tp = device_config.get("enable_tp", False)
        self.enable_ddp = device_config.get("enable_ddp", False)

        # we currently support only [1, N] mesh shapes
        assert self.mesh_shape[0] == 1, f"Only [1, N] mesh shapes are supported, got {self.mesh_shape}"

    def total_devices(self) -> int:
        """Get total number of devices in mesh.

        Returns:
            Total device count
        """
        return self.mesh_shape[0] * self.mesh_shape[1]


class TrainingConfig:
    """Configuration for training hyperparameters."""

    def __init__(self, yaml_config: dict):
        """Initialize training configuration from YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """
        tc = yaml_config.get("training_config", {})
        self.batch_size = int(tc.get("batch_size", 64))
        self.steps = int(tc.get("max_steps", 1000))
        self.eval_every = int(tc.get("eval_every", 200))
        self.gradient_accumulation_steps = int(tc.get("gradient_accumulation_steps", 1))

        tcfg = tc.get("transformer_config", yaml_config.get("transformer_config", {}))
        self.seq_len = int(tcfg.get("max_sequence_length", 256))
