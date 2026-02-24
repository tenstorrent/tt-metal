# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Utility functions for transformer training."""

from __future__ import annotations

import os, random
from time import time
import numpy as np
import ttnn
import ttml


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    ttml.autograd.AutoContext.get_instance().set_seed(seed)


def get_tt_metal_home() -> str:
    """Get the TT-Metal home directory.

    Returns:
        Path to TT-Metal home directory
    """
    tt_metal_home = (
        os.environ["TT_METAL_HOME"]
        if "TT_METAL_HOME" in os.environ
        else os.path.expanduser("~/.tt-metal")
    )
    return tt_metal_home


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
    if device_config.total_devices() > 1:
        ttml.core.distributed.enable_fabric(device_config.total_devices())
    ttml.autograd.AutoContext.get_instance().open_device(
        device_config.mesh_shape, device_config.device_ids
    )


def parse_optimizer_config(yaml_config: dict) -> dict:
    """Parse the optimizer config YAML referenced by the training config.

    Args:
        yaml_config: Top-level YAML config dict containing training_config.optimizer_config

    Returns:
        Dictionary with optimizer hyperparameters and type
    """
    from ttml.common.config import load_config

    tc = yaml_config.get("training_config", {})
    optimizer_config_path = tc.get("optimizer_config")
    if optimizer_config_path is None:
        raise ValueError(
            "training_config must specify 'optimizer_config' path "
            "(e.g. 'configs/optimizer_configs/adamw.yaml')"
        )
    tt_train_root = f"{get_tt_metal_home()}/tt-train"
    return load_config(optimizer_config_path, tt_train_root)


def create_optimizer(model, yaml_config: dict):
    """Create an optimizer from the optimizer config YAML.

    Reads the optimizer_config YAML file referenced in training_config
    and creates the appropriate optimizer based on the 'type' field.

    Args:
        model: Model to optimize
        yaml_config: Top-level YAML config dict

    Returns:
        Optimizer instance
    """
    cfg = parse_optimizer_config(yaml_config)
    opt_type = cfg.get("type", "AdamW")

    lr = float(cfg.get("lr", 3e-4))
    beta1 = float(cfg.get("beta1", 0.9))
    beta2 = float(cfg.get("beta2", 0.999))
    epsilon = float(cfg.get("epsilon", 1e-8))
    weight_decay = float(cfg.get("weight_decay", 0.01))

    if opt_type in ("AdamW", "MorehAdamW"):
        if opt_type == "MorehAdamW":
            adamw_cfg = ttml.optimizers.AdamWCompositeConfig.make(
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            )
            return ttml.optimizers.MorehAdamW(model.parameters(), adamw_cfg)
        adamw_cfg = ttml.optimizers.AdamWConfig.make(
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
        )
        return ttml.optimizers.AdamW(model.parameters(), adamw_cfg)

    if opt_type == "SGD":
        momentum = float(cfg.get("momentum", 0.0))
        dampening = float(cfg.get("dampening", 0.0))
        nesterov = bool(cfg.get("nesterov", False))
        sgd_cfg = ttml.optimizers.SGDConfig.make(
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
        )
        return ttml.optimizers.SGD(model.parameters(), sgd_cfg)

    raise ValueError(f"Unsupported optimizer type: {opt_type}")


def get_loss_over_devices(loss):
    """Aggregate loss over all devices and return mean."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    loss_numpy = loss.to_numpy(composer=composer)
    return loss_numpy.mean()


def build_logits_mask(vocab_size: int, padded_vocab_size: int) -> ttml.autograd.Tensor:
    logits_mask = np.zeros((1, 1, 1, padded_vocab_size), dtype=np.float32)
    logits_mask[:, :, :, vocab_size:] = 1e4
    return ttml.autograd.Tensor.from_numpy(
        logits_mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )  # [1,1,1,T], bfloat16


class PerformanceMeter:
    def __init__(self, cfg, window_size=10):
        self.cfg = cfg
        self.steps = []
        self.window_size = window_size

    def step(self):
        self.steps.append(time())
        if len(self.steps) > self.window_size:
            self.steps.pop(0)

    def get_metrics(self):
        time_window = self.steps[-1] - self.steps[0]
        if time_window == 0:
            return 0, 0

        samples = (
            len(self.steps) * self.cfg.batch_size * self.cfg.gradient_accumulation_steps
        )
        samples_per_second = samples / time_window
        tokens_per_second = samples * self.cfg.seq_len / time_window
        return samples_per_second, tokens_per_second


class no_grad:
    """Context manager and decorator to disable gradient computation.

    Usage as context manager:
        with no_grad():
            # code here runs without gradients

    Usage as decorator:
        @no_grad()
        def my_function():
            # function runs without gradients
    """

    def __init__(self):
        self._ctx = None
        self._prev = None

    def __enter__(self):
        self._ctx = ttml.autograd.AutoContext.get_instance()
        self._prev = (
            self._ctx.get_gradient_mode()
            if hasattr(self._ctx, "get_gradient_mode")
            else None
        )
        self._ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._prev is not None:
            self._ctx.set_gradient_mode(self._prev)
        else:
            self._ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
        return False

    def __call__(self, func):
        """Allow using as a decorator."""

        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper
