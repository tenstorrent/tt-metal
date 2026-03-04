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


def create_optimizer(model, yaml_config: dict):
    """Create AdamW or MorehAdamW optimizer from configuration.

    Args:
        model: Model to optimize
        yaml_config: Dictionary containing optimizer configuration

    Returns:
        AdamW or MorehAdamW optimizer instance based on configuration
    """
    optimizer_config = yaml_config.get("training_config", {})

    lr = optimizer_config.get("learning_rate", 0.0003)
    beta1 = optimizer_config.get("beta1", 0.9)
    beta2 = optimizer_config.get("beta2", 0.999)
    eps = optimizer_config.get("eps", 1e-8)
    weight_decay = optimizer_config.get("weight_decay", 0.01)
    use_moreh_adamw = optimizer_config.get("use_moreh_adamw", False)

    adamw_cfg = ttml.optimizers.AdamWConfig.make(
        float(lr),
        float(beta1),
        float(beta2),
        float(eps),
        float(weight_decay),
    )

    if use_moreh_adamw:
        return ttml.optimizers.MorehAdamW(model.parameters(), adamw_cfg)
    else:
        return ttml.optimizers.AdamW(model.parameters(), adamw_cfg)


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
