# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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
    """Get the TT-Metal root directory.

    Returns:
        Path to TT-Metal runtime root.
    """
    runtime_root = os.environ.get("TT_METAL_RUNTIME_ROOT")
    if runtime_root:
        return runtime_root
    raise RuntimeError(
        "TT_METAL_RUNTIME_ROOT is not set. Please export TT_METAL_RUNTIME_ROOT "
        "to the tt-metal repository root before running training utilities."
    )


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
    """Create an optimizer from the optimizer config YAML.

    Accepts either a top-level YAML config dict (with training_config.optimizer)
    or the optimizer dict directly (detected by the presence of a "type" key).
    Passes the dict straight to the C++ optimizer factory.

    Args:
        model: Model to optimize
        yaml_config: Top-level YAML config dict or optimizer dict

    Returns:
        Optimizer instance
    """
    if "type" in yaml_config:
        optimizer_cfg = yaml_config
    else:
        tc = yaml_config.get("training_config", {})
        optimizer_cfg = tc.get("optimizer")
        if optimizer_cfg is None:
            raise ValueError("training_config must contain an 'optimizer' section")
    return ttml.optimizers.create_optimizer(optimizer_cfg, model.parameters())


def get_loss_over_devices(loss):
    """Aggregate loss over all devices and return mean."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    loss_numpy = loss.to_numpy(ttnn.DataType.FLOAT32, composer=composer)
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


def summary(model) -> None:
    """Print a torchsummary-style overview of model parameters.

    Parameters are grouped: trainable first, then non-trainable.
    """
    params = model.parameters()

    trainable = []
    frozen = []
    for name, tensor in sorted(params.items()):
        shape = tuple(tensor.shape())
        n = 1
        for d in shape:
            n *= d
        entry = (name, shape, n, tensor.get_requires_grad())
        if entry[3]:
            trainable.append(entry)
        else:
            frozen.append(entry)

    entries = trainable + frozen

    col_name = "Parameter"
    col_shape = "Shape"
    col_nparams = "# Params"
    col_train = "Trainable"

    name_w = (
        max(len(col_name), *(len(e[0]) for e in entries)) if entries else len(col_name)
    )
    shape_w = (
        max(len(col_shape), *(len(str(e[1])) for e in entries))
        if entries
        else len(col_shape)
    )
    nparams_w = (
        max(len(col_nparams), *(len(f"{e[2]:,}") for e in entries))
        if entries
        else len(col_nparams)
    )
    train_w = len(col_train)

    total_w = name_w + shape_w + nparams_w + train_w + 9  # 3 separators + padding

    sep = "=" * total_w
    thin_sep = "-" * total_w

    header = (
        f" {col_name:<{name_w}} | {col_shape:<{shape_w}} "
        f"| {col_nparams:>{nparams_w}} | {col_train}"
    )

    lines = [sep, header, sep]

    prev_trainable = None
    for name, shape, n, is_train in entries:
        if prev_trainable is not None and prev_trainable != is_train:
            lines.append(thin_sep)
        prev_trainable = is_train
        mark = "Yes" if is_train else "No"
        lines.append(
            f" {name:<{name_w}} | {str(shape):<{shape_w}} "
            f"| {n:>{nparams_w},} | {mark}"
        )

    total = sum(e[2] for e in entries)
    total_train = sum(e[2] for e in trainable)
    total_frozen = total - total_train

    lines.append(sep)
    lines.append(f"Total params:          {total:,}")
    lines.append(f"Trainable params:      {total_train:,}")
    lines.append(f"Non-trainable params:  {total_frozen:,}")
    lines.append(thin_sep)

    print("\n".join(lines))


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
