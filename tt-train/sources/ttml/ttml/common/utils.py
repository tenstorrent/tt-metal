# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Utility functions for transformer training."""

from __future__ import annotations

import os, random
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
    ttml.manual_seed(seed)


def build_mesh(device_config) -> "ttml.Mesh":
    """Construct a named mesh from ``device_config``; axis names follow the C++ ``(dp -> fsdp -> tp)`` order.

    First enabled name claims axis 0, the next claims axis 1, etc. -- matching PyTorch FSDP2's HSDP
    convention (replicate/``dp`` outermost, shard/``fsdp`` inner) so ``ttml.fsdp.fully_shard`` and
    ``ttml.sync_gradients`` bind to the same physical axes the C++ trainer would. A line mesh
    (<=1 dim > 1) needs exactly one of enable_ddp/enable_fsdp/enable_tp. HSDP+TP (a full 3D mesh)
    is not supported by the dataloader batch sharding in the training scripts.

    ``device_config`` is a :class:`ttml.common.config.DeviceConfig` (accessed by duck typing -- it
    is not imported here to avoid a circular import, since ``config`` imports from this module).
    """
    shape = tuple(int(s) for s in device_config.mesh_shape)
    n = len(shape)
    nontrivial = [i for i, s in enumerate(shape) if s > 1]
    is_line = len(nontrivial) <= 1

    # Ordered (dp -> fsdp -> tp): maps mesh axis index -> parallelism name.
    enabled_names: list[str] = []
    if device_config.enable_ddp:
        enabled_names.append("dp")
    if device_config.enable_fsdp:
        enabled_names.append("fsdp")
    if device_config.enable_tp:
        enabled_names.append("tp")

    # ttml.Mesh requires a name per axis; "_i" is a placeholder for unassigned axes.
    axis_names = [f"_{i}" for i in range(n)]
    if not enabled_names:
        return ttml.Mesh(shape, tuple(axis_names))

    if is_line:
        if len(enabled_names) != 1:
            raise ValueError(
                f"Line mesh {shape} requires exactly one of enable_ddp / enable_fsdp / enable_tp; "
                f"got enabled={enabled_names}"
            )
        active = nontrivial[0] if nontrivial else 0
        axis_names[active] = enabled_names[0]
    else:
        if len(enabled_names) != n:
            raise ValueError(
                f"Mesh {shape} requires {n} parallelisms enabled "
                f"(any subset of enable_ddp / enable_fsdp / enable_tp); got enabled={enabled_names}"
            )
        for i, name in enumerate(enabled_names):
            axis_names[i] = name
    return ttml.Mesh(shape, tuple(axis_names))


def get_tt_metal_runtime_root() -> str:
    """Return TT-Metal runtime root from the environment.

    Returns:
        Value of TT_METAL_RUNTIME_ROOT.

    Raises:
        RuntimeError: If TT_METAL_RUNTIME_ROOT is not set.
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


def resolve_padded_load_shape(src_shape, target_shape, expected_shape=None, *, name: str = "") -> tuple:
    """Validate a weight being loaded and return the (tile-padded) shape to pad UP to.

    Shared by every HF -> ttml weight loader so the "how do checkpoint dims map
    onto a param's tile-padded shape" policy lives in one place.

    Args:
        src_shape:      shape of the source tensor (after any layout transform).
        target_shape:   the destination param's physical, tile-padded dims to pad
                        up to. The caller is responsible for reconstructing the
                        GLOBAL shape for sharded params (e.g. ``dim *= tp_size``)
                        so this is always a global-vs-global comparison.
        expected_shape: the config-implied LOGICAL (un-tile-padded) shape, when
                        known. Comparing against this -- not the tile-padded
                        ``target_shape`` -- is the only way to catch a checkpoint
                        whose dim (e.g. vocab_size) diverges from the config by
                        LESS than a tile (both round to the same tile).

    Returns ``target_shape`` (as an int tuple) when the load is legitimate; the
    caller then zero-pads the source up to it.

    Raises ``RuntimeError`` on: divergence from ``expected_shape``, a transposed
    layout, a target smaller than the source (would crop), or -- when no
    ``expected_shape`` is given -- a source that is not an exact tile-alignment of
    the target. Never crops. When it must tile-pad without an ``expected_shape``
    to verify against, it prints a warning (that padding is unverified).
    """
    src = tuple(int(x) for x in src_shape)
    tgt = tuple(int(x) for x in target_shape)
    if len(src) != len(tgt):
        raise RuntimeError(f"{name}: source rank {len(src)} does not match target rank {len(tgt)}.")

    if expected_shape is not None:
        exp = tuple(int(x) for x in expected_shape)
        if src != exp:
            raise RuntimeError(
                f"{name}: checkpoint shape {src} does not match the config-implied shape {exp}; "
                f"the checkpoint diverges from the model config."
            )
        if any(t < s for t, s in zip(tgt, src)):
            raise RuntimeError(
                f"{name}: parameter shape {tgt} is smaller than the config-implied shape {src}; cannot fit."
            )
        return tgt

    # No config-derived shape to validate against: permit ONLY tile-alignment
    # padding (target == source rounded up to the tile on every dim).
    if src == tgt:
        return tgt
    if all(t == round_up_to_tile(s) for t, s in zip(tgt, src)):
        print(
            f"  Warning: tile-padding {name} from {src} to {tgt} without a config-derived shape to "
            f"verify against; confirm this is only tile alignment."
        )
        return tgt
    raise RuntimeError(
        f"Unexpected shape for {name}: got {src}, expected {tgt} or its pre-tile-padding source. "
        f"Check that the model config matches the checkpoint."
    )


def initialize_device(yaml_config: dict):
    """Initialize device mesh from configuration.

    Args:
        yaml_config: Dictionary containing device configuration
    """
    from ttml.common.config import DeviceConfig

    device_config = DeviceConfig(yaml_config)
    if device_config.total_devices() > 1:
        ttml.core.distributed.enable_fabric(device_config.total_devices())
    ttml.autograd.AutoContext.get_instance().open_device(device_config.mesh_shape, device_config.device_ids)


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
    return ttml.autograd.Tensor.from_numpy(logits_mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)  # [1,1,1,T], bfloat16


def build_causal_mask(T: int, device: bool = False):
    """Square causal attention mask ``[1, 1, T, T]`` (1 = attend).

    By default returns a host numpy array. With ``device=True`` the mask is
    uploaded as a replicated TILE bf16 ttml tensor (no mesh mapper).
    """
    m = np.tril(np.ones((1, 1, T, T), dtype=np.float32))
    if not device:
        return m
    return ttml.autograd.Tensor.from_numpy(m, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)


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

    name_w = max(len(col_name), *(len(e[0]) for e in entries)) if entries else len(col_name)
    shape_w = max(len(col_shape), *(len(str(e[1])) for e in entries)) if entries else len(col_shape)
    nparams_w = max(len(col_nparams), *(len(f"{e[2]:,}") for e in entries)) if entries else len(col_nparams)
    train_w = len(col_train)

    total_w = name_w + shape_w + nparams_w + train_w + 9  # 3 separators + padding

    sep = "=" * total_w
    thin_sep = "-" * total_w

    header = f" {col_name:<{name_w}} | {col_shape:<{shape_w}} " f"| {col_nparams:>{nparams_w}} | {col_train}"

    lines = [sep, header, sep]

    prev_trainable = None
    for name, shape, n, is_train in entries:
        if prev_trainable is not None and prev_trainable != is_train:
            lines.append(thin_sep)
        prev_trainable = is_train
        mark = "Yes" if is_train else "No"
        lines.append(f" {name:<{name_w}} | {str(shape):<{shape_w}} " f"| {n:>{nparams_w},} | {mark}")

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
        self._prev = self._ctx.get_gradient_mode() if hasattr(self._ctx, "get_gradient_mode") else None
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
