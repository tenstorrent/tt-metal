# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading Oobleck VAE checkpoints into TTNN.

PyTorch's ``torch.nn.utils.weight_norm`` (legacy) stores two tensors per conv:
``weight_g`` (the gain, shape ``[out, 1, 1]``) and ``weight_v`` (the direction,
same shape as the underlying weight).  The runtime weight is

    weight = weight_g * weight_v / ||weight_v||_dim

where the norm is taken over every dimension *except* ``dim`` (``dim=0`` is the
output-channel axis for Conv1d / ConvTranspose1d).

This module exposes a single host-side function ``fuse_weight_norm`` that takes
a tensor pair and returns the fused weight as a NumPy array.  The bulk helper
``fused_oobleck_decoder_weights`` walks a state dict (with ``weight_g`` /
``weight_v`` keys) and emits a flat dict of fused weights so that
``TtOobleckDecoder`` does not need to know about ``weight_norm``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _as_numpy(t) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t.astype(np.float32, copy=False)
    try:
        import torch

        if isinstance(t, torch.Tensor):
            return t.detach().to(dtype=torch.float32, device="cpu").numpy()
    except Exception:
        pass
    return np.asarray(t, dtype=np.float32)


def fuse_weight_norm(weight_g, weight_v) -> np.ndarray:
    """Fuse the ``weight_g``/``weight_v`` pair into a single conv weight.

    Args:
        weight_g: Gain tensor of shape ``[out, 1, 1]`` (Conv1d / ConvTranspose1d).
        weight_v: Direction tensor with the same shape as the underlying weight.

    Returns:
        Fused weight as a contiguous float32 NumPy array.
    """
    g = _as_numpy(weight_g)
    v = _as_numpy(weight_v)
    if v.ndim != 3:
        raise ValueError(f"Expected rank-3 weight_v for Conv1d/ConvTranspose1d, got {v.shape}")
    norm = np.sqrt((v * v).sum(axis=(1, 2), keepdims=True))
    norm = np.maximum(norm, 1e-12)
    fused = g * v / norm
    return np.ascontiguousarray(fused.astype(np.float32))


def _fused_or_passthrough(sd: Mapping[str, object], prefix: str) -> np.ndarray:
    """Resolve ``prefix.weight`` from either a fused or a (weight_g, weight_v) pair."""
    if f"{prefix}.weight_g" in sd and f"{prefix}.weight_v" in sd:
        return fuse_weight_norm(sd[f"{prefix}.weight_g"], sd[f"{prefix}.weight_v"])
    plain = sd.get(f"{prefix}.weight")
    if plain is None:
        raise KeyError(
            f"Missing {prefix}.weight (and no weight_g/weight_v) in state dict; " "VAE checkpoint may be malformed."
        )
    return _as_numpy(plain)


def _maybe_bias(sd: Mapping[str, object], prefix: str):
    b = sd.get(f"{prefix}.bias")
    if b is None:
        return None
    return _as_numpy(b)


def fused_oobleck_decoder_weights(
    state_dict: Mapping[str, object],
    *,
    upsampling_ratios,
    decoder_prefix: str = "",
) -> dict[str, np.ndarray]:
    """Walk an Oobleck decoder state dict and emit fused weights for TTNN.

    Snake parameters retain their original shape ``[1, C, 1]``; conv weights
    retain their PyTorch layout (``[out, in, k]`` for Conv1d,
    ``[in, out, k]`` for ConvTranspose1d).
    """
    pref = decoder_prefix
    out: dict[str, np.ndarray] = {}

    out["conv1.weight"] = _fused_or_passthrough(state_dict, f"{pref}conv1")
    b = _maybe_bias(state_dict, f"{pref}conv1")
    if b is not None:
        out["conv1.bias"] = b

    for i, _stride in enumerate(upsampling_ratios):
        bp = f"{pref}block.{i}"
        out[f"block.{i}.snake1.alpha"] = _as_numpy(state_dict[f"{bp}.snake1.alpha"])
        out[f"block.{i}.snake1.beta"] = _as_numpy(state_dict[f"{bp}.snake1.beta"])
        out[f"block.{i}.conv_t1.weight"] = _fused_or_passthrough(state_dict, f"{bp}.conv_t1")
        b = _maybe_bias(state_dict, f"{bp}.conv_t1")
        if b is not None:
            out[f"block.{i}.conv_t1.bias"] = b

        for ru in (1, 2, 3):
            rp = f"{bp}.res_unit{ru}"
            out[f"block.{i}.res_unit{ru}.snake1.alpha"] = _as_numpy(state_dict[f"{rp}.snake1.alpha"])
            out[f"block.{i}.res_unit{ru}.snake1.beta"] = _as_numpy(state_dict[f"{rp}.snake1.beta"])
            out[f"block.{i}.res_unit{ru}.snake2.alpha"] = _as_numpy(state_dict[f"{rp}.snake2.alpha"])
            out[f"block.{i}.res_unit{ru}.snake2.beta"] = _as_numpy(state_dict[f"{rp}.snake2.beta"])
            out[f"block.{i}.res_unit{ru}.conv1.weight"] = _fused_or_passthrough(state_dict, f"{rp}.conv1")
            b = _maybe_bias(state_dict, f"{rp}.conv1")
            if b is not None:
                out[f"block.{i}.res_unit{ru}.conv1.bias"] = b
            out[f"block.{i}.res_unit{ru}.conv2.weight"] = _fused_or_passthrough(state_dict, f"{rp}.conv2")
            b = _maybe_bias(state_dict, f"{rp}.conv2")
            if b is not None:
                out[f"block.{i}.res_unit{ru}.conv2.bias"] = b

    out["snake1.alpha"] = _as_numpy(state_dict[f"{pref}snake1.alpha"])
    out["snake1.beta"] = _as_numpy(state_dict[f"{pref}snake1.beta"])

    out["conv2.weight"] = _fused_or_passthrough(state_dict, f"{pref}conv2")
    b = _maybe_bias(state_dict, f"{pref}conv2")
    if b is not None:
        out["conv2.bias"] = b

    return out
