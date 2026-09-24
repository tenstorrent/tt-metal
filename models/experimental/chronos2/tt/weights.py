# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
#
# Checkpoint loading (safetensors -> FP32 numpy) and the exact FP32 host-side
# weight folds used before upload.

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_weights_fp32(weights_path: str | Path) -> dict[str, np.ndarray]:
    wp = Path(weights_path)
    if wp.is_dir():
        wp = wp / "model.safetensors"
    try:
        from safetensors.numpy import load_file
    except ImportError as e:  # pragma: no cover
        raise ImportError("safetensors is required to load the checkpoint") from e
    tensors = load_file(str(wp))
    return {k: v.astype(np.float32) for k, v in tensors.items()}


def fuse_group_attention(weights: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """W_ov = W_o @ W_v — exact replacement of GroupSelfAttention for independent rows.

    With one series per group, softmax runs over a single key (=1.0 identically),
    so the block output is o(v(rms(x))) with no q/k dependence.
    """
    v = weights[f"{prefix}.self_attention.v.weight"]  # [inner, d_model]
    o = weights[f"{prefix}.self_attention.o.weight"]  # [d_model, inner]
    return (o @ v).astype(np.float32)


def rotate_half_columns(w_t: np.ndarray, num_heads: int, d_kv: int) -> np.ndarray:
    """[K, H*dk] -> columns such that h @ out == rotate_half(h @ w_t) per head."""
    w = w_t.reshape(w_t.shape[0], num_heads, d_kv)
    half = d_kv // 2
    return np.concatenate([-w[..., half:], w[..., :half]], axis=-1).reshape(w_t.shape)
