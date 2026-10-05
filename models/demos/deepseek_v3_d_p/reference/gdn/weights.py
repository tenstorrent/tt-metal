# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Canonical host-weight schema of one GDN layer: the checkpoint's layer-local ``linear_attn.*`` names, unchanged.

``in_proj_qkv`` and ``conv1d`` keep the checkpoint's fused, contiguous ``[q | k | v]`` row blocks, each head-major.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig

GDN_WEIGHT_NAMES = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "out_proj.weight",
    "conv1d.weight",
    "A_log",
    "dt_bias",
    "norm.weight",
)


def expected_gdn_weight_shapes(config: GDNConfig) -> dict[str, tuple[int, ...]]:
    """Return the canonical shape of every host weight for ``config``."""
    hidden, value_heads = config.hidden_size, config.num_value_heads
    return {
        "in_proj_qkv.weight": (config.conv_dim, hidden),
        "in_proj_z.weight": (config.v_dim, hidden),
        "in_proj_a.weight": (value_heads, hidden),
        "in_proj_b.weight": (value_heads, hidden),
        "out_proj.weight": (hidden, config.v_dim),
        "conv1d.weight": (config.conv_dim, 1, config.conv_kernel_size),
        "A_log": (value_heads,),
        "dt_bias": (value_heads,),
        "norm.weight": (config.head_v_dim,),
    }


def validate_gdn_weights(weights: Mapping[str, torch.Tensor], config: GDNConfig) -> None:
    """Require exactly the canonical weights of ``config``, each with its canonical shape."""
    expected = expected_gdn_weight_shapes(config)
    missing = [name for name in expected if name not in weights]
    if missing:
        raise ValueError(f"missing GDN weights: {missing}")
    extra = sorted(set(weights) - set(expected))
    if extra:
        raise ValueError(f"unexpected GDN weights: {extra}")
    for name, shape in expected.items():
        if tuple(weights[name].shape) != shape:
            raise ValueError(f"{name} shape {tuple(weights[name].shape)} != {shape}")
