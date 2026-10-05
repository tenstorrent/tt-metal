# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-K-group head slices of one GDN layer: one TP rank's work, run without TP.

LoudBox LB-B (8x1, SP8xTP1) runs on each chip the heads one Galaxy TP4 rank owns. A GDN rank holds K heads
``[s, s + n)`` and exactly the V heads that read them, ``[s * G, (s + n) * G)`` with ``G = Nv / Nk`` (V head ``j``
reads K head ``j // G``), so slices are whole K-head groups:

* the q and k row blocks of ``in_proj_qkv`` and ``conv1d`` are split by K heads, the v block, ``in_proj_z``,
  ``in_proj_a``, ``in_proj_b``, ``A_log`` and ``dt_bias`` by V heads, each block contiguous per rank;
* ``norm.weight`` is per value channel and shared by all heads, so it is replicated;
* ``out_proj`` is split by its input columns, so the sliced layer's output is the rank's partial sum before the TP
  reduce, and the full-layer output is the sum of the partials over all ranks.

Heads do not interact before ``out_proj``, so the reference evaluated on the slice is that rank's oracle: its
recurrent state is the full state's V heads ``[s * G, (s + n) * G)``, its convolution state the full state's
``[q_s | k_s | v_s]`` columns, and its output the rank partial above.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch

from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.reference.gdn.weights import validate_gdn_weights


def gdn_head_slice_config(config: GDNConfig, num_key_heads: int) -> GDNConfig:
    """Return ``config`` restricted to ``num_key_heads`` K heads and their V heads; other fields are unchanged."""
    if not 0 < num_key_heads <= config.num_key_heads:
        raise ValueError(f"num_key_heads must be in [1, {config.num_key_heads}], got {num_key_heads}")
    return replace(config, num_key_heads=num_key_heads, num_value_heads=num_key_heads * config.group)


def gdn_head_slice_channels(config: GDNConfig, *, key_head_start: int, num_key_heads: int) -> torch.Tensor:
    """Indices of the slice's ``[q | k | v]`` channels within the full layer's fused ``conv_dim`` channels."""
    _check_range(config, key_head_start, num_key_heads)
    k, v, group = config.head_k_dim, config.head_v_dim, config.group
    stop = key_head_start + num_key_heads
    q = torch.arange(key_head_start * k, stop * k)
    return torch.cat(
        [q, config.q_dim + q, config.q_dim + config.k_dim + torch.arange(key_head_start * group * v, stop * group * v)]
    )


def slice_gdn_heads(
    weights: Mapping[str, torch.Tensor],
    config: GDNConfig,
    *,
    key_head_start: int,
    num_key_heads: int,
) -> dict[str, torch.Tensor]:
    """Return the canonical weights of K heads ``[key_head_start, key_head_start + num_key_heads)`` and their V heads.

    ``weights`` is one layer's canonical host weights for ``config``. The result is validated against
    ``gdn_head_slice_config(config, num_key_heads)``. Its content differs from the full layer, so callers key derived
    caches by its own identity.
    """
    sliced_config = gdn_head_slice_config(config, num_key_heads)
    channels = gdn_head_slice_channels(config, key_head_start=key_head_start, num_key_heads=num_key_heads)
    validate_gdn_weights(weights, config)
    v_heads = slice(key_head_start * config.group, (key_head_start + num_key_heads) * config.group)
    v_channels = slice(v_heads.start * config.head_v_dim, v_heads.stop * config.head_v_dim)
    sliced = {
        "in_proj_qkv.weight": weights["in_proj_qkv.weight"][channels],
        "in_proj_z.weight": weights["in_proj_z.weight"][v_channels],
        "in_proj_a.weight": weights["in_proj_a.weight"][v_heads],
        "in_proj_b.weight": weights["in_proj_b.weight"][v_heads],
        "out_proj.weight": weights["out_proj.weight"][:, v_channels],
        "conv1d.weight": weights["conv1d.weight"][channels],
        "A_log": weights["A_log"][v_heads],
        "dt_bias": weights["dt_bias"][v_heads],
        "norm.weight": weights["norm.weight"],
    }
    # Copies, not views: a slice is an independent weight set (callers may cast or mutate it).
    sliced = {name: tensor.clone(memory_format=torch.contiguous_format) for name, tensor in sliced.items()}
    validate_gdn_weights(sliced, sliced_config)
    return sliced


def _check_range(config: GDNConfig, key_head_start: int, num_key_heads: int) -> None:
    if not 0 < num_key_heads <= config.num_key_heads:
        raise ValueError(f"num_key_heads must be in [1, {config.num_key_heads}], got {num_key_heads}")
    if not 0 <= key_head_start <= config.num_key_heads - num_key_heads:
        raise ValueError(
            f"K-head range [{key_head_start}, {key_head_start + num_key_heads}) exceeds {config.num_key_heads} K heads"
        )
