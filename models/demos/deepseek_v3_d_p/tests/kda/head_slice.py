# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Contiguous head slices of one KDA layer, used to run one TP rank's work without TP.

LoudBox LB-B (8x1, SP8xTP1) emulates the per-chip work of Galaxy SP8xTP4 by running a
quarter of the heads on each chip. A contiguous head range ``[s, s + n)`` with
``s = r * n`` and ``n = H / tp`` is exactly TP rank ``r``'s shard in ``tt/kda/weights.py``:

* q/k/v projections and convolutions, ``f_b``, ``dt_bias``, ``A_log``, ``b_proj`` and the
  output gate (``g_proj``, or ``g_b`` of the low-rank gate) are split by heads, outer
  ``(h d)`` layout, contiguous per rank;
* ``f_a`` and ``g_a`` (the low-rank factors that project from ``hidden``) are replicated
  on every rank, so the folded low-rank gate ``g_b[rows_r] @ g_a`` is that rank's rows;
* ``o_norm`` is per head dimension and shared by all heads, so it is replicated;
* ``o_proj`` is split by its input columns, so the sliced layer's output is rank ``r``'s
  partial sum before the TP reduce-scatter, and the full-layer output is the sum of the
  partials over all ranks.

Heads do not interact before ``o_proj``, so the CPU reference evaluated on the slice is the
oracle for that rank: its recurrent and convolution states are the full layer's state
restricted to the slice, and its output is the partial sum above.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict, validate_kda_weights


def kda_head_slice_config(config: KDAConfig, num_heads: int) -> KDAConfig:
    """Return ``config`` restricted to ``num_heads`` heads; all other fields are unchanged."""
    if not 0 < num_heads <= config.num_heads:
        raise ValueError(f"num_heads must be in [1, {config.num_heads}], got {num_heads}")
    return replace(config, num_heads=num_heads)


# LoudBox LB-B (8x1, SP8xTP1) runs on each chip the heads one chip owns on Galaxy (TP4).
GALAXY_TENSOR_PARALLEL_SIZE = 4


def galaxy_chip_head_slice_config(config: KDAConfig) -> KDAConfig:
    """Return the LB-B per-chip config: one Galaxy TP rank's share of ``config``'s heads."""
    if config.num_heads % GALAXY_TENSOR_PARALLEL_SIZE:
        raise ValueError(f"num_heads {config.num_heads} is not divisible by Galaxy TP{GALAXY_TENSOR_PARALLEL_SIZE}")
    return kda_head_slice_config(config, config.num_heads // GALAXY_TENSOR_PARALLEL_SIZE)


def slice_kda_heads(
    state_dict: Mapping[str, torch.Tensor],
    config: KDAConfig,
    *,
    num_heads: int,
    head_start: int = 0,
) -> dict[str, torch.Tensor]:
    """Return the canonical weights of heads ``[head_start, head_start + num_heads)``.

    ``state_dict`` is one layer's checkpoint-local weights for ``config`` (for example the
    output of ``load_kda_layer_state_dict``); checkpoint head padding is normalized first.
    The result is validated against ``kda_head_slice_config(config, num_heads)``. Its
    content differs from the full layer, so callers key derived caches by its own identity.
    """
    sliced_config = kda_head_slice_config(config, num_heads)
    if not 0 <= head_start <= config.num_heads - num_heads:
        raise ValueError(f"head range [{head_start}, {head_start + num_heads}) exceeds {config.num_heads} heads")
    weights = normalize_kda_state_dict(state_dict, config)
    stop = head_start + num_heads

    def rows(name: str, head_dim: int) -> torch.Tensor:
        return weights[name][head_start * head_dim : stop * head_dim]

    k, v = config.head_k_dim, config.head_v_dim
    sliced = {
        "q_proj.weight": rows("q_proj.weight", k),
        "k_proj.weight": rows("k_proj.weight", k),
        "v_proj.weight": rows("v_proj.weight", v),
        "q_conv1d.weight": rows("q_conv1d.weight", k),
        "k_conv1d.weight": rows("k_conv1d.weight", k),
        "v_conv1d.weight": rows("v_conv1d.weight", v),
        "A_log": weights["A_log"][:, :, head_start:stop],
        "f_a_proj.weight": weights["f_a_proj.weight"],
        "f_b_proj.weight": rows("f_b_proj.weight", k),
        "dt_bias": rows("dt_bias", k),
        "b_proj.weight": rows("b_proj.weight", 1),
        "o_norm.weight": weights["o_norm.weight"],
        "o_proj.weight": weights["o_proj.weight"][:, head_start * v : stop * v],
    }
    if config.use_full_rank_gate:
        sliced["g_proj.weight"] = rows("g_proj.weight", v)
    else:
        sliced["g_a_proj.weight"] = weights["g_a_proj.weight"]
        sliced["g_b_proj.weight"] = rows("g_b_proj.weight", v)
    sliced = {name: tensor.contiguous() for name, tensor in sliced.items()}
    validate_kda_weights(sliced, sliced_config)
    return sliced
