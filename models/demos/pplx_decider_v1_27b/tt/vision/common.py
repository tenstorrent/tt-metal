# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared vision helpers: weight placement and the biased projection every vision module uses."""

from __future__ import annotations

import ttnn
from models.demos.pplx_decider_v1_27b.tt.common import resolve
from models.demos.pplx_decider_v1_27b.tt.optimizations import VisionOptimizations
from models.demos.pplx_decider_v1_27b.tt.vision.weights import LayerNormWeights, LinearWeights


def resolve_linear(w: LinearWeights, device) -> LinearWeights:
    return LinearWeights(weight=resolve(w.weight, device), bias=resolve(w.bias, device))


def resolve_layer_norm(w: LayerNormWeights, device) -> LayerNormWeights:
    return LayerNormWeights(
        weight=resolve(w.weight, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        bias=resolve(w.bias, device, layout=ttnn.ROW_MAJOR_LAYOUT),
    )


def device_linear(w: LinearWeights) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    return w.weight.get_device_weight(), w.bias.get_device_weight()


def vision_linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    role: str,
    opts: VisionOptimizations,
    *,
    activation: str | None = None,
) -> ttnn.Tensor:
    """[1, 1, S, K] x [K, N] + bias [1, N] -> [1, 1, S, N] BF16 in DRAM, fp32 accumulation.

    Bias and ``activation`` ("gelu_tanh" for the ViT MLP, "gelu" = exact erf for the merger) run in
    the matmul epilogue, on the fp32 accumulator.
    """
    kwargs = {"activation": activation, "core_grid": opts.core_grid} if activation else {}
    return ttnn.linear(
        x,
        weight,
        bias=bias,
        compute_kernel_config=opts.compute_kernel_cfg[role],
        memory_config=opts.output_memcfg,
        dtype=opts.output_dtype,
        **kwargs,
    )
