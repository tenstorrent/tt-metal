# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chain all 5 DFlash drafter layers. Context and RoPE tables are fixed across layers;
only the noise/draft-block representation threads from one layer to the next. Each
layer's mask depends only on (is_causal, sliding_window) -- built once per distinct
combination and reused (models/demos/gemma4/docs/dflash_design.md section 1: layers 1-4
are sliding/causal, layer 5 is full/bidirectional)."""

from __future__ import annotations

import ttnn
from models.demos.gemma4.tt.dflash.attention import (
    build_attention_mask_additive_device,
    build_attention_mask_additive_device_dynamic,
    combine_attention_mask_dynamic,
)
from models.demos.gemma4.tt.dflash.layer import dflash_layer_forward
from models.demos.gemma4.tt.dflash.weights import Gemma4DFlashWeights


def dflash_drafter_forward(
    context: ttnn.Tensor,
    noise: ttnn.Tensor,
    weights: Gemma4DFlashWeights,
    cos_full: ttnn.Tensor,
    sin_full: ttnn.Tensor,
    mesh_device,
    mesh_config,
    ccl_manager,
    num_local_heads: int,
    num_local_kv_heads: int,
    head_dim: int,
    eps: float,
    layer_configs: list[tuple[bool, int | None]],  # (is_causal, sliding_window) per layer
    context_valid_len_tt: ttnn.Tensor | None = None,
    mask_static_parts: dict[tuple[bool, int | None], "DynamicMaskStaticParts"] | None = None,
) -> ttnn.Tensor:
    """``context_valid_len_tt``: when given (a ``[1,1]`` int32 device tensor), ``context``
    is treated as a FIXED-size window (its own shape, e.g. the drafter's block_size) whose
    first ``context_valid_len_tt`` rows are real and the rest are masked-out padding --
    see attention.py's ``build_attention_mask_additive_device_dynamic``. This is the
    steady-state (every generation iteration after the first) shape a Metal trace needs,
    since the REAL number of valid context rows varies iteration to iteration but a
    trace's tensor shapes cannot. When ``None`` (the default, used for the first
    iteration's real, variably-sized prefill context), the ordinary static mask is used
    instead, exactly as before.

    ``mask_static_parts``: when given alongside ``context_valid_len_tt`` (one
    ``DynamicMaskStaticParts`` per distinct ``(is_causal, sliding_window)`` pair in
    ``layer_configs``, from ``attention.build_attention_mask_static_parts``), masks are
    recombined via ``combine_attention_mask_dynamic`` -- pure elementwise ops on
    already-built tensors, safe inside a captured Metal trace -- instead of being rebuilt
    from scratch (``ttnn.arange``/``ones``/``zeros``/``full``, each a host->device write)
    every call, which a trace capture rejects (``TT_FATAL: Writes are not supported during
    trace capture``). Required for ``generate.py``'s ``_traced_steady_state``; ordinary
    (non-traced) callers can omit it."""
    ctx_len = context.shape[-2]
    q_len = noise.shape[-2]

    mask_cache: dict[tuple[bool, int | None], ttnn.Tensor] = {}

    def mask_for(is_causal, sliding_window):
        key = (is_causal, sliding_window)
        if key not in mask_cache:
            if mask_static_parts is not None:
                mask_cache[key] = combine_attention_mask_dynamic(mask_static_parts[key], context_valid_len_tt)
            elif context_valid_len_tt is not None:
                mask_cache[key] = build_attention_mask_additive_device_dynamic(
                    mesh_device, ctx_len, q_len, is_causal, sliding_window, context_valid_len_tt
                )
            else:
                mask_cache[key] = build_attention_mask_additive_device(
                    mesh_device, ctx_len, q_len, is_causal, sliding_window
                )
        return mask_cache[key]

    for layer_weights, (is_causal, sliding_window) in zip(weights.layers, layer_configs):
        mask = mask_for(is_causal, sliding_window)
        noise = dflash_layer_forward(
            context,
            noise,
            layer_weights,
            cos_full,
            sin_full,
            mask,
            mesh_config,
            ccl_manager,
            num_local_heads,
            num_local_kv_heads,
            head_dim,
            eps,
        )
    return noise
