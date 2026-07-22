# tt/tst_io.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

"""
Torch -> TTNN conversion helpers for model inputs. Shared by tst_model.py
(untraced/single-fused-trace generation) and tst_model_cached_additions.py
(per-layer-trace KV-cache generation) -- lives here instead of in either
so neither reaches into the other's private names.
"""

import torch

import ttnn

from .tst_config import D_MODEL


def _apply_layernorm_ttnn(x, ln_weights, orig_dim=D_MODEL):
    return ttnn.layer_norm(x, weight=ln_weights["weight"], bias=ln_weights["bias"])


def _inputs_to_ttnn(
    device, past_values, past_time_features, past_observed_mask, static_categorical_features, static_real_features
):
    pv = ttnn.from_torch(past_values.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    pt = ttnn.from_torch(past_time_features.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    pm = ttnn.from_torch(past_observed_mask.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    sc = ttnn.from_torch(
        static_categorical_features.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    sr = ttnn.from_torch(static_real_features.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return pv, pt, pm, sc, sr


def _future_time_to_ttnn(device, future_time_features):
    return ttnn.from_torch(
        future_time_features.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def _past_values_repeated_to_ttnn(device, past_values_raw):
    return ttnn.from_torch(past_values_raw.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _future_vals_k_to_ttnn(device, future_vals_k):
    return ttnn.from_torch(future_vals_k.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
