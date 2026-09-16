# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Semantic layer ids for a contiguous Gemma prefill slice."""


def prefill_layer_ids(num_layers: int, first_layer_idx: int, total_layers: int) -> tuple[int, ...]:
    if num_layers <= 0 or first_layer_idx < 0 or first_layer_idx + num_layers > total_layers:
        raise ValueError(
            f"prefill layer range [{first_layer_idx}, {first_layer_idx + num_layers}) "
            f"must be nonempty and within [0, {total_layers})"
        )
    return tuple(range(first_layer_idx, first_layer_idx + num_layers))
