# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Helpers to extract sub-dicts from nested state dicts (layer/weight loading)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def substate(state: dict[str, torch.Tensor] | None, key: str) -> dict[str, torch.Tensor] | None:
    """Extract a sub-dictionary from a state dict based on a key prefix.

    Args:
        state: The source state dictionary, or None (pass-through for cache-only loading).
        key: The prefix key to filter by (e.g., "q_proj" to get all "q_proj.*" entries).

    Returns:
        A dictionary with the prefix removed from keys, or None if state is None.
    """
    if state is None:
        return None
    prefix = f"{key}."
    prefix_len = len(prefix)
    return {k[prefix_len:]: v for k, v in state.items() if k.startswith(prefix)}
