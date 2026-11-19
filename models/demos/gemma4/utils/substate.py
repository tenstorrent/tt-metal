# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for working with nested state dictionaries.

This module provides helper functions to extract sub-dictionaries from state dicts,
which is useful for loading layer-specific weights and configurations.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class _PlaceholderSubstate(dict):
    """A substate of a warm-cache placeholder state dict (``is_placeholder``): its tensors are dataless too."""

    is_placeholder = True


def substate(state: dict[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    """
    Extract a sub-dictionary from a state dict based on a key prefix.

    Args:
        state: The source state dictionary
        key: The prefix key to filter by (e.g., "q_proj" to get all "q_proj.*" entries)

    Returns:
        A dictionary with the prefix removed from keys. A substate of a warm-cache placeholder state dict is a
        placeholder too: the expert loaders read ``is_placeholder`` to load from the cache instead of converting
        the dataless tensors (fp32 copies of every expert, ~3 GB of host memory per 26B layer).
    """
    prefix = f"{key}."
    prefix_len = len(prefix)

    sub = {k[prefix_len:]: v for k, v in state.items() if k.startswith(prefix)}
    return _PlaceholderSubstate(sub) if getattr(state, "is_placeholder", False) else sub


def has_substate(state: dict[str, torch.Tensor], key: str) -> bool:
    """
    Check if a state dict contains any keys with the given prefix.
    """
    prefix = f"{key}."

    return any(k.startswith(prefix) for k in state)


def indexed_substates(state: dict[str, torch.Tensor], key: str) -> list[dict[str, torch.Tensor]]:
    """
    Extract a list of indexed sub-states (e.g., "layer.0", "layer.1", ...).
    """
    result = []
    for i in itertools.count():
        s = substate(state, f"{key}.{i}")
        if not s:
            return result
        result.append(s)

    return []
