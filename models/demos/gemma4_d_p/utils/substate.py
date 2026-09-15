# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for working with nested state dictionaries.

This module provides helper functions to extract sub-dictionaries from state dicts,
which is useful for loading layer-specific weights and configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def substate(state: dict[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    """
    Extract a sub-dictionary from a state dict based on a key prefix.

    Args:
        state: The source state dictionary
        key: The prefix key to filter by (e.g., "q_proj" to get all "q_proj.*" entries)

    Returns:
        A dictionary with the prefix removed from keys
    """
    prefix = f"{key}."
    prefix_len = len(prefix)

    return {k[prefix_len:]: v for k, v in state.items() if k.startswith(prefix)}
