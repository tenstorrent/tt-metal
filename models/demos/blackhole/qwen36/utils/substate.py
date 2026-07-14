# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for slicing nested state dicts."""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def substate(state: dict[str, "torch.Tensor"], key: str) -> dict[str, "torch.Tensor"]:
    """Return the sub-dict of entries whose keys start with `key.`, with that prefix removed."""
    prefix = f"{key}."
    prefix_len = len(prefix)
    return {k[prefix_len:]: v for k, v in state.items() if k.startswith(prefix)}


def has_substate(state: dict[str, "torch.Tensor"], key: str) -> bool:
    """True if any key starts with `key.`."""
    prefix = f"{key}."
    return any(k.startswith(prefix) for k in state)


def indexed_substates(state: dict[str, "torch.Tensor"], key: str) -> list[dict[str, "torch.Tensor"]]:
    """Extract a list of indexed sub-states (e.g. `key.0`, `key.1`, ...)."""
    result = []
    for i in itertools.count():
        s = substate(state, f"{key}.{i}")
        if not s:
            return result
        result.append(s)
    return []
