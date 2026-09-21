# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefix-slicing helpers for state dicts — how each module receives only its own weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def substate(state: dict[str, "torch.Tensor"], key: str) -> dict[str, "torch.Tensor"]:
    """``{"q_proj.weight": t}`` with ``key="q_proj"`` -> ``{"weight": t}``."""
    prefix = f"{key}."
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def has_substate(state: dict[str, "torch.Tensor"], key: str) -> bool:
    prefix = f"{key}."
    return any(k.startswith(prefix) for k in state)
