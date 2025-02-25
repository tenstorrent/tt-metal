# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def substate(state: dict[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    prefix = f"{key}."
    prefix_len = len(prefix)

    return {k[prefix_len:]: v for k, v in state.items() if k.startswith(prefix)}


def has_substate(state: dict[str, torch.Tensor], key: str) -> bool:
    prefix = f"{key}."

    return any(k.startswith(prefix) for k in state)


def indexed_substates(state: dict[str, torch.Tensor], key: str) -> list[dict[str, torch.Tensor]]:
    result = []
    for i in itertools.count():
        s = substate(state, f"{key}.{i}")
        if not s:
            return result
        result.append(s)

    return []
