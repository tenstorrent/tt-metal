# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

    import torch


def substate(state: Mapping[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    prefix = f"{key}."
    prefix_len = len(prefix)

    return {k[prefix_len:]: v for k, v in state.items() if k.startswith(prefix)}


def has_substate(state: Mapping[str, torch.Tensor], key: str) -> bool:
    prefix = f"{key}."

    return any(k.startswith(prefix) for k in state)


def pop_substate(state: MutableMapping[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    prefix = f"{key}."
    return {k.removeprefix(prefix): state.pop(k) for k in list(state) if k.startswith(prefix)}


def indexed_substates(state: Mapping[str, torch.Tensor], key: str) -> list[dict[str, torch.Tensor]]:
    result = []
    for i in itertools.count():
        s = substate(state, f"{key}.{i}")
        if not s:
            return result
        result.append(s)

    return []
