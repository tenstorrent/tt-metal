# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import torch

import ttnn

LoadResult = dict[str, list[str]]


def to_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(torch.float32)


def upload(tensor: torch.Tensor, *, device, dtype: ttnn.DataType) -> ttnn.Tensor:
    return ttnn.from_torch(tensor.contiguous(), device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def substate(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not prefix:
        return dict(state)
    head = f"{prefix}."
    return {key[len(head) :]: value for key, value in state.items() if key.startswith(head)}


def pick_roots(state: Mapping[str, torch.Tensor], roots: Sequence[str]) -> dict[str, torch.Tensor]:
    """Keep only entries whose first path component is in ``roots``."""
    allowed = set(roots)
    return {key: value for key, value in state.items() if key.split(".", 1)[0] in allowed}


def merge_results(
    results: Sequence[tuple[str, LoadResult]],
    *,
    state: Optional[Mapping[str, torch.Tensor]] = None,
    claimed: Optional[Sequence[str]] = None,
) -> LoadResult:
    """Re-prefix and merge per-module load reports.

    Children are handed disjoint slices of ``state``, so pass ``state`` and the set of
    ``claimed`` root names to have the parent report anything no child was offered. Without
    them, only the children's own reports are merged.
    """
    missing: list[str] = []
    unexpected: list[str] = []
    for prefix, result in results:
        head = f"{prefix}." if prefix else ""
        missing.extend(f"{head}{key}" for key in result["missing_keys"])
        unexpected.extend(f"{head}{key}" for key in result["unexpected_keys"])
    if state is not None and claimed is not None:
        allowed = set(claimed)
        unexpected.extend(key for key in state if key.split(".", 1)[0] not in allowed)
    return {"missing_keys": missing, "unexpected_keys": sorted(set(unexpected))}


__all__ = [
    "LoadResult",
    "merge_results",
    "pick_roots",
    "substate",
    "to_float_tensor",
    "upload",
]
