# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable, Optional

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


def load_tensors(
    module: object,
    state: Mapping[str, torch.Tensor],
    mapping: Sequence[tuple[str, str]],
    *,
    device,
    dtype: ttnn.DataType,
    strict: bool,
    label: str,
    transform: Optional[Callable[[str, torch.Tensor], torch.Tensor]] = None,
) -> LoadResult:
    """Copy ``state[key]`` onto ``module.<attr>`` (device) and ``module.<attr>_torch`` (host).

    ``mapping`` is a sequence of ``(state_key, attribute_name)`` pairs. Returns the usual
    ``{"missing_keys", "unexpected_keys"}`` report so callers can aggregate across modules.
    """
    used: set[str] = set()
    missing: list[str] = []
    for key, attr in mapping:
        tensor = state.get(key)
        if tensor is None:
            missing.append(key)
            continue
        used.add(key)
        value = to_float_tensor(tensor)
        if transform is not None:
            value = transform(attr, value)
        setattr(module, f"{attr}_torch", value)
        setattr(module, attr, upload(value, device=device, dtype=dtype))
    unexpected = sorted(key for key in state if key not in used)
    if strict and missing:
        raise ValueError(f"Missing {label} weights: {missing}")
    return {"missing_keys": missing, "unexpected_keys": unexpected}


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
    "load_tensors",
    "merge_results",
    "pick_roots",
    "substate",
    "to_float_tensor",
    "upload",
]
