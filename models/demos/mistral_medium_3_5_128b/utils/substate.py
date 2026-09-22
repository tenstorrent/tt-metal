# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""State-dict prefix helpers. Same two functions every package in ``models/demos`` carries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def substate(state: dict[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    """The sub-dict of ``state`` under ``key.``, with the prefix stripped.

    A ``state`` that defines its own ``substate`` method answers for itself. That one-line hook is
    what makes the 88-layer real-weights load possible: a plain dict of the whole checkpoint is
    ~250 GB of host bf16, and every consumer in this package reaches its weights through this
    function, so intercepting here lets
    :class:`~...reference.checkpoint.CheckpointStateDict` materialize one layer at a time and let
    it fall out of scope again. Plain dicts are unaffected.
    """
    own = getattr(state, "substate", None)
    if own is not None:
        return own(key)
    prefix = f"{key}."
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def has_substate(state: dict[str, torch.Tensor], key: str) -> bool:
    """True if any key in ``state`` lives under ``key.``."""
    prefix = f"{key}."
    return any(k.startswith(prefix) for k in state)
