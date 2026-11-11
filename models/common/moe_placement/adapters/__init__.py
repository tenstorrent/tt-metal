# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Adapter registry: model name -> PlacementAdapter.

Supported today (device stages): the deepseek_v3_d_p family, see deepseek_v3_d_p.MODELS.
Any other expert-parallel MoE can use the host stages right away with ``--spec spec.json`` and routing
imported with ``import-routing``; device stages need a ``PlacementAdapter`` subclass (see base.py).

Not candidates: MoE implementations that shard every expert over all chips (tensor-parallel experts,
e.g. 1xN meshes) - every chip does the same share of every expert, so there is no per-chip imbalance.
"""

from __future__ import annotations

from .base import PlacementAdapter


def available() -> dict:
    from .deepseek_v3_d_p import MODELS

    return {
        name: dict(
            family="deepseek_v3_d_p",
            verified=e.verified,
            experts=e.num_experts,
            top_k=e.top_k,
            transport=e.transport,
            layers=e.num_layers,
        )
        for name, e in MODELS.items()
    }


def get_adapter(model: str) -> PlacementAdapter:
    from .deepseek_v3_d_p import MODELS, DeepseekV3DPAdapter

    if model in MODELS:
        return DeepseekV3DPAdapter(model)
    raise KeyError(
        f"no placement adapter for {model!r}; known: {sorted(available())}. Host-only stages work with --spec."
    )
