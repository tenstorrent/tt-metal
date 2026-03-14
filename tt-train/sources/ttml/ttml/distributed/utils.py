# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Topology and mesh helper utilities.

Topology repair after collectives is handled in the C++ layer
(ttnn_fixed/distributed/ttnn_ops.cpp) via ``repair_topology``,
``make_replicated_on_axis``, and ``make_sharded_on_axis``.
No Python-side topology rewriting is needed.
"""

from __future__ import annotations

from .layout import Layout, Shard, Replicate


def is_distributed(tensor) -> bool:
    """True if the tensor lives on a mesh device (more than one shard)."""
    try:
        topology = tensor.get_value().tensor_topology()
        dist_shape = list(topology.distribution_shape())
        return any(d > 1 for d in dist_shape)
    except Exception:
        return False
