# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compute-grid classes that select the S512 configs.

A Blackhole p150a exposes 13 compute columns (13x10). One chip of a Blackhole
Galaxy exposes 12 (12x10). The Galaxy S512 paths (model-local SDPA and
LayerNorm, L1 embeddings, the Galaxy chunk plan) apply only below
P150_GRID_COLUMNS. A 13-column grid keeps the p150a configs.
"""

P150_GRID_COLUMNS = 13


def is_galaxy_grid(mesh_device) -> bool:
    """True when the device grid is narrower than a p150a (for example Galaxy)."""
    return int(mesh_device.compute_with_storage_grid_size().x) < P150_GRID_COLUMNS
