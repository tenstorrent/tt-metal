# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-fixture profiles and mesh helpers for the Llama-3.1-8B prefill tests.

Each profile returns a fresh dict, so no two ``pytest.param``s can share mutable fixture state.
"""

from __future__ import annotations

import ttnn


def drop_sp_replicas(composed, rows: int):
    """Keep one SP-axis replica of a tensor composed with ``dims=(0, -1)``.

    The single-shot module tests (MLP, attention, decoder) replicate their input across the SP
    axis and shard only the TP axis, so every SP row recomputes the same thing.
    ``ConcatMesh2dToTensor`` has no "take one replica" mode -- it only concatenates -- so composing
    an ``sp x tp`` mesh stacks ``sp`` identical copies on dim 0, and the reference comparison wants
    exactly one. On a 1xN loudbox mesh this is a no-op; it is the Galaxy's 4x8 that makes it load
    bearing, where the alternative is a reshape of four copies into one copy's shape.
    """
    return composed[:1] if rows > 1 else composed


def fabric_1d_device_params(**overrides) -> dict:
    """Line-fabric profile for the loudbox/quietbox shapes (1x8, 4x2, 2x2)."""
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}
    params.update(overrides)
    return params


def galaxy_torus_xy_device_params(**overrides) -> dict:
    """The only profile a single BH Galaxy will open, and only at all 32 chips.

    Measured on bh-glx-120-c10u08, sweeping descriptor x fabric x shape:

        4x8 / 8x4  FABRIC_2D_TORUS_XY   open + reduce_scatter ok on both axes
        4x2, 2x2   FABRIC_2D_TORUS_XY   cannot open
        1x8, 8x1, 4x2, 2x2  FABRIC_1D   cannot open

    so a Galaxy runs the production 4x8 (SP=4 rows x TP=8 cols) and nothing smaller -- every
    partial shape dies in fabric router sync. The narrower arms belong on a loudbox/quietbox.

    Two things this profile must keep:

    * ``RELAXED_INIT``. The fixture defaults to ``STRICT_INIT``, which fails outright on the
      link-training flakiness a Galaxy has and a loudbox does not.
    * No ``TT_MESH_GRAPH_DESC_PATH``. Auto-discovery is what
      ``deepseek_v3_d_p/tests/fabric_profiles.py`` prescribes for this profile: a torus descriptor
      declares its channel counts ``policy: STRICT``, which prunes any physical pair carrying
      fewer channels than declared, so one degraded link fails the whole map, while
      auto-discovery validates RELAXED and still maps the wrap. Both opened the full mesh here,
      but only auto-discovery is safe on a box that is one link down.

    A ``FABRIC_1D_RING`` request reaches the same fabric by a different road --
    ``get_fabric_type`` maps it to TORUS_XY when the cluster is a UBB galaxy -- but it is legal
    only against a torus descriptor, since a FabricConfig may restrict an MGD's topology and
    never add links. TORUS_XY is asked for directly to keep that out of the picture.
    """
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    params.update(overrides)
    return params
