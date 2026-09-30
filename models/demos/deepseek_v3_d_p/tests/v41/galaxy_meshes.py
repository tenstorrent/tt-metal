# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Galaxy mesh parameters for the V4.1 device tests (bead 8y7.13.4; spec: artifacts galaxy-acceptance.md).

Two layouts of the 32-chip Blackhole Galaxy (SP rows x TP columns): ``torus-xy-8x4`` (production, SP8 x TP4, Ring on
both axes; the id and fabric of ``FABRIC_2D_PREFILL_BLOCK_MESH_PARAMS`` in tests/conftest.py) and ``fabric2d-mesh-4x8``
(capacity fallback, SP4 x TP8). tests/conftest.py permits only FABRIC_1D / FABRIC_2D for a 4x8 mesh on a Galaxy
(its TorusXY descriptor check is 8x4 Ring/Ring), so 4x8 runs unwrapped Fabric2D and ``V41Collectives`` realizes
Linear on both axes. The ``requires_mesh_topology`` marks skip both on any other box (LoudBox: 8 devices).

Galaxy chunks are multiples of 1024 (``V41MeshLayout.check_chunk``: 32 * sp * tp query split), so the small-dims
cases (512 tokens) cannot run there; tests deselect them with ``uncollect_if`` and ``on_galaxy``.
"""

import pytest

from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params

GALAXY_SHAPES = ((8, 4), (4, 8))


def galaxy_meshes(**device_overrides) -> list:
    """``mesh_device, device_params`` params for 8x4 TorusXY and 4x8 Fabric2D; fresh dicts per call."""
    return [
        pytest.param(
            (8, 4),
            torus_xy_device_params(**device_overrides),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
        pytest.param(
            (4, 8),
            fabric2d_device_params(**device_overrides),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 8), topology="mesh-4x8"),
            id="fabric2d-mesh-4x8",
        ),
    ]


def on_galaxy(mesh_device) -> bool:
    """Whether a ``mesh_device`` parameter (collection-time shape tuple) is a Galaxy layout."""
    return tuple(mesh_device) in GALAXY_SHAPES
