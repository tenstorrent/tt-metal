# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's KDA state geometry, built in one place.

The engine's slabs, the model's scratch carry and the migration readers all address the same bytes,
so they must agree on it; building it here is what makes that a fact rather than a coincidence.
"""

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry


def kimi_k3_kda_geometry(mesh_shape, sp_axis: int, tp_axis: int) -> KdaContractGeometry:
    return KdaContractGeometry.from_kda_config(
        kimi_k3_kda_config(), mesh_shape=tuple(mesh_shape), sp_axis=sp_axis, tp_axis=tp_axis
    )
