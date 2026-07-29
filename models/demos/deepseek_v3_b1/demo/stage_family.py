# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Iterable

import ttnn


class StageFamily(str, Enum):
    STAGE_4X2 = "4x2"
    STAGE_4X4 = "4x4"
    STAGE_8X4 = "8x4"


def _shape_tuple(mesh_shape: Iterable[int]) -> tuple[int, int]:
    shape = tuple(int(dim) for dim in mesh_shape)
    if len(shape) != 2:
        raise ValueError(f"Stage family requires a 2D mesh shape, got {shape}")
    return shape


def stage_family_from_shape(mesh_shape: Iterable[int]) -> StageFamily:
    """Classify the global stage mesh shape into a supported stage family."""

    shape = _shape_tuple(mesh_shape)
    if shape == (4, 2):
        return StageFamily.STAGE_4X2
    if shape == (4, 4):
        return StageFamily.STAGE_4X4
    if shape == (8, 4):
        return StageFamily.STAGE_8X4
    raise ValueError(f"Unsupported DeepSeek V3 B1 stage mesh shape: {shape}")


def query_global_stage_mesh_shape() -> ttnn.MeshShape:
    """Return the MGD-derived global stage mesh shape before opening the mesh."""

    return ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()


def fabric_config_for_stage_family(
    stage_family: StageFamily,
    *,
    num_stages: int | None = None,
) -> ttnn.FabricConfig:
    """Derive the fabric mode for known stage-family deployments.

    The 4-stage single-galaxy 4x2 deployment keeps its historical FABRIC_2D
    mode. Larger 4x2 deployments use Y-torus; 4x4 and 8x4 use XY-torus.
    """

    if stage_family == StageFamily.STAGE_4X2:
        if num_stages == 4:
            return ttnn.FabricConfig.FABRIC_2D
        return ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    if stage_family in (StageFamily.STAGE_4X4, StageFamily.STAGE_8X4):
        return ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    raise ValueError(f"Unsupported stage family for fabric config: {stage_family!r}")
