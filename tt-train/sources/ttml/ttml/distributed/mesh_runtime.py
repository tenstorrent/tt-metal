# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Mesh runtime context -- replaces ParallelismContext for the new dispatch layer.

MeshRuntime holds the mesh device reference, axis assignments for TP/DP/CP,
and a reference to the plan cache.  It is the single source of truth that
dispatch and redistribute consult at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .cache import PlanCache


@dataclass
class MeshRuntime:
    """Runtime context for distributed dispatch.

    Args:
        mesh_device: The ttnn MeshDevice to execute on.
        tp_axis: Mesh axis used for tensor parallelism (None = disabled).
        dp_axis: Mesh axis used for data parallelism (None = disabled).
        cp_axis: Mesh axis used for context parallelism (None = disabled).
    """

    mesh_device: object
    tp_axis: Optional[int] = None
    dp_axis: Optional[int] = None
    cp_axis: Optional[int] = None
    plan_cache: PlanCache = field(default_factory=PlanCache)

    @property
    def mesh_shape(self):
        return self.mesh_device.shape

    @property
    def tp_size(self) -> int:
        if self.tp_axis is None:
            return 1
        return int(self.mesh_shape[self.tp_axis])

    @property
    def dp_size(self) -> int:
        if self.dp_axis is None:
            return 1
        return int(self.mesh_shape[self.dp_axis])

    @property
    def cp_size(self) -> int:
        if self.cp_axis is None:
            return 1
        return int(self.mesh_shape[self.cp_axis])

    @property
    def num_devices(self) -> int:
        return int(self.mesh_device.get_num_devices())

    @property
    def is_tp_enabled(self) -> bool:
        return self.tp_axis is not None and self.tp_size > 1

    @property
    def is_dp_enabled(self) -> bool:
        return self.dp_axis is not None and self.dp_size > 1

    @property
    def is_cp_enabled(self) -> bool:
        return self.cp_axis is not None and self.cp_size > 1


# ---------------------------------------------------------------------------
# Global runtime singleton
# ---------------------------------------------------------------------------

_RUNTIME: Optional[MeshRuntime] = None


def get_runtime() -> Optional[MeshRuntime]:
    """Return the active MeshRuntime, or None if not initialized."""
    return _RUNTIME


def set_runtime(runtime: Optional[MeshRuntime]) -> None:
    """Set the active MeshRuntime (or clear it with None)."""
    global _RUNTIME
    _RUNTIME = runtime
