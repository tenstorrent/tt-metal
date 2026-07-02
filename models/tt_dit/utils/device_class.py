# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Model-agnostic Tenstorrent board / device-class taxonomy.

`DeviceClass` is a shared primitive: it names the boards and clusters tt_dit
runs on, with no knowledge of which models are supported on which board. That
model-support matrix (e.g. BOARD_SPECS / per-model deployment shapes) belongs
next to the model or the serving layer that consumes this enum, not here.

The taxonomy mirrors tt-inference-server/workflows/workflow_types.py:DeviceTypes
so that `DeviceClass.from_string("p300")` resolves the same names a user would
pass on the inference-server CLI.
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import FrozenSet


class DeviceClass(IntEnum):
    """Tenstorrent board / cluster identity.

    Mirrors tt-inference-server/workflows/workflow_types.py:DeviceTypes so
    `DeviceClass.from_string("p300")` resolves the same names a user would
    pass on the inference-server CLI.
    """

    # Wormhole_B0
    N150 = auto()
    N300 = auto()
    T3K = auto()
    GALAXY_T3K = auto()
    GALAXY = auto()
    DUAL_GALAXY = auto()
    QUAD_GALAXY = auto()

    # Blackhole
    P100 = auto()
    P150 = auto()
    P150X2 = auto()
    P150X4 = auto()
    P150X8 = auto()
    P300 = auto()
    P300X2 = auto()

    @classmethod
    def from_string(cls, name: str) -> "DeviceClass":
        """Parse 'p300', 'P300', 'p300x2', 'P300X2' etc."""
        try:
            return cls[name.upper()]
        except KeyError:
            valid = ", ".join(m.name.lower() for m in cls)
            raise ValueError(f"Unknown device class '{name}'. Valid: {valid}")


GALAXY_BOARDS: FrozenSet[DeviceClass] = frozenset(
    {DeviceClass.GALAXY, DeviceClass.GALAXY_T3K, DeviceClass.DUAL_GALAXY, DeviceClass.QUAD_GALAXY}
)


def is_galaxy(board: DeviceClass) -> bool:
    """True if `board` is a Galaxy-class cluster."""
    return board in GALAXY_BOARDS
