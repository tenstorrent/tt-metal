# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh device open/close helpers for generator readiness runners."""

from __future__ import annotations

import argparse
from typing import Any

#: Label → (mesh rows, mesh cols). N300 (2 chips) is openable as N150 by
#: requesting (1, 1) — the device manager picks one of the two chips. T3K
#: and TG follow the same row-major convention used in the demo fixtures.
#: P300X2 is a QuietBox 2: two dual-die P300 cards exposed as a 1x4 mesh.
MESH_SHAPES: dict[str, tuple[int, int]] = {
    "N150": (1, 1),
    "N300": (1, 2),
    "P300X2": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
}

FABRIC_CONFIG_CHOICES = ("FABRIC_1D", "FABRIC_1D_RING", "FABRIC_2D")


def add_mesh_device_args(parser: argparse.ArgumentParser) -> None:
    """Register --mesh-device and --fabric-config on a readiness runner parser."""
    parser.add_argument(
        "--mesh-device",
        type=str,
        required=True,
        choices=sorted(MESH_SHAPES.keys()),
        help="Mesh device label. Mapped to a ttnn.MeshShape internally.",
    )
    parser.add_argument(
        "--fabric-config",
        default=None,
        choices=FABRIC_CONFIG_CHOICES,
        help=(
            "Call ttnn.set_fabric_config(...) before open_mesh_device on multi-device "
            "meshes. Omit for single-chip (N150) or when the model does not need fabric."
        ),
    )


def open_readiness_mesh_device(mesh_device_label: str, fabric_config: str | None = None) -> Any:
    """Open a mesh device, optionally enabling fabric first."""
    import ttnn  # noqa: WPS433 — lazy

    shape = MESH_SHAPES.get(mesh_device_label)
    if shape is None:
        raise ValueError(f"Unknown --mesh-device {mesh_device_label!r}. Supported: {sorted(MESH_SHAPES)}.")

    num_devices = shape[0] * shape[1]
    if fabric_config is not None:
        if num_devices == 1:
            raise ValueError("--fabric-config is only valid for multi-device meshes")
        fabric = {
            "FABRIC_1D": ttnn.FabricConfig.FABRIC_1D,
            "FABRIC_1D_RING": ttnn.FabricConfig.FABRIC_1D_RING,
            "FABRIC_2D": ttnn.FabricConfig.FABRIC_2D,
        }[fabric_config]
        ttnn.set_fabric_config(fabric)

    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape))


def close_readiness_mesh_device(mesh_device: Any, fabric_config: str | None = None) -> None:
    """Close a mesh device and reset fabric when it was enabled."""
    import ttnn  # noqa: WPS433 — lazy

    ttnn.close_mesh_device(mesh_device)
    if fabric_config is not None:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
