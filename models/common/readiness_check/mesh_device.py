# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh device open/close helpers for generator readiness runners."""

from __future__ import annotations

import argparse
import os
from typing import Any

#: Label → (mesh rows, mesh cols). N300 (2 chips) is openable as N150 by
#: requesting (1, 1) — the device manager picks one of the two chips. T3K
#: and TG follow the same row-major convention used in the demo fixtures.
MESH_SHAPES: dict[str, tuple[int, int]] = {
    "N150": (1, 1),
    "N300": (1, 2),
    "P300_X2": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
}

FABRIC_CONFIG_CHOICES = ("FABRIC_1D", "FABRIC_1D_RING", "FABRIC_2D")

#: Environment overrides for the three `open_mesh_device` knobs a traced
#: multi-chip model needs but the runner CLIs do not expose. They default to
#: metal's own defaults, so setting none of them reproduces the historical
#: behaviour exactly.
#:
#: `trace_region_size` is the load-bearing one: it defaults to 0, and the
#: teacher-forcing runner requires `generate(..., enable_trace=True)`, so a
#: model whose decode must be traced cannot use these runners without it.
ENV_TRACE_REGION_SIZE = "TT_READINESS_TRACE_REGION_SIZE"
ENV_L1_SMALL_SIZE = "TT_READINESS_L1_SMALL_SIZE"
ENV_FABRIC_PACKET_PAYLOAD_BYTES = "TT_READINESS_FABRIC_PACKET_PAYLOAD_BYTES"


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


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


def open_readiness_mesh_device(
    mesh_device_label: str,
    fabric_config: str | None = None,
    *,
    trace_region_size: int | None = None,
    l1_small_size: int | None = None,
    fabric_packet_payload_bytes: int | None = None,
) -> Any:
    """Open a mesh device, optionally enabling fabric first.

    The three keyword arguments fall back to the `TT_READINESS_*` environment
    variables and then to metal's defaults, so callers that pass nothing (every
    runner CLI) behave exactly as before. Models whose decode path must be traced
    need a non-zero `trace_region_size`; multi-chip models with per-program CCL
    semaphores may also need a specific `l1_small_size` and fabric packet size.
    """
    import ttnn  # noqa: WPS433 — lazy

    shape = MESH_SHAPES.get(mesh_device_label)
    if shape is None:
        raise ValueError(f"Unknown --mesh-device {mesh_device_label!r}. Supported: {sorted(MESH_SHAPES)}.")

    if trace_region_size is None:
        trace_region_size = _env_int(ENV_TRACE_REGION_SIZE)
    if l1_small_size is None:
        l1_small_size = _env_int(ENV_L1_SMALL_SIZE)
    if fabric_packet_payload_bytes is None:
        fabric_packet_payload_bytes = _env_int(ENV_FABRIC_PACKET_PAYLOAD_BYTES)

    num_devices = shape[0] * shape[1]
    if fabric_config is not None:
        if num_devices == 1:
            raise ValueError("--fabric-config is only valid for multi-device meshes")
        fabric = {
            "FABRIC_1D": ttnn.FabricConfig.FABRIC_1D,
            "FABRIC_1D_RING": ttnn.FabricConfig.FABRIC_1D_RING,
            "FABRIC_2D": ttnn.FabricConfig.FABRIC_2D,
        }[fabric_config]
        if fabric_packet_payload_bytes is not None:
            router = ttnn.FabricRouterConfig()
            router.max_packet_payload_size_bytes = fabric_packet_payload_bytes
            ttnn.set_fabric_config(fabric, router_config=router)
        else:
            ttnn.set_fabric_config(fabric)

    open_kwargs: dict[str, Any] = {"mesh_shape": ttnn.MeshShape(*shape)}
    if trace_region_size is not None:
        open_kwargs["trace_region_size"] = trace_region_size
    if l1_small_size is not None:
        open_kwargs["l1_small_size"] = l1_small_size
    return ttnn.open_mesh_device(**open_kwargs)


def close_readiness_mesh_device(mesh_device: Any, fabric_config: str | None = None) -> None:
    """Close a mesh device and reset fabric when it was enabled."""
    import ttnn  # noqa: WPS433 — lazy

    ttnn.close_mesh_device(mesh_device)
    if fabric_config is not None:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
