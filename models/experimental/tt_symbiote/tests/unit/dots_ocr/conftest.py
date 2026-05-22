# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for dots.ocr bottom-up unit tests.

Provides a ``mesh_device_t3k_dp`` fixture that opens the same ``(8, 1)``
DP-on-T3K mesh used by ``test_dots_ocr.py`` so op-level tests can reuse
captured per-device shapes verbatim.

This conftest **wraps** the global ``mesh_device`` / ``device_params`` indirect
fixtures from the repo-root ``conftest.py`` (see
``/home/ttuser/salnahari/tt-metal/conftest.py:527`` and
``:301``); we only inject the parameter values.

The capture pass used ``trace_region_size=300_000_000`` and
``FABRIC_1D_RING`` because the production pipeline traces; these unit tests
do NOT trace, so we use a far smaller ``trace_region_size=1_000_000`` to
keep host-side memory pressure low. Fabric must stay enabled because the
mesh has multiple devices.
"""

from __future__ import annotations

import os

import pytest
import ttnn


# Map mirrors ``test_dots_ocr.py:18`` / ``:31`` so a contributor can run
# the same MESH_DEVICE env var and get the same mesh.
_MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}

_DOTS_OCR_DP_MESH_DEVICE_MAP = {
    "N300": (2, 1),
    "T3K": (8, 1),
}


def _resolve_mesh_device_shape():
    """Resolve mesh shape from ``MESH_DEVICE`` env var (DP-aware).

    Mirrors ``test_dots_ocr._resolve_mesh_device_shape`` so unit tests pick
    the same mesh the capture used.
    """
    mesh_device = os.environ.get("MESH_DEVICE")
    if os.environ.get("DOTS_OCR_PARALLELISM", "").upper() == "DP":
        return _DOTS_OCR_DP_MESH_DEVICE_MAP.get(
            mesh_device, _MESH_DEVICE_MAP.get(mesh_device, len(ttnn.get_device_ids()))
        )
    return _MESH_DEVICE_MAP.get(mesh_device, len(ttnn.get_device_ids()))


def _dots_ocr_mesh_num_devices() -> int:
    sh = _resolve_mesh_device_shape()
    if isinstance(sh, int):
        return max(1, int(sh))
    if isinstance(sh, (tuple, list)):
        if len(sh) >= 2:
            return int(sh[0]) * int(sh[1])
        if len(sh) == 1:
            return int(sh[0])
    return 1


def _dots_ocr_unit_test_device_params() -> dict:
    """Per Phase 1 plan: small trace region, fabric on for multi-device meshes."""
    dp = {"trace_region_size": 1_000_000, "num_command_queues": 1}
    if _dots_ocr_mesh_num_devices() > 1:
        dp["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
    else:
        dp["fabric_config"] = ttnn.FabricConfig.DISABLED
    return dp


@pytest.fixture(scope="function")
def mesh_device_t3k_dp(request):
    """A ``(8, 1)`` DP mesh on T3K (or whatever ``MESH_DEVICE`` resolves to).

    This fixture is implemented as a wrapper around the upstream ``mesh_device``
    indirect fixture by re-invoking it with the unit-test ``device_params``
    and the resolved mesh shape. Tests should request this fixture by name —
    they do NOT need their own ``@pytest.mark.parametrize`` for ``mesh_device``
    / ``device_params``.
    """
    # We reuse the global mesh_device fixture by directly invoking its
    # implementation. The simplest, most robust approach is to install the
    # indirect params on the request and ask pytest for the wrapped fixture.
    # Because indirect-parametrize must happen at collection time, we
    # instead replicate the global fixture body locally — the body is small
    # and stable (see tt-metal/conftest.py:527).
    import ttnn as _ttnn
    from tests.scripts.common import get_updated_device_params

    # Defer the optional helper used by upstream — not needed when no trace
    # marker is set.
    request.node.pci_ids = _ttnn.get_pcie_device_ids()

    shape = _resolve_mesh_device_shape()
    if isinstance(shape, tuple):
        mesh_shape = _ttnn.MeshShape(*shape)
        num_requested = shape[0] * shape[1]
    else:
        mesh_shape = _ttnn.MeshShape(1, int(shape))
        num_requested = int(shape)

    if not _ttnn.using_distributed_env() and num_requested > _ttnn.get_num_devices():
        pytest.skip(f"Requested {num_requested} devices but only {_ttnn.get_num_devices()} available.")

    device_params = _dots_ocr_unit_test_device_params()
    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    fabric_manager = updated_device_params.pop("fabric_manager", None)
    fabric_router_config = updated_device_params.pop("fabric_router_config", None)

    # Re-use the same set_fabric / reset_fabric helpers the global fixture uses.
    # They're defined at the top of /home/ttuser/salnahari/tt-metal/conftest.py.
    # Import locally so this file is still importable on machines without the
    # tt-metal repo root on the path.
    from conftest import set_fabric, reset_fabric  # type: ignore  # noqa: E402

    set_fabric(
        fabric_config,
        reliability_mode,
        fabric_tensix_config,
        fabric_manager,
        fabric_router_config,
    )

    mesh_device = _ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        _ttnn.close_mesh_device(submesh)
    _ttnn.close_mesh_device(mesh_device)
    reset_fabric(fabric_config)
    del mesh_device
