# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for TTML Python tests."""

import os
from typing import Optional

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_device: mark test as requiring a Tenstorrent device to run",
    )


def pytest_collection_modifyitems(config, items):
    """Skip device-requiring tests if no device is available."""
    import pathlib

    device_available = (
        any(pathlib.Path("/dev/tenstorrent/").iterdir()) if pathlib.Path("/dev/tenstorrent/").exists() else False
    )

    if not device_available:
        skip_device = pytest.mark.skip(reason="Tenstorrent device not available")
        for item in items:
            if "requires_device" in item.keywords:
                item.add_marker(skip_device)


# ---------------------------------------------------------------------------
# Shared [1, 2] tensor-parallel mesh
# ---------------------------------------------------------------------------
#
# Session-scoped on purpose. Opening a mesh costs a fabric bring-up, and closing one
# invalidates the JIT kernel cache, so a per-module fixture makes every module after
# the first pay a full recompile -- measured at 23 minutes for two modules that take
# ~20 seconds when the mesh stays warm.
#
# Opening once per session solves both. Modules that need this shape request the
# ``tp_mesh`` fixture instead of managing a mesh themselves.

TP_MESH_SHAPE = (1, 2)

_TTML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MGD_FOR_ARCH_AND_SHAPE = {
    ("blackhole", TP_MESH_SHAPE): os.path.join(_TTML_ROOT, "configs", "mgd", "bh_galaxy_1_2_line_line.textproto"),
    ("wormhole_b0", TP_MESH_SHAPE): os.path.join(_TTML_ROOT, "configs", "mgd", "n300_1_2_line_line.textproto"),
}


def _detect_arch() -> Optional[str]:
    import ttnn

    try:
        name = ttnn.get_arch_name().lower()
    except Exception:  # noqa: BLE001
        return None
    if "blackhole" in name:
        return "blackhole"
    if "wormhole_b0" in name:
        return "wormhole_b0"
    return None


def _ensure_mgd_path(shape) -> Optional[str]:
    """Point TT_MESH_GRAPH_DESC_PATH at a bundled descriptor unless the caller set one."""
    previous = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    if previous:
        return previous
    arch = _detect_arch()
    if arch is None:
        return previous
    candidate = _MGD_FOR_ARCH_AND_SHAPE.get((arch, shape))
    if candidate and os.path.isfile(candidate):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = candidate
    return previous


def _restore_mgd_path(previous: Optional[str]) -> None:
    if previous is None:
        os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
    else:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = previous


def _close_device_mesh_quietly() -> None:
    import ttml

    try:
        ttml.close_device_mesh()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(scope="session")
def tp_mesh():
    """A ``[1, 2]`` mesh with axes ``("dp", "tp")``, opened once per session.

    Skips the requesting tests if two devices on the ``"tp"`` axis are unavailable.
    The parallelism context is initialised here too, since the qwen3 model paths
    resolve their TP size through it.
    """
    import ttml

    previous_mgd = _ensure_mgd_path(TP_MESH_SHAPE)
    _close_device_mesh_quietly()
    try:
        ttml.open_device_mesh(ttml.Mesh(TP_MESH_SHAPE, ("dp", "tp")))
        ctx = ttml.autograd.AutoContext.get_instance()
        if not ctx.is_parallelism_context_initialized():
            ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=False, enable_tp=True))
    except Exception as e:  # noqa: BLE001
        _restore_mgd_path(previous_mgd)
        pytest.skip(f"needs {TP_MESH_SHAPE[1]} devices on the 'tp' axis: {e}")

    yield ttml.mesh()

    _close_device_mesh_quietly()
    _restore_mgd_path(previous_mgd)
