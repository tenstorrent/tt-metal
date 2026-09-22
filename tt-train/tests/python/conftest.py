# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for TTML Python tests."""

import math
import os
import pathlib
from typing import Optional, Sequence

import pytest

import ttnn
import ttml


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_device: mark test as requiring a Tenstorrent device to run",
    )


def pytest_collection_modifyitems(config, items):
    """Skip device-requiring tests if no device is available."""
    device_available = (
        any(pathlib.Path("/dev/tenstorrent/").iterdir()) if pathlib.Path("/dev/tenstorrent/").exists() else False
    )

    if not device_available:
        skip_device = pytest.mark.skip(reason="Tenstorrent device not available")
        for item in items:
            if "requires_device" in item.keywords:
                item.add_marker(skip_device)


# ---------------------------------------------------------------------------
# Host capability checks
# ---------------------------------------------------------------------------
#
# A system that is too small for the mesh a test wants should skip it. A system
# that has the devices but still fails to open the mesh must fail.


def _num_available_devices() -> Optional[int]:
    """Chips in the global system mesh (all hosts), or ``None`` if it can't be queried."""
    try:
        return math.prod(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    except Exception:  # noqa: BLE001
        return None


def _host_supports_mesh(shape: Sequence[int]) -> bool:
    """Checks whether the system has enough chips for ``shape``."""
    available = _num_available_devices()
    return available is None or available >= math.prod(shape)


def _skip_if_host_too_small(shape: Sequence[int], what: str) -> None:
    """Skip when the system is too small for ``shape``, otherwise return normally."""

    if _host_supports_mesh(shape):
        return
    pytest.skip(
        f"{what} needs a {tuple(shape)} mesh with ({math.prod(shape)} devices); the system has {_num_available_devices()}"
    )


@pytest.fixture(scope="session")
def skip_if_host_too_small():
    """``skip_if_host_too_small(shape, what)`` -- skip unless this host has the chips for ``shape``.

    Session-scoped so fixtures of any scope can request it.
    """
    return _skip_if_host_too_small


# ---------------------------------------------------------------------------
# Shared [1, 2] tensor-parallel mesh
# ---------------------------------------------------------------------------
#
# One definition for every module that wants this shape, instead of the copy that used
# to sit in each of them.
#
# Deliberately module-scoped, not session-scoped. Holding the mesh open for the whole
# session is much faster -- closing it invalidates the JIT cache, so a reopen costs a
# full recompile (measured at 23 minutes for two modules that take ~20 seconds warm) --
# but it also means the mesh outlives the module that asked for it, and sibling modules
# that open a plain single device via ``AutoContext.open_device`` then fail with
# "open_device was called after the device was created". Correctness first: release the
# mesh at module teardown and pay the reopen.
#
# Making this session-scoped requires first teaching the single-device fixtures to
# cooperate with an already-open mesh.

TP_MESH_SHAPE = (1, 2)

_TTML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MGD_FOR_ARCH_AND_SHAPE = {
    ("blackhole", TP_MESH_SHAPE): os.path.join(_TTML_ROOT, "configs", "mgd", "bh_galaxy_1_2_line_line.textproto"),
    ("wormhole_b0", TP_MESH_SHAPE): os.path.join(_TTML_ROOT, "configs", "mgd", "n300_1_2_line_line.textproto"),
}


def _detect_arch() -> Optional[str]:
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


@pytest.fixture(scope="session")
def reset_metal_env_quietly():
    """``reset_metal_env_quietly()`` -- ``ttml.reset_metal_env()``, ignoring errors so cleanup
    never hides the failure that triggered it.

    A fixture because test modules can't import from this file: ``--import-mode=importlib``
    loads it as ``tt-train.tests.python.conftest``, so ``from conftest import ...`` would
    load a second copy. Session-scoped so fixtures of any scope can request it.
    """

    def reset() -> None:
        try:
            ttml.reset_metal_env()
        except Exception:  # noqa: BLE001
            pass

    return reset


@pytest.fixture(scope="module")
def tp_mesh(reset_metal_env_quietly):
    """A ``[1, 2]`` mesh with axes ``("dp", "tp")``, per requesting module.

    Skips the requesting tests on a host with too few devices for the shape. A host
    that has the devices but fails to open the mesh is a real failure and therefore,
    not skipped. The parallelism context is initialised here too, since the qwen3
    model paths resolve their TP size through it.
    """
    dp_expected, tp_expected = TP_MESH_SHAPE
    _skip_if_host_too_small(TP_MESH_SHAPE, "tensor-parallel tests")
    previous_mgd = _ensure_mgd_path(TP_MESH_SHAPE)
    # The host-size check above already created the process-wide MetalEnv, and a MetalEnv
    # reads TT_MESH_GRAPH_DESC_PATH only once, when it is created. Drop it now that
    # _ensure_mgd_path has set the descriptor, so the open below builds a new one from it;
    # otherwise the fabric control plane is built from the wrong descriptor.
    ttml.reset_metal_env()
    try:
        ttml.open_device_mesh(ttml.Mesh(TP_MESH_SHAPE, ("dp", "tp")))
        ctx = ttml.autograd.AutoContext.get_instance()
        if ctx.is_parallelism_context_initialized():
            # ParallelismContext is a one-shot singleton with no reset hook, so an
            # earlier module's may still be installed. It is only usable here if it
            # describes this same shape -- reusing a mismatched one (a dp-enabled GRPO
            # context, say) silently shards the model for the wrong device count
            # instead of failing.
            pctx = ctx.get_parallelism_context()
            actual = (pctx.get_ddp_size(), pctx.get_tp_size())
            if actual != (dp_expected, tp_expected):
                raise RuntimeError(
                    f"this process already installed a ParallelismContext for DP={actual[0]}, "
                    f"TP={actual[1]} and it cannot be reset; needed DP={dp_expected}, TP={tp_expected}"
                )
        else:
            ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=False, enable_tp=True))
    except Exception:  # noqa: BLE001
        reset_metal_env_quietly()
        _restore_mgd_path(previous_mgd)
        raise

    yield ttml.mesh()

    reset_metal_env_quietly()
    _restore_mgd_path(previous_mgd)
