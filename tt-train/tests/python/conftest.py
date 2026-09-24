# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for TTML Python tests."""

import contextlib
import math
import os
import pathlib
from typing import Iterator, Optional, Sequence

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


# ---------------------------------------------------------------------------
# Mesh graph descriptors
# ---------------------------------------------------------------------------
#
# A bundled descriptor is only filled in when TT_MESH_GRAPH_DESC_PATH is unset, so a
# user-provided value always wins.

_MGD_ENV = "TT_MESH_GRAPH_DESC_PATH"
_TTML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MGD_DIR = os.path.join(_TTML_ROOT, "configs", "mgd")
_BUNDLED_MGD = {
    ("blackhole", (1, 2)): "bh_galaxy_1_2_line_line.textproto",
    ("blackhole", (2, 2)): "bh_galaxy_2_2_line_line.textproto",
    # The galaxy fabric is a torus in X; a LINE/LINE descriptor faults with SIGBUS.
    ("blackhole", (8, 4)): "bh_galaxy_8_4_torus_x.textproto",
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


def _bundled_mgd(shape: Sequence[int]) -> Optional[str]:
    """The bundled descriptor for the host arch and ``shape``, or ``None`` if there isn't one."""
    name = _BUNDLED_MGD.get((_detect_arch(), tuple(shape)))
    if name is None:
        return None
    path = os.path.join(_MGD_DIR, name)
    return path if os.path.isfile(path) else None


def _restore_mgd_path(previous: Optional[str]) -> None:
    if previous is None:
        os.environ.pop(_MGD_ENV, None)
    else:
        os.environ[_MGD_ENV] = previous


# ---------------------------------------------------------------------------
# Fresh device mesh
# ---------------------------------------------------------------------------


def _reset_metal_env_quietly() -> None:
    try:
        ttml.reset_metal_env()
    except Exception:  # noqa: BLE001
        pass


@contextlib.contextmanager
def _fresh_device_mesh(
    shape: Sequence[int],
    axis_names: Optional[Sequence[str]] = None,
    *,
    what: str,
    require_mgd: bool = False,
) -> Iterator["ttml.Mesh"]:
    """Open a ``shape`` mesh on a new ``MetalEnv``, which is reset on exit along with the MGD path.

    Skips when the host has too few devices for ``shape``, or, with ``require_mgd``, when
    the host arch is known but there is neither a bundled descriptor for it nor a
    user-provided one. Any other failure to open the mesh will raise an exception.

    A failure inside the ``with`` body (or while opening) resets the ``MetalEnv`` quietly
    and re-raises, so a cleanup error can't replace the original failure as the reported
    error. On a normal exit, a failure to reset is raised rather than ignored, because the
    failure usually means live device references blocked ``ReleaseOwnership``, and the
    stale ``MetalEnv`` left behind would break later modules.
    """
    _skip_if_host_too_small(shape, what)
    previous_mgd = os.environ.get(_MGD_ENV)
    mgd = _bundled_mgd(shape)
    arch = _detect_arch()
    if require_mgd and not previous_mgd and mgd is None and arch is not None:
        pytest.skip(
            f"{what} need a mesh graph descriptor for arch={arch!r} shape={tuple(shape)}; "
            f"add one under tt-train/configs/mgd/ or export {_MGD_ENV}"
        )
    if mgd and not previous_mgd:
        os.environ[_MGD_ENV] = mgd
    try:
        try:
            # A MetalEnv reads TT_MESH_GRAPH_DESC_PATH only when it is created, and the
            # host-size check above has already created one.
            ttml.reset_metal_env()
            ttml.open_device_mesh(ttml.Mesh(tuple(shape), tuple(axis_names)) if axis_names else tuple(shape))
            yield ttml.mesh()
        except BaseException:
            _reset_metal_env_quietly()
            raise
        ttml.reset_metal_env()
    finally:
        _restore_mgd_path(previous_mgd)


@pytest.fixture(scope="session")
def fresh_device_mesh():
    """``with fresh_device_mesh(shape, axis_names, what=...) as mesh:`` -- see ``_fresh_device_mesh``.

    Session-scoped so fixtures of any scope can request it.
    """
    return _fresh_device_mesh


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


@pytest.fixture(scope="module")
def tp_mesh():
    """A ``[1, 2]`` mesh with axes ``("dp", "tp")``, per requesting module.

    Skips the requesting tests on a host with too few devices for the shape. A host
    that has the devices but fails to open the mesh is a real failure and therefore,
    not skipped. The parallelism context is initialised here too, since the qwen3
    model paths resolve their TP size through it.
    """
    dp_expected, tp_expected = TP_MESH_SHAPE
    with _fresh_device_mesh(TP_MESH_SHAPE, ("dp", "tp"), what="tensor-parallel tests") as mesh:
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
        yield mesh
