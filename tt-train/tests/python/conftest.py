# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for TTML Python tests."""

import os
import pathlib
import sys
from typing import Optional

import pytest

import ttnn
import ttml

# pytest.ini selects --import-mode=importlib, which leaves this directory off sys.path;
# helper modules next to the tests (bf16_ulp) need it there.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


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


def _close_device_mesh_quietly() -> None:
    try:
        ttml.close_device_mesh()
    except Exception:  # noqa: BLE001
        pass


# TTML_SP_OVERLAP=backward runs every module that uses ``tp_mesh`` with the sequence-parallel linears'
# backward scheduled across two command queues (ttml.ops.distributed.set_sp_overlap), on a CCL sub-device
# given by TTML_SP_CCL ("rows=1", the default, or "columns=1", ...). That is how the SP suite is checked
# under the overlap without a second copy of it; test_sp_overlap.py holds the overlap-specific tests.
def sp_overlap_mode_from_env() -> str:
    return os.environ.get("TTML_SP_OVERLAP", "off")


def ccl_sub_device_from_env() -> tuple[int, int]:
    """(columns, rows) of the CCL sub-device requested by TTML_SP_CCL."""
    spec = os.environ.get("TTML_SP_CCL", "rows=1")
    kind, _, count = spec.partition("=")
    if kind not in ("rows", "columns") or not count.isdigit():
        raise ValueError(f"TTML_SP_CCL must be rows=<n> or columns=<n>, got {spec!r}")
    return (int(count), 0) if kind == "columns" else (0, int(count))


def enable_sp_overlap_from_env() -> None:
    """After the mesh is open: split the grid and switch the overlap on when TTML_SP_OVERLAP asks for it."""
    mode = sp_overlap_mode_from_env()
    if mode == "off":
        return
    ctx = ttml.autograd.AutoContext.get_instance()
    if not ctx.has_ccl_sub_device():
        columns, rows = ccl_sub_device_from_env()
        ctx.enable_ccl_sub_device(columns, rows)
    ttml.ops.distributed.set_sp_overlap(mode)


@pytest.fixture(scope="module")
def tp_mesh():
    """A ``[1, 2]`` mesh with axes ``("dp", "tp")``, per requesting module.

    Skips the requesting tests if two devices on the ``"tp"`` axis are unavailable.
    The parallelism context is initialised here too, since the qwen3 model paths
    resolve their TP size through it.
    """
    dp_expected, tp_expected = TP_MESH_SHAPE
    previous_mgd = _ensure_mgd_path(TP_MESH_SHAPE)
    _close_device_mesh_quietly()
    try:
        ttml.open_device_mesh(
            ttml.Mesh(TP_MESH_SHAPE, ("dp", "tp")),
            num_command_queues=2 if sp_overlap_mode_from_env() != "off" else 1,
        )
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
        enable_sp_overlap_from_env()
    except Exception as e:  # noqa: BLE001
        _close_device_mesh_quietly()
        _restore_mgd_path(previous_mgd)
        pytest.skip(f"needs a [{dp_expected}, {tp_expected}] 'tp' mesh: {e}")

    yield ttml.mesh()

    ttml.ops.distributed.set_sp_overlap("off")
    _close_device_mesh_quietly()
    _restore_mgd_path(previous_mgd)
