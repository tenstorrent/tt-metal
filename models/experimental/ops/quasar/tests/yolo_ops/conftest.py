# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Local conftest for the YOLOv8 per-op suite: tags emulator-appropriate cases.

Same mechanism as tests/ops/conftest.py — an ``emulator`` marker inferred from each
case's actual parametrized values, so:

    pytest models/experimental/ops/quasar/tests/yolo_ops/ -m emulator

selects the subset small enough for the 2-node emulator. For YOLO most feature maps
are large; a case is emulator-appropriate only when its spatial size / element count
is small (see op_utils.EMU_MAX_HW / EMU_MAX_ELEMS) and it targets a single (1,1) mesh.

Classification (a case is NOT emulator when any of):
  * it targets a non-(1,1) mesh,
  * an ``hw`` / ``h`` / ``w`` / ``res`` int param exceeds EMU_MAX_HW,
  * a ``shape`` / ``in_shape`` / ``out_shape`` tuple's element count exceeds EMU_MAX_ELEMS.
"""

from models.experimental.ops.quasar.tests.yolo_ops.op_utils import EMU_MAX_ELEMS, EMU_MAX_HW

_HW_PARAMS = ("hw", "h", "w", "height", "width", "res", "resolution", "in_h", "in_w")
_SHAPE_PARAMS = ("shape", "in_shape", "out_shape")


def _elems(shape) -> int:
    n = 1
    for d in shape:
        if isinstance(d, int):
            n *= d
    return n


def _fits_emulator(item) -> bool:
    # Model-faithful tests reproduce the model's real sharded/TILE/L1 state (often on
    # 25-80 core grids) — Blackhole-scale, never the 2-node emulator subset.
    if item.get_closest_marker("blackhole_scale") is not None:
        return False
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return True
    params = callspec.params
    mesh = params.get("ttnn_mesh_device")
    if isinstance(mesh, dict):  # {"mesh_shape": (..), "l1_small_size": ..} form
        mesh = mesh.get("mesh_shape")
    if mesh is not None and tuple(mesh) != (1, 1):
        return False
    for name, val in params.items():
        if name in _HW_PARAMS and isinstance(val, int) and val > EMU_MAX_HW:
            return False
        if name in _SHAPE_PARAMS and isinstance(val, (tuple, list)) and _elems(val) > EMU_MAX_ELEMS:
            return False
    return True


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "emulator: YOLOv8 op-test case that fits the 2-node Quasar emulator (small feature map / single-device)",
    )
    config.addinivalue_line(
        "markers",
        "blackhole_scale: reproduces the model's exact sharded/TILE/L1 input state (Blackhole-scale; excluded from -m emulator)",
    )


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        if _fits_emulator(item):
            item.add_marker(pytest.mark.emulator)
