# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Local conftest for the VAD-variant per-op suite: tags emulator-appropriate cases.

Same mechanism as tests/yolo_ops/conftest.py — an ``emulator`` marker inferred from each
case's parametrized values, so:

    pytest models/experimental/ops/quasar/tests/vad_variant_ops/ -m emulator

selects the subset small enough for the 2-node emulator.

A case is NOT emulator when any of:
  * it is marked ``blackhole_scale`` (model-faithful 20-core sharded state),
  * it targets a non-(1,1) mesh,
  * an ``hw`` / ``h`` / ``w`` int param exceeds EMU_MAX_HW,
  * a ``shape`` / ``in_shape`` / ``out_shape`` / ``a_shape`` tuple's element count exceeds EMU_MAX_ELEMS,
  * a conv/pool case's input volume ``n * h * w * (in_ch | c)`` exceeds EMU_MAX_ELEMS.

Tests that allocate large tensors in their body without exposing the shape as a parameter
must be marked ``blackhole_scale`` explicitly.
"""

from models.experimental.ops.quasar.tests.vad_variant_ops.op_utils import EMU_MAX_ELEMS, EMU_MAX_HW

_HW_PARAMS = ("hw", "h", "w", "height", "width", "in_h", "in_w")
_SHAPE_PARAMS = ("shape", "in_shape", "out_shape", "a_shape")


def _elems(shape) -> int:
    n = 1
    for d in shape:
        if isinstance(d, int):
            n *= d
    return n


def _fits_emulator(item) -> bool:
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
    # conv/pool: hw alone misses deep-channel or batched cases (e.g. 24x40x1024, 6 cameras).
    if all(isinstance(params.get(k), int) for k in ("n", "h", "hw")):
        ch = params.get("in_ch", params.get("c"))
        if isinstance(ch, int) and params["n"] * params["h"] * params["hw"] * ch > EMU_MAX_ELEMS:
            return False
    return True


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "emulator: VAD-variant op-test case that fits the 2-node Quasar emulator (small tensor / single-device)",
    )
    config.addinivalue_line(
        "markers",
        "blackhole_scale: reproduces the model's exact 20-core sharded input state (excluded from -m emulator)",
    )


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        if _fits_emulator(item):
            item.add_marker(pytest.mark.emulator)
