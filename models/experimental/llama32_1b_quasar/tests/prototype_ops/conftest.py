# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Local conftest for the per-op suite: tags emulator-appropriate cases.

The 2-node Quasar emulator can only run small / batch-1, single-device shapes.
This hook marks every op-test *case* that fits the emulator with the ``emulator``
marker, so you can select the whole emulator subset with:

    pytest models/experimental/llama32_1b_quasar/tests/ops/ -m emulator

A case is classified from its **actual parametrized values** (not display-name
substrings), so oversized cases without a size token in their id — e.g.
``test_reshape[mlp_prefill_fold]`` (1024 rows) or ``test_minimal_matmul[512]`` —
are correctly excluded. A case is NOT emulator-appropriate when any of:
  * it targets a non-(1,1) mesh (multi-device CCL cases),
  * a batch/user param > 1,
  * a sequence/row param (seq, seq_len, m) > 128,
  * a shape param whose leading (row) dims multiply to > 128.
A leading-token denylist is also honored as a backstop.
"""

# Backstop id substrings that always mean "too large for the emulator".
_LARGE_ID_TOKENS = ("batch32", "b32", "seq512", "seq1024", "seq500")
# Parameter names whose scalar value is a row/sequence count.
_ROW_PARAMS = ("seq", "seq_len", "m")
# Parameter names holding a full tensor shape tuple.
_SHAPE_PARAMS = ("shape", "in_shape", "out_shape")
_EMU_MAX_ROWS = 128


def _rows_of(shape) -> int:
    """Product of all but the last (hidden) dim — the tile-row count that drives cores/L1."""
    dims = [d for d in shape if isinstance(d, int)]
    rows = 1
    for d in dims[:-1]:
        rows *= d
    return rows


def _fits_emulator(item) -> bool:
    nid = item.nodeid.lower()
    if any(tok in nid for tok in _LARGE_ID_TOKENS):
        return False
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return True
    params = callspec.params
    mesh = params.get("ttnn_mesh_device")
    if mesh is not None and tuple(mesh) != (1, 1):
        return False
    for name, val in params.items():
        if name == "batch" and isinstance(val, int) and val > 1:
            return False
        if name in _ROW_PARAMS and isinstance(val, int) and val > _EMU_MAX_ROWS:
            return False
        if name in _SHAPE_PARAMS and isinstance(val, (tuple, list)) and _rows_of(val) > _EMU_MAX_ROWS:
            return False
    return True


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "emulator: op-test case that fits the 2-node Quasar emulator (batch-1 / small / single-device)",
    )


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        if _fits_emulator(item):
            item.add_marker(pytest.mark.emulator)
