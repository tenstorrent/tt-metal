# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pure-Python tests for ``models.demos.dots_ocr.tt.mesh``.

These tests are device-free and validate the topology resolution / clamping
logic that keeps dots.mocr runnable on N150 / N300 / T3K (auto-clamped to a
1x2 submesh because ``dots.mocr`` has ``num_key_value_heads=2``).
"""

from __future__ import annotations

import pytest

from models.demos.dots_ocr.tt.mesh import (
    DOTS_MAX_DP_ROWS,
    DOTS_MAX_TP_COLS,
    resolve_mesh_shape,
    resolve_supported_mesh_shape,
)


def test_resolve_mesh_shape_known_names():
    assert resolve_mesh_shape("N150") == (1, 1)
    assert resolve_mesh_shape("N300") == (1, 2)
    assert resolve_mesh_shape("T3K") == (1, 8)
    assert resolve_mesh_shape("T3K_2X4") == (2, 4)
    assert resolve_mesh_shape("TG") == (8, 4)


def test_resolve_mesh_shape_aliases():
    assert resolve_mesh_shape("n150") == (1, 1)
    assert resolve_mesh_shape("t3k") == (1, 8)
    assert resolve_mesh_shape("t3000") == (1, 8)
    assert resolve_mesh_shape("galaxy") == (8, 4)


def test_resolve_mesh_shape_unknown_falls_back_to_single_chip():
    assert resolve_mesh_shape("NONEXISTENT") == (1, 1)
    assert resolve_mesh_shape("") == (1, 1)


def test_supported_mesh_shape_passthrough_for_n150_n300():
    """N150 / N300 already fit dots.mocr's TP ceiling — no clamping."""
    assert resolve_supported_mesh_shape("N150") == (1, 1)
    assert resolve_supported_mesh_shape("N300") == (1, 2)


@pytest.mark.parametrize(
    "selector",
    ["T3K", "T3K_1X8", "T3K_2X4", "N150X4"],
)
def test_supported_mesh_shape_clamps_multi_chip(selector):
    """
    T3K / N150x4 must be clamped because dots.mocr has ``n_kv_heads=2`` and the
    base ``ModelArgs`` requires ``n_kv_heads % cluster_shape[1] == 0``. The
    clamp opens a 1x2 submesh on the same hardware.
    """
    rows, cols = resolve_supported_mesh_shape(selector)
    assert rows <= DOTS_MAX_DP_ROWS
    assert cols <= DOTS_MAX_TP_COLS


def test_supported_mesh_shape_galaxy_clamps_rows_and_cols():
    """TG (Galaxy) resolves to (8, 4) which must clamp on both axes."""
    rows, cols = resolve_supported_mesh_shape("TG")
    assert rows == DOTS_MAX_DP_ROWS  # clamped from 8 to 1
    assert cols == DOTS_MAX_TP_COLS  # clamped from 4 to 2


def test_dots_tp_ceiling_reflects_dots_mocr_n_kv_heads():
    """
    Sanity-check that the documented TP ceiling matches dots.mocr's
    ``num_key_value_heads``. If the architecture ever changes, this test
    forces the constant to be re-examined.
    """
    assert DOTS_MAX_TP_COLS == 2, (
        "dots.mocr has num_key_value_heads=2; update DOTS_MAX_TP_COLS if the " "model architecture changes."
    )
    assert DOTS_MAX_DP_ROWS == 1, "No data-parallel support yet; keep DP=1."
