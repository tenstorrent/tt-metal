# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import contextlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
TT_DIR = ROOT / "models" / "autoports" / "google_gemma_4_31b" / "tt"
IRREGULAR_BATCHES_BY_GRID = {
    (11, 10): {13, 17, 19, 23, 26, 29, 31},
    (14, 10): {17, 19, 23, 29, 31},
}


@pytest.fixture
def expect_error():
    """Host-only equivalent of the repository fixture for isolated tests."""

    @contextlib.contextmanager
    def expect_error_(error, message):
        try:
            yield
        except error as exception:
            assert message in str(exception)
        else:
            pytest.fail(f"Expected {error.__name__} matching {message!r}")

    return expect_error_


@dataclass(frozen=True, order=True)
class _CoreCoord:
    x: int
    y: int


@dataclass(frozen=True)
class _CoreRange:
    start: _CoreCoord
    end: _CoreCoord


class _CoreRangeSet:
    def __init__(self, ranges):
        self._ranges = sorted(ranges, key=lambda item: (item.start.y, item.start.x))

    def ranges(self):
        return self._ranges

    def num_cores(self):
        return sum((item.end.x - item.start.x + 1) * (item.end.y - item.start.y + 1) for item in self._ranges)


def _num_cores_to_corerangeset(num_cores, grid, *, row_wise):
    assert row_wise is True
    full_rows, remainder = divmod(num_cores, grid.x)
    ranges = set()
    if full_rows:
        ranges.add(_CoreRange(_CoreCoord(0, 0), _CoreCoord(grid.x - 1, full_rows - 1)))
    if remainder:
        ranges.add(_CoreRange(_CoreCoord(0, full_rows), _CoreCoord(remainder - 1, full_rows)))
    return _CoreRangeSet(ranges)


@pytest.fixture(scope="module")
def decode_grid_module():
    fake_ttnn = ModuleType("ttnn")
    fake_ttnn.CoreCoord = _CoreCoord
    fake_ttnn.CoreRange = _CoreRange
    fake_ttnn.CoreRangeSet = _CoreRangeSet
    fake_ttnn.num_cores_to_corerangeset = _num_cores_to_corerangeset

    module_name = "gemma4_decode_head_grid_cpu_test"
    spec = importlib.util.spec_from_file_location(module_name, TT_DIR / "decode_head_grid.py")
    module = importlib.util.module_from_spec(spec)
    previous_ttnn = sys.modules.get("ttnn")
    sys.modules["ttnn"] = fake_ttnn
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous_ttnn is None:
            sys.modules.pop("ttnn", None)
        else:
            sys.modules["ttnn"] = previous_ttnn


def _mesh(width, height):
    return SimpleNamespace(compute_with_storage_grid_size=lambda: _CoreCoord(width, height))


@pytest.mark.parametrize("grid_size", [(11, 10), (14, 10)])
@pytest.mark.parametrize("batch_size", range(1, 33))
def test_decode_head_grid_covers_every_active_user_within_bounds(decode_grid_module, grid_size, batch_size):
    width, height = grid_size
    mesh = _mesh(width, height)
    core_grid = decode_grid_module.decode_head_core_grid(mesh, batch_size)

    assert core_grid.num_cores() == batch_size
    for core_range in core_grid.ranges():
        assert _CoreCoord(0, 0) <= core_range.start <= core_range.end
        assert core_range.end.x < width
        assert core_range.end.y < height


def test_rectangular_batches_preserve_existing_14x10_geometry(decode_grid_module):
    mesh = _mesh(14, 10)
    expected = {
        **{batch_size: (batch_size, 1) for batch_size in range(1, 15)},
        15: (5, 3),
        16: (8, 2),
        18: (9, 2),
        20: (10, 2),
        21: (7, 3),
        22: (11, 2),
        24: (12, 2),
        25: (5, 5),
        26: (13, 2),
        27: (9, 3),
        28: (14, 2),
        30: (10, 3),
        32: (8, 4),
    }

    for batch_size, (width, height) in expected.items():
        ranges = decode_grid_module.decode_head_core_grid(mesh, batch_size).ranges()
        assert ranges == [_CoreRange(_CoreCoord(0, 0), _CoreCoord(width - 1, height - 1))]


@pytest.mark.parametrize(
    ("batch_size", "expected_ranges"),
    [
        (17, [(0, 0, 13, 0), (0, 1, 2, 1)]),
        (19, [(0, 0, 13, 0), (0, 1, 4, 1)]),
        (23, [(0, 0, 13, 0), (0, 1, 8, 1)]),
        (29, [(0, 0, 13, 1), (0, 2, 0, 2)]),
        (31, [(0, 0, 13, 1), (0, 2, 2, 2)]),
    ],
)
def test_irregular_14x10_batches_use_exact_row_wise_ranges(decode_grid_module, batch_size, expected_ranges):
    mesh = _mesh(14, 10)
    ranges = decode_grid_module.decode_head_core_grid(mesh, batch_size).ranges()
    assert [(item.start.x, item.start.y, item.end.x, item.end.y) for item in ranges] == expected_ranges


@pytest.mark.parametrize(
    ("batch_size", "expected_ranges"),
    [
        (13, [(0, 0, 10, 0), (0, 1, 1, 1)]),
        (17, [(0, 0, 10, 0), (0, 1, 5, 1)]),
        (19, [(0, 0, 10, 0), (0, 1, 7, 1)]),
        (23, [(0, 0, 10, 1), (0, 2, 0, 2)]),
        (26, [(0, 0, 10, 1), (0, 2, 3, 2)]),
        (29, [(0, 0, 10, 1), (0, 2, 6, 2)]),
        (31, [(0, 0, 10, 1), (0, 2, 8, 2)]),
    ],
)
def test_irregular_actual_11x10_batches_use_exact_row_wise_ranges(decode_grid_module, batch_size, expected_ranges):
    mesh = _mesh(11, 10)
    ranges = decode_grid_module.decode_head_core_grid(mesh, batch_size).ranges()
    assert [(item.start.x, item.start.y, item.end.x, item.end.y) for item in ranges] == expected_ranges


@pytest.mark.parametrize("grid_size", [(11, 10), (14, 10)])
@pytest.mark.parametrize("batch_size", range(1, 33))
def test_only_irregular_batches_select_full_grid_subcores(decode_grid_module, grid_size, batch_size):
    width, height = grid_size
    mesh = _mesh(width, height)
    core_grid = decode_grid_module.decode_head_core_grid(mesh, batch_size)
    sub_core_grids = decode_grid_module.decode_head_sub_core_grids(mesh, core_grid)

    if batch_size in IRREGULAR_BATCHES_BY_GRID[grid_size]:
        assert sub_core_grids.ranges() == [_CoreRange(_CoreCoord(0, 0), _CoreCoord(width - 1, height - 1))]
    else:
        assert sub_core_grids is None


@pytest.mark.parametrize("batch_size", [0, 111])
def test_decode_head_grid_rejects_out_of_bounds_batches(decode_grid_module, batch_size, expect_error):
    mesh = _mesh(11, 10)
    with expect_error(ValueError, "worker-grid capacity"):
        decode_grid_module.decode_head_core_grid(mesh, batch_size)


@pytest.mark.parametrize("filename", ["multichip_decoder.py", "optimized_decoder.py", "fused_decoder.py"])
def test_concat_heads_decode_is_wired_to_the_matching_core_grid(filename):
    tree = ast.parse((TT_DIR / filename).read_text())
    core_grid_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "core_grid" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "decode_head_core_grid"
    ]
    concat_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "nlp_concat_heads_decode"
    ]

    assert len(core_grid_assignments) == 1
    assert len(concat_calls) == 1
    subcore_keyword = next(keyword for keyword in concat_calls[0].keywords if keyword.arg == "sub_core_grids")
    assert isinstance(subcore_keyword.value, ast.Call)
    assert isinstance(subcore_keyword.value.func, ast.Name)
    assert subcore_keyword.value.func.id == "decode_head_sub_core_grids"
    assert isinstance(subcore_keyword.value.args[1], ast.Name)
    assert subcore_keyword.value.args[1].id == "core_grid"
