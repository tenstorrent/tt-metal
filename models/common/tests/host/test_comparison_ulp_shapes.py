# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""comp_ulp must reject a shape mismatch instead of broadcasting or raising.

ULP distance is a per-element measure, so a comparison between differently shaped outputs is
meaningless. Before the guard, comp_ulp on a (2, 3) golden and a (3, 2) result died inside
torch.abs with a broadcasting RuntimeError, and comp_ulp on a (4, 4) golden and a (1, 4) result
broadcast the smaller operand and returned a verdict for a comparison that never happened.

The test loads ttnn/ttnn/comparison.py by path with a stub ttnn, the same way
models/common/tests/host/test_ttnn_library_import_cycle.py does, so it needs no device.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPARISON = _REPO_ROOT / "ttnn" / "ttnn" / "comparison.py"


@pytest.fixture(scope="module")
def comparison():
    pytest.importorskip("torch")
    if "ttnn" not in sys.modules:
        stub = types.ModuleType("ttnn")
        stub.Tensor = type("Tensor", (), {})
        stub.to_torch = lambda tensor, **kwargs: tensor
        stub.from_torch = lambda tensor, **kwargs: tensor
        sys.modules["ttnn"] = stub
    spec = importlib.util.spec_from_file_location("_comparison_ulp_shapes", _COMPARISON)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_same_shape_still_passes(comparison):
    torch = pytest.importorskip("torch")
    golden = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    matches, message = comparison.comp_ulp(golden, golden.clone(), ulp_threshold=1)
    assert bool(matches) is True, message


def test_transposed_shape_is_rejected(comparison):
    torch = pytest.importorskip("torch")
    golden = torch.zeros(2, 3)
    calculated = torch.zeros(3, 2)
    matches, message = comparison.comp_ulp(golden, calculated, ulp_threshold=1)
    assert bool(matches) is False
    assert "Shape mismatch" in message and "(2, 3)" in message and "(3, 2)" in message


def test_broadcastable_shape_is_rejected(comparison):
    torch = pytest.importorskip("torch")
    golden = torch.ones(4, 4)
    calculated = torch.ones(1, 4)
    matches, message = comparison.comp_ulp(golden, calculated, ulp_threshold=1)
    assert bool(matches) is False, "a broadcast comparison must not be reported as a ULP verdict"
    assert "Shape mismatch" in message


def test_equal_rank_difference_is_rejected(comparison):
    torch = pytest.importorskip("torch")
    golden = torch.ones(1, 1, 32, 64)
    calculated = torch.ones(32, 64)
    matches, message = comparison.comp_ulp(golden, calculated, ulp_threshold=1)
    assert bool(matches) is False
    assert "Shape mismatch" in message


def test_matching_empty_tensors_still_pass(comparison):
    torch = pytest.importorskip("torch")
    empty = torch.empty(0, 32)
    matches, message = comparison.comp_ulp(empty, empty.clone(), ulp_threshold=1)
    assert bool(matches) is True, message


def test_empty_tensors_of_different_shapes_are_rejected(comparison):
    torch = pytest.importorskip("torch")
    matches, message = comparison.comp_ulp(torch.empty(0, 32), torch.empty(0, 16), ulp_threshold=1)
    assert bool(matches) is False
    assert "Shape mismatch" in message
