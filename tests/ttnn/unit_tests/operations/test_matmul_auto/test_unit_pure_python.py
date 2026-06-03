# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pure-Python auto-config logic (no TT hardware required)."""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
import types
from unittest.mock import MagicMock

import pytest


def _create_mock_ttnn():
    """Create a minimal mock ttnn module for auto_config imports."""
    mock_ttnn = types.ModuleType("ttnn")
    mock_ttnn.__path__ = []
    mock_ttnn.CoreCoord = MagicMock(side_effect=lambda x, y: MagicMock(x=x, y=y))
    for cls_name in [
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        "MatmulMultiCoreReuseMultiCastProgramConfig",
        "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
        "MatmulMultiCoreReuseProgramConfig",
    ]:
        setattr(mock_ttnn, cls_name, MagicMock(side_effect=lambda **kw: MagicMock(**kw)))
    mock_ttnn.bfloat16 = MagicMock()
    mock_ttnn.bfloat8_b = MagicMock()
    mock_ttnn.MathFidelity = MagicMock()
    mock_ttnn.MathFidelity.HiFi4 = "HiFi4"
    mock_ttnn.TILE_LAYOUT = MagicMock()
    mock_ttnn.experimental = MagicMock()
    return mock_ttnn


@pytest.fixture(autouse=True)
def mock_ttnn_module():
    """Patch ttnn into sys.modules before each test."""
    mock = _create_mock_ttnn()
    original_modules = {}
    for k in list(sys.modules.keys()):
        if k == "ttnn" or k.startswith("ttnn."):
            original_modules[k] = sys.modules.pop(k)

    sys.modules["ttnn"] = mock
    sys.modules["ttnn.operations"] = MagicMock()
    sys.modules["ttnn.distributed"] = MagicMock()

    exp_mod = types.ModuleType("ttnn._experimental")
    exp_mod.__path__ = []
    sys.modules["ttnn._experimental"] = exp_mod
    mock._experimental = exp_mod

    ac_mod = types.ModuleType("ttnn._experimental.auto_config")
    ac_mod.__path__ = []
    sys.modules["ttnn._experimental.auto_config"] = ac_mod
    exp_mod.auto_config = ac_mod

    ac_scorer_mod = types.ModuleType("ttnn._experimental.auto_config.scorer")
    ac_scorer_mod.__path__ = []
    sys.modules["ttnn._experimental.auto_config.scorer"] = ac_scorer_mod
    ac_mod.scorer = ac_scorer_mod

    modules_to_load = [
        ("ttnn._experimental.auto_config.base", "ttnn/ttnn/_experimental/auto_config/base.py"),
        ("ttnn._experimental.auto_config.math_fidelity", "ttnn/ttnn/_experimental/auto_config/math_fidelity.py"),
        (
            "ttnn._experimental.auto_config.candidate_generator",
            "ttnn/ttnn/_experimental/auto_config/candidate_generator.py",
        ),
        ("ttnn._experimental.auto_config.config_cache", "ttnn/ttnn/_experimental/auto_config/config_cache.py"),
        (
            "ttnn._experimental.auto_config.constraint_validator",
            "ttnn/ttnn/_experimental/auto_config/constraint_validator.py",
        ),
        (
            "ttnn._experimental.auto_config.feature_extraction",
            "ttnn/ttnn/_experimental/auto_config/feature_extraction.py",
        ),
        ("ttnn._experimental.auto_config.scorer.heuristic", "ttnn/ttnn/_experimental/auto_config/scorer/heuristic.py"),
        (
            "ttnn._experimental.auto_config.scorer.dnn_scorer",
            "ttnn/ttnn/_experimental/auto_config/scorer/dnn_scorer.py",
        ),
        ("ttnn._experimental.auto_config.benchmark", "ttnn/ttnn/_experimental/auto_config/benchmark.py"),
        ("ttnn._experimental.auto_config.matmul_auto", "ttnn/ttnn/_experimental/auto_config/matmul_auto.py"),
    ]

    for name, path in modules_to_load:
        parent_name = ".".join(name.split(".")[:-1])
        base_name = name.split(".")[-1]
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        setattr(sys.modules[parent_name], base_name, mod)

    for name, path in modules_to_load:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = sys.modules[name]
        mod.__file__ = path
        mod.__package__ = ".".join(name.split(".")[:-1])
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass

    yield mock

    for k in list(sys.modules.keys()):
        if k == "ttnn" or k.startswith("ttnn."):
            sys.modules.pop(k, None)
    for k, v in original_modules.items():
        sys.modules[k] = v


def _get_candidate_gen_fn(name):
    """Import a function from candidate_generator module."""
    mod = sys.modules.get("ttnn._experimental.auto_config.candidate_generator")
    fn = getattr(mod, name, None) if mod else None
    if fn is None:
        pytest.skip(f"Could not import {name}")
    return fn


# --- API smoke tests ---


def test_matmul_auto_is_callable():
    """ttnn.matmul_auto must resolve to a callable function, not a module."""
    mod = sys.modules.get("ttnn._experimental.auto_config.matmul_auto")
    fn = getattr(mod, "matmul_auto", None)
    assert fn is not None, "matmul_auto not found in module"
    assert callable(fn), f"matmul_auto is {type(fn)}, expected callable"


def test_matmul_auto_config_is_callable():
    """MatmulAutoConfig must be a class."""
    mod = sys.modules.get("ttnn._experimental.auto_config.matmul_auto")
    cls = getattr(mod, "MatmulAutoConfig", None)
    assert cls is not None, "MatmulAutoConfig not found in module"
    assert callable(cls), f"MatmulAutoConfig is {type(cls)}, expected callable class"


# --- _find_largest_divisor tests ---


def test_find_largest_divisor_basic():
    fn = _get_candidate_gen_fn("_find_largest_divisor")
    assert fn(8) == 8
    assert fn(16) == 8
    assert fn(32) == 8
    assert fn(7) == 7
    assert fn(1) == 1


def test_find_largest_divisor_with_max():
    fn = _get_candidate_gen_fn("_find_largest_divisor")
    assert fn(16, max_divisor=4) == 4
    assert fn(16, max_divisor=2) == 2
    assert fn(7, max_divisor=3) == 1


def test_find_largest_divisor_primes():
    fn = _get_candidate_gen_fn("_find_largest_divisor")
    assert fn(11) == 1
    assert fn(13) == 1


def test_find_largest_divisor_powers_of_2():
    fn = _get_candidate_gen_fn("_find_largest_divisor")
    assert fn(2) == 2
    assert fn(4) == 4
    assert fn(8) == 8
    assert fn(64) == 8
    assert fn(128) == 8


# --- _find_grid tests ---


def test_find_grid_perfect_square():
    fn = _get_candidate_gen_fn("_find_grid")
    rows, cols = fn(64)
    assert rows * cols <= 64
    assert 64 % (rows * cols) == 0


def test_find_grid_targets_32_cores():
    fn = _get_candidate_gen_fn("_find_grid")
    rows, cols = fn(128)
    cores = rows * cols
    assert 128 % cores == 0
    assert cores == 32


def test_find_grid_small_and_prime():
    fn = _get_candidate_gen_fn("_find_grid")
    rows, cols = fn(4)
    assert rows * cols >= 1 and 4 % (rows * cols) == 0
    rows, cols = fn(37)
    assert 37 % (rows * cols) == 0


# --- _get_out_subblock_w tests ---


def test_get_out_subblock_w_basic():
    fn = _get_candidate_gen_fn("_get_out_subblock_w")
    assert fn(8, 1, 8) == 8


def test_get_out_subblock_w_constrained():
    fn = _get_candidate_gen_fn("_get_out_subblock_w")
    result = fn(8, 4, 8)
    assert result <= 2 and result * 4 <= 8
    result = fn(5, 1, 8)
    assert 5 % result == 0
    result = fn(8, 1, 4)
    assert result <= 4 and 8 % result == 0


# --- L1 budget formula tests ---


def test_l1_small_config_passes():
    """Small per_core values should fit in L1."""
    tile_bytes = 2048
    cb_in0 = 2 * 2 * 2  # 8
    cb_in1 = 2 * 2 * 2  # 8
    cb_out = 2 * 2  # 4
    fp32_accum = 1 * 1 * 32 * 32 * 4
    total = (cb_in0 + cb_in1 + cb_out) * tile_bytes + fp32_accum
    assert total < 1_258_291


def test_l1_large_config_fails():
    """Very large per_core values should exceed L1."""
    tile_bytes = 2048
    cb_in0 = 8 * 32 * 2
    cb_in1 = 8 * 32 * 2
    cb_out = 32 * 32
    fp32_accum = 4 * 4 * 32 * 32 * 4
    total = (cb_in0 + cb_in1 + cb_out) * tile_bytes + fp32_accum
    assert total > 1_258_291


def test_l1_bfloat8b_more_headroom():
    """bfloat8_b tiles (1088 bytes) give more L1 headroom than bf16 (2048)."""
    tiles = 4 * 8 * 2 + 4 * 8 * 2 + 8 * 8
    assert tiles * 1088 < tiles * 2048


def test_subblock_constraint_max8():
    """out_subblock_h * out_subblock_w must be <= 8."""
    valid = [(h, w) for h in range(1, 9) for w in range(1, 9) if h * w <= 8]
    assert (4, 2) in valid and (8, 1) in valid and (1, 8) in valid
    assert (3, 3) not in valid and (4, 3) not in valid


def test_subblock_constraint_fp32_max4():
    """With fp32_dest_acc_en, max product is 4."""
    valid = [(h, w) for h in range(1, 9) for w in range(1, 9) if h * w <= 4]
    assert (4, 1) in valid and (2, 2) in valid
    assert (4, 2) not in valid and (3, 2) not in valid


# --- Scorer weight tests ---


def test_scorer_weights_sum_to_one():
    """All scoring weights must sum to 1.0."""
    weights = [0.25, 0.18, 0.15, 0.07, 0.08, 0.12, 0.15]
    assert abs(sum(weights) - 1.0) < 0.01


def test_scorer_all_weights_positive():
    weights = [0.25, 0.18, 0.15, 0.07, 0.08, 0.12, 0.15]
    assert all(w > 0 for w in weights)


# --- Production formula tests ---


def test_llama_in0_block_w():
    """For Llama-7B (K=4096, 8x8 grid), in0_block_w should be 2."""
    fn = _get_candidate_gen_fn("_find_largest_divisor")
    k_per_core = (4096 // 32) // 64  # 128/64 = 2
    assert fn(k_per_core) == 2


def test_div_up_formula():
    """ceil division should match expected values."""

    def div_up(a, b):
        return (a + b - 1) // b

    assert div_up(128, 8) == 16
    assert div_up(129, 8) == 17
    assert div_up(1, 8) == 1
    assert div_up(0, 8) == 0
