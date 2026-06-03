# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for math fidelity module (CPU-only, no hardware required)."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

AUTO_CONFIG_DIR = Path(__file__).resolve().parents[5] / "ttnn" / "ttnn" / "_experimental" / "auto_config"


def _load_auto_config_modules():
    """Load pure-Python auto_config modules without importing compiled ttnn."""
    package_names = [
        "ttnn",
        "ttnn._experimental",
        "ttnn._experimental.auto_config",
        "ttnn._experimental.auto_config.math_fidelity",
        "ttnn._experimental.auto_config.constraint_validator",
    ]
    saved_modules = {name: sys.modules.get(name) for name in package_names}

    try:
        for name in package_names[:3]:
            if name not in sys.modules:
                module = types.ModuleType(name)
                module.__path__ = []
                sys.modules[name] = module

        def load_module(name: str, filename: str):
            spec = importlib.util.spec_from_file_location(name, AUTO_CONFIG_DIR / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        math_fidelity = load_module("ttnn._experimental.auto_config.math_fidelity", "math_fidelity.py")
        constraint_validator = load_module(
            "ttnn._experimental.auto_config.constraint_validator",
            "constraint_validator.py",
        )
        return math_fidelity, constraint_validator
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


_math_fidelity, _constraint_validator = _load_auto_config_modules()

CYCLES_PER_TILE = _math_fidelity.CYCLES_PER_TILE
DTYPE_FIDELITY_CONSTRAINTS = _math_fidelity.DTYPE_FIDELITY_CONSTRAINTS
MAX_CYCLES_PER_TILE = _math_fidelity.MAX_CYCLES_PER_TILE
MathFidelity = _math_fidelity.MathFidelity
default_fidelity = _math_fidelity.default_fidelity
fidelity_cycle_cost = _math_fidelity.fidelity_cycle_cost
fidelity_to_ttnn_string = _math_fidelity.fidelity_to_ttnn_string
valid_fidelities = _math_fidelity.valid_fidelities


def test_enum_properties():
    """Consolidated: values, ordering, and count."""
    assert len(MathFidelity) == 4
    assert MathFidelity.LoFi.value == 0
    assert MathFidelity.HiFi2.value == 1
    assert MathFidelity.HiFi3.value == 2
    assert MathFidelity.HiFi4.value == 3
    assert MathFidelity.LoFi < MathFidelity.HiFi2 < MathFidelity.HiFi3 < MathFidelity.HiFi4


def test_known_cycle_values():
    """Verify cycle costs and monotonicity."""
    assert CYCLES_PER_TILE[MathFidelity.LoFi] == 16
    assert CYCLES_PER_TILE[MathFidelity.HiFi2] == 32
    assert CYCLES_PER_TILE[MathFidelity.HiFi3] == 48
    assert CYCLES_PER_TILE[MathFidelity.HiFi4] == 64
    costs = [CYCLES_PER_TILE[f] for f in MathFidelity]
    assert costs == sorted(costs) and len(set(costs)) == len(costs)
    assert MAX_CYCLES_PER_TILE == 64 == max(CYCLES_PER_TILE.values())


@pytest.mark.parametrize(
    "dtype_a,dtype_b,must_include,must_exclude",
    [
        ("BFLOAT8_B", "BFLOAT8_B", [MathFidelity.HiFi2], [MathFidelity.LoFi]),
        ("BFLOAT16", "BFLOAT16", [MathFidelity.HiFi4], [MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi3]),
        ("BFLOAT16", "BFLOAT4_B", [MathFidelity.HiFi3], [MathFidelity.HiFi2]),
        ("BFLOAT16", "BFLOAT8_B", [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4], [MathFidelity.LoFi]),
        ("BFLOAT4_B", "BFLOAT4_B", [MathFidelity.LoFi], []),
    ],
)
def test_dtype_fidelity_constraints(dtype_a, dtype_b, must_include, must_exclude):
    """Parametrized dtype constraint validation."""
    valid = valid_fidelities(dtype_a, dtype_b)
    for fid in must_include:
        assert fid in valid, f"{fid} should be valid for {dtype_a} x {dtype_b}"
    for fid in must_exclude:
        assert fid not in valid, f"{fid} should NOT be valid for {dtype_a} x {dtype_b}"


def test_unknown_dtype_returns_defaults():
    """Unknown dtypes should still return some valid fidelities."""
    valid = valid_fidelities("fp32", "fp32")
    assert len(valid) > 0


def test_normalization():
    """Case-insensitive and DataType. prefix stripped."""
    v1 = valid_fidelities("bfloat16", "bfloat16")
    v2 = valid_fidelities("BFLOAT16", "BFLOAT16")
    v3 = valid_fidelities("DataType.BFLOAT16", "DataType.BFLOAT16")
    assert v1 == v2 == v3


def test_all_constraint_entries_nonempty():
    """Every constraint entry should have at least one valid fidelity."""
    for key, fids in DTYPE_FIDELITY_CONSTRAINTS.items():
        assert len(fids) > 0, f"Empty fidelity list for {key}"


def test_hifi4_always_valid():
    """HiFi4 should always be valid — it's the most accurate."""
    for a, b in DTYPE_FIDELITY_CONSTRAINTS:
        assert MathFidelity.HiFi4 in valid_fidelities(a, b), f"HiFi4 not valid for {a} x {b}"


def test_default_fidelity():
    """Default should be first valid; check specific known defaults."""
    for (a, b), fids in DTYPE_FIDELITY_CONSTRAINTS.items():
        assert default_fidelity(a, b) == fids[0]
    assert default_fidelity("BFLOAT16", "BFLOAT4_B") == MathFidelity.HiFi3
    assert default_fidelity("BFLOAT16", "BFLOAT16") == MathFidelity.HiFi4


@pytest.mark.parametrize(
    "fid,expected",
    [
        (MathFidelity.LoFi, 16),
        (MathFidelity.HiFi2, 32),
        (MathFidelity.HiFi3, 48),
        (MathFidelity.HiFi4, 64),
    ],
)
def test_fidelity_cycle_cost(fid, expected):
    """Verify fidelity_cycle_cost returns correct values."""
    assert fidelity_cycle_cost(fid) == expected


def test_fidelity_to_string_all():
    """Verify string conversion for all fidelity levels."""
    assert fidelity_to_ttnn_string(MathFidelity.LoFi) == "MathFidelity.LoFi"
    assert fidelity_to_ttnn_string(MathFidelity.HiFi2) == "MathFidelity.HiFi2"
    assert fidelity_to_ttnn_string(MathFidelity.HiFi3) == "MathFidelity.HiFi3"
    assert fidelity_to_ttnn_string(MathFidelity.HiFi4) == "MathFidelity.HiFi4"
