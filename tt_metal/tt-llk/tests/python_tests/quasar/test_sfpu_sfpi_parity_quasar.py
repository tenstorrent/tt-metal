# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Source and functional-coverage audit for the 57 Blackhole SFPI families.

This is deliberately an audit rather than another kernel harness.  Each family
is tied to the executable test which reaches it and to the C++ include point
which proves that the test dispatches the Quasar header.  Helper-only headers
are tied to an executable consumer instead of being presented as tile-level
operations of their own.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import MathOperation
from helpers.tile_constants import SUPPORTED_TILE_SIZES

SFPI_PARITY_FAMILIES = (
    "activations",
    "add1",
    "addcdiv",
    "alt_complex_rotate90",
    "atan2",
    "binary_fmod",
    "binary_pow",
    "binary_remainder",
    "bitwise",
    "cast_fp32_to_fp16a",
    "cbrt",
    "celu",
    "clamp",
    "conversions",
    "digamma",
    "div_int32",
    "div_int32_floor",
    "elu",
    "erf",
    "erfc",
    "erfinv",
    "exp2",
    "expm1",
    "hardmish",
    "hardshrink",
    "hardtanh",
    "heaviside",
    "i0",
    "i1",
    "identity",
    "int_sum",
    "isclose",
    "lerp",
    "lgamma",
    "log",
    "log1p",
    "logsigmoid",
    "mask",
    "piecewise_rational",
    "polygamma",
    "prelu",
    "rdiv",
    "rpow",
    "selu",
    "sign",
    "snake_beta",
    "softplus",
    "softshrink",
    "softsign",
    "sqrt_custom",
    "tanhshrink",
    "tiled_prod",
    "trigonometry",
    "unary_comp",
    "unary_power",
    "unary_shift",
    "xielu",
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_LLK_ROOT = Path(__file__).resolve().parents[3]
_BH_SFPU_DIR = _REPO_ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"
_QSR_SFPU_DIR = _REPO_ROOT / "tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu"

_UNARY_TEST = _LLK_ROOT / "tests/python_tests/quasar/test_eltwise_unary_sfpu_quasar.py"
_UNARY_SOURCE = _LLK_ROOT / "tests/sources/quasar/eltwise_unary_sfpu_quasar_test.cpp"
_UNARY_DISPATCH = _LLK_ROOT / "tests/helpers/include/sfpu_operations_quasar.h"

_BINARY_TEST = (
    _LLK_ROOT / "tests/python_tests/quasar/test_eltwise_binary_sfpu_quasar.py"
)
_BINARY_SOURCE = _LLK_ROOT / "tests/sources/quasar/eltwise_binary_sfpu_quasar_test.cpp"
_BINARY_DISPATCH = _UNARY_DISPATCH

_TERNARY_TEST = _LLK_ROOT / "tests/python_tests/quasar/test_sfpu_ternary_sfpi_quasar.py"
_TERNARY_SOURCE = _LLK_ROOT / "tests/sources/quasar/sfpu_ternary_sfpi_quasar_test.cpp"

_SPECIAL_TEST = _LLK_ROOT / "tests/python_tests/quasar/test_sfpu_sfpi_special_quasar.py"
_SPECIAL_SOURCE = _LLK_ROOT / "tests/sources/quasar/sfpu_sfpi_special_quasar_test.cpp"

_TRIGONOMETRY_OPS = (
    MathOperation.Sin,
    MathOperation.Cos,
    MathOperation.Tan,
    MathOperation.Atan,
    MathOperation.Asin,
    MathOperation.Acos,
    MathOperation.Sinh,
    MathOperation.Cosh,
    MathOperation.Acosh,
    MathOperation.Asinh,
    MathOperation.Atanh,
)


@dataclass(frozen=True)
class FunctionalCoverage:
    """One executable route from a parity family to its Quasar header."""

    suite: str
    operations: tuple[MathOperation, ...]
    python_test: Path
    test_functions: tuple[str, ...]
    cpp_source: Path
    include_sources: tuple[Path, ...]
    helper_only: bool = False


def _unary(*operations: MathOperation, helper_only: bool = False, include_sources=None):
    return FunctionalCoverage(
        suite="unary",
        operations=tuple(operations),
        python_test=_UNARY_TEST,
        test_functions=("test_eltwise_unary_sfpu_quasar",),
        cpp_source=_UNARY_SOURCE,
        include_sources=tuple(include_sources or (_UNARY_DISPATCH,)),
        helper_only=helper_only,
    )


def _binary_float(
    *operations: MathOperation, helper_only: bool = False, include_sources=None
):
    return FunctionalCoverage(
        suite="binary_float",
        operations=tuple(operations),
        python_test=_BINARY_TEST,
        test_functions=("test_eltwise_binary_sfpi_parity_float_quasar",),
        cpp_source=_BINARY_SOURCE,
        include_sources=tuple(include_sources or (_BINARY_DISPATCH,)),
        helper_only=helper_only,
    )


def _binary_int(*operations: MathOperation):
    return FunctionalCoverage(
        suite="binary_int",
        operations=tuple(operations),
        python_test=_BINARY_TEST,
        test_functions=("test_eltwise_binary_sfpi_parity_int_div_quasar",),
        cpp_source=_BINARY_SOURCE,
        include_sources=(_BINARY_DISPATCH,),
    )


def _ternary(*operations: MathOperation):
    return FunctionalCoverage(
        suite="ternary",
        operations=tuple(operations),
        python_test=_TERNARY_TEST,
        test_functions=("test_sfpu_ternary_sfpi_quasar",),
        cpp_source=_TERNARY_SOURCE,
        include_sources=(_TERNARY_SOURCE,),
    )


def _special(*test_functions: str):
    return FunctionalCoverage(
        suite="special",
        operations=(),
        python_test=_SPECIAL_TEST,
        test_functions=tuple(test_functions),
        cpp_source=_SPECIAL_SOURCE,
        include_sources=(_SPECIAL_SOURCE,),
    )


# All 57 families must appear here.  conversions and piecewise_rational are
# helper-only: binary POW and accurate GELU are their executable consumers.
# The four special entries use bespoke full-tile/layout harnesses and therefore
# have no MathOperation value to pretend is a conventional unary/binary op.
EXECUTABLE_PARITY_COVERAGE = {
    "activations": _unary(MathOperation.Hardsigmoid),
    "add1": _unary(MathOperation.Add1),
    "addcdiv": _ternary(MathOperation.SfpuAddcdiv),
    "alt_complex_rotate90": _special("test_sfpu_alt_complex_rotate90_quasar"),
    "atan2": _binary_float(MathOperation.SfpuAtan2),
    "binary_fmod": _binary_float(MathOperation.SfpuBinaryFmod),
    "binary_pow": _binary_float(MathOperation.SfpuElwpow),
    "binary_remainder": _binary_float(MathOperation.SfpuBinaryRemainder),
    "bitwise": _special("test_sfpu_unary_bitwise_quasar"),
    "cast_fp32_to_fp16a": _unary(MathOperation.CastFp32ToFp16a),
    "cbrt": _unary(MathOperation.Cbrt),
    "celu": _unary(MathOperation.Celu),
    "clamp": _unary(MathOperation.Clamp),
    "conversions": _binary_float(
        MathOperation.SfpuElwpow,
        helper_only=True,
        include_sources=(_QSR_SFPU_DIR / "ckernel_sfpu_binary_pow.h",),
    ),
    "digamma": _unary(MathOperation.Digamma),
    "div_int32": _binary_int(MathOperation.SfpuDivInt32),
    "div_int32_floor": _binary_int(MathOperation.SfpuDivInt32Floor),
    "elu": _unary(MathOperation.Elu),
    "erf": _unary(MathOperation.Erf),
    "erfc": _unary(MathOperation.Erfc),
    "erfinv": _unary(MathOperation.Erfinv),
    "exp2": _unary(MathOperation.Exp2),
    "expm1": _unary(MathOperation.Expm1),
    "hardmish": _unary(MathOperation.Hardmish),
    "hardshrink": _unary(MathOperation.Hardshrink),
    "hardtanh": _unary(MathOperation.Hardtanh),
    "heaviside": _unary(MathOperation.Heaviside),
    "i0": _unary(MathOperation.I0),
    "i1": _unary(MathOperation.I1),
    "identity": _unary(MathOperation.Identity),
    "int_sum": _special(
        "test_sfpu_int_sum_col_quasar",
        "test_sfpu_int_sum_row_quasar",
        "test_sfpu_int_sum_add_quasar",
    ),
    "isclose": _binary_float(MathOperation.SfpuIsclose),
    "lerp": _ternary(MathOperation.SfpuLerp),
    "lgamma": _unary(MathOperation.Lgamma),
    "log": _unary(MathOperation.Log),
    "log1p": _unary(MathOperation.Log1p),
    "logsigmoid": _binary_float(MathOperation.SfpuLogsigmoid),
    "mask": _binary_float(MathOperation.SfpuMask),
    "piecewise_rational": _unary(
        MathOperation.Gelu,
        helper_only=True,
        include_sources=(_QSR_SFPU_DIR / "ckernel_sfpu_gelu.h",),
    ),
    "polygamma": _unary(MathOperation.Polygamma),
    "prelu": _unary(MathOperation.Prelu),
    "rdiv": _unary(MathOperation.Rdiv),
    "rpow": _unary(MathOperation.Rpow),
    "selu": _unary(MathOperation.Selu),
    "sign": _unary(MathOperation.Sign),
    "snake_beta": _ternary(MathOperation.SfpuSnakeBeta),
    "softplus": _unary(MathOperation.Softplus),
    "softshrink": _unary(MathOperation.Softshrink),
    "softsign": _unary(MathOperation.Softsign),
    "sqrt_custom": _unary(MathOperation.SqrtCustom),
    "tanhshrink": _unary(MathOperation.Tanhshrink),
    "tiled_prod": _special("test_sfpu_tiled_prod_quasar"),
    "trigonometry": _unary(*_TRIGONOMETRY_OPS),
    "unary_comp": _unary(
        MathOperation.UnaryGt,
        MathOperation.UnaryLt,
        MathOperation.UnaryGe,
        MathOperation.UnaryLe,
        MathOperation.UnaryEq,
        MathOperation.UnaryNe,
    ),
    "unary_power": _unary(MathOperation.UnaryPower),
    "unary_shift": _unary(MathOperation.LeftShift, MathOperation.RightShift),
    "xielu": _unary(MathOperation.Xielu),
}


def _kernel_header(directory: Path, family: str) -> Path:
    return directory / f"ckernel_sfpu_{family}.h"


def _defined_test_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _assigned_math_operations(path: Path, assignment: str) -> set[MathOperation]:
    """Read a literal operation registry without importing another test module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else ()
        if isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        if any(
            isinstance(target, ast.Name) and target.id == assignment
            for target in targets
        ):
            values.append(node.value)
    assert len(values) == 1, f"Expected one assignment to {assignment} in {path}"
    operation_names = {
        node.attr
        for node in ast.walk(values[0])
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "MathOperation"
    }
    return {getattr(MathOperation, name) for name in operation_names}


def _assigned_data_formats(path: Path, assignment: str) -> set[DataFormat]:
    """Read a literal format registry without importing another test module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else ()
        if isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        if any(
            isinstance(target, ast.Name) and target.id == assignment
            for target in targets
        ):
            values.append(node.value)
    assert len(values) == 1, f"Expected one assignment to {assignment} in {path}"
    format_names = {
        node.attr
        for node in ast.walk(values[0])
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "DataFormat"
    }
    return {getattr(DataFormat, name) for name in format_names}


def _all_named_math_operations(path: Path) -> set[MathOperation]:
    """Return every MathOperation symbol named by a Python test module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    operation_names = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "MathOperation"
    }
    return {getattr(MathOperation, name) for name in operation_names}


def _direct_operations(suite: str) -> set[MathOperation]:
    return {
        operation
        for coverage in EXECUTABLE_PARITY_COVERAGE.values()
        if coverage.suite == suite and not coverage.helper_only
        for operation in coverage.operations
    }


@pytest.mark.quasar
def test_blackhole_sfpi_parity_inventory_matches_sources():
    """All 57 dashboard entries still resolve to local pure-SFPI BH headers."""
    assert len(SFPI_PARITY_FAMILIES) == 57
    assert len(set(SFPI_PARITY_FAMILIES)) == 57

    missing = []
    no_sfpi_marker = []
    for family in SFPI_PARITY_FAMILIES:
        header = _kernel_header(_BH_SFPU_DIR, family)
        if not header.is_file():
            missing.append(family)
            continue
        source = header.read_text(encoding="utf-8")
        if "sfpi::" not in source and "using namespace sfpi" not in source:
            no_sfpi_marker.append(family)

    assert not missing, f"Parity families without a Blackhole header: {missing}"
    assert not no_sfpi_marker, (
        "Parity families no longer visibly implemented with SFPI: " f"{no_sfpi_marker}"
    )


@pytest.mark.quasar
def test_every_implemented_quasar_parity_family_has_functional_coverage():
    """A Quasar port must have an explicit executable route in this audit."""
    implemented = {
        family
        for family in SFPI_PARITY_FAMILIES
        if _kernel_header(_QSR_SFPU_DIR, family).is_file()
    }
    assert implemented == set(EXECUTABLE_PARITY_COVERAGE), (
        "Update EXECUTABLE_PARITY_COVERAGE and add a functional Quasar test for "
        f"the changed parity set (implemented={sorted(implemented)})"
    )

    unary_registry = (
        _assigned_math_operations(_UNARY_TEST, "SFPI_PARITY_NEW_UNARY_OPS")
        | _assigned_math_operations(_UNARY_TEST, "SFPI_PARITY_SHIFT_OPS")
        | {
            MathOperation.Clamp,
            MathOperation.Log,
            MathOperation.Log1p,
            MathOperation.Softplus,
            MathOperation.SqrtCustom,
            *_TRIGONOMETRY_OPS,
        }
    )
    assert _direct_operations("unary") == unary_registry
    assert unary_registry <= _all_named_math_operations(_UNARY_TEST)

    binary_float_registry = _assigned_math_operations(
        _BINARY_TEST, "_SFPI_PARITY_BINARY_FLOAT_OPS"
    )
    assert _direct_operations("binary_float") == binary_float_registry

    binary_int_registry = _assigned_math_operations(
        _BINARY_TEST, "_SFPI_PARITY_INT_DIV_OPS"
    )
    assert _direct_operations("binary_int") == binary_int_registry
    assert _direct_operations("ternary") == _assigned_math_operations(
        _TERNARY_TEST, "_OPERATIONS"
    )


@pytest.mark.quasar
def test_parity_coverage_resolves_to_real_tests_sources_and_quasar_headers():
    """Coverage records must name a collected test and reach the named family header."""
    function_cache = {}
    text_cache = {}

    def source_text(path: Path) -> str:
        if path not in text_cache:
            assert path.is_file(), f"Missing parity coverage source: {path}"
            text_cache[path] = path.read_text(encoding="utf-8")
        return text_cache[path]

    for family, coverage in EXECUTABLE_PARITY_COVERAGE.items():
        assert (
            coverage.python_test.is_file()
        ), f"{family} names missing test module {coverage.python_test}"
        assert (
            coverage.cpp_source.is_file()
        ), f"{family} names missing C++ harness {coverage.cpp_source}"

        if coverage.python_test not in function_cache:
            function_cache[coverage.python_test] = _defined_test_functions(
                coverage.python_test
            )
        missing_functions = (
            set(coverage.test_functions) - function_cache[coverage.python_test]
        )
        assert not missing_functions, (
            f"{family} names tests not defined in {coverage.python_test}: "
            f"{sorted(missing_functions)}"
        )

        source_reference = coverage.cpp_source.relative_to(
            _LLK_ROOT / "tests"
        ).as_posix()
        assert source_reference in source_text(
            coverage.python_test
        ), f"{family}'s Python test no longer references {source_reference}"

        header_marker = f"ckernel_sfpu_{family}.h"
        assert any(
            header_marker in source_text(include_source)
            for include_source in coverage.include_sources
        ), (
            f"{family}'s executable route no longer includes {header_marker}; "
            f"checked {[str(path) for path in coverage.include_sources]}"
        )

    assert {
        family
        for family, coverage in EXECUTABLE_PARITY_COVERAGE.items()
        if coverage.helper_only
    } == {"conversions", "piecewise_rational"}


@pytest.mark.quasar
def test_quasar_parity_sweeps_tensix_formats_and_tile_sizes():
    """The float/MX parity paths retain the documented L1 formats and tile shapes."""
    expected_formats = {
        DataFormat.Float32,
        DataFormat.Tf32,
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    }
    unary_formats = _assigned_data_formats(
        _UNARY_TEST, "SFPU_UNARY_FORMATS"
    ) | _assigned_data_formats(_UNARY_TEST, "SFPU_PARITY_L1_FORMATS")
    assert unary_formats == expected_formats
    assert (
        _assigned_data_formats(_BINARY_TEST, "_SFPI_PARITY_BINARY_FLOAT_FORMATS")
        == expected_formats
    )

    assert SUPPORTED_TILE_SIZES
    for operation in _direct_operations("unary"):
        assert operation in _all_named_math_operations(_UNARY_TEST)

    # The executable suite derives the parity geometry registry from the shared
    # source of truth rather than copying the eight shapes into this audit.
    unary_source = _UNARY_TEST.read_text(encoding="utf-8")
    assert (
        "PARITY_TILE_CASES = tuple((shape, shape) for shape in SUPPORTED_TILE_SIZES)"
        in unary_source
    )
