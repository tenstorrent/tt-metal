# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Source and functional-coverage audit for the Blackhole-SFPI parity set.

The dashboard identifies 57 Blackhole kernel families that are pure SFPI and
were not present on Quasar. Keep the inventory explicit: when another matching
Quasar header appears, this test requires its functional coverage to be wired
before the source audit can pass.
"""

from pathlib import Path

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import MathOperation
from helpers.tile_constants import SUPPORTED_TILE_SIZES
from quasar.test_eltwise_unary_sfpu_quasar import (
    OP_CONFIG_BY_MATHOP,
    SFPU_PARITY_FLOAT_FORMATS,
    TRIGONOMETRY_OPS,
)

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

# piecewise_rational is a helper rather than a tile-level operation. Accurate
# FP32 GELU calls it directly, so that path is its functional parity test.
EXECUTABLE_PARITY_COVERAGE = {
    "clamp": (MathOperation.Clamp,),
    "log": (MathOperation.Log,),
    "log1p": (MathOperation.Log1p,),
    "piecewise_rational": (MathOperation.Gelu,),
    "softplus": (MathOperation.Softplus,),
    "sqrt_custom": (MathOperation.SqrtCustom,),
    "trigonometry": tuple(TRIGONOMETRY_OPS),
}

_REPO_ROOT = Path(__file__).resolve().parents[5]
_BH_SFPU_DIR = _REPO_ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"
_QSR_SFPU_DIR = _REPO_ROOT / "tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu"


def _kernel_header(directory: Path, family: str) -> Path:
    return directory / f"ckernel_sfpu_{family}.h"


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
    """A new Quasar port must acquire tests in the same change."""
    implemented = {
        family
        for family in SFPI_PARITY_FAMILIES
        if _kernel_header(_QSR_SFPU_DIR, family).is_file()
    }
    assert implemented == set(EXECUTABLE_PARITY_COVERAGE), (
        "Update EXECUTABLE_PARITY_COVERAGE and add a functional Quasar test for "
        f"the changed parity set (implemented={sorted(implemented)})"
    )

    configured_ops = set(OP_CONFIG_BY_MATHOP)
    expected_ops = {
        operation
        for operations in EXECUTABLE_PARITY_COVERAGE.values()
        for operation in operations
    }
    assert expected_ops <= configured_ops


@pytest.mark.quasar
def test_quasar_parity_sweeps_tensix_formats_and_tile_sizes():
    """The executable parity paths cover every representable documented float/MX L1 format."""
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
    assert {
        formats.input_format for formats in SFPU_PARITY_FLOAT_FORMATS
    } == expected_formats

    expected_tile_sizes = set(SUPPORTED_TILE_SIZES)
    for operations in EXECUTABLE_PARITY_COVERAGE.values():
        for operation in operations:
            configured_tile_sizes = {
                tile_dimensions
                for _, tile_dimensions in OP_CONFIG_BY_MATHOP[operation].tile_cases
            }
            assert expected_tile_sizes <= configured_tile_sizes, operation
