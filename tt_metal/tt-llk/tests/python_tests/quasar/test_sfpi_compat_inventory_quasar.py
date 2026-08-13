# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Coverage manifest for the 57 SFPI families in tt-metal issue #52947."""

from quasar.test_sfpi_compat_binary_quasar import SFPI_BINARY_CASES
from quasar.test_sfpi_compat_special_quasar import SFPI_SPECIAL_CASES
from quasar.test_sfpi_compat_ternary_quasar import SFPI_TERNARY_CASES
from quasar.test_sfpi_compat_unary_quasar import SFPI_UNARY_CASES

ISSUE_52947_FAMILIES = {
    "activations",
    "add1",
    "addcdiv",
    "addcmul",
    "alt_complex_rotate90",
    "atan2",
    "binary_bitwise",
    "binary_fmod",
    "binary_pow",
    "binary_remainder",
    "bitwise",
    "bitwise_not",
    "cast_fp32_to_fp16a",
    "cbrt",
    "celu",
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
    "fmod",
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
    "logical_not",
    "logsigmoid",
    "mask",
    "polygamma",
    "prelu",
    "rdiv",
    "remainder",
    "rpow",
    "rsub_int32",
    "selu",
    "sign",
    "snake_beta",
    "softshrink",
    "softsign",
    "tanhshrink",
    "tiled_prod",
    "unary_comp",
    "unary_power",
    "unary_shift",
    "xielu",
}


def test_issue_52947_inventory_is_complete():
    functional_families = {
        *(case.kernel for case in SFPI_UNARY_CASES),
        *(case.kernel for case in SFPI_BINARY_CASES),
        *(case.kernel for case in SFPI_TERNARY_CASES),
        *(case.kernel for case in SFPI_SPECIAL_CASES),
    }
    # conversions is a helper body rather than an independently callable
    # kernel.  It is covered through unary_power and binary_pow, both of which
    # include ckernel_sfpu_conversions.h in the production implementation.
    covered_families = functional_families | {"conversions"}

    assert len(ISSUE_52947_FAMILIES) == 57
    assert covered_families == ISSUE_52947_FAMILIES
    assert "unary_power" in functional_families
    assert "binary_pow" in functional_families
