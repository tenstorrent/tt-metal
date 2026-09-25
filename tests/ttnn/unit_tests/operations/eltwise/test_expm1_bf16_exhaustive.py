# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Strict numerical supplement to the existing BF16 unary category tests.

The category tests cover TTNN golden behavior; this checks every raw BF16
input against the compiler's finite mathematical and terminal contract.
Paired qualification remains the authority for observed TTNN specials.
"""

import importlib
import numpy as np
import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from tests.ttnn.utils_for_testing import generate_all_bfloat16_bitpatterns


pytestmark = pytest.mark.use_module_device


_REFERENCE_MODULE = "torch"
_REFERENCE_FUNCTION = "expm1"
_RAW_TO_REFERENCE_INPUT_BY_ARCH = {
    "blackhole": {
        "pos_zero": "pos_zero",
        "neg_zero": "pos_zero",
        "pos_subnormal": "pos_zero",
        "neg_subnormal": "pos_zero",
        "finite_other": "finite_other",
        "pos_inf": "pos_inf",
        "neg_inf": "neg_inf",
        "pos_nan": "pos_inf",
        "neg_nan": "pos_inf",
    },
    "wormhole_b0": {
        "pos_zero": "pos_zero",
        "neg_zero": "pos_zero",
        "pos_subnormal": "pos_zero",
        "neg_subnormal": "pos_zero",
        "finite_other": "finite_other",
        "pos_inf": "pos_inf",
        "neg_inf": "neg_inf",
        "pos_nan": "pos_inf",
        "neg_nan": "neg_inf",
    },
}


def _raw_to_reference_input():
    if is_blackhole():
        return _RAW_TO_REFERENCE_INPUT_BY_ARCH["blackhole"]
    if is_wormhole_b0():
        return _RAW_TO_REFERENCE_INPUT_BY_ARCH["wormhole_b0"]
    raise AssertionError("no compiled ingress contract for current architecture")


def _words(values):
    return values.contiguous().view(torch.uint16).cpu().numpy().reshape(-1)


def _bf16_round_ftz(values):
    rounded = torch.from_numpy(values).to(torch.bfloat16).to(torch.float64).numpy()
    subnormal = (np.abs(rounded) < 2.0**-126) & (rounded != 0.0)
    return np.where(subnormal, np.copysign(0.0, rounded), rounded)


def _ulp_spacing(values):
    words = (np.abs(values).astype(np.float32).view(np.uint32) >> 16).astype(np.uint32)
    upper = (np.minimum(words + 1, 0x7F80) << 16).view(np.float32)
    lower = (words << 16).view(np.float32)
    spacing = (upper - lower).astype(np.float64)
    return np.where(np.isinf(upper), np.float64(2.0**120), spacing)


def _raw_classes(words):
    magnitude = words & np.uint16(0x7FFF)
    exponent = magnitude & np.uint16(0x7F80)
    mantissa = magnitude & np.uint16(0x007F)
    negative = (words & np.uint16(0x8000)) != 0
    classes = np.full(words.shape, "finite_other", dtype="<U16")
    classes[(magnitude == 0) & ~negative] = "pos_zero"
    classes[(magnitude == 0) & negative] = "neg_zero"
    classes[(exponent == 0) & (mantissa != 0) & ~negative] = "pos_subnormal"
    classes[(exponent == 0) & (mantissa != 0) & negative] = "neg_subnormal"
    classes[(exponent == 0x7F80) & (mantissa == 0) & ~negative] = "pos_inf"
    classes[(exponent == 0x7F80) & (mantissa == 0) & negative] = "neg_inf"
    classes[(exponent == 0x7F80) & (mantissa != 0) & ~negative] = "pos_nan"
    classes[(exponent == 0x7F80) & (mantissa != 0) & negative] = "neg_nan"
    return classes


def _reference_inputs(raw_classes, values):
    """Apply the compiler target's typed ingress before the math reference."""
    canonical = {
        "pos_zero": np.float64(0.0),
        "neg_zero": np.float64(-0.0),
        "pos_subnormal": np.float64(2.0**-133),
        "neg_subnormal": np.float64(-(2.0**-133)),
        "pos_inf": np.float64(np.inf),
        "neg_inf": np.float64(-np.inf),
        "pos_nan": np.asarray([0x7FF8000000000000], dtype=np.uint64).view(np.float64)[0],
        "neg_nan": np.asarray([0xFFF8000000000000], dtype=np.uint64).view(np.float64)[0],
    }
    effective = values.copy()
    for raw_class, effective_class in _raw_to_reference_input().items():
        if effective_class != "finite_other":
            effective[raw_classes == raw_class] = canonical[effective_class]
    return effective


def _reference(values):
    module = importlib.import_module(_REFERENCE_MODULE)
    if _REFERENCE_MODULE == "numpy":
        result = getattr(module, _REFERENCE_FUNCTION)(values.numpy(), **{})
        return torch.from_numpy(np.asarray(result, dtype=np.float64))
    return getattr(module, _REFERENCE_FUNCTION)(input=values, **{})


_NUMERIC_TERMINALS = ((), (("above", 89.0, True, "return_class", "pos_inf"), ("below", -6.25, True, "constant", -1.0)))


def _real_domain_mask(values):
    return np.ones(values.shape, dtype=bool)


def _assert_finite_math(raw_words, result_words, reference_values):
    """Check finite mathematical results, including zero and saturation tails.

    Paired qualification checks candidate <= stock ULP and observed-TTNN
    special parity. This standalone does not assign exceptional output classes.
    """
    scored = (raw_words & np.uint16(0x7FFF)) < np.uint16(0x7F80)
    late_nonfinite, domain_rows = _NUMERIC_TERMINALS
    scored &= ~np.isin(_raw_classes(raw_words), late_nonfinite)
    with np.errstate(all="ignore"):
        scored &= _real_domain_mask(reference_values)
        golden = _reference(torch.from_numpy(reference_values[scored].astype(np.float64))).numpy()
        coordinate = reference_values[scored]
        resolved = np.zeros(coordinate.shape, dtype=bool)
        for direction, bound, inclusive, kind, value in domain_rows:
            if direction == "below":
                owned = coordinate <= bound if inclusive else coordinate < bound
            else:
                owned = coordinate >= bound if inclusive else coordinate > bound
            owned &= np.isfinite(coordinate) & ~resolved
            resolved |= owned
            if kind == "constant":
                golden[owned] = _bf16_round_ftz(np.full(np.count_nonzero(owned), value, dtype=np.float64))
        rounded = _bf16_round_ftz(golden)
        numeric = np.isfinite(golden) & np.isfinite(rounded)
        selected_words = result_words[scored][numeric]
        got = (selected_words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        golden_ftz = np.where(rounded[numeric] == 0.0, np.copysign(0.0, golden[numeric]), golden[numeric])
        pure_ulp = np.abs(golden_ftz - got) / _ulp_spacing(rounded[numeric])
    assert np.isfinite(pure_ulp).all()
    assert not pure_ulp.size or float(pure_ulp.max()) < 1.0
    scored[np.flatnonzero(scored)] = numeric
    names, counts = np.unique(_raw_classes(result_words[~scored]), return_counts=True)
    print("Exceptional output classes (observed-stock parity checked in qualification):", dict(zip(names, counts)))


@pytest.mark.skipif(
    not (is_blackhole() or is_wormhole_b0()),
    reason="compiler-generated BF16 kernel ships on Blackhole and Wormhole B0",
)
def test_expm1_bf16_exhaustive(device):
    host = generate_all_bfloat16_bitpatterns()
    assert host.numel() == 65536
    device_input = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.expm1(device_input, **{})).to(torch.bfloat16)
    result_words = _words(result)

    raw = _words(host)
    values = host.to(torch.float32).cpu().numpy().reshape(-1)
    raw_classes = _raw_classes(raw)
    with np.errstate(invalid="ignore"):
        reference_values = _reference_inputs(raw_classes, values.astype(np.float64))
    _assert_finite_math(raw, result_words, reference_values)
