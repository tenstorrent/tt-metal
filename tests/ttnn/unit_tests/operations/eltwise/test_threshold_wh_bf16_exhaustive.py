# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""All 65,536 BF16 encodings under the compiler-emitted terminal plan."""

import importlib
import numpy as np
import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0


_REFERENCE_MODULE = "torch.nn.functional"
_REFERENCE_FUNCTION = "threshold"
_RAW_CLASS_OUTPUTS = {
    "pos_zero": "pos_zero",
    "neg_zero": "pos_zero",
    "pos_subnormal": "pos_zero",
    "neg_subnormal": "pos_zero",
    "finite_other": "finite_other",
    "pos_inf": "pos_inf",
    "neg_inf": "pos_zero",
    "pos_nan": "pos_inf",
    "neg_nan": "pos_zero",
}
_RAW_TO_REFERENCE_INPUT = {
    "pos_zero": "pos_zero",
    "neg_zero": "pos_zero",
    "pos_subnormal": "pos_zero",
    "neg_subnormal": "pos_zero",
    "finite_other": "finite_other",
    "pos_inf": "pos_inf",
    "neg_inf": "neg_inf",
    "pos_nan": "pos_inf",
    "neg_nan": "neg_inf",
}
_RESULT_TO_EGRESS = {
    "finite_other": "finite_other",
    "nan": "nan",
    "neg_inf": "neg_inf",
    "neg_zero": "pos_zero",
    "pos_inf": "pos_inf",
    "pos_zero": "pos_zero",
}
_ACTIONS = ()
_DOMAIN_ACTIONS = ()
_LATE_RAW_CLASSES = ()
_EXACT_CLASS_WORD = {
    "pos_inf": np.uint16(0x7F80),
    "neg_inf": np.uint16(0xFF80),
    "pos_zero": np.uint16(0x0000),
    "neg_zero": np.uint16(0x8000),
}


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
    for raw_class, effective_class in _RAW_TO_REFERENCE_INPUT.items():
        if effective_class != "finite_other":
            effective[raw_classes == raw_class] = canonical[effective_class]
    return effective


def _expected_classes(words, values):
    raw_classes = _raw_classes(words)
    expected = np.asarray([_RAW_CLASS_OUTPUTS[name] for name in raw_classes])
    owned = expected != "finite_other"
    for direction, bound_bits, inclusive, return_class in _ACTIONS:
        bound = np.asarray([bound_bits], dtype=np.uint32).view(np.float32)[0]
        if direction == "below":
            selected = values <= bound if inclusive else values < bound
        else:
            selected = values >= bound if inclusive else values > bound
        selected &= ~owned
        expected[selected] = return_class
        owned[selected] = True
    return expected


def _evaluated_zero_sign_matches(result_words, golden, rounded):
    zero_lanes = rounded == 0.0
    if not np.any(zero_lanes):
        return True
    result_classes = np.where(np.signbit(golden[zero_lanes]), "neg_zero", "pos_zero")
    expected_words = np.asarray(
        [_EXACT_CLASS_WORD[_RESULT_TO_EGRESS[name]] for name in result_classes],
        dtype=np.uint16,
    )
    return np.array_equal(result_words[zero_lanes], expected_words)


def _reference(values):
    module = importlib.import_module(_REFERENCE_MODULE)
    return getattr(module, _REFERENCE_FUNCTION)(values, **{"threshold": 0.0, "value": 0.0})


def _physical_words(values):
    rounded = _bf16_round_ftz(np.asarray(values, dtype=np.float64))
    words = (rounded.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    classes = _raw_classes(words)
    for name in _EXACT_CLASS_WORD:
        selected = classes == name
        after = _RESULT_TO_EGRESS[name]
        assert after in _EXACT_CLASS_WORD
        words[selected] = _EXACT_CLASS_WORD[after]
    return words


def _domain_expectations(words, values):
    """First matching action owns the lane; explicit late raw classes win last."""
    expected = _expected_classes(words, values)
    original = expected.copy()
    owned = np.zeros(words.shape, dtype=bool)
    exact = np.zeros(words.shape, dtype=np.uint16)
    exact_owned = np.zeros(words.shape, dtype=bool)
    for direction, bound, inclusive, kind, payload in _DOMAIN_ACTIONS:
        if direction == "below":
            selected = values <= bound if inclusive else values < bound
        else:
            selected = values >= bound if inclusive else values > bound
        selected &= ~owned
        owned[selected] = True
        if kind in ("constant", "identity"):
            result = (
                np.full(int(selected.sum()), np.float32(payload), dtype=np.float64)
                if kind == "constant"
                else values[selected]
            )
            encoded = _physical_words(result)
            classes = _raw_classes(encoded)
            expected[selected] = np.where(np.isin(classes, ("pos_nan", "neg_nan")), "nan", classes)
            exact[selected] = encoded
            exact_owned[selected] = ~np.isin(classes, ("pos_nan", "neg_nan"))
        else:
            classes = (
                np.where(np.signbit(values[selected]), "neg_inf", "pos_inf")
                if kind == "signed_inf"
                else np.full(int(selected.sum()), payload)
            )
            expected[selected] = np.asarray([_RESULT_TO_EGRESS[name] for name in classes])
    # Ordered domain actions claim lanes before explicit late raw overrides.
    # A default raw-class expectation is not itself a late terminal.
    late = np.isin(_raw_classes(words), _LATE_RAW_CLASSES)
    expected[late] = original[late]
    exact_owned[late] = False
    return expected, exact, exact_owned


def _assert_finite_math(raw_words, result_words, reference_values):
    """Every finite raw lane retains mathematical scoring, including tails."""
    scored = (raw_words & np.uint16(0x7FFF)) < np.uint16(0x7F80)
    golden = _reference(torch.from_numpy(reference_values[scored].astype(np.float64))).numpy()
    rounded = _bf16_round_ftz(golden)
    selected_words = result_words[scored]
    for positive, name in ((True, "pos_inf"), (False, "neg_inf")):
        overflow = np.isinf(rounded) & (np.signbit(rounded) != positive)
        after = _RESULT_TO_EGRESS[name]
        assert after in _EXACT_CLASS_WORD
        assert np.all(selected_words[overflow] == _EXACT_CLASS_WORD[after])
    # An all-real unary reference must not manufacture NaNs for finite inputs.
    assert not np.isnan(golden).any()
    numeric = np.isfinite(rounded)
    got = (selected_words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    assert _evaluated_zero_sign_matches(selected_words[numeric], golden[numeric], rounded[numeric])
    golden_ftz = np.where(rounded[numeric] == 0.0, np.copysign(0.0, golden[numeric]), golden[numeric])
    pure_ulp = np.abs(golden_ftz - got[numeric]) / _ulp_spacing(rounded[numeric])
    assert np.isfinite(pure_ulp).all()
    assert not pure_ulp.size or float(pure_ulp.max()) < 1.0


@pytest.mark.skipif(
    not (is_wormhole_b0()),
    reason="compiler-generated BF16 kernel ships on Wormhole B0",
)
def test_threshold_bf16_exhaustive(device):
    input_words = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
    host = input_words.view(torch.bfloat16).reshape(256, 256)
    device_input = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.threshold(device_input, **{"threshold": 0.0, "value": 0.0})).to(torch.bfloat16)
    result_words = _words(result)

    raw = input_words.cpu().numpy().astype(np.uint16)
    values = host.to(torch.float32).cpu().numpy().reshape(-1)
    raw_classes = _raw_classes(raw)
    with np.errstate(invalid="ignore"):
        reference_values = _reference_inputs(raw_classes, values.astype(np.float64))
    expected = _expected_classes(raw, reference_values)
    expected, terminal_words, terminal_owned = _domain_expectations(raw, reference_values)
    assert np.array_equal(result_words[terminal_owned], terminal_words[terminal_owned])
    for result_class, exact_word in _EXACT_CLASS_WORD.items():
        lanes = expected == result_class
        assert np.array_equal(result_words[lanes], np.full(lanes.sum(), exact_word, dtype=np.uint16))
    nan_lanes = expected == "nan"
    nan_words = result_words[nan_lanes]
    assert np.all((nan_words & 0x7F80) == 0x7F80)
    assert np.all((nan_words & 0x007F) != 0)

    exceptional_finite = (expected == "finite_other") & ~np.isfinite(values)
    assert np.all(_raw_classes(result_words[exceptional_finite]) == "finite_other")
    _assert_finite_math(raw, result_words, reference_values)
