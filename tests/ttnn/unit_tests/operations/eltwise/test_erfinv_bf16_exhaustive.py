# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""All 65,536 BF16 encodings under the compiler-emitted terminal plan."""

import importlib
import numpy as np
import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0


_REFERENCE_MODULE = "torch"
_REFERENCE_FUNCTION = "erfinv"
_RAW_CLASS_OUTPUTS = {
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
    "nan": "pos_inf",
    "neg_inf": "neg_inf",
    "neg_zero": "pos_zero",
    "pos_inf": "pos_inf",
    "pos_zero": "pos_zero",
}
_ACTIONS = (
    ("below", 0xBF800000, False, "neg_inf"),
    ("above", 0x3F800000, False, "pos_inf"),
    ("below", 0xBF800000, True, "neg_inf"),
    ("above", 0x3F800000, True, "pos_inf"),
)
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
    return getattr(module, _REFERENCE_FUNCTION)(values, **{})


@pytest.mark.skipif(
    not (is_blackhole() or is_wormhole_b0()),
    reason="compiler-generated BF16 kernel ships on Blackhole and Wormhole B0",
)
def test_erfinv_bf16_exhaustive(device):
    input_words = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
    host = input_words.view(torch.bfloat16).reshape(256, 256)
    device_input = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.erfinv(device_input, **{})).to(torch.bfloat16)
    result_words = _words(result)

    raw = input_words.cpu().numpy().astype(np.uint16)
    values = host.to(torch.float32).cpu().numpy().reshape(-1)
    expected = _expected_classes(raw, values)
    for result_class, exact_word in _EXACT_CLASS_WORD.items():
        lanes = expected == result_class
        assert np.array_equal(result_words[lanes], np.full(lanes.sum(), exact_word, dtype=np.uint16))
    nan_lanes = expected == "nan"
    nan_words = result_words[nan_lanes]
    assert np.all((nan_words & 0x7F80) == 0x7F80)
    assert np.all((nan_words & 0x007F) != 0)

    scored = expected == "finite_other"
    x = host.to(torch.float64).numpy().reshape(-1)[scored]
    golden = _reference(torch.from_numpy(x)).numpy()
    rounded = _bf16_round_ftz(golden)
    golden_ftz = np.where(rounded == 0.0, np.copysign(0.0, golden), golden)
    got = result.to(torch.float32).cpu().numpy().reshape(-1)[scored].astype(np.float64)
    assert _evaluated_zero_sign_matches(result_words[scored], golden, rounded)
    pure_ulp = np.abs(golden_ftz - got) / _ulp_spacing(rounded)
    assert np.isfinite(pure_ulp).all()
    assert float(pure_ulp.max()) < 1.0
