# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 situ_glu cases, beside the unary category suites."""

import pytest
import torch
import ttnn
from loguru import logger
from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.eltwise.test_situ_glu_sfpu import TILE_ELEMS, _run


_GENERATED_HELD = {
    "gate": [
        -1.0002555517425873e30,
        -200.0,
        -87.5,
        -87.0,
        -19.5,
        -4.0,
        -1.0,
        -0.3046875,
        -7.52316384526264e-37,
        7.52316384526264e-37,
        0.0078125,
        1.0,
        3.140625,
        4.0,
        19.5,
        200.0,
        1.0002555517425873e30,
    ],
    "up": [
        -1.0002555517425873e30,
        -200.0,
        -122.0,
        -25.0,
        -1.0,
        -0.3046875,
        -7.52316384526264e-37,
        7.52316384526264e-37,
        0.0078125,
        1.0,
        3.140625,
        25.0,
        122.0,
        200.0,
        1.0002555517425873e30,
    ],
}


_GENERATED_CLASSES = {
    "+0": 0.0,
    "-0": -0.0,
    "+inf": float("inf"),
    "-inf": -float("inf"),
    "nan": float("nan"),
    "+1": 1.0,
    "-1": -1.0,
    "-100": -100.0,
    "+200": 200.0,
}


# (gate class, up class) -> the result's class: torch's, except where the stock path and torch
# differ (signed zero, subnormal inputs read as zero, results the SFPU flushes) and the kernel keeps the stock one.
_GENERATED_SPECIAL_RESULTS = {
    ("+0", "+0"): "+0",
    ("+0", "+1"): "+0",
    ("+0", "+200"): "+0",
    ("+0", "+inf"): "+0",
    ("+0", "-0"): "+0",
    ("+0", "-1"): "+0",
    ("+0", "-100"): "+0",
    ("+0", "-inf"): "+0",
    ("+0", "nan"): "nan",
    ("+1", "+0"): "+0",
    ("+1", "+inf"): "finite",
    ("+1", "-0"): "+0",
    ("+1", "-inf"): "finite",
    ("+1", "nan"): "nan",
    ("+200", "+0"): "+0",
    ("+200", "+inf"): "finite",
    ("+200", "-0"): "+0",
    ("+200", "-inf"): "finite",
    ("+200", "nan"): "nan",
    ("+inf", "+0"): "+0",
    ("+inf", "+1"): "finite",
    ("+inf", "+200"): "finite",
    ("+inf", "+inf"): "finite",
    ("+inf", "-0"): "+0",
    ("+inf", "-1"): "finite",
    ("+inf", "-100"): "finite",
    ("+inf", "-inf"): "finite",
    ("+inf", "nan"): "nan",
    ("-0", "+0"): "+0",
    ("-0", "+1"): "+0",
    ("-0", "+200"): "+0",
    ("-0", "+inf"): "+0",
    ("-0", "-0"): "+0",
    ("-0", "-1"): "+0",
    ("-0", "-100"): "+0",
    ("-0", "-inf"): "+0",
    ("-0", "nan"): "nan",
    ("-1", "+0"): "+0",
    ("-1", "+inf"): "finite",
    ("-1", "-0"): "+0",
    ("-1", "-inf"): "finite",
    ("-1", "nan"): "nan",
    ("-100", "+0"): "+0",
    ("-100", "+inf"): "+0",
    ("-100", "-0"): "+0",
    ("-100", "-inf"): "+0",
    ("-100", "nan"): "nan",
    ("-inf", "+0"): "+0",
    ("-inf", "+1"): "+0",
    ("-inf", "+200"): "+0",
    ("-inf", "+inf"): "+0",
    ("-inf", "-0"): "+0",
    ("-inf", "-1"): "+0",
    ("-inf", "-100"): "+0",
    ("-inf", "-inf"): "+0",
    ("-inf", "nan"): "nan",
    ("nan", "+0"): "nan",
    ("nan", "+1"): "nan",
    ("nan", "+200"): "nan",
    ("nan", "+inf"): "nan",
    ("nan", "-0"): "nan",
    ("nan", "-1"): "nan",
    ("nan", "-100"): "nan",
    ("nan", "-inf"): "nan",
    ("nan", "nan"): "nan",
}


# (input, bound): at or below it the logistic factor's e^-|x| is below the smallest normal, as in the stock sigmoid.
_GENERATED_LOGISTIC_FLUSH = (0, -87.5)


_GENERATED_BOUNDARY_FLUSH = 0.0  # relative distance above the smallest normal a result may flush from


# With FP32 DEST, TT-NN's copy_tile hands the SFPU x + 2^-127: below this input magnitude the sweep checks the
# result's class and sign, and a zero passes only in the declared flush band.
_GENERATED_FP32_DEST_TINY_INPUT = 2.0**-110


_SMALLEST_NORMAL = 2.0**-126


# (gate, up, out) DEST tiles each case calls the kernel on: adjacent, the fused routed-expert kernels' (j, c + j, j),
# a separate output, and an up tile before the gate (the compiled body's path).
_GENERATED_SLOTS = [(0, 1, 0), (0, 4, 0), (3, 7, 3), (0, 1, 2), (1, 0, 1)]


def _generated_exact(first, second):
    """The exact product in float64 of the inputs as the SFPU reads them (a subnormal as 0, as stock does): the
    leaves have no cancellation, so float64 holds it far below 1 ULP."""
    a = first.to(torch.float64)
    b = second.to(torch.float64)
    a = torch.where(a.abs() < _SMALLEST_NORMAL, torch.zeros_like(a), a)
    b = torch.where(b.abs() < _SMALLEST_NORMAL, torch.zeros_like(b), b)
    return (4 * torch.tanh(a / 4) / (1 + torch.exp(-a))) * (25 * torch.tanh(b / 25))


def _generated_finite_bf16():
    bits = torch.arange(65536, dtype=torch.int32)
    values = (bits << 16).view(torch.float32)
    return values[torch.isfinite(values)].to(torch.bfloat16)


def _generated_fp32_dest_ingress(x):
    """A BF16 input as the SFPU reads it from FP32 DEST after TT-NN's copy_tile: x + 2^-127 in FP32, an exponent-zero
    x read as 0 first, and 0 where the sum is below the smallest normal."""
    x = x.to(torch.float32)
    x = torch.where(x.abs() < _SMALLEST_NORMAL, torch.zeros_like(x), x) + 2.0**-127
    return torch.where(x.abs() < _SMALLEST_NORMAL, torch.zeros_like(x), x)


def _generated_pad(first, second):
    pad = (-first.numel()) % TILE_ELEMS
    return (
        torch.cat([first, torch.ones(pad, dtype=first.dtype)]),
        torch.cat([second, torch.ones(pad, dtype=second.dtype)]),
        first.numel(),
    )


def _generated_ulp(actual, first, second, fp32_dest=False):
    """Pure ULP of each scored lane, the count of lanes in the declared flush classes, and the lanes of the FP32 DEST
    tiny-input class that lose the exact product's class or sign (with the class's count)."""
    exact = _generated_exact(first, second)
    a = actual.to(torch.float64)
    normal = (exact.abs() >= _SMALLEST_NORMAL) & (exact.abs() <= 3.3895313892515355e38)
    tiny = torch.zeros_like(normal)
    zero = torch.zeros_like(normal)
    if fp32_dest:
        smaller = torch.minimum(first.to(torch.float64).abs(), second.to(torch.float64).abs())
        tiny = normal & (smaller < _GENERATED_FP32_DEST_TINY_INPUT)
        read = _generated_exact(_generated_fp32_dest_ingress(first), _generated_fp32_dest_ingress(second))
        flushable = read.abs() < _SMALLEST_NORMAL * (1.0 + _GENERATED_BOUNDARY_FLUSH)
        zero = (a == 0) & flushable
    lost = tiny & ~(torch.isfinite(a) & (zero | (torch.sign(a) == torch.sign(exact))))
    missed = [
        f"({first[i].item():.6e}, {second[i].item():.6e}) -> {a[i].item():.6e}, exact {exact[i].item():.6e}"
        for i in lost.nonzero().flatten()[:8].tolist()
    ]
    if missed:
        missed.insert(0, f"{int(lost.sum())} lanes, the first (first, second) -> result:")
    normal = normal & ~tiny
    declared = normal & (a == 0) & (exact.abs() < _SMALLEST_NORMAL * (1.0 + _GENERATED_BOUNDARY_FLUSH))
    if _GENERATED_LOGISTIC_FLUSH is not None:
        index, bound = _GENERATED_LOGISTIC_FLUSH
        declared |= normal & (a == 0) & ((first, second)[index].to(torch.float64) <= bound)
    scored = normal & ~declared
    ulp = torch.exp2(torch.floor(torch.log2(exact.abs().clamp(min=_SMALLEST_NORMAL))) - 7)
    return ((a - exact).abs() / ulp)[scored], int(declared.sum()), (missed, int(tiny.sum()))


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu SFPU op is implemented for Blackhole only")
@pytest.mark.parametrize("swept", ["gate", "up"])
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["bf16_dst", "fp32_dst"])
@pytest.mark.parametrize("slots", _GENERATED_SLOTS, ids=lambda s: "dst%d_%d_%d" % s)
def test_situ_glu_sfpu_generated_bf16_sweep(device, swept, fp32_dest, slots):
    """Every finite BF16 encoding of one input against the declared values of the other, in pure ULP."""
    if fp32_dest and max(slots) >= 4:
        pytest.skip("FP32 DEST holds 4 tiles in half sync")
    every = _generated_finite_bf16()
    other = "up" if swept == "gate" else "gate"
    held = torch.tensor(_GENERATED_HELD[other], dtype=torch.float32).to(torch.bfloat16)
    swept_values = every.repeat(held.numel())
    held_values = held.repeat_interleave(every.numel())
    first, second = (swept_values, held_values) if swept == "gate" else (held_values, swept_values)
    first_t, second_t, n = _generated_pad(first, second)
    actual = _run(
        device,
        first_t,
        second_t,
        ttnn.bfloat16,
        TILE_ELEMS * 2,
        fp32_dest,
        dst_gate=slots[0],
        dst_up=slots[1],
        dst_out=slots[2],
    )[:n]
    ulp, declared, (missed, tiny) = _generated_ulp(actual, first, second, fp32_dest)
    logger.info(
        f"swept {swept} fp32_dst={fp32_dest}: max {ulp.max().item():.4f} ULP, {declared} declared flush lanes, "
        f"{tiny} FP32 DEST tiny-input lanes"
    )
    assert ulp.max().item() < 1.0
    assert not missed, "FP32 DEST tiny-input lanes lost the exact product's class or sign: " + "; ".join(missed)


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu SFPU op is implemented for Blackhole only")
@pytest.mark.parametrize("slots", _GENERATED_SLOTS, ids=lambda s: "dst%d_%d_%d" % s)
def test_situ_glu_sfpu_generated_specials(device, slots):
    """NaN gives NaN (stored as +inf); the other classes as torch, or as the stock path where the two differ."""
    names = list(_GENERATED_CLASSES)
    pairs = [(a, b) for a in names for b in names if (a, b) in _GENERATED_SPECIAL_RESULTS]
    first = torch.tensor([_GENERATED_CLASSES[a] for a, _ in pairs], dtype=torch.float32).to(torch.bfloat16)
    second = torch.tensor([_GENERATED_CLASSES[b] for _, b in pairs], dtype=torch.float32).to(torch.bfloat16)
    first_t, second_t, n = _generated_pad(first, second)
    actual = _run(
        device,
        first_t,
        second_t,
        ttnn.bfloat16,
        TILE_ELEMS * 2,
        False,
        dst_gate=slots[0],
        dst_up=slots[1],
        dst_out=slots[2],
    )[:n].to(torch.float32)

    def kind(v):
        if torch.isnan(v):
            return "nan"
        if torch.isinf(v):
            return "+inf" if v > 0 else "-inf"
        if v == 0:
            return "-0" if torch.signbit(v) else "+0"
        return "finite"

    # The BF16 pack stores a NaN as +inf, and the SFPU a subnormal result as +0.
    stored = {"nan": "+inf", "subnormal": "+0"}
    got = {pair: kind(actual[i]) for i, pair in enumerate(pairs)}
    want = {pair: stored.get(_GENERATED_SPECIAL_RESULTS[pair], _GENERATED_SPECIAL_RESULTS[pair]) for pair in pairs}
    assert got == want
