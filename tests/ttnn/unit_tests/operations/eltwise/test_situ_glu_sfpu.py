# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone device test for the fused situ_glu binary SFPU op (api/compute/situ_glu.h).

Drives the op through ttnn.generic_op with a minimal binary test kernel (no
production op wired), reaching both dst-accumulator modes, and compares against
the torch reference:

    situ_a  = beta_gate * tanh(gate / beta_gate) * sigmoid(gate)
    up_half = beta_up * tanh(up / beta_up)
    result  = situ_a * up_half
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
from models.common.utility_functions import is_blackhole

BETA_GATE = 4.0  # Kimi K3 gate-half beta
BETA_UP = 25.0  # Kimi K3 up-half beta
TILE_ELEMS = 32 * 32

# (ttnn dtype, tile page bytes). bfp8_b: 1 mantissa byte/datum + 1 shared exp byte / 16.
IN_DTYPES = {
    "bf16": (ttnn.bfloat16, TILE_ELEMS * 2),
    "bfp8_b": (ttnn.bfloat8_b, TILE_ELEMS + TILE_ELEMS // 16),
}
OUT_PAGE_BYTES = TILE_ELEMS * 2  # op rounds its result to bf16

# The op always packs bf16, so the bf16 arm is gated in ULP (measured worst case: 1.4).
BF16_ULP = 4
BF16_PCC = 0.999
BFP8_PCC = 0.99


def situ_glu_reference(gate, up):
    g = gate.to(torch.float32)
    u = up.to(torch.float32)
    situ_a = BETA_GATE * torch.tanh(g / BETA_GATE) * torch.sigmoid(g)
    up_half = BETA_UP * torch.tanh(u / BETA_UP)
    return situ_a * up_half


def _coverage_inputs(num_tiles, seed=0):
    """Sweeps that force each half's tanh saturation clamp to be entered, plus a
    heavy tail, shuffled so every tile spans the range."""
    n = num_tiles * TILE_ELEMS
    torch.manual_seed(seed)
    gate = torch.cat([torch.linspace(-6 * BETA_GATE, 6 * BETA_GATE, n // 2), torch.randn(n - n // 2) * (2 * BETA_GATE)])
    up = torch.cat([torch.linspace(-6 * BETA_UP, 6 * BETA_UP, n // 2), torch.randn(n - n // 2) * (2 * BETA_UP)])
    perm = torch.randperm(n)
    return gate[perm].to(torch.bfloat16), up[perm].to(torch.bfloat16)


def _run(device, gate_t, up_t, in_dtype, page_bytes, fp32_dest, dst_out=0):
    num_tiles = gate_t.numel() // TILE_ELEMS
    shape = [1, num_tiles, 32, 32]

    gate = ttnn.from_torch(
        gate_t.reshape(shape),
        dtype=in_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    up = ttnn.from_torch(
        up_t.reshape(shape),
        dtype=in_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cb_gate, cb_up, cb_out = 0, 1, 16

    def cb(idx, fmt, page):
        return ttnn.CBDescriptor(
            total_size=2 * page,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
        )

    cbs = [
        cb(cb_gate, in_dtype, page_bytes),
        cb(cb_up, in_dtype, page_bytes),
        cb(cb_out, ttnn.bfloat16, OUT_PAGE_BYTES),
    ]

    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [gate.buffer_address(), up.buffer_address(), num_tiles, 0]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [output.buffer_address(), num_tiles, 0]

    reader_cta = (
        ttnn.TensorAccessorArgs(gate).get_compile_time_args() + ttnn.TensorAccessorArgs(up).get_compile_time_args()
    )
    writer_cta = [cb_out] + ttnn.TensorAccessorArgs(output).get_compile_time_args()

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source="tests/tt_metal/tt_metal/test_kernels/dataflow/reader_situ_glu.cpp",
            core_ranges=core,
            compile_time_args=reader_cta,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=writer_cta,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source="tests/tt_metal/tt_metal/test_kernels/compute/situ_glu.cpp",
            core_ranges=core,
            compile_time_args=[num_tiles, dst_out],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest),
        ),
    ]

    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    out = ttnn.generic_op([gate, up, output], program)
    return ttnn.to_torch(out).reshape(gate_t.shape)


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu SFPU op is implemented for Blackhole only")
@pytest.mark.parametrize("in_name", list(IN_DTYPES), ids=list(IN_DTYPES))
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["bf16_dst", "fp32_dst"])
# out_tile_idx aliasing the gate operand is what an expert kernel does; a separate dst slot is
# what catches a kernel that ignores out_tile_idx.
@pytest.mark.parametrize("dst_out", [0, 2], ids=["out_aliases_gate", "out_separate"])
def test_situ_glu_sfpu(device, in_name, fp32_dest, dst_out):
    in_dtype, page_bytes = IN_DTYPES[in_name]
    num_tiles = 8
    gate_t, up_t = _coverage_inputs(num_tiles)

    golden = situ_glu_reference(gate_t, up_t)
    actual = _run(device, gate_t, up_t, in_dtype, page_bytes, fp32_dest, dst_out)

    is_bfp8 = in_name == "bfp8_b"
    # |situ_a| <= beta_gate, |up_half| <= beta_up -> |result| <= their product.
    bound = BETA_GATE * BETA_UP * (1.0 + (5e-2 if is_bfp8 else 2**-8))
    assert actual.to(torch.float32).abs().max().item() <= bound

    g = golden.to(torch.float32)
    a = actual.to(torch.float32)
    logger.debug(f"{in_name} fp32_dst={fp32_dest}: max abs err {(a - g).abs().max().item():.4e}")

    if is_bfp8:
        # bfp8_b inputs quantize before the op runs, so the output carries hundreds of bf16 ULP
        # no matter how accurate the SFPU is; ULP only says something about the bf16 arm.
        assert_with_pcc(g, a, pcc=BFP8_PCC)
    else:
        assert_with_ulp(expected_result=golden, actual_result=actual, ulp_threshold=BF16_ULP)
        assert_with_pcc(g, a, pcc=BF16_PCC)


# Generated by tt-polynomial-fitter from activations/products/situ_glu.json; do not edit below this line.
#
# The kernel is certified over every finite BF16 gate x up pair against the exact product (max
# 0.547 ULP). These tests run every finite BF16 encoding of one input against declared
# values of the other on the device, and the special-value classes.

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
# With FP32 DEST, TT-NN's path into DEST hands the SFPU an input below about 2^-118 perturbed (by up to about
# 2^-127): (1.0, 7.5e-37) gives 5.436661e-37 for the stock kernel and this one alike, against 5.388087e-37 exact,
# while the LLK harness in FP32 DEST matches the kernel's model bit for bit. Below this input magnitude the FP32
# DEST sweep checks the result's class and sign, not its ULP.
_GENERATED_FP32_DEST_TINY_INPUT = 2.0**-110
_SMALLEST_NORMAL = 2.0**-126


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


def _generated_pad(first, second):
    pad = (-first.numel()) % TILE_ELEMS
    return (
        torch.cat([first, torch.ones(pad, dtype=first.dtype)]),
        torch.cat([second, torch.ones(pad, dtype=second.dtype)]),
        first.numel(),
    )


def _generated_ulp(actual, first, second, fp32_dest=False):
    """Pure ULP of each scored lane, the count of lanes in the declared flush classes, and whether every lane of
    the FP32 DEST tiny-input class keeps the exact product's class and sign (with its count)."""
    exact = _generated_exact(first, second)
    a = actual.to(torch.float64)
    normal = (exact.abs() >= _SMALLEST_NORMAL) & (exact.abs() <= 3.3895313892515355e38)
    tiny = torch.zeros_like(normal)
    if fp32_dest:
        smaller = torch.minimum(first.to(torch.float64).abs(), second.to(torch.float64).abs())
        tiny = normal & (smaller < _GENERATED_FP32_DEST_TINY_INPUT)
    kept = bool(
        torch.all(torch.isfinite(a[tiny]) & ((a[tiny] == 0) | (torch.sign(a[tiny]) == torch.sign(exact[tiny]))))
    )
    normal = normal & ~tiny
    declared = normal & (a == 0) & (exact.abs() < _SMALLEST_NORMAL * (1.0 + _GENERATED_BOUNDARY_FLUSH))
    if _GENERATED_LOGISTIC_FLUSH is not None:
        index, bound = _GENERATED_LOGISTIC_FLUSH
        declared |= normal & (a == 0) & ((first, second)[index].to(torch.float64) <= bound)
    scored = normal & ~declared
    ulp = torch.exp2(torch.floor(torch.log2(exact.abs().clamp(min=_SMALLEST_NORMAL))) - 7)
    return ((a - exact).abs() / ulp)[scored], int(declared.sum()), (kept, int(tiny.sum()))


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu SFPU op is implemented for Blackhole only")
@pytest.mark.parametrize("swept", ["gate", "up"])
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["bf16_dst", "fp32_dst"])
def test_situ_glu_sfpu_generated_bf16_sweep(device, swept, fp32_dest):
    """Every finite BF16 encoding of one input against the declared values of the other, in pure ULP."""
    every = _generated_finite_bf16()
    other = "up" if swept == "gate" else "gate"
    held = torch.tensor(_GENERATED_HELD[other], dtype=torch.float32).to(torch.bfloat16)
    swept_values = every.repeat(held.numel())
    held_values = held.repeat_interleave(every.numel())
    first, second = (swept_values, held_values) if swept == "gate" else (held_values, swept_values)
    first_t, second_t, n = _generated_pad(first, second)
    actual = _run(device, first_t, second_t, ttnn.bfloat16, TILE_ELEMS * 2, fp32_dest)[:n]
    ulp, declared, (kept, tiny) = _generated_ulp(actual, first, second, fp32_dest)
    logger.info(
        f"swept {swept} fp32_dst={fp32_dest}: max {ulp.max().item():.4f} ULP, {declared} declared flush lanes, "
        f"{tiny} FP32 DEST tiny-input lanes"
    )
    assert ulp.max().item() < 1.0
    assert kept


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu SFPU op is implemented for Blackhole only")
def test_situ_glu_sfpu_generated_specials(device):
    """NaN gives NaN (stored as +inf); the other classes as torch, or as the stock path where the two differ."""
    names = list(_GENERATED_CLASSES)
    pairs = [(a, b) for a in names for b in names if (a, b) in _GENERATED_SPECIAL_RESULTS]
    first = torch.tensor([_GENERATED_CLASSES[a] for a, _ in pairs], dtype=torch.float32).to(torch.bfloat16)
    second = torch.tensor([_GENERATED_CLASSES[b] for _, b in pairs], dtype=torch.float32).to(torch.bfloat16)
    first_t, second_t, n = _generated_pad(first, second)
    actual = _run(device, first_t, second_t, ttnn.bfloat16, TILE_ELEMS * 2, False)[:n].to(torch.float32)

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
