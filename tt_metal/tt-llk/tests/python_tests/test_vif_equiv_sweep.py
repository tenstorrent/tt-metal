# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive bit-exact check for the Wormhole metal SFPU comparison kernels.

These kernels reach a 0/1 (or pass-through) result through predication, and
several have been rewritten into unpredicated forms -- a `copysgn` sign stamp, a
comparison rearranged onto a native condition code, or a default written up
front so the `v_else` can go. Rewrites like that are exact or they are wrong;
there is no "close". So every output bit pattern is compared with `==` against a
golden, over the whole input domain rather than a sampled tolerance.

What this covers that the existing suites do not:

* **Exhaustively, not by sampling.** All 65,279 distinct finite bf16 values per
  op. `test_eltwise_unary_sfpu` samples these ops; for kernels that dispatch on
  a sign bit and on zero-ness, the interesting inputs are precisely the ones a
  sample is likely to miss.
* **Bit patterns, not floats.** `NaN != NaN` under float comparison, and signed
  zeros compare equal, so a tolerance check cannot see either -- and a sign-bit
  rewrite is exactly the kind of change that would disturb them.
* **Int32 boundaries.** `test_unary_zero_comp_ttnn` covers the six
  `calculate_comp_int` modes end to end, but draws `randint(-5, 5)`, so it never
  sees INT_MIN, INT_MAX or 0x80000000 -- the sign-magnitude trap the
  `relu_clamp_int` comment calls out. Those are pinned here.

Output format is bf16 in / fp32 out throughout: that shows the kernel's own
result before the pack quantises it, so a one-bit difference cannot hide. A
bf16-out pass was measured during development and its result was, for every op
and every input class, exactly the fp32 result narrowed to bf16 -- it added no
information and is not carried.

Two hardware behaviours the fp32 golden does not model, both measured identical
on the pre-rewrite headers and therefore properties of this SFPU path rather
than of any rewrite:

* **bf16 subnormals are flushed to zero** entering DEST. In fp32 they are
  ordinary non-zero numbers, so `UnarySFPUGolden` would call them non-zero and
  disagree with the kernel on all 254 of them for `sign` and `heaviside`. The
  golden is fed subnormal-flushed inputs -- see `_flush_subnormals`.
* **Non-finite inputs** are pinned by the explicit `_NONFINITE_EXPECTED` table
  rather than by the golden, which canonicalises NaN. Note what that table
  records: NaN behaves exactly like `+inf` for all nine kernels, i.e. it reaches
  the SFPU with a *clear* sign bit, so the sign-bit dispatch these kernels rely
  on sends it down the positive arm.

A note on non-finite coverage, because it is easy to overstate: the stimulus
asks for all 256 exp==0xFF bf16 patterns, but only three distinct patterns
survive the host-to-device path -- torch canonicalises NaN payloads when it
narrows fp32 to bf16. `test_nonfinite_stimulus_reaches_device` pins that, so the
claim cannot silently rot. What is covered is the *classes* +inf, -inf and a
quiet NaN, not 254 separate payloads.

Wormhole only; the kernels are in hw/ckernels/wormhole_b0.
"""

import math

import numpy as np
import pytest
import torch
from conftest import wormhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    DestSync,
    generate_input_dim,
)

_FACE_ELEMENTS = 16 * 16

# ─────────────────────────────────────────────────────────────────────────────
# The metal comparison kernels, grouped by the input domain they need.
#
# relu_min / relu_max are deliberately absent, and not for a harness reason:
# every float relu entry point in tt_metal/hw/inc/api/compute/eltwise_unary/relu.h
# passes _relu_min_/_relu_max_ to SFPU_UNARY_CALL, i.e. tt-llk's own kernels in
# tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h. The metal
# sfpu::relu_min / sfpu::relu_max in llk_sfpu/ckernel_sfpu_relu.h have no caller
# at all, so there is nothing here for a sweep to cover. sfpu_operations.h routes
# MathOperation.ReluMin/ReluMax to the same tt-llk pair, which is what
# test_eltwise_unary_sfpu already exercises.
# ─────────────────────────────────────────────────────────────────────────────

_FLOAT_OPS = {
    "sign": MathOperation.Sign,
    "heaviside": MathOperation.Heaviside,
    "hardshrink": MathOperation.Hardshrink,
    "unary_eq": MathOperation.UnaryEq,
    "unary_ne": MathOperation.UnaryNe,
    "unary_gt": MathOperation.UnaryGt,
    "unary_lt": MathOperation.UnaryLt,
    "unary_ge": MathOperation.UnaryGe,
    "unary_le": MathOperation.UnaryLe,
}

_INT_OPS = {
    "eqz_int": MathOperation.EqualZero,
    "nez_int": MathOperation.NotEqualZero,
    "ltz_int": MathOperation.LessThanZero,
    "gtz_int": MathOperation.GreaterThanZero,
    "lez_int": MathOperation.LessThanEqualZero,
    "gez_int": MathOperation.GreaterThanEqualZero,
}

# calculate_comp_int maps each element to 0 or 1 by comparing it against zero as
# a signed two's-complement int32. Written directly rather than via
# UnarySFPUGolden -- see the note in test_equiv_int32.
_INT_GOLDEN = {
    "eqz_int": lambda x: (x == 0).astype(np.int64),
    "nez_int": lambda x: (x != 0).astype(np.int64),
    "ltz_int": lambda x: (x < 0).astype(np.int64),
    "gtz_int": lambda x: (x > 0).astype(np.int64),
    "lez_int": lambda x: (x <= 0).astype(np.int64),
    "gez_int": lambda x: (x >= 0).astype(np.int64),
}

assert set(_INT_GOLDEN) == set(_INT_OPS)

# bf16 in / fp32 out: the kernel's own result before the pack quantises it, so a
# one-bit difference cannot hide. See the module docstring on why bf16-out is not
# also swept.
_FLOAT_FORMAT = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32)

_INT_FORMAT = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)

# Measured behaviour on the three non-finite classes that actually reach the SFPU,
# expressed as fp32 result values; the bf16-out case is exactly this narrowed to
# bf16, which _nonfinite_want() does. Every row was captured on the pre-rewrite
# headers as well and is unchanged by the rewrites.
#
# The "nan" column is the interesting one: it equals the "+inf" column for all
# nine kernels. These kernels dispatch on the sign bit (SFPSETCC tests it), so
# that says the NaN arrives with its sign bit clear and takes the positive arm --
# which is why none of the sign-bit rewrites could disturb it.
# Smallest normal bf16: exponent field 1, mantissa 0.
_BF16_MIN_NORMAL = 2.0**-126

_INF = math.inf
_NONFINITE_EXPECTED = {
    #              +inf,  -inf,   nan
    "sign": (1.0, -1.0, 1.0),
    "heaviside": (1.0, 0.0, 1.0),
    "hardshrink": (_INF, -_INF, _INF),
    "unary_eq": (0.0, 0.0, 0.0),
    "unary_ne": (1.0, 1.0, 1.0),
    "unary_gt": (1.0, 0.0, 1.0),
    "unary_lt": (0.0, 1.0, 0.0),
    "unary_ge": (1.0, 0.0, 1.0),
    "unary_le": (0.0, 1.0, 0.0),
}

assert set(_NONFINITE_EXPECTED) == set(_FLOAT_OPS)


def _flush_subnormals(t: torch.Tensor) -> torch.Tensor:
    """Model the DEST flush-to-zero the golden's fp32 arithmetic does not have.

    Flushed to +0.0 rather than to a signed zero: the kernels that branch on
    zero write a literal +0.0 in that arm, so a negative subnormal comes out as
    +0.0 and not -0.0. Verified against the device for all 254 subnormal
    patterns; without this the golden calls them non-zero and disagrees.
    """
    return torch.where(t.abs() < _BF16_MIN_NORMAL, torch.zeros_like(t), t)


def _nonfinite_patterns():
    """The 256 bf16 patterns with exp == 0xFF: 2 infinities and 254 NaNs."""
    bits = torch.arange(0, 2**16, dtype=torch.int32)
    vals = (bits.to(torch.int16)).view(torch.bfloat16).to(torch.float32)
    sel = ~torch.isfinite(vals)
    out = vals[sel]
    assert out.numel() == _FACE_ELEMENTS
    return out


# StimuliSpec.custom writes its values at the start of ONE face and zero-fills the
# rest, so a long list has to be handed over as custom_faces: 256 values per face,
# 4 faces per tile. 4 tiles = 16 faces = 4096 values, which sits comfortably inside
# the DEST budget.
_INT_TILES = 4
_INT_FACES = _INT_TILES * 4
_INT_VALUES = _INT_FACES * _FACE_ELEMENTS


def _int32_probe_values():
    """Crafted int32 domain for the comparison-to-zero kernels.

    2^32 is not enumerable, so this is a sample rather than a sweep: 19 explicit
    representation boundaries (INT_MIN/INT_MAX, 0x80000000, the 16-bit carries),
    1,024 dense small magnitudes either side of zero, and a fixed-seed random
    bulk fill across the full range for the remaining 3,053 slots.

    Comparison to zero only distinguishes {negative, zero, positive}, so those
    three equivalence classes are covered many times over; what the boundaries
    buy is coverage of the sign-magnitude / two's-complement handling the kernel
    reaches its verdict through.
    """
    special = [
        0,
        1,
        -1,
        2,
        -2,
        0x7FFFFFFF,  # INT_MAX
        -0x80000000,  # INT_MIN, unrepresentable in sign-magnitude
        0x40000000,
        -0x40000000,
        0x00FFFFFF,
        -0x00FFFFFF,
        0x00010000,
        -0x00010000,
        0x7FFFFFFE,
        -0x7FFFFFFF,
        0x0000FFFF,
        -0x0000FFFF,
        0x00008000,
        -0x00008000,
    ]
    # Dense small magnitudes either side of zero: the region where sign-magnitude
    # and two's-complement encodings diverge most cheaply.
    dense = [v for m in range(1, 513) for v in (m, -m)]
    rng = np.random.default_rng(20260902)
    remaining = _INT_VALUES - len(special) - len(dense)
    assert remaining > 0
    bulk = rng.integers(
        -(2**31), 2**31 - 1, size=remaining, dtype=np.int64, endpoint=True
    ).tolist()
    vals = special + dense + bulk
    assert len(vals) == _INT_VALUES
    return vals


def _int32_face_spec(vals):
    return StimuliSpec.custom_faces(
        {
            f: vals[f * _FACE_ELEMENTS : (f + 1) * _FACE_ELEMENTS]
            for f in range(_INT_FACES)
        }
    )


def _unpack_to_dest(input_format: DataFormat, dest_acc: DestAccumulation) -> bool:
    return input_format.is_32_bit() and dest_acc == DestAccumulation.Yes


def _drive(mathop, formats, spec_A, num_tiles, dest_acc, want_golden=True):
    """Run one kernel over one stimulus set.

    Returns (src_A, result tensor, golden tensor or None).
    """
    input_dimensions = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * num_tiles]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_A,
    )

    actual_dimensions = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * tile_cnt_A]

    golden_tensor = None
    if want_golden:
        generate_golden = get_golden_generator(UnarySFPUGolden)
        golden_tensor = generate_golden(
            mathop,
            _flush_subnormals(src_A),
            formats.output_format,
            dest_acc,
            formats.input_format,
            actual_dimensions,
        )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        actual_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(actual_dimensions, actual_dimensions),
            APPROX_MODE(ApproximationMode.No),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=_unpack_to_dest(formats.input_format, dest_acc),
    )
    res = configuration.run().result
    res_tensor = torch.tensor(res, dtype=format_dict[formats.output_format])
    return src_A, res_tensor, golden_tensor


def _as_bits(t: torch.Tensor, fmt: DataFormat) -> np.ndarray:
    """Reinterpret a result tensor as an integer bit pattern array."""
    if fmt == DataFormat.Float32:
        return t.to(torch.float32).view(torch.int32).numpy().astype(np.int64)
    if fmt == DataFormat.Float16_b:
        return (
            t.to(torch.float32).to(torch.bfloat16).view(torch.int16).numpy()
        ).astype(np.int64)
    if fmt == DataFormat.Int32:
        return t.to(torch.int32).numpy().astype(np.int64)
    raise AssertionError(f"no bit view for {fmt}")


def _assert_bit_exact(case, x_bits, got_bits, want_bits, x_mask=0xFFFF):
    """Compare against the golden with ==, reporting the first few divergences."""
    assert (
        got_bits.shape == want_bits.shape
    ), f"{case}: result has {got_bits.shape} points, golden has {want_bits.shape}"
    diff = got_bits != want_bits
    ndiff = int(diff.sum())
    if not ndiff:
        return
    idx = np.flatnonzero(diff)[:8]
    detail = ", ".join(
        f"x={int(x_bits[i]) & x_mask:#x} got={int(got_bits[i]) & 0xFFFFFFFF:#x} "
        f"want={int(want_bits[i]) & 0xFFFFFFFF:#x}"
        for i in idx
    )
    raise AssertionError(
        f"{case}: {ndiff}/{got_bits.size} outputs differ from the golden bit pattern "
        f"(first {len(idx)}: {detail})"
    )


def _bf16_class(pattern: int) -> str:
    """Classify a delivered bf16 bit pattern as +inf / -inf / nan."""
    if pattern & 0x7FFF == 0x7F80:
        return "-inf" if pattern & 0x8000 else "+inf"
    return "nan"


def _nonfinite_want(op_name, x_bits, out_format):
    """Expected result bits for each delivered non-finite input."""
    plus_inf, minus_inf, nan = _NONFINITE_EXPECTED[op_name]
    lookup = {"+inf": plus_inf, "-inf": minus_inf, "nan": nan}
    vals = torch.tensor(
        [lookup[_bf16_class(int(b) & 0xFFFF)] for b in x_bits], dtype=torch.float32
    )
    return _as_bits(vals, out_format)


@wormhole_only
@pytest.mark.parametrize("op_name", list(_FLOAT_OPS))
def test_equiv_float_finite(op_name):
    """All 65,279 distinct finite bf16 values, against the golden, bit for bit."""
    # ulp_sweep sorts and dedupes, so it yields the distinct finite bf16 values
    # and drops the exp==0xFF patterns -- hence the separate non-finite test.
    spec_A = StimuliSpec.ulp_sweep(low=-math.inf, high=math.inf)
    expected = 65279

    src_A, res, golden = _drive(
        _FLOAT_OPS[op_name], _FLOAT_FORMAT, spec_A, 64, DestAccumulation.Yes
    )

    x_bits = _as_bits(src_A.to(torch.float32), DataFormat.Float16_b)[:expected]
    got_bits = _as_bits(res, _FLOAT_FORMAT.output_format)[:expected]
    want_bits = _as_bits(golden, _FLOAT_FORMAT.output_format)[:expected]

    _assert_bit_exact(f"{op_name}__finite", x_bits, got_bits, want_bits)


@wormhole_only
@pytest.mark.parametrize("op_name", list(_FLOAT_OPS))
def test_equiv_float_nonfinite(op_name):
    """+inf, -inf and NaN, against the measured table rather than the golden.

    The golden canonicalises a generated NaN's sign, so it cannot arbitrate this
    class; _NONFINITE_EXPECTED records what the SFPU actually does, captured on
    both the pre- and post-rewrite headers.
    """
    spec_A = StimuliSpec.custom(values=_nonfinite_patterns().tolist(), seed=0)
    expected = _FACE_ELEMENTS

    src_A, res, _ = _drive(
        _FLOAT_OPS[op_name],
        _FLOAT_FORMAT,
        spec_A,
        1,
        DestAccumulation.Yes,
        want_golden=False,
    )

    x_bits = _as_bits(src_A.to(torch.float32), DataFormat.Float16_b)[:expected]
    got_bits = _as_bits(res, _FLOAT_FORMAT.output_format)[:expected]
    want_bits = _nonfinite_want(op_name, x_bits, _FLOAT_FORMAT.output_format)

    _assert_bit_exact(f"{op_name}__nonfinite", x_bits, got_bits, want_bits)


@wormhole_only
def test_nonfinite_stimulus_reaches_device():
    """Pin what the non-finite stimulus actually delivers.

    The request is all 256 exp==0xFF bf16 patterns, but torch canonicalises NaN
    payloads when it narrows fp32 to bf16, so only the three classes below
    survive. This test exists so the coverage claim in the module docstring
    cannot quietly become false: if the harness ever starts carrying payloads,
    this fails and the docstring gets updated with it.
    """
    spec_A = StimuliSpec.custom(values=_nonfinite_patterns().tolist(), seed=0)
    src_A, _, _ = _drive(
        _FLOAT_OPS["sign"],
        _FLOAT_FORMAT,
        spec_A,
        1,
        DestAccumulation.Yes,
        want_golden=False,
    )
    x_bits = _as_bits(src_A.to(torch.float32), DataFormat.Float16_b)[:_FACE_ELEMENTS]
    classes = {_bf16_class(int(b) & 0xFFFF) for b in x_bits}
    assert classes == {"+inf", "-inf", "nan"}, (
        f"non-finite stimulus delivered classes {sorted(classes)}; "
        f"distinct patterns {sorted(hex(int(b) & 0xFFFF) for b in set(x_bits.tolist()))}"
    )


@wormhole_only
@pytest.mark.parametrize("op_name", list(_INT_OPS))
def test_equiv_int32(op_name):
    spec_A = _int32_face_spec(_int32_probe_values())
    # UnarySFPUGolden cannot be used here: for an (Int32, Int32) pair it falls
    # through to convert_nan_to_inf, which builds a float inf and overflows the
    # int tensor. calculate_comp_int's semantics are a one-liner and its output
    # is 0/1 in the same format as its input, so there is no pack behaviour to
    # model -- the golden below is the whole specification.
    src_A, res, _ = _drive(
        _INT_OPS[op_name],
        _INT_FORMAT,
        spec_A,
        _INT_TILES,
        DestAccumulation.Yes,
        want_golden=False,
    )

    x_bits = _as_bits(src_A, DataFormat.Int32)
    got_bits = _as_bits(res, DataFormat.Int32)
    want_bits = _INT_GOLDEN[op_name](x_bits).astype(np.int64)

    _assert_bit_exact(f"{op_name}__int32", x_bits, got_bits, want_bits)
