# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-segment error bounds on the three retuned Wormhole SFPU LUT kernels.

`tanh` (APPROXIMATION_MODE), `sigmoid_appx` and `gelu_appx` get their piecewise-linear
coefficients from `SFPLUT` / `SFPLUTFP32` immediates loaded in the kernel's `*_init()`.
Those immediates were retuned from interpolants to per-segment minimax lines
(`tech_reports/SFPU_LUT_Retune_Wormhole/`), and nothing in the pre-existing suites fails
if they are put back:

* the tt-llk sweep gives `sigmoid_appx` and `gelu_appx` `(atol, rtol) = (0.13, 0.05)`,
  which the old `sigmoid_appx` defect (0.1036) sits inside, and skips `tanh` in
  approximation mode entirely;
* `ttnn.tanh(..., fast_and_approximate_mode=True)` *is* exercised by
  `tests/ttnn/unit_tests/operations/eltwise/test_unary.py::test_unary_tanh_approx_ttnn`,
  but at `atol = 0.15`, which the old table's 0.1447 also passes.

So this test measures instead of correlating: a ladder over every LUT segment on
Float32 -> Float32 / dest_acc=Yes (the only configuration where the pack does not quantise
the answer to ~2^-8 and hide the difference), and a `max|err|` bound per segment.

Each bound is derived, not observed. For a segment fitted by `y = A*a + B` the exact
worst-case error is `sup |A*a + B - f(a)|` over that segment, which is computable to full
precision from the decoded coefficients, and it matches the measured hardware maxima to
six decimals. Most bounds below therefore sit strictly between the retuned segment's
analytic sup and the pre-retune one, so restoring the old immediates fails that row. The
rest -- the three segments the retune deliberately left alone, plus the one where old and
new are too close to separate -- sit at ~1.35x the sup as a plain regression guard. The
`discriminates` column says which of the two a row is.

Wormhole only: Blackhole ships its own copy of these tables and was deliberately left on
the old coefficients, since the fit is arch-independent but the measurement is not.
"""

import math

import pytest
import torch
from conftest import wormhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
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

# StimuliSpec.custom writes its values at the start of each 16x16 face and zero-fills the
# rest, so the whole ladder -- both signs -- has to fit in one face.
_FACE_ELEMENTS = 16 * 16

# Far-field probes, shared by all three ops: past the last breakpoint every table is a
# pinned constant (or, for gelu_appx, the identity), so these check saturation rather
# than fit quality.
_SATURATION = [8.0, 16.0, 128.0]


def _exact_sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _exact_gelu(x):
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Per-segment bounds.
#
#   (lo, hi, bound, retuned_sup, pre_retune_sup, discriminates)
#
# `retuned_sup` / `pre_retune_sup` are the analytic sup|A*a + B - f(a)| of the segment's
# line against the exact function -- the numbers the bound is placed between, recorded so
# a future retune can re-derive the bound instead of guessing at it. `discriminates` is
# True when the bound is below the pre-retune sup, i.e. when restoring the old immediates
# fails this row rather than merely being allowed by it.
# ─────────────────────────────────────────────────────────────────────────────

_TANH_SEGMENTS = [
    (0.0, 1.0, 0.0700, 0.056339, 0.144653, True),
    (1.0, 2.0, 0.0650, 0.050906, 0.144656, True),
    # Pinned tail: the constant 1.0 is unchanged, so this row is a guard only.
    (2.0, math.inf, 0.0400, 0.035972, 0.035972, False),
]

_SIGMOID_SEGMENTS = [
    (0.0, 1.0, 0.0070, 0.005275, 0.009755, True),
    # The defect: the old line's slope exceeded segment 0's, which a concave target
    # cannot use, and the error ran to 0.1036 by the top of the segment.
    (1.0, 2.0, 0.0120, 0.007223, 0.103577, True),
    # Structural and untouched: with the last breakpoint fixed at |x| = 2 this segment
    # must be a constant, and sigmoid(inf) = 1.0 forces it to 0.5 against
    # sigmoid(2) = 0.8808. Guard only.
    (2.0, math.inf, 0.1250, 0.119203, 0.119203, False),
]

_GELU_SEGMENTS = [
    (0.0, 0.5, 0.0160, 0.011609, 0.023424, True),
    (0.5, 1.0, 0.0079, 0.006836, 0.009281, True),
    (1.0, 1.5, 0.0018, 0.001501, 0.002145, True),
    # Old and new sups are only 1.25x apart here, too close to separate reliably. Guard.
    (1.5, 2.0, 0.0020, 0.001443, 0.001800, False),
    (2.0, 3.0, 0.0055, 0.004717, 0.006500, True),
    # Pinned at (0.5, 0.0), which makes the kernel exact here (x for x >= 3, 0 for
    # x <= -3) and leaves only gelu's own distance from its asymptote. Unchanged; guard.
    (3.0, math.inf, 0.0055, 0.004050, 0.004050, False),
]

_CASES = {
    "tanh": (
        MathOperation.Tanh,
        ApproximationMode.Yes,
        math.tanh,
        _TANH_SEGMENTS,
        40,
    ),
    "sigmoid_appx": (
        MathOperation.SigmoidAppx,
        ApproximationMode.No,
        _exact_sigmoid,
        _SIGMOID_SEGMENTS,
        40,
    ),
    "gelu_appx": (
        MathOperation.GeluAppx,
        ApproximationMode.No,
        _exact_gelu,
        _GELU_SEGMENTS,
        20,
    ),
}


def _ladder(segments, per_segment):
    """Probe points for one op: `per_segment` samples inside every LUT segment, mirrored.

    Points land *just inside* each segment rather than on its breakpoint. The comparison
    the hardware uses at a breakpoint is not documented as `<` versus `<=`, and a probe
    sitting exactly on one would be attributed to whichever side this file guessed at.
    Nudging in costs nothing: every sup that lives at a segment edge is approached to
    within a part in a thousand of the segment width, which is far inside the margin
    each bound carries.
    """
    points = [0.0]
    for lo, hi, *_ in segments:
        # The open-ended tail is sampled to 8; _SATURATION carries it further out.
        top = 8.0 if math.isinf(hi) else hi
        eps = (top - lo) * 1e-3
        start, stop = lo + eps, top - eps
        step = (stop - start) / (per_segment - 1)
        points += [start + step * i for i in range(per_segment)]
    points += _SATURATION
    points = sorted(set(points))
    mirrored = [-p for p in reversed(points) if p != 0.0] + points
    assert len(mirrored) <= _FACE_ELEMENTS, (
        f"{len(mirrored)} probes exceed the {_FACE_ELEMENTS}-element face that "
        "StimuliSpec.custom fills"
    )
    return mirrored


def _run(mathop, approx_mode, probes):
    """Drive one op over `probes` and return (x, y) as float32 tensors.

    Deliberately not routed through `eltwise_unary_sfpu`: that helper asserts against the
    golden generator with the sweep's own tolerances and discards the raw result, and the
    raw result per probe point is the entire subject of this test.
    """
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes
    input_dimensions = [64, 64]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.custom(values=probes, seed=0),
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(approx_mode),
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
        unpack_to_dest=True,
    )

    res_from_L1 = configuration.run().result
    y = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format]).to(
        torch.float32
    )
    x = src_A.to(torch.float32)
    assert len(x) == len(y), "result and stimulus differ in length"

    # Every face carries the same ladder followed by zero fill; the leading probes are
    # enough, and taking only them keeps the zero padding out of segment 0's statistics.
    n = len(probes)
    return x[:n], y[:n]


@wormhole_only
@pytest.mark.parametrize("op", list(_CASES))
def test_lut_retune_error_bounds(op):
    mathop, approx_mode, exact, segments, per_segment = _CASES[op]
    probes = _ladder(segments, per_segment)
    x, y = _run(mathop, approx_mode, probes)

    failures = []
    for lo, hi, bound, retuned_sup, pre_retune_sup, discriminates in segments:
        a = x.abs()
        mask = (a >= lo) & (a < hi)
        assert bool(mask.any()), f"{op}: no probe landed in [{lo}, {hi})"

        err = torch.tensor(
            [abs(float(yi) - exact(float(xi))) for xi, yi in zip(x[mask], y[mask])]
        )
        worst = int(err.argmax())
        max_err = float(err[worst])
        if max_err > bound:
            kind = "regressed" if discriminates else "guard breached"
            failures.append(
                f"  |x| in [{lo}, {hi}): max|err| = {max_err:.6f} > {bound:.4f} "
                f"({kind}; retuned sup {retuned_sup:.6f}, pre-retune sup "
                f"{pre_retune_sup:.6f}) at x = {float(x[mask][worst]):.6f}, "
                f"got {float(y[mask][worst]):.6f}, want "
                f"{exact(float(x[mask][worst])):.6f}"
            )

    assert not failures, (
        f"{op}: retuned LUT coefficients do not meet their per-segment error bounds. "
        f"The most likely cause is that the immediates in the kernel's *_init() were "
        f"changed or reverted; see tech_reports/SFPU_LUT_Retune_Wormhole/.\n"
        + "\n".join(failures)
    )


@wormhole_only
@pytest.mark.parametrize("op", list(_CASES))
def test_lut_retune_invariants(op):
    """The structural properties the retuned tables are documented to preserve.

    These are what makes the retune a drop-in rather than a behaviour change, and none of
    them is implied by the error bounds above: a table can meet every bound and still
    lose monotonicity, drift off its saturation value, or move the value at zero.
    """
    mathop, approx_mode, exact, segments, per_segment = _CASES[op]
    probes = _ladder(segments, per_segment)
    x, y = _run(mathop, approx_mode, probes)

    order = torch.argsort(x)
    xs, ys = x[order], y[order]

    at_zero = float(ys[int((xs == 0.0).nonzero()[0][0])])
    far = {
        float(xi): float(yi) for xi, yi in zip(xs, ys) if abs(float(xi)) in _SATURATION
    }

    # Monotone non-decreasing, for the two ops whose target is. `gelu_appx` is excluded
    # because gelu genuinely is not monotone -- it dips below zero around x = -0.75 -- and
    # the table reproduces that dip: on [1, 3) the fitted slope exceeds 0.5, so the
    # 0.5*x + fit(|x|) sum decreases on the negative side. Requiring monotonicity there
    # would be asserting a property the function does not have.
    if op != "gelu_appx":
        diffs = ys[1:] - ys[:-1]
        bad = int((diffs < 0).nonzero()[0][0]) if bool((diffs < 0).any()) else None
        assert bad is None, (
            f"{op} is not monotone non-decreasing: y({float(xs[bad]):.6f}) = "
            f"{float(ys[bad]):.6f} > y({float(xs[bad + 1]):.6f}) = "
            f"{float(ys[bad + 1]):.6f}"
        )

    if op == "tanh":
        # Segment 0's intercept byte is 0xFF, which reads back as exactly 0.0.
        assert at_zero == 0.0, f"tanh_appx(0) must be exactly 0, got {at_zero}"
        for xi, yi in far.items():
            assert yi == math.copysign(1.0, xi), (
                f"tanh_appx must saturate at exactly {math.copysign(1.0, xi)} "
                f"for x = {xi}, got {yi}"
            )
    elif op == "sigmoid_appx":
        assert at_zero == 0.5, f"sigmoid_appx(0) must be exactly 0.5, got {at_zero}"
        for xi, yi in far.items():
            want = 1.0 if xi > 0 else 0.0
            assert yi == want, (
                f"sigmoid_appx must saturate at exactly {want} "
                f"for x = {xi}, got {yi}"
            )
    else:
        # gelu_appx is 0.5*x + lut2_sign(x), and lut2_sign is SFPLUTFP32 with SGN_UPDATE,
        # so the table term is a function of |x| only -- even, not odd, despite the name.
        # With the [3, inf) segment pinned at (A, B) = (0.5, 0.0) that makes the tail
        # exact on both sides to the last bit: 0.5*x + 0.5*|x| is x for x >= 3 and 0 for
        # x <= -3, against a true gelu of x - 0.004 and -0.004 respectively.
        #
        # Any other slope there makes the absolute error grow without bound -- a free
        # minimax fit proposes 0.5004883 because it halves the error at x = 3, and at
        # x = 128 that returns 128.06. No error-bound row above would catch it: they all
        # stop at a finite probe.
        for i in (x.abs() >= 3.0).nonzero().flatten().tolist():
            xi, yi = float(x[i]), float(y[i])
            want = xi if xi > 0 else 0.0
            assert yi == want, (
                f"gelu_appx's [3, inf) segment is pinned at (0.5, 0.0), so "
                f"gelu_appx({xi}) must be exactly {want}, got {yi}"
            )
