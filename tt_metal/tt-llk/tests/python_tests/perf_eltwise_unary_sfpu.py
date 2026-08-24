# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
    PerfRunType,
    StableSort,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.sfpu_domains import sfpu_unary_ops
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    FRESH_CPP_IMPL,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    STABLE_SORT,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

_OPS_WITHOUT_DEST_ACC = {
    MathOperation.Abs,
    # Acosh/Asinh now select their log1p polynomial precision from the dest-accum
    # (is_fp32_dest_acc_en) flag, so both modes are exercised.
    MathOperation.Celu,
    MathOperation.Cos,
    MathOperation.Elu,
    MathOperation.Exp2,
    MathOperation.Exp,
    MathOperation.Fill,
    MathOperation.Gelu,
    MathOperation.GeluTanh,
    MathOperation.Hardsigmoid,
    MathOperation.Log,
    MathOperation.Neg,
    MathOperation.Silu,
    MathOperation.Sin,
    MathOperation.Square,
    MathOperation.Threshold,
    MathOperation.ReluMax,
    MathOperation.ReluMin,
}

_OPS_WITH_FAST_MODE = {
    MathOperation.Exp,
    MathOperation.Rsqrt,
    MathOperation.Sqrt,
}

_OPS_WITH_STABLE_SORT = {
    MathOperation.TopKLocalSort,
    MathOperation.TopKMerge,
    MathOperation.TopKRebuild,
}


def _get_dest_acc_modes(mathop):
    if mathop in _OPS_WITHOUT_DEST_ACC:
        return [DestAccumulation.No]
    return [DestAccumulation.Yes, DestAccumulation.No]


def _get_fast_modes(mathop):
    if mathop in _OPS_WITH_FAST_MODE:
        return [FastMode.Yes, FastMode.No]
    return [FastMode.No]


def _get_stable_sort_modes(mathop):
    if mathop in _OPS_WITH_STABLE_SORT:
        return [StableSort.Yes, StableSort.No]
    return [StableSort.No]


# Every op with a unary SFPU kernel, taken from the same registry the correctness sweep
# in test_sfpu_unary.py drives, so an op cannot be added there and silently skip perf.
# _UNARY_OPS_NOT_SWEPT is deliberately *not* subtracted: those ops (the topk halves) are
# exempt from the correctness sweep precisely because they are perf-only, so they belong
# here. Sorted so the parametrize ids are stable across runs.
PERF_SWEEP_OPS = sorted(sfpu_unary_ops(), key=lambda op: op.name)

# Five PerfRunTypes per variant, so all 97 registry ops against all 16 format pairs is
# ~30k ELF builds and profiled runs on llk_perf_tests.yaml's five shards, against ~6.4k
# before the reroute -- and it buys little, since an SFPU kernel's math cost is its
# instruction sequence while the format pair moves unpack/pack cycles, which these ops
# already characterise. So every op is still swept (with its own dest_acc / fast_mode /
# stable_sort / approx_mode), but only the pre-reroute set carries the full 16-pair matrix.
_FULL_FORMAT_MATRIX_OPS = frozenset(
    {
        MathOperation.Reciprocal,
        MathOperation.Sqrt,
        MathOperation.Rsqrt,
        MathOperation.Silu,
        MathOperation.Gelu,
        MathOperation.GeluTanh,
        MathOperation.Exp,
        MathOperation.TopKLocalSort,
        MathOperation.TopKMerge,
        MathOperation.TopKRebuild,
    }
)

_FULL_FORMATS = [
    DataFormat.Float32,
    DataFormat.Float16,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
]

# Float16_b in and out: the SFPU's native 16-bit exponent-B format, so the measurement is
# the kernel's own cost with no unpack/pack conversion folded in.
_REPRESENTATIVE_FORMAT = [DataFormat.Float16_b]

_FULL_FORMAT_PAIRS = input_output_formats(_FULL_FORMATS)
_REPRESENTATIVE_FORMAT_PAIRS = input_output_formats(_REPRESENTATIVE_FORMAT)

# An op named here but no longer in the sweep would silently stop being measured on the
# full matrix, which is the one regression this split can cause.
_UNSWEPT_FULL_MATRIX_OPS = sorted(
    op.name for op in _FULL_FORMAT_MATRIX_OPS - set(PERF_SWEEP_OPS)
)
assert not _UNSWEPT_FULL_MATRIX_OPS, (
    "these ops are declared as carrying the full format matrix but are not in "
    f"PERF_SWEEP_OPS: {_UNSWEPT_FULL_MATRIX_OPS}"
)


def _get_formats(mathop):
    if mathop in _FULL_FORMAT_MATRIX_OPS:
        return _FULL_FORMAT_PAIRS
    return _REPRESENTATIVE_FORMAT_PAIRS


def _run(
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    fast_mode,
    stable_sort,
    input_dimensions,
    fresh_cpp_impl=0,
):
    # Calculate tile count from input dimensions
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    # A 32-bit (fp32) input with dest_acc ON unpacks straight into the 32-bit Dest
    # register. With dest_acc OFF it goes through the source registers (converted to 16-bit)
    # and is copied into Dest for the SFPU op.
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = PerfConfig(
        "sources/eltwise_unary_sfpu_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
            FAST_MODE(fast_mode),
            STABLE_SORT(stable_sort),
            CLAMP_NEGATIVE(False),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_count_A),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(num_faces=faces_to_generate),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count_A,
            tile_count_B=tile_count_B,
            tile_count_res=tile_count_A,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    return configuration


@pytest.mark.perf
@parametrize(
    formats=lambda mathop: _get_formats(mathop),
    approx_mode=[
        ApproximationMode.Yes,
        ApproximationMode.No,
    ],
    mathop=PERF_SWEEP_OPS,
    dest_acc=lambda mathop: _get_dest_acc_modes(mathop),
    loop_factor=[
        16,
    ],  # Number of iterations to run the test in order to minimize profiler overhead in measurement
    iterations=[
        32,
    ],  # Number of SFPU iterations
    fast_mode=lambda mathop: _get_fast_modes(mathop),
    stable_sort=lambda mathop: _get_stable_sort_modes(mathop),
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],  # Specifying different input sizes to cover different tile counts
)
def test_perf_eltwise_unary_sfpu(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    fast_mode,
    stable_sort,
    input_dimensions,
):
    _run(
        formats,
        mathop,
        approx_mode,
        dest_acc,
        loop_factor,
        iterations,
        fast_mode,
        stable_sort,
        input_dimensions,
    ).run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    fresh_cpp_impl=[0, 1],
)
def test_perf_exp_fresh_cpp(
    perf_report,
    formats,
    dest_acc,
    fresh_cpp_impl,
):
    _run(
        formats,
        MathOperation.Exp,
        ApproximationMode.No,
        dest_acc,
        16,
        32,
        FastMode.No,
        StableSort.No,
        [128, 64],
        fresh_cpp_impl,
    ).run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    # 0 = production, 1 = fresh semantic cubic, 2 = fresh semantic 3-range
    # magnitude dispatch tree (the LUT-eligible shape).
    fresh_cpp_impl=[0, 1, 2],
)
def test_perf_sigmoid_appx_fresh_cpp(
    perf_report,
    formats,
    dest_acc,
    fresh_cpp_impl,
):
    _run(
        formats,
        MathOperation.SigmoidAppx,
        ApproximationMode.No,
        dest_acc,
        16,
        32,
        FastMode.No,
        StableSort.No,
        [128, 64],
        fresh_cpp_impl,
    ).run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    mathop=[MathOperation.UnaryMax, MathOperation.UnaryMin],
    fresh_cpp_impl=[0, 1],
)
def test_perf_unary_max_min_fresh_cpp(
    perf_report,
    formats,
    dest_acc,
    mathop,
    fresh_cpp_impl,
):
    _run(
        formats,
        mathop,
        ApproximationMode.No,
        dest_acc,
        16,
        32,
        FastMode.No,
        StableSort.No,
        [128, 64],
        fresh_cpp_impl,
    ).run(perf_report)


# Lane BR causal-tier lift perf vehicles: fresh typed-C++ semantic selectors
# (impl 1) A/B'd against the hand-shaped production bodies (impl 0) on the
# same node family — identical stimuli, marker, and metric.  Formats mirror
# each op's measured sweep-row perf leg (Float16_b, dest_acc No — the corr
# axis stays on the row's corr format, the expm1 corr/perf-format precedent).
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    mathop=[
        MathOperation.Ceil,
        MathOperation.EqualZero,
        MathOperation.Clamp,
        MathOperation.Hardtanh,
        MathOperation.Tanh,
        MathOperation.TanhDerivativeLut,
        MathOperation.Silu,
        # Batch 2 (Lane BR): remaining causal-tier lifts.
        MathOperation.Fmod,
        MathOperation.Remainder,
        MathOperation.Xielu,
        MathOperation.UnaryPower,
        MathOperation.Expm1,
        MathOperation.Log,
        MathOperation.Sqrt,
        MathOperation.Rsqrt,
        # Batch 3 (Lane BR).
        MathOperation.Sigmoid,
        MathOperation.Cbrt,
        MathOperation.Softplus,
        MathOperation.Hardsigmoid,
        MathOperation.Gelu,
        MathOperation.Expm1Cw,
        MathOperation.I1,
        # Storm S4 (fresh bodies in fresh_cpp/<op>.h).
        MathOperation.Rdiv,
        MathOperation.Rpow,
        MathOperation.Selu,
        MathOperation.Sign,
        MathOperation.ReluMax,
        MathOperation.Floor,
        MathOperation.Trunc,
        MathOperation.Frac,
        # Storm lane S3 (fresh_cpp/ per-op headers).
        MathOperation.Log1p,
        # Storm lane S1 (fresh_cpp/<op>.h bodies).
        MathOperation.Abs,
        MathOperation.Add1,
        MathOperation.CastFp32ToFp16a,
        MathOperation.Celu,
        # Storm S2 (agent/storm-s2, fresh_cpp/<op>.h canonical bodies).
        MathOperation.Erf,
        MathOperation.Erfc,
        MathOperation.Erfinv,
        MathOperation.Digamma,
        MathOperation.Hardmish,
        MathOperation.Hardshrink,
        MathOperation.Heaviside,
        MathOperation.Elu,
        MathOperation.Exp2,
        MathOperation.Fill,
        # Storm S5.
        MathOperation.Softshrink,
        MathOperation.Softsign,
        MathOperation.Square,
        MathOperation.Tanhshrink,
        MathOperation.Threshold,
        MathOperation.UnaryGe,
        MathOperation.Acosh,
        # laneED sem-only audit: NotEqualZero's production float body is the
        # all-raw-TTI calculate_comp hand kernel; GeluAppx's is the 6-segment
        # SFPLUTFP32 hand kernel.  impl 1 = the new fresh semantic arms.
        MathOperation.NotEqualZero,
        MathOperation.GeluAppx,
    ],
    fresh_cpp_impl=[0, 1],
)
def test_perf_causal_lift_fresh_cpp(
    perf_report,
    formats,
    dest_acc,
    mathop,
    fresh_cpp_impl,
):
    _run(
        formats,
        mathop,
        ApproximationMode.No,
        dest_acc,
        16,
        32,
        FastMode.No,
        StableSort.No,
        [128, 64],
        fresh_cpp_impl,
    ).run(perf_report)


# laneED sem-only audit: hand-LUT-variant pairs.  impl 3 = the byte-untouched
# hand LUT kernel (Tanh: production calculate_tanh<APPROXIMATION_MODE=true>,
# reached by building the impl-3 leg with approx mode ON — no cpp branch;
# Sigmoid: the legacy tt-llk 6-segment SFPLUTFP32 _calculate_sigmoid_, wired
# by the impl-3 selector + init override in eltwise_unary_sfpu_perf.cpp).
# impl 1 = the fresh semantic body (same bodies the causal-lift family
# measures; separate node ids so each LUT row owns its cells).
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    mathop=[
        MathOperation.Tanh,
        MathOperation.Sigmoid,
        # Lane GI (owner ratification 2026-08-24 item 2): GeluAppx joins so
        # the licensed impl-4 sem arm gets a perf node in this family (its
        # hand perf node stays test_perf_causal_lift_fresh_cpp impl 0,
        # byte-untouched).  impl 3 for GeluAppx falls through to the
        # production dispatch (hand-equivalent, unswept).
        MathOperation.GeluAppx,
        # Lane GI gelu-255: impl 4 = the licensed slim body
        # (fresh_cpp/gelu_255_licensed.h); impl 3 falls through to the
        # production typed gelu (hand-equivalent, unswept).
        MathOperation.Gelu,
    ],
    # impl 4 = the lane-GI LICENSED semantic arms (fresh_cpp/*_licensed.h).
    fresh_cpp_impl=[1, 3, 4],
)
def test_perf_lut_variant_fresh_cpp(
    perf_report,
    formats,
    dest_acc,
    mathop,
    fresh_cpp_impl,
):
    approx = (
        ApproximationMode.Yes
        if (fresh_cpp_impl == 3 and mathop == MathOperation.Tanh)
        else ApproximationMode.No
    )
    _run(
        formats,
        mathop,
        approx,
        dest_acc,
        16,
        32,
        FastMode.No,
        StableSort.No,
        [128, 64],
        fresh_cpp_impl,
    ).run(perf_report)


# Lane CM fitted-kernel placeholders (tt-polynomial-fitter frontier
# selections, PLACEHOLDER-PENDING-UPSTREAM-MERGE; provenance headers in
# fresh_cpp/*_fitted.h).  Dedicated perf family, all-new node ids: impl 2 =
# fitted body (sem arm), impl 0 = production hand kernel (hand arm).
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    mathop=[
        MathOperation.Tanh,
        MathOperation.Sigmoid,
        MathOperation.Gelu,
        MathOperation.TanhDerivative,
        # Lane CR wave 2 (same contract; provenance in fresh_cpp/*_fitted.h).
        MathOperation.Digamma,
        MathOperation.Lgamma,
        MathOperation.Polygamma,
        MathOperation.I0,
        MathOperation.I1,
        MathOperation.Mish,
        MathOperation.Log,
        MathOperation.Log1p,
        MathOperation.Exp,
        MathOperation.Rsqrt,
        MathOperation.Celu,
        MathOperation.Elu,
        MathOperation.Selu,
        # Lane CW wave 3 (rlibm rounding-interval refits of the CR wave-2
        # honest-outs; provenance in fresh_cpp/*_fitted.h).
        MathOperation.Threshold,
        MathOperation.Expm1,
        MathOperation.Acosh,
    ],
    fresh_cpp_impl=[0, 2],
)
def test_perf_fitted_cpp(
    perf_report,
    formats,
    dest_acc,
    mathop,
    fresh_cpp_impl,
):
    _run(
        formats,
        mathop,
        ApproximationMode.No,
        dest_acc,
        16,
        32,
        FastMode.No,
        StableSort.No,
        [128, 64],
        fresh_cpp_impl,
    ).run(perf_report)


# Perf vehicle for the corr-only isinf/isnan corpus row (Lane BK measurement-
# gap close): the comp-class predicate ops live outside the sfpu_unary_ops()
# domain registry (their functional sweep is the dedicated
# test_eltwise_unary_sfpu_isinf_isnan with specials-injecting stimuli), so
# PERF_SWEEP_OPS never picks them up. One MATH_ISOLATE node per predicate at
# the representative Float16_b pair (the corr axis stays F32 — the expm1
# corr/perf-format precedent).
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    mathop=[
        MathOperation.Isinf,
        MathOperation.Isposinf,
        MathOperation.Isneginf,
        MathOperation.Isnan,
        MathOperation.Isfinite,
    ],
)
def test_perf_isinf_isnan(
    perf_report,
    formats,
    dest_acc,
    mathop,
):
    _run(
        formats,
        mathop,
        ApproximationMode.No,
        dest_acc,
        16,
        32,
        FastMode.No,
        StableSort.No,
        [128, 64],
    ).run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float32]),
    dest_acc=[DestAccumulation.No],
    fresh_cpp_impl=[0, 1],
)
def test_perf_signbit_fresh_cpp(
    perf_report,
    formats,
    dest_acc,
    fresh_cpp_impl,
):
    _run(
        formats,
        MathOperation.Signbit,
        ApproximationMode.No,
        dest_acc,
        16,
        32,
        FastMode.No,
        StableSort.No,
        [128, 64],
        fresh_cpp_impl,
    ).run(perf_report)
