# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
    StableSort,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, PerfConfig
from helpers.sfpu_domains import sfpu_unary_ops
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
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
# in test_eltwise_unary_sfpu.py drives, so an op cannot be added there and silently skip perf.
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
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
            FAST_MODE(fast_mode),
            STABLE_SORT(stable_sort),
            CLAMP_NEGATIVE(False),
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

    configuration.run(perf_report)


# Extra slices for formats the main sweep does not cover. They must use
# ALL_PERF_RUN_TYPES so this module's single CSV has one column schema.
_EXTRA_SLICE_DIMS = [[128, 64]]  # tile_cnt: 8

# AddInt32/SubInt32 are perf-only until the int32-unary functional sweep is
# unblocked (tt-llk #495); integer add/sub is covered via the binary path.
_INT32_UNARY_OPS = [
    MathOperation.AddInt32,
    MathOperation.SubInt32,
    MathOperation.AbsInt32,
    MathOperation.ReluMin,
    MathOperation.BitwiseNot,
    MathOperation.LogicalNot,
    MathOperation.Fill,
]

# uint16 (calculate_comp_uint16) and uint32 (calculate_eqz/nez_uint32) only support
# equality against zero.
_UINT_COMP_OPS = [
    MathOperation.EqualZero,
    MathOperation.NotEqualZero,
]


def _extra_slice_config(formats, mathop, dest_acc, unpack_to_dest, input_dimensions):
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    return PerfConfig(
        "sources/eltwise_unary_sfpu_perf.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FAST_MODE(FastMode.No),
            STABLE_SORT(StableSort.No),
            CLAMP_NEGATIVE(False),
        ],
        runtimes=[
            TILE_COUNT(tile_count_A),
            LOOP_FACTOR(16),
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


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32], same=True),
    mathop=_INT32_UNARY_OPS,
    input_dimensions=_EXTRA_SLICE_DIMS,
)
def test_perf_eltwise_unary_sfpu_int32(perf_report, formats, mathop, input_dimensions):
    _extra_slice_config(
        formats,
        mathop,
        DestAccumulation.No,
        formats.input_format.is_32_bit(),
        input_dimensions,
    ).run(perf_report)


def _comp_dest_acc(formats):
    # 32-bit inputs (UInt32) are unpacked straight into DEST and the uint32 kernels do
    # full-word DataLayout::U32 accesses, so they require a 32-bit DEST. Mirror the
    # correctness test (test_eltwise_unary_sfpu_int) and constraints.py, which force
    # dest_acc=Yes for 32-bit formats; hardcoding No here would measure an unsupported config.
    if formats.input_format.is_32_bit():
        return DestAccumulation.Yes
    return DestAccumulation.No


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.UInt16], same=True),
    mathop=_UINT_COMP_OPS,
    input_dimensions=_EXTRA_SLICE_DIMS,
)
def test_perf_eltwise_unary_sfpu_comp_uint16(
    perf_report, formats, mathop, input_dimensions
):
    dest_acc = _comp_dest_acc(formats)
    _extra_slice_config(
        formats,
        mathop,
        dest_acc,
        formats.input_format.is_32_bit(),
        input_dimensions,
    ).run(perf_report)


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.UInt32], same=True),
    mathop=_UINT_COMP_OPS,
    input_dimensions=_EXTRA_SLICE_DIMS,
)
def test_perf_eltwise_unary_sfpu_comp_uint32(
    perf_report, formats, mathop, input_dimensions
):
    dest_acc = _comp_dest_acc(formats)
    _extra_slice_config(
        formats,
        mathop,
        dest_acc,
        formats.input_format.is_32_bit(),
        input_dimensions,
    ).run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float32]),
    mathop=[MathOperation.Erfinv],
    input_dimensions=_EXTRA_SLICE_DIMS,
)
def test_perf_eltwise_unary_sfpu_erfinv_fp32(
    perf_report, formats, mathop, input_dimensions
):
    dest_acc = DestAccumulation.Yes
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    _extra_slice_config(
        formats, mathop, dest_acc, unpack_to_dest, input_dimensions
    ).run(perf_report)
