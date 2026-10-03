# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.constraints import distinct_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)


def get_dest_accum_modes(formats):
    if formats.input_format.is_32_bit() and formats.input_format.is_integer():
        return [DestAccumulation.No]
    # TestConfig promotes dest_acc=No to Yes for outlier format combos, so asking
    # for both would record two rows with an identical key (the same kernel twice).
    return distinct_dest_accumulation_modes(
        formats, [DestAccumulation.Yes, DestAccumulation.No]
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    approx_mode=[
        ApproximationMode.Yes,
        ApproximationMode.No,
    ],
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
        MathOperation.SfpuElwdiv,
        MathOperation.SfpuElwrsub,
        MathOperation.SfpuElwpow,
        MathOperation.SfpuXlogy,
    ],
    dest_acc=lambda formats: get_dest_accum_modes(formats),
    loop_factor=[
        16,
    ],  # Number of iterations to run the test in order to minimize profiler overhead in measurement
    iterations=[
        32,
    ],
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],  # Specifying different input sizes to cover different tile counts
)
def test_perf_eltwise_binary_sfpu_float(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    approx_mode=[
        ApproximationMode.Yes,
        ApproximationMode.No,
    ],
    mathop=[
        MathOperation.SfpuElwRightShift,
        MathOperation.SfpuElwLeftShift,
        MathOperation.SfpuElwLogicalRightShift,
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
    ],
    dest_acc=lambda formats: get_dest_accum_modes(formats),
    loop_factor=[
        16,
    ],
    iterations=[
        32,
    ],
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],
)
def test_perf_eltwise_binary_sfpu_int(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.UInt32,
        ],
        same=True,
    ),
    approx_mode=[
        ApproximationMode.Yes,
        ApproximationMode.No,
    ],
    mathop=[
        MathOperation.SfpuAddTopRow,
    ],
    dest_acc=lambda formats: get_dest_accum_modes(formats),
    loop_factor=[
        16,
    ],
    iterations=[
        32,
    ],
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],
)
def test_perf_eltwise_binary_sfpu_add_top_row(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    chip_arch = get_chip_architecture()

    # Skip DestAccumulation.No on Blackhole for SfpuAddTopRow
    if chip_arch == ChipArchitecture.BLACKHOLE and dest_acc == DestAccumulation.No:
        pytest.skip(
            "DestAccumulation.No is not supported for SfpuAddTopRow on Blackhole"
        )

    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.Yes:
        pytest.skip("SfpuAddTopRow does not support Float32 with DestAccumulation.Yes")

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report)


import numpy as np


def _paired_reference_gelu_bw(x):
    def _declared_derivative(x):
        import math

        erfc = np.vectorize(math.erfc, otypes=[np.float64])
        exp = np.exp
        pi = np.pi
        sqrt = np.sqrt
        return np.broadcast_to(
            np.asarray(
                0.5 * erfc(-x / sqrt(2)) + x * exp(-(x**2) / 2) / sqrt(2 * pi),
                dtype=np.float64,
            ),
            x.shape,
        )

    return _declared_derivative(x)


def _paired_reference_softsign_bw(x):
    def _declared_derivative(x):
        abs = np.abs
        return np.broadcast_to(
            np.asarray(1 / (1 + abs(x)) ** 2, dtype=np.float64), x.shape
        )

    return _declared_derivative(x)


def _paired_reference_relu_bw(x):
    raw = np.asarray(x, dtype=np.float32).view(np.uint32) >> 16
    result = np.zeros(raw.shape, dtype=np.uint32)
    for first, stop, word in ((0, 1, 0), (1, 32641, 16256), (32641, 65536, 0)):
        result[(raw >= first) & (raw < stop)] = word << 16
    return result.view(np.float32).astype(np.float64)


def _paired_reference_tanhshrink_bw(x):
    def _declared_derivative(x):
        tanh = np.tanh
        return np.broadcast_to(np.asarray(tanh(x) ** 2, dtype=np.float64), x.shape)

    return _declared_derivative(x)


def _paired_reference_erf_bw(x):
    def _declared_derivative(x):
        exp = np.exp
        pi = np.pi
        sqrt = np.sqrt
        return np.broadcast_to(
            np.asarray(2 * exp(-(x**2)) / sqrt(pi), dtype=np.float64), x.shape
        )

    return _declared_derivative(x)


def _paired_reference_leaky_relu_bw(x):
    raw = np.asarray(x, dtype=np.float32).view(np.uint32) >> 16
    result = np.zeros(raw.shape, dtype=np.uint32)
    for first, stop, word in ((0, 1, 15396), (1, 32641, 16256), (32641, 65536, 15396)):
        result[(raw >= first) & (raw < stop)] = word << 16
    return result.view(np.float32).astype(np.float64)


def _paired_reference_elu_bw(x):
    def _declared_derivative(x):
        abs = np.abs
        exp = np.exp
        return np.broadcast_to(
            np.asarray(exp((x - abs(x)) / 2), dtype=np.float64), x.shape
        )

    return _declared_derivative(x)


def _paired_reference_selu_bw(x):
    def _declared_piece_0(x):
        exp = np.exp
        return np.broadcast_to(
            np.asarray(
                1.0507009873554805 * 1.6732632423543772 * exp(x), dtype=np.float64
            ),
            x.shape,
        )

    def _declared_piece_1(x):
        return np.broadcast_to(
            np.asarray(1.0507009873554805, dtype=np.float64), x.shape
        )

    def _declared_derivative(x):
        result = np.full(x.shape, np.nan)
        finite = np.isfinite(x)
        bins = np.searchsorted((0.0,), x, side="right")
        bins[finite & (x == 0.0)] = 0
        active = finite & (bins == 0)
        result[active] = _declared_piece_0(x[active])
        active = finite & (bins == 1)
        result[active] = _declared_piece_1(x[active])
        result[np.isnan(x)] = np.nan
        result[np.isneginf(x)] = 0.0
        result[np.isposinf(x)] = 1.0507009873554805
        return result

    return _declared_derivative(x)


def _paired_reference_celu_bw(x):
    def _declared_derivative(x):
        abs = np.abs
        exp = np.exp
        return np.broadcast_to(
            np.asarray(exp((x - abs(x)) / 2), dtype=np.float64), x.shape
        )

    return _declared_derivative(x)


def _paired_reference_hardshrink_bw(x):
    raw = np.asarray(x, dtype=np.float32).view(np.uint32) >> 16
    result = np.zeros(raw.shape, dtype=bool)
    for first, stop in ((16129, 32768), (48897, 65536)):
        result |= (raw >= first) & (raw < stop)
    return result.astype(np.float64) * 1.0


def _paired_reference_hardtanh_bw(x):
    raw = np.asarray(x, dtype=np.float32).view(np.uint32) >> 16
    result = np.zeros(raw.shape, dtype=bool)
    for first, stop in ((0, 16256), (32641, 49024), (65409, 65536)):
        result |= (raw >= first) & (raw < stop)
    return result.astype(np.float64) * 1.0


def _paired_reference_hardswish_bw(x):
    def _declared_piece_0(x):
        return np.broadcast_to(np.asarray(0, dtype=np.float64), x.shape)

    def _declared_piece_1(x):
        return np.broadcast_to(np.asarray(x / 3 + 1 / 2, dtype=np.float64), x.shape)

    def _declared_piece_2(x):
        return np.broadcast_to(np.asarray(x / 3 + 1 / 2, dtype=np.float64), x.shape)

    def _declared_piece_3(x):
        return np.broadcast_to(np.asarray(1, dtype=np.float64), x.shape)

    def _declared_derivative(x):
        result = np.full(x.shape, np.nan)
        finite = np.isfinite(x)
        bins = np.searchsorted((-3.0, -1.5, 3.0), x, side="right")
        bins[finite & (x == 3.0)] = 2
        active = finite & (bins == 0)
        result[active] = _declared_piece_0(x[active])
        active = finite & (bins == 1)
        result[active] = _declared_piece_1(x[active])
        active = finite & (bins == 2)
        result[active] = _declared_piece_2(x[active])
        active = finite & (bins == 3)
        result[active] = _declared_piece_3(x[active])
        result[np.isnan(x)] = 1.0
        result[np.isneginf(x)] = 0.0
        result[np.isposinf(x)] = 1.0
        return result

    return _declared_derivative(x)


def _paired_reference_silu_bw(x):
    def _declared_derivative(x):
        exp = np.exp
        return np.broadcast_to(
            np.asarray(
                1 / (1 + exp(-x)) * (1 + x * (1 - 1 / (1 + exp(-x)))), dtype=np.float64
            ),
            x.shape,
        )

    return _declared_derivative(x)


def _paired_reference_softplus_bw(x):
    def _declared_derivative(x):
        exp = np.exp
        return np.broadcast_to(np.asarray(1 / (1 + exp(-x)), dtype=np.float64), x.shape)

    return _declared_derivative(x)


def _paired_reference_log_sigmoid_bw(x):
    def _declared_derivative(x):
        exp = np.exp
        return np.broadcast_to(np.asarray(1 / (1 + exp(x)), dtype=np.float64), x.shape)

    return _declared_derivative(x)


def _paired_reference_hardsigmoid_bw(x):
    raw = np.asarray(x, dtype=np.float32).view(np.uint32) >> 16
    result = np.zeros(raw.shape, dtype=bool)
    for first, stop in ((0, 16448), (32641, 49216), (65409, 65536)):
        result |= (raw >= first) & (raw < stop)
    return result.astype(np.float64) * 0.1666666716337204


def _paired_reference_softshrink_bw(x):
    raw = np.asarray(x, dtype=np.float32).view(np.uint32) >> 16
    result = np.zeros(raw.shape, dtype=bool)
    for first, stop in ((16129, 32641), (48897, 65409)):
        result |= (raw >= first) & (raw < stop)
    return result.astype(np.float64) * 1.0


import importlib

import numpy as np


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


def _apply_finite_constants(golden, coordinate, domain_rows):
    resolved = np.zeros(coordinate.shape, dtype=bool)
    for direction, bound, inclusive, kind, value in domain_rows:
        if direction == "below":
            owned = coordinate <= bound if inclusive else coordinate < bound
        else:
            owned = coordinate >= bound if inclusive else coordinate > bound
        owned &= np.isfinite(coordinate) & ~resolved
        resolved |= owned
        if kind == "constant":
            golden[owned] = _bf16_round_ftz(
                np.full(np.count_nonzero(owned), value, dtype=np.float64)
            )
    return golden


def _tt_poly_reference_hardswish(x):
    def _declared_piece_0(x):
        return np.broadcast_to(np.asarray(0, dtype=np.float64), x.shape)

    def _declared_piece_1(x):
        return np.broadcast_to(np.asarray(x * (x / 6 + 0.5), dtype=np.float64), x.shape)

    def _declared_piece_2(x):
        return np.broadcast_to(np.asarray(x, dtype=np.float64), x.shape)

    def _declared_forward(x):
        result = np.full(x.shape, np.nan)
        finite = np.isfinite(x)
        bins = np.searchsorted((-3.0, 3.0), x, side="right")
        active = finite & (bins == 0)
        result[active] = _declared_piece_0(x[active])
        active = finite & (bins == 1)
        result[active] = _declared_piece_1(x[active])
        active = finite & (bins == 2)
        result[active] = _declared_piece_2(x[active])
        return result

    return torch.from_numpy(_declared_forward(x.double().numpy()))


def _tt_poly_reference_lgamma(x):
    return getattr(importlib.import_module("torch"), "lgamma")(x.double(), **{})


def _tt_poly_reference_logit(x):
    return getattr(importlib.import_module("torch"), "logit")(x.double(), **{})


def _tt_poly_reference_logsigmoid(x):
    return getattr(importlib.import_module("torch.nn.functional"), "logsigmoid")(
        x.double(), **{}
    )


def _tt_poly_reference_multigammaln(x):
    return getattr(importlib.import_module("torch.special"), "multigammaln")(
        x.double(), **{"p": 4}
    )


def _tt_poly_reference_multigammaln_p4(x):
    return getattr(importlib.import_module("torch.special"), "multigammaln")(
        x.double(), **{"p": 4}
    )


_TT_POLY_FORWARD_REFERENCES = {
    "hardswish": (
        _tt_poly_reference_hardswish,
        ((0, 1), (128, 32640), (32768, 32769), (32896, 65408)),
        (16447, 16448, 16449, 49215, 49216, 49217),
        (),
    ),
    "lgamma": (
        _tt_poly_reference_lgamma,
        ((0, 1), (128, 31813), (32768, 32769), (32896, 65408)),
        (31812,),
        (),
    ),
    "logit": (_tt_poly_reference_logit, ((128, 16256),), (16255,), ()),
    "logsigmoid": (
        _tt_poly_reference_logsigmoid,
        ((0, 1), (128, 32640), (32768, 32769), (32896, 65408)),
        (17081, 17082, 17083, 32638, 32639, 49439, 49440, 49441),
        (
            ("above", 3.3895313892515355e38, False, "return_class", "pos_zero"),
            ("below", -10.0, False, "identity", None),
            ("above", 93.0, True, "constant", 0.0),
        ),
    ),
    "multigammaln": (
        _tt_poly_reference_multigammaln,
        ((16321, 31560),),
        (16321, 31559),
        (),
    ),
    "multigammaln_p4": (
        _tt_poly_reference_multigammaln_p4,
        ((16321, 31560),),
        (16321, 31559),
        (),
    ),
}


def _tt_poly_forward_arguments(op):
    if op not in _TT_POLY_FORWARD_REFERENCES:
        return {}
    reference, intervals, boundaries, domain_rows = _TT_POLY_FORWARD_REFERENCES[op]

    def golden_for(inputs):
        golden = reference(inputs).double().numpy()
        golden = _apply_finite_constants(golden, inputs.double().numpy(), domain_rows)
        return torch.from_numpy(golden).to(torch.bfloat16)

    raw = np.concatenate(
        [np.arange(first, stop, dtype=np.uint32) for first, stop in intervals]
    )
    values = torch.from_numpy((raw << 16).view(np.float32))
    with np.errstate(all="ignore"):
        golden = golden_for(values)
    finite = torch.isfinite(golden).numpy()
    raw, values = raw[finite], values[finite]
    assert len(raw), "declared reference has no finite BF16 LLK probes"
    boundary_values = values[np.isin(raw, boundaries)]

    def distribution(size, dtype, generator):
        indices = np.linspace(0, len(raw) - 1, size, dtype=np.int64)
        samples = values[indices].clone()
        samples[: len(boundary_values)] = boundary_values
        return samples.to(dtype)

    return {
        "spec_A": StimuliSpec(distribution=distribution, seed=0),
        "declared_golden": golden_for,
    }


_PAIRED_FORWARD = (
    "hardswish",
    "lgamma",
    "logit",
    "logsigmoid",
    "multigammaln",
    "multigammaln_p4",
)
_PAIRED_FORWARD_FP32 = {
    "hardswish": (),
    "logsigmoid": (),
    "logit": (),
    "lgamma": (),
    "multigammaln": ("blackhole", "wormhole"),
    "multigammaln_p4": ("blackhole", "wormhole"),
}

_PAIRED_CONSTANTS = {
    "erf_bw": (1.1283791670955126,),
    "leaky_relu_bw": (0.01,),
    "elu_bw": (1.0,),
    "selu_bw": (1.0507, 1.67326),
    "celu_bw": (1.0, 0.0),
    "hardtanh_bw": (-1.0, 1.0),
    "hardswish_bw": (-3.0, 3.0, 0.3333, 0.5),
    "silu_bw": (1.0,),
    "softplus_bw": (1.0, -20.0, 1.0),
    "log_sigmoid_bw": (1.0,),
    "hardsigmoid_bw": (-3.0, 3.0, 0.16666666666666666),
    "softshrink_bw": (-0.5, 0.5),
    "multigammaln": (0.5, 1.0, 1.5, 3.434189657547),
    "multigammaln_p4": (0.5, 1.0, 1.5, 3.434189657547),
}
_PAIRED_STOCK_STAGES = {
    "gelu_bw": (((0, 1), 3, ("GeluDerivative",), (1, 0, 0)),),
    "softsign_bw": (
        ((0,), 2, ("Abs", (0, 1065353216), "Square", "Reciprocal"), ()),
        ((2, 1), 3, (), (1, 0, 0)),
    ),
    "relu_bw": (((0,), 2, ("GreaterThanZero",), ()), ((2, 1), 3, (), (0, 1, 0))),
    "tanhshrink_bw": (
        ((0,), 2, ("Tanh",), ()),
        ((2,), 3, ("Square",), ()),
        ((3, 1), 4, (), (1, 0, 0)),
    ),
    "erf_bw": (
        ((0,), 2, ("Square", "Neg", ("Exp", False, True)), ()),
        ((2, 1), 3, (), (0, 1, 0)),
        ((3, -1), 4, (), (0, 1, 0)),
    ),
    "leaky_relu_bw": (
        ((0,), 2, ("GreaterThanZero",), ()),
        ((1, -1), 3, (), (0, 1, 0)),
        ((2, 1, 3), 4, (), ("where", (0, 1, 2, 0))),
    ),
    "elu_bw": (
        ((0,), 2, ("GreaterThanZero",), ()),
        ((0,), 3, (("Exp", False, True),), ()),
        ((3, -1), 4, (), (0, 1, 0)),
        ((1, 4), 5, (), (0, 1, 0)),
        ((2, 1, 5), 6, (), ("where", (0, 1, 2, 0))),
    ),
    "selu_bw": (
        ((1, -1), 2, (), (0, 1, 0)),
        ((0,), 3, ("GreaterThanZero",), ()),
        ((2, -2), 4, (), (0, 1, 0)),
        ((0,), 5, (("Exp", False, True),), ()),
        ((4, 5), 6, (), (0, 1, 0)),
        ((3, 2, 6), 7, (), ("where", (0, 1, 2, 0))),
    ),
    "celu_bw": (
        ((0, -1), 2, (), (0, 1, 0)),
        ((2,), 3, (("Exp", False, True),), ()),
        ((0, -2), 4, (), ("GT", (0, 1, 0))),
        ((1, 3), 5, (), (0, 1, 0)),
        ((4, 1, 5), 6, (), ("where", (0, 1, 2, 0))),
    ),
    "hardshrink_bw": (
        ((0,), 2, (("Hardshrink", 1056964608),), ()),
        ((2,), 3, ("EqualZero",), ()),
        ((3, 1), 4, (), ("where_fill", (1, 0.0))),
    ),
    "hardtanh_bw": (
        ((0, -1), 2, (), ("LE", (0, 1, 0))),
        ((0, -2), 3, (), ("GE", (0, 1, 0))),
        ((3, 1), 4, (), ("where_fill", (1, 0.0))),
        ((2, 4), 5, (), ("where_fill", (1, 0.0))),
    ),
    "hardswish_bw": (
        ((0, -1), 2, (), ("LT", (0, 1, 0))),
        ((0, -2), 3, (), ("LE", (0, 1, 0))),
        ((0, -3), 4, (), (0, 1, 0)),
        ((4, -4), 5, (), ("ADD", (0, 1, 0))),
        ((1, 5), 6, (), (0, 1, 0)),
        ((3, 6, 1), 7, (), ("where", (0, 1, 2, 0))),
        ((2, 7), 8, (), ("where_fill", (1, 0.0))),
    ),
    "silu_bw": (
        ((0,), 2, ("Sigmoid",), ()),
        ((1, 2), 3, (), (0, 1, 0)),
        ((2, -1), 4, (), ("RSUB", (0, 1, 0))),
        ((4, 0), 5, (), (0, 1, 0)),
        ((5, -1), 6, (), ("ADD", (0, 1, 0))),
        ((3, 6), 7, (), (0, 1, 0)),
    ),
    "softplus_bw": (
        ((0, -1), 2, (), (0, 1, 0)),
        ((2,), 3, (("Exp", False, True),), ()),
        ((2, -2), 4, (), ("ADD", (0, 1, 0))),
        ((1, 3), 5, (), (0, 1, 0)),
        ((3, -3), 6, (), ("ADD", (0, 1, 0))),
        ((6,), 7, ("Reciprocal",), ()),
        ((5, 7), 8, (), (0, 1, 0)),
        ((4,), 9, ("GreaterThanZero",), ()),
        ((9, 1, 8), 10, (), ("where", (0, 1, 2, 0))),
    ),
    "log_sigmoid_bw": (
        ((0,), 2, ("LessThanZero",), ()),
        ((2,), 3, (), ("where_fill_bits", ((1, 1065353216), (2, 0)))),
        ((0,), 4, ("LessThanZero",), ()),
        ((4,), 5, (), ("where_fill_bits", ((1, 1065353216), (2, 3212836864)))),
        ((0,), 6, ("Abs",), ()),
        ((6,), 7, ("Neg",), ()),
        ((7,), 8, (("Exp", False, True),), ()),
        ((8, -1), 9, (), ("ADD", (0, 1, 0))),
        ((9,), 10, ("Reciprocal",), ()),
        ((8, 10), 11, (), (0, 1, 0)),
        ((5, 11), 12, (), (0, 1, 0)),
        ((3, 12), 13, (), ("SUB", (0, 1, 0))),
        ((1, 13), 14, (), (0, 1, 0)),
    ),
    "hardsigmoid_bw": (
        ((0, -1), 2, (), ("LE", (0, 1, 0))),
        ((0, -2), 3, (), ("GE", (0, 1, 0))),
        ((2,), 4, ("NotEqualZero",), ()),
        ((3,), 5, ("NotEqualZero",), ()),
        ((4, 5), 6, (), ("FPU_ADD_NEZ", ())),
        ((1, -3), 7, (), (0, 1, 0)),
        ((6, 7), 8, (), ("where_fill", (1, 0.0))),
    ),
    "softshrink_bw": (
        ((0, -1), 2, (), ("LT", (0, 1, 0))),
        ((0, -2), 3, (), ("GT", (0, 1, 0))),
        ((2,), 4, ("NotEqualZero",), ()),
        ((3,), 5, ("NotEqualZero",), ()),
        ((4, 5), 6, (), ("FPU_ADD_NEZ", ())),
        ((6, 1), 7, (), ("where_fill", (2, 0.0))),
    ),
    "logsigmoid": (
        (
            (0, 0),
            2,
            (
                ("Neg", False, False, 1),
                ("Exp", True, True, 1),
                ("binary", "LOGSIGMOID", (0, 1, 0)),
            ),
            (),
        ),
    ),
    "hardswish": (
        (
            (0, 0),
            2,
            (("load", 0), ("Hardsigmoid", False, False, 0), ("dest_reuse_mul", 0)),
            (),
        ),
    ),
    "logit": (
        ((0,), 2, (), ()),
        (
            (2, 2),
            3,
            (
                (4, 1065353216, 0),
                ("binary", "DIV", (1, 0, 0)),
                ("Log", False, False, 0),
            ),
            (),
        ),
    ),
    "lgamma": (
        (
            (0, 0, 0, 0, 0),
            2,
            (
                ("load", 0),
                ("load", 1),
                ("Lgamma", False, False, 0),
                ("fill", 2, 3.141592653589793),
                ("Frac", False, False, 1),
                ("binary", "MUL", (1, 2, 1)),
                ("Sin", False, False, 1),
                ("load", 2),
                ("load", 3),
                ("Floor", False, False, 3),
                ("binary", "EQ", (2, 3, 2)),
                ("fill", 3, 0.0),
                ("where", (2, 3, 1, 1)),
                ("Abs", False, False, 1),
                ("Log", False, False, 1),
                ("load", 2),
                ("lgamma_adjusted", (0, 1, 2, 0)),
            ),
            (),
        ),
    ),
    "multigammaln": (
        (
            (0, 0, 0, 0, 0),
            2,
            (
                ("load", 0),
                ("load", 1),
                ("Lgamma", False, False, 0),
                ("fill", 2, 3.141592653589793),
                ("Frac", False, False, 1),
                ("binary", "MUL", (1, 2, 1)),
                ("Sin", False, False, 1),
                ("load", 2),
                ("load", 3),
                ("Floor", False, False, 3),
                ("binary", "EQ", (2, 3, 2)),
                ("fill", 3, 0.0),
                ("where", (2, 3, 1, 1)),
                ("Abs", False, False, 1),
                ("Log", False, False, 1),
                ("load", 2),
                ("lgamma_adjusted", (0, 1, 2, 0)),
            ),
            (),
        ),
        ((0, -1), 3, (), ("SUB", (0, 1, 0))),
        (
            (3, 3, 3, 3, 3),
            4,
            (
                ("load", 0),
                ("load", 1),
                ("Lgamma", False, False, 0),
                ("fill", 2, 3.141592653589793),
                ("Frac", False, False, 1),
                ("binary", "MUL", (1, 2, 1)),
                ("Sin", False, False, 1),
                ("load", 2),
                ("load", 3),
                ("Floor", False, False, 3),
                ("binary", "EQ", (2, 3, 2)),
                ("fill", 3, 0.0),
                ("where", (2, 3, 1, 1)),
                ("Abs", False, False, 1),
                ("Log", False, False, 1),
                ("load", 2),
                ("lgamma_adjusted", (0, 1, 2, 0)),
            ),
            (),
        ),
        ((2, 4), 5, (), ("ADD", (0, 1, 0))),
        ((0, -2), 6, (), ("SUB", (0, 1, 0))),
        (
            (6, 6, 6, 6, 6),
            7,
            (
                ("load", 0),
                ("load", 1),
                ("Lgamma", False, False, 0),
                ("fill", 2, 3.141592653589793),
                ("Frac", False, False, 1),
                ("binary", "MUL", (1, 2, 1)),
                ("Sin", False, False, 1),
                ("load", 2),
                ("load", 3),
                ("Floor", False, False, 3),
                ("binary", "EQ", (2, 3, 2)),
                ("fill", 3, 0.0),
                ("where", (2, 3, 1, 1)),
                ("Abs", False, False, 1),
                ("Log", False, False, 1),
                ("load", 2),
                ("lgamma_adjusted", (0, 1, 2, 0)),
            ),
            (),
        ),
        ((5, 7), 8, (), ("ADD", (0, 1, 0))),
        ((0, -3), 9, (), ("SUB", (0, 1, 0))),
        (
            (9, 9, 9, 9, 9),
            10,
            (
                ("load", 0),
                ("load", 1),
                ("Lgamma", False, False, 0),
                ("fill", 2, 3.141592653589793),
                ("Frac", False, False, 1),
                ("binary", "MUL", (1, 2, 1)),
                ("Sin", False, False, 1),
                ("load", 2),
                ("load", 3),
                ("Floor", False, False, 3),
                ("binary", "EQ", (2, 3, 2)),
                ("fill", 3, 0.0),
                ("where", (2, 3, 1, 1)),
                ("Abs", False, False, 1),
                ("Log", False, False, 1),
                ("load", 2),
                ("lgamma_adjusted", (0, 1, 2, 0)),
            ),
            (),
        ),
        ((8, 10), 11, (), ("ADD", (0, 1, 0))),
        ((11, -4), 12, (), ("ADD", (0, 1, 0))),
    ),
    "multigammaln_p4": (
        (
            (0, 0, 0, 0, 0),
            2,
            (
                ("load", 0),
                ("load", 1),
                ("Lgamma", False, False, 0),
                ("fill", 2, 3.141592653589793),
                ("Frac", False, False, 1),
                ("binary", "MUL", (1, 2, 1)),
                ("Sin", False, False, 1),
                ("load", 2),
                ("load", 3),
                ("Floor", False, False, 3),
                ("binary", "EQ", (2, 3, 2)),
                ("fill", 3, 0.0),
                ("where", (2, 3, 1, 1)),
                ("Abs", False, False, 1),
                ("Log", False, False, 1),
                ("load", 2),
                ("lgamma_adjusted", (0, 1, 2, 0)),
            ),
            (),
        ),
        ((0, -1), 3, (), ("SUB", (0, 1, 0))),
        (
            (3, 3, 3, 3, 3),
            4,
            (
                ("load", 0),
                ("load", 1),
                ("Lgamma", False, False, 0),
                ("fill", 2, 3.141592653589793),
                ("Frac", False, False, 1),
                ("binary", "MUL", (1, 2, 1)),
                ("Sin", False, False, 1),
                ("load", 2),
                ("load", 3),
                ("Floor", False, False, 3),
                ("binary", "EQ", (2, 3, 2)),
                ("fill", 3, 0.0),
                ("where", (2, 3, 1, 1)),
                ("Abs", False, False, 1),
                ("Log", False, False, 1),
                ("load", 2),
                ("lgamma_adjusted", (0, 1, 2, 0)),
            ),
            (),
        ),
        ((2, 4), 5, (), ("ADD", (0, 1, 0))),
        ((0, -2), 6, (), ("SUB", (0, 1, 0))),
        (
            (6, 6, 6, 6, 6),
            7,
            (
                ("load", 0),
                ("load", 1),
                ("Lgamma", False, False, 0),
                ("fill", 2, 3.141592653589793),
                ("Frac", False, False, 1),
                ("binary", "MUL", (1, 2, 1)),
                ("Sin", False, False, 1),
                ("load", 2),
                ("load", 3),
                ("Floor", False, False, 3),
                ("binary", "EQ", (2, 3, 2)),
                ("fill", 3, 0.0),
                ("where", (2, 3, 1, 1)),
                ("Abs", False, False, 1),
                ("Log", False, False, 1),
                ("load", 2),
                ("lgamma_adjusted", (0, 1, 2, 0)),
            ),
            (),
        ),
        ((5, 7), 8, (), ("ADD", (0, 1, 0))),
        ((0, -3), 9, (), ("SUB", (0, 1, 0))),
        (
            (9, 9, 9, 9, 9),
            10,
            (
                ("load", 0),
                ("load", 1),
                ("Lgamma", False, False, 0),
                ("fill", 2, 3.141592653589793),
                ("Frac", False, False, 1),
                ("binary", "MUL", (1, 2, 1)),
                ("Sin", False, False, 1),
                ("load", 2),
                ("load", 3),
                ("Floor", False, False, 3),
                ("binary", "EQ", (2, 3, 2)),
                ("fill", 3, 0.0),
                ("where", (2, 3, 1, 1)),
                ("Abs", False, False, 1),
                ("Log", False, False, 1),
                ("load", 2),
                ("lgamma_adjusted", (0, 1, 2, 0)),
            ),
            (),
        ),
        ((8, 10), 11, (), ("ADD", (0, 1, 0))),
        ((11, -4), 12, (), ("ADD", (0, 1, 0))),
    ),
}
_PAIRED_GENERATED = {}


import torch
from helpers.format_config import InputOutputFormat
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import PerfRunType
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import BuildMode, ProfilerBuild, TestConfig
from helpers.utils import passed_test


class _PairedDerivative(MATH_OP):
    # Inherit ordinary MATH_OP columns; routing never becomes a comparison key.
    def convert_to_cpp(self):
        stages = _PAIRED_STOCK_STAGES[self.mathop]
        result_slot = max(stage[1] for stage in stages)
        selected = self.mathop in _PAIRED_GENERATED
        forward = self.mathop in _PAIRED_FORWARD
        first = next((stage[2][0] for stage in stages if stage[2]), None)
        code = "#define TT_POLY_LLK_PERF_PAIRED\n"
        if first is not None and not forward:
            first = getattr(
                MathOperation, first[0] if isinstance(first, tuple) else first
            )
            code += MATH_OP(mathop=first).convert_to_cpp() + "\n"
        code += MATH_OP(mathop=MathOperation.SfpuElwmul).convert_to_cpp() + "\n"
        code += f"constexpr unsigned TT_POLY_LLK_INPUT_ARITY = {1 if forward else 2};\n"
        if selected and forward:
            from test_eltwise_unary_sfpu import (
                _GENERATED_UNARY_CASES,
                _TTPolyGeneratedBF16,
            )

            row = next(row for row in _GENERATED_UNARY_CASES if row[1] == self.mathop)
            native, op, initialize, replace_init, iterations, vector_mode, header = row
            code += _TTPolyGeneratedBF16(
                op,
                initialize,
                replace_init,
                iterations,
                vector_mode,
                header,
                native_enum=native.cpp_enum_value if native is not None else "unused",
            ).convert_to_cpp()
            stages = (((0,), result_slot, (), ()),)
        elif selected:
            from test_eltwise_binary_sfpu import _TTPolyBackwardFactor

            where_factor = _PAIRED_GENERATED[self.mathop] == "backward_where_factor"
            code += _TTPolyBackwardFactor(
                self.mathop, mask_only=where_factor
            ).convert_to_cpp()
            if where_factor:
                stages = (
                    ((0,), 2, (), ()),
                    ((2, 1), result_slot, (), ("where_fill", (2, 0.0))),
                )
            else:
                stages = (((0, 1), result_slot, (), (1, 0, 0)),)
            code += (
                "constexpr bool TT_POLY_LLK_STAGE_GENERATED[] = {"
                + ",".join("true" if i == 0 else "false" for i in range(len(stages)))
                + "};\n"
            )
        fpu = [bool(stage[3] and stage[3][0] == "FPU_ADD_NEZ") for stage in stages]
        dest_reuse = [
            any(
                isinstance(step, tuple) and step[0] == "dest_reuse_mul"
                for step in stage[2]
            )
            for stage in stages
        ]
        if any(dest_reuse):
            code += "#define TT_POLY_LLK_PERF_DEST_REUSE\n"
            code += (
                "constexpr bool TT_POLY_LLK_STAGE_DEST_REUSE[] = {"
                + ",".join(str(value).lower() for value in dest_reuse)
                + "};\n"
            )
        inline_load = [
            any(isinstance(step, tuple) and step[0] == "load" for step in stage[2])
            for stage in stages
        ]
        if any(inline_load):
            code += "#define TT_POLY_LLK_PERF_INLINE_LOAD\n"
            code += (
                "constexpr bool TT_POLY_LLK_STAGE_INLINE_LOAD[] = {"
                + ",".join(str(value).lower() for value in inline_load)
                + "};\n"
            )
        if any(fpu):
            code += "#define TT_POLY_LLK_PERF_HAS_FPU\n"
            code += (
                "constexpr bool TT_POLY_LLK_STAGE_FPU[] = {"
                + ",".join(str(value).lower() for value in fpu)
                + "};\n"
            )
        code += f"constexpr unsigned TT_POLY_LLK_STAGE_COUNT = {len(stages)};\n"
        code += f"constexpr unsigned TT_POLY_LLK_RESULT_SLOT = {result_slot};\n"
        code += (
            "constexpr unsigned TT_POLY_LLK_STAGE_ARITY[] = {"
            + ",".join(str(len(s[0])) for s in stages)
            + "};\n"
        )
        arity = max(len(stage[0]) for stage in stages)
        code += (
            f"constexpr int TT_POLY_LLK_STAGE_INPUT[][{arity}] = "
            + "{"
            + ",".join(
                "{" + ",".join(map(str, s[0] + (0,) * (arity - len(s[0])))) + "}"
                for s in stages
            )
            + "};\n"
        )
        destinations = []
        for stage in stages:
            action = stage[3]
            if action and action[0] == "where_fill":
                row = tuple(i for i in range(3) if i != action[1][0])
            elif action and action[0] == "where_fill_bits":
                row = tuple(
                    i for i in range(3) if i not in {item[0] for item in action[1]}
                )
            else:
                row = tuple(range(len(stage[0])))
            destinations.append(row)
        code += (
            f"constexpr unsigned TT_POLY_LLK_STAGE_DST[][{arity}] = "
            + "{"
            + ",".join(
                "{" + ",".join(map(str, row + (0,) * (arity - len(row)))) + "}"
                for row in destinations
            )
            + "};\n"
        )
        code += (
            "constexpr unsigned TT_POLY_LLK_STAGE_OUTPUT[] = {"
            + ",".join(str(s[1]) for s in stages)
            + "};\n"
        )
        if len(stages) > 1:
            code += "#define TT_POLY_LLK_PERF_STAGED\n"
        if len(stages) > 1 or (forward and not selected):
            code += "#define TT_POLY_LLK_PERF_STOCK_CHAIN\n"
            initialize, calculate = [], []
            for index, (_, _, chain, multiply) in enumerate(stages):
                init, calc = [], []
                if any(fpu) and not fpu[index]:
                    init.append(
                        "_llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(num_faces, formats.math);"
                    )
                for operation in chain:
                    if (
                        isinstance(operation, tuple)
                        and operation[0] == "dest_reuse_mul"
                    ):
                        calc.append(
                            "_llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MathFidelity::HiFi4, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(DEFAULT_TENSOR_SHAPE, false);"
                        )
                        calc.append(
                            f"_llk_math_eltwise_binary_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, DST_SYNC_MODE, is_fp32_dest_acc_en, MathFidelity::HiFi4, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(DEFAULT_TENSOR_SHAPE, {operation[1]}, true);"
                        )
                        continue
                    if isinstance(operation, tuple) and operation[0] == "load":
                        calc.append(
                            "_llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(num_faces, formats.math);"
                        )
                        calc.append(
                            f"_llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>({operation[1]}, formats.math, formats.math);"
                        )
                        continue
                    if isinstance(operation, tuple) and operation[0] == "fill":
                        calc.append(
                            "test_utils::call_unary_sfpu_operation_init<SfpuType::fill, false, is_fp32_dest_acc_en, 8>();"
                        )
                        calc.append(
                            f"SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_fill_, (false, 8), {operation[1]}, VectorMode::RC, {operation[2]!r}f);"
                        )
                        continue
                    if isinstance(operation, tuple) and operation[0] == "binary":
                        opcode, operands = operation[1:]
                        calc.append(
                            f"test_utils::call_binary_sfpu_operation_init<false, is_fp32_dest_acc_en, BinaryOp::{opcode}, 8>();"
                        )
                        calc.append(
                            f"test_utils::call_binary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, false, BinaryOp::{opcode}, 8, formats.math>("
                            + ", ".join(map(str, operands))
                            + ");"
                        )
                        continue
                    if isinstance(operation, tuple) and operation[0] == "where":
                        calc.append(
                            "test_utils::call_ternary_sfpu_operation_init<SfpuType::where, false, is_fp32_dest_acc_en>();"
                        )
                        calc.append(
                            "test_utils::call_ternary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, SfpuType::where, false, is_fp32_dest_acc_en, (is_fp32_dest_acc_en ? DataFormat::Float32 : DataFormat::Float16_b), 8>("
                            + ", ".join(map(str, operation[1]))
                            + ");"
                        )
                        continue
                    if (
                        isinstance(operation, tuple)
                        and operation[0] == "lgamma_adjusted"
                    ):
                        calc.append("SFPU_TERNARY_INIT(lgamma);")
                        calc.append(
                            "SFPU_TERNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_lgamma_adjusted, (false, is_fp32_dest_acc_en), "
                            + ", ".join(map(str, operation[1]))
                            + ", VectorMode::RC);"
                        )
                        continue
                    if isinstance(operation, tuple) and isinstance(operation[0], int):
                        mode, word = operation[:2]
                        destination = operation[2] if len(operation) == 3 else 0
                        count, vector = (
                            ("8", "RC")
                            if len(operation) == 3
                            else ("ITERATIONS", "None")
                        )
                        calc.append(
                            "ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, is_fp32_dest_acc_en>();"
                        )
                        calc.append(
                            f"SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_binop_with_scalar, (false, {mode}, {count}, is_fp32_dest_acc_en), {destination}, VectorMode::{vector}, {word}u);"
                        )
                        continue
                    approx, clamp, parameter = False, False, None
                    destination, count = 0, "ITERATIONS"
                    if isinstance(operation, tuple):
                        if len(operation) == 2:
                            operation, parameter = operation
                        elif len(operation) == 4:
                            operation, approx, clamp, destination = operation
                            count = "8"
                        else:
                            operation, approx, clamp = operation
                    mode = f"{str(approx).lower()}, is_fp32_dest_acc_en, {count}, false, false, {str(clamp).lower()}"
                    symbol = getattr(MathOperation, operation).cpp_enum_value
                    calc.append(
                        f"test_utils::call_unary_sfpu_operation_init<SfpuType::{symbol}, {mode}>();"
                    )
                    if parameter is None:
                        vector = ", 5.0f, VectorMode::RC" if count == "8" else ""
                        calc.append(
                            f"test_utils::call_unary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, SfpuType::{symbol}, {mode}>({destination}, formats.math{vector});"
                        )
                    else:
                        calc.append(
                            f"SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_{symbol}, (false, ITERATIONS), 0, VectorMode::None, {parameter}u);"
                        )
                if multiply and multiply[0] == "FPU_ADD_NEZ":
                    init.append(
                        "_llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MathFidelity::LoFi>(DEFAULT_TENSOR_SHAPE, 0);"
                    )
                    calc.append(
                        "_llk_math_eltwise_binary_<EltwiseBinaryType::ELWADD, BroadcastType::NONE, DST_SYNC_MODE, is_fp32_dest_acc_en, MathFidelity::LoFi>(DEFAULT_TENSOR_SHAPE, 0, false);"
                    )
                    calc.append(
                        "test_utils::call_unary_sfpu_operation_init<SfpuType::not_equal_zero, false, is_fp32_dest_acc_en, ITERATIONS>();"
                    )
                    calc.append(
                        "test_utils::call_unary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, SfpuType::not_equal_zero, false, is_fp32_dest_acc_en, ITERATIONS>(0, formats.math);"
                    )
                elif multiply and multiply[0] == "where_fill_bits":
                    for destination, bits in multiply[1]:
                        calc.append(
                            "test_utils::call_unary_sfpu_operation_init<SfpuType::fill, false, is_fp32_dest_acc_en, 8>();"
                        )
                        calc.append(
                            f"SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_fill_bitcast_, (false, 8), {destination}, VectorMode::RC, {bits}u);"
                        )
                    calc.append(
                        "test_utils::call_ternary_sfpu_operation_init<SfpuType::where, false, is_fp32_dest_acc_en>();"
                    )
                    calc.append(
                        "test_utils::call_ternary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, SfpuType::where, false, is_fp32_dest_acc_en, DataFormat::Float16_b, 8>(0, 1, 2, 0);"
                    )
                elif multiply and multiply[0] == "where_fill":
                    destination, value = multiply[1]
                    init.append(
                        "test_utils::call_ternary_sfpu_operation_init<SfpuType::where, false, is_fp32_dest_acc_en>();"
                    )
                    calc.append(
                        "test_utils::call_unary_sfpu_operation_init<SfpuType::fill, false, is_fp32_dest_acc_en, 8>();"
                    )
                    calc.append(
                        f"SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_fill_, (false, 8), {destination}, VectorMode::RC, {value!r}f);"
                    )
                    calc.append(
                        "test_utils::call_ternary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, SfpuType::where, false, is_fp32_dest_acc_en, DataFormat::Float16_b, 8>(0, 1, 2, 0);"
                    )
                elif multiply and multiply[0] == "where":
                    init.append(
                        "test_utils::call_ternary_sfpu_operation_init<SfpuType::where, false, is_fp32_dest_acc_en>();"
                    )
                    calc.append(
                        "test_utils::call_ternary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, SfpuType::where, false, is_fp32_dest_acc_en, DataFormat::Float16_b, 8>("
                        + ", ".join(map(str, multiply[1]))
                        + ");"
                    )
                elif multiply:
                    opcode, operands = (
                        ("BinaryOp::" + multiply[0], multiply[1])
                        if isinstance(multiply[0], str)
                        else ("SFPU_BINARY_OPERATION", multiply)
                    )
                    init.append(
                        f"test_utils::call_binary_sfpu_operation_init<false, is_fp32_dest_acc_en, {opcode}, ITERATIONS>();"
                    )
                    calc.append(
                        f"test_utils::call_binary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, false, {opcode}, ITERATIONS, formats.math>("
                        + ", ".join(map(str, operands))
                        + ");"
                    )
                initialize.append(f"if ((stage) == {index}) {{" + " ".join(init) + "}")
                calculate.append(f"if ((stage) == {index}) {{" + " ".join(calc) + "}")
            code += (
                "#define TT_POLY_LLK_STOCK_STAGE_INIT(stage) "
                + " ".join(initialize)
                + "\n"
            )
            code += (
                "#define TT_POLY_LLK_STOCK_STAGE_CALC(stage) "
                + " ".join(calculate)
                + "\n"
            )
        return code


@pytest.mark.perf
@pytest.mark.parametrize("operation", tuple(_PAIRED_STOCK_STAGES))
def test_perf_paired_derivative_gradient(perf_report, operation):
    # Compare source-owned L1 compositions, including their BF16 tensor boundaries.
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    forward = operation in _PAIRED_FORWARD
    output_tiles = 8
    if forward:
        reference = _tt_poly_forward_arguments(operation)
        values, _, _, _ = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=[128, 64],
            stimuli_format_B=formats.input_format,
            input_dimensions_B=[128, 64],
            spec_A=reference["spec_A"],
        )
        paired = values
    else:
        values = torch.linspace(-4, 4, 8192).to(torch.bfloat16)
        gradients = ((torch.arange(8192) % 13) - 6).to(torch.bfloat16) / 4
        paired = torch.stack(
            (values.reshape(8, 1024), gradients.reshape(8, 1024)), dim=1
        ).flatten()
    dest_acc = (
        DestAccumulation.Yes
        if forward
        and str(TestConfig.CHIP_ARCH) in _PAIRED_FORWARD_FP32.get(operation, ())
        else DestAccumulation.No
    )
    constants = _PAIRED_CONSTANTS.get(operation, ())
    constant_tiles = (
        torch.tensor(constants, dtype=torch.bfloat16).repeat_interleave(1024)
        if constants
        else None
    )
    scratch_tiles = output_tiles * (
        max(stage[1] for stage in _PAIRED_STOCK_STAGES[operation]) - 2
    )
    arguments = dict(
        test_name="sources/eltwise_binary_sfpu_perf.cpp",
        formats=formats,
        templates=[
            _PairedDerivative(mathop=operation),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
        ],
        runtimes=[
            TILE_COUNT(8),
            LOOP_FACTOR(16),
            NUM_FACES(num_faces=4),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            paired,
            formats.input_format,
            constant_tiles,
            formats.input_format,
            formats.output_format,
            tile_count_A=8 if forward else 16,
            tile_count_B=len(constants),
            tile_count_res=output_tiles,
            buffer_C=torch.zeros(scratch_tiles * 1024, dtype=torch.bfloat16),
            stimuli_C_format=formats.output_format,
            tile_count_C=scratch_tiles,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )
    functional = create_test_or_perf_config(
        is_perf=False,
        run_types=[PerfRunType.L1_TO_L1],
        test_config_kwargs={**arguments, "profiler_build": ProfilerBuild.Yes},
    )
    functional.prepare()
    if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        actual = torch.tensor(functional.run().result, dtype=torch.bfloat16).flatten()
        declared = globals().get("_paired_reference_" + operation)
        if forward:
            expected = reference["declared_golden"](values)
        elif declared is not None:
            factor = torch.tensor(
                declared(values.double().numpy()), dtype=torch.bfloat16
            )
        else:
            factor = get_golden_generator(UnarySFPUGolden)(
                MathOperation.GeluDerivative,
                values,
                formats.output_format,
                DestAccumulation.No,
                formats.input_format,
                [64, 64],
            ).to(torch.bfloat16)
        if not forward:
            expected = (factor.float() * gradients.float()).to(torch.bfloat16)
        assert actual.numel() == expected.numel()
        assert passed_test(
            expected, actual, formats.output_format
        ), "paired derivative/gradient transport failed"
    configuration = create_test_or_perf_config(
        is_perf=True, run_types=[PerfRunType.L1_TO_L1], test_config_kwargs=arguments
    )
    configuration.run(perf_report)
