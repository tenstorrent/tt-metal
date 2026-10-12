# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import test_eltwise_binary_sfpu as _func
from helpers.constraints import distinct_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
)
from helpers.llk_params import BroadcastType as LlkBroadcastType
from helpers.llk_params import (
    DestAccumulation,
    DstRoundingMode,
    MathOperation,
    SfpuMulImpl,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES

_PERF_AXES = dict(
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[16],
    iterations=[32],
    approx_mode=[ApproximationMode.No],
    is_perf=[True],
)

_ATAN2_PERF_AXES = {
    **_PERF_AXES,
    "approx_mode": [ApproximationMode.Yes, ApproximationMode.No],
}

_PERF_EXCLUDED_MATHOPS = {
    # TODO(#58145): profiler register pressure prevents a production-equivalent build.
    MathOperation.SfpuDivInt32,
    MathOperation.SfpuDivInt32Floor,
}

# A functional sweep dict shared below becomes perf cases, times the run types.
# Narrow it here when a point should stay functional-only, as with the ops below.
_INT_UNIFORM_PERF_SWEEP = {
    **_func.INT_UNIFORM_SWEEP,
    "mathop": [
        mathop
        for mathop in _func.INT_UNIFORM_SWEEP["mathop"]
        if mathop not in _PERF_EXCLUDED_MATHOPS
    ],
}


_BOTH_DEST_ACC = [DestAccumulation.No, DestAccumulation.Yes]


def _distinct_dest_acc(sweep):
    """Drop dest_acc modes that TestConfig promotes onto the same kernel.

    Outlier format combos record dest_acc=Yes for a requested No, so a [No, Yes]
    sweep would publish two rows with one measurement. Functional tests still
    ask for both; only the perf copy narrows the axis.
    """
    if sweep.get("dest_acc") != _BOTH_DEST_ACC:
        return sweep
    return {
        **sweep,
        "dest_acc": lambda formats: distinct_dest_accumulation_modes(
            formats, list(_BOTH_DEST_ACC)
        ),
    }


def _without_float16(sweep):
    """Drop Float16 from a perf copy of a functional sweep.

    No ttnn dtype maps to it, Blackhole cannot run it at dest_acc=No, and the
    Quasar variant exists only at dest_acc=No. Functional sweeps keep it.
    """
    return {
        **sweep,
        "formats": [
            fmt
            for fmt in sweep["formats"]
            if DataFormat.Float16
            not in (fmt.input_format, fmt.output_format, fmt.input_format_B)
        ],
    }


def _perf_kwargs(perf_report, run_types, loop_factor, iterations, approx_mode, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        iterations=iterations,
        approx_mode=approx_mode,
    )


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_without_float16(_func.FLOAT_SWEEP)),
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_float(
    perf_report,
    formats,
    dest_acc,
    mathop,
    bcast_dim,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_float(
        formats,
        dest_acc,
        mathop,
        bcast_dim,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


# mul_binary_tile on WH/BH. The float sweep above stays on the generic kernel so
# its rows still join Quasar. One same-format row per remaining float format.
_MUL_PRODUCTION_SWEEP = dict(
    formats=input_output_formats(
        [DataFormat.Float32, DataFormat.Float16_b, DataFormat.Bfp8_b],
        same=True,
    ),
    bcast_dim=[LlkBroadcastType.None_],
    mathop=[MathOperation.SfpuElwmul],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mul_kernel=[SfpuMulImpl.Production],
)


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_MUL_PRODUCTION_SWEEP),
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_mul_production(
    perf_report,
    formats,
    dest_acc,
    mathop,
    bcast_dim,
    mul_kernel,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_float(
        formats,
        dest_acc,
        mathop,
        bcast_dim,
        mul_kernel=mul_kernel,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


# ADD/SUB/RSUB on Float16_b with dest_acc=No, the only place DstRoundingMode
# changes the store. Default stays on the float sweep; this family carries both.
_BF16_RNE_SWEEP = dict(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    bcast_dim=[LlkBroadcastType.None_],
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwrsub,
    ],
    dest_acc=[DestAccumulation.No],
    dst_rounding=[DstRoundingMode.Default, DstRoundingMode.NearestEven],
)


@pytest.mark.perf
@parametrize(
    **_BF16_RNE_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_bf16_rne(
    perf_report,
    formats,
    dest_acc,
    mathop,
    bcast_dim,
    dst_rounding,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_float(
        formats,
        dest_acc,
        mathop,
        bcast_dim,
        dst_rounding_mode=dst_rounding,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


# Production depth is 1 tile, or 2 when sharded. Two tiles, add and mul only.
_SHALLOW_SWEEP = dict(
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32],
        same=True,
    ),
    bcast_dim=[LlkBroadcastType.None_],
    mathop=[MathOperation.SfpuElwadd, MathOperation.SfpuElwmul],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 32]],
)


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_SHALLOW_SWEEP),
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_float_shallow(
    perf_report,
    formats,
    dest_acc,
    mathop,
    bcast_dim,
    input_dimensions,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_float(
        formats,
        dest_acc,
        mathop,
        bcast_dim,
        input_dimensions=input_dimensions,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_without_float16(_func.DIV_SWEEP)),
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_div(
    perf_report,
    formats,
    dest_acc,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_div(
        formats,
        dest_acc,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_without_float16(_func.FLOAT_EXTENDED_SWEEP)),
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_float_extended(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_float_extended(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_func.MASK_SWEEP),
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_mask(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_mask(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_func.ATAN2_SWEEP),
    **_ATAN2_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_atan2(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_atan2(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_without_float16(_func.EQ_NE_SWEEP)),
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_eq_ne(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_eq_ne(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_without_float16(_func.FLOAT_COMPARISON_SWEEP)),
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_float_comparison(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_float_comparison(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_func.ISCLOSE_SWEEP),
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_isclose(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_isclose(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_func.LOGSIGMOID_SWEEP),
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_logsigmoid(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_logsigmoid(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.INT_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_int(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_int(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.BITWISE_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_bitwise(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_bitwise(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_INT_UNIFORM_PERF_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_int_uniform(
    perf_report,
    mathop,
    dest_acc,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_int_uniform(
        mathop,
        dest_acc,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.RSUB_INT32_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_rsub_int32(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_rsub_int32(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.EQ_NE_INT_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_eq_ne_int(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_eq_ne_int(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_func.ADD_TOP_ROW_SWEEP),
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_add_top_row(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_add_top_row(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_distinct_dest_acc(_without_float16(_func.BCAST_SWEEP)),
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_bcast(
    perf_report,
    formats,
    bcast_dim,
    mathop,
    dest_acc,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_bcast(
        formats,
        bcast_dim,
        mathop,
        dest_acc,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )
