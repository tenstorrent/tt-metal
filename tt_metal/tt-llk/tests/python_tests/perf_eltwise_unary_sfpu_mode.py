# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cycles per tile of each SfpuMode instruction sequence.

SfpuMode exists to trade accuracy and special-value handling for throughput, so the trade is
measured here per mode rather than inferred from instruction counts. Blackhole p100a, bf16 input,
MATH_ISOLATE, TILE_LOOP (dest_acc=Yes runs the fp32-Dest sequences):

    op          dest_acc  PreciseStrict  PreciseRelaxed  ApproximateStrict  ApproximateRelaxed
    softsign    No            731.6         540.7 (L)          517.7              325.8
    softsign    Yes           731.6         540.5 (L)          513.6              321.6
    exp2        No            869.8 (L)     773.8              = Precise          = Precise
    exp2        Yes           993.6 (L)     763.5              865.6              635.5
    reciprocal  No            638.6         285.7 (L)          350.6               78.0 (L approx)
    reciprocal  Yes           606.5         193.5 (L)          350.5               65.9 (L approx)

(L) marks the mode the legacy, mode-less call runs. Reciprocal's Strict modes are slower because
the Relaxed ones are fixed-schedule SFPLOADMACRO pipelines that overwrite each input in place, so
the NaN guard needs a separate sfpi sequence.
"""

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, MathOperation, SfpuMode
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_eltwise_unary_sfpu import eltwise_unary_sfpu

# Ops whose kernel API takes an SfpuMode.
SFPU_MODE_OPS = [MathOperation.Softsign, MathOperation.Exp2, MathOperation.Reciprocal]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    mathop=SFPU_MODE_OPS,
    sfpu_mode=list(SfpuMode),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_eltwise_unary_sfpu_mode(
    perf_report, formats, mathop, sfpu_mode, dest_acc, input_dimensions
):
    eltwise_unary_sfpu(
        formats=formats,
        dest_acc=dest_acc,
        # The legacy axis the mode's precision half maps to; the kernel call itself is
        # selected by SFPU_MODE_OVERRIDE.
        approx_mode=sfpu_mode.approximation_mode,
        mathop=mathop,
        input_dimensions=input_dimensions,
        clamp_negative=False,
        loop_factor=16,
        unpack_to_dest=False,
        is_perf=True,
        perf_report=perf_report,
        run_types=ALL_PERF_RUN_TYPES,
        check_golden=False,
        sfpu_mode=sfpu_mode,
    )
