# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Perf for `calculate_comp_int`, which the shared sweep structurally cannot reach.

`perf_eltwise_unary_sfpu.py` asserts its format matrix against `PERF_SWEEP_OPS`
and does not carry Int32, but `calculate_comp_int` is reachable only with an
Int32 input — so these six comparison-to-zero modes have no perf coverage
anywhere. This module is the extra slice that supplies it, exactly as
`test_perf_eltwise_unary_sfpu_comp_uint16` and `_comp_uint32` already do for
their formats: a slice that bypasses `PERF_SWEEP_OPS` rather than widening it.

Nothing here duplicates the shared sweep. Every float op these kernels touch is
already in `_OP_DOMAIN_REGISTRY`, hence in `PERF_SWEEP_OPS`, and measured at
these same parameters; `run_llk_perf_wormhole.sh` collects the whole directory,
so carrying them here too would measure those rows twice in every perf shard.

loop_factor/iterations/dimensions match the shared sweep so the numbers stay
directly comparable with it.

When A/B-ing header variants, wipe the ELF cache between runs.
`TestConfig.variant_id` hashes the `-I` include-directory paths, not header
*content*, and `build_elfs()` skips outright once `.build_complete` is set, so
an unwiped $RUNNER_TEMP/tt-llk-build silently replays the previous variant's
ELFs and reports a 1.00x delta that means nothing.
"""

import pytest
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    DestSync,
    FastMode,
    MathOperation,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_eltwise_unary_sfpu import eltwise_unary_sfpu

_DIMS = [[128, 64]]  # tile_cnt: 8, same as the main perf sweep

# calculate_comp_int: reached only when the runtime math_format is Int32.
_INT_COMP_OPS = [
    MathOperation.EqualZero,
    MathOperation.NotEqualZero,
    MathOperation.LessThanZero,
    MathOperation.GreaterThanZero,
    MathOperation.LessThanEqualZero,
    MathOperation.GreaterThanEqualZero,
]


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32], same=True),
    mathop=_INT_COMP_OPS,
    input_dimensions=_DIMS,
)
def test_perf_vif_comp_int32(perf_report, formats, mathop, input_dimensions):
    # Int32 unpacks straight into a 32-bit DEST, mirroring the correctness path.
    eltwise_unary_sfpu(
        formats=formats,
        dest_acc=DestAccumulation.Yes,
        approx_mode=ApproximationMode.No,
        mathop=mathop,
        fast_mode=FastMode.No,
        input_dimensions=input_dimensions,
        dest_sync=DestSync.Half,
        clamp_negative=False,
        unpack_to_dest=True,
        is_perf=True,
        perf_report=perf_report,
        run_types=ALL_PERF_RUN_TYPES,
        check_golden=False,
        loop_factor=16,
    )
