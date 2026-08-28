# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul performance across every MX (microscaling) L1 data format on Quasar.

Conditions are identical to perf_matmul_formats_quasar.py, so rows from the two
reports are directly comparable: same output block, LoFi, DestSync.Half, 16-bit
destination, no transpose, loop factor 32. Input format always equals output
format, giving exactly one row per format.

There are six MX formats that can live in L1. The enum also carries MxFp4_2x_A
and MxFp4_2x_B, but those are src-register-only formats -- the unpacker produces
them from an MxFp4 tile in L1 -- so they are not measurable as an L1 format.
MxFp6R / MxFp6P exist in the Quasar hardware enum but are not modelled in this
test harness, so they cannot be swept here either.
"""

import pytest
from helpers.format_config import DataFormat
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from quasar.test_matmul_quasar import test_matmul as run_matmul

# Every MX format that exists as an L1 format, with its tile footprint in bytes
# (1024 elements plus one shared exponent per 32-element block):
#   MxFp8R  E5M2 + block exp   1056 B
#   MxFp8P  E4M3 + block exp   1056 B
#   MxInt8  S1.6 + block exp   1056 B
#   MxFp4   E2M1 + block exp    544 B
#   MxInt4  S1.2 + block exp    544 B
#   MxInt2  S1.0 + block exp    288 B
MX_FORMATS = [
    DataFormat.MxFp8R,
    DataFormat.MxFp8P,
    DataFormat.MxInt8,
    DataFormat.MxFp4,
    DataFormat.MxInt4,
    DataFormat.MxInt2,
]

# Same fixed output block as the Float16_b / MxFp8P comparison: a dest-full 1x8
# tile output accumulated over kt=4.
MT_DIM, NT_DIM, KT_DIM = 1, 8, 4
DIMENSIONS = [
    [MT_DIM * TILE_DIM, KT_DIM * TILE_DIM],
    [KT_DIM * TILE_DIM, NT_DIM * TILE_DIM],
]


def mx_register_format_hint(format):
    """Pin the src-register format the unpacker must produce for this L1 format.

    Plain MxFp4 has no non-2x matmul path -- the FPU consumes it only through the
    2x-packed src-register formats -- so it needs an explicit hint. Pin the
    MxFp4_2x_B (Float16_b exponent family) variant so every format in the sweep
    ends up in the same exponent family in the src registers. Every other MX
    format unpacks straight to Float16_b and needs no hint.
    """
    return (
        [DataFormat.MxFp4_2x_B] if format.input_format == DataFormat.MxFp4 else [None]
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    format=input_output_formats(MX_FORMATS, same=True),
    # MX inputs are already full precision at LoFi, so the HiFi phases would only
    # multiply the MVMUL count without changing the represented values.
    math_fidelity=[MathFidelity.LoFi],
    dest_sync_mode=[DestSync.Half],
    dest_acc=[DestAccumulation.No],
    dimensions=[DIMENSIONS],
    implied_math_format=[ImpliedMathFormat.Yes],
    register_format_hint=mx_register_format_hint,
    enable_direct_indexing=[False],
    transpose=[Transpose.No],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_matmul_mx_formats_quasar(
    perf_report,
    format,
    math_fidelity,
    dest_sync_mode,
    dest_acc,
    dimensions,
    implied_math_format,
    register_format_hint,
    enable_direct_indexing,
    transpose,
    run_types,
    loop_factor,
    is_perf,
):
    run_matmul(
        math_fidelity,
        dest_sync_mode,
        dest_acc,
        dimensions,
        format,
        implied_math_format,
        register_format_hint,
        enable_direct_indexing,
        transpose,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
