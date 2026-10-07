# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Round 3 eltwise binary CI (measurement only): the dest-reuse forms on 8x32 tiles (two 8-row faces) with the dest-reuse unpack,
# the per-face program (per_face_handoff True) against the whole-tile program for 8-row faces (False), with 32x32 beside them. Eight
# output tiles per DEST section, folded once (two input tiles per output tile) or three times (four).

import pytest
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
    MathFidelity,
    MathOperation,
)
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_eltwise_binary import (
    _run_eltwise_binary_dest_reuse_test,
    get_dest_reuse_formats,
)

PERF_LOOP_FACTOR = 32
INPUTS = {(8, 32): [[128, 32], [256, 32]], (32, 32): [[512, 32], [1024, 32]]}
OUTPUTS = {(8, 32): [[64, 32]], (32, 32): [[256, 32]]}


def _fidelities(formats, math_op):
    if math_op != MathOperation.Elwmul:
        return [MathFidelity.LoFi]
    return [MathFidelity.LoFi, MathFidelity.HiFi2]


@pytest.mark.perf
@parametrize(
    reuse_dest_type=[
        EltwiseBinaryReuseDestType.DEST_TO_SRCA,
        EltwiseBinaryReuseDestType.DEST_TO_SRCB,
    ],
    math_op=[MathOperation.Elwadd, MathOperation.Elwmul],
    formats=lambda math_op: [f for f in get_dest_reuse_formats(math_op) if f.input_format.name == "Float16_b"],
    dest_acc=[DestAccumulation.No],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    math_fidelity=_fidelities,
    tile_dimensions=[[8, 32], [32, 32]],
    input_dimensions=lambda tile_dimensions: INPUTS[tuple(tile_dimensions)],
    output_dimensions=lambda tile_dimensions: OUTPUTS[tuple(tile_dimensions)],
    per_face_handoff=[True, False],
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)
def test_perf_eltwise_binary_dest_reuse_dr8(
    perf_report,
    reuse_dest_type,
    math_op,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    math_fidelity,
    tile_dimensions,
    input_dimensions,
    output_dimensions,
    per_face_handoff,
    run_types,
    loop_factor,
    is_perf,
):
    _run_eltwise_binary_dest_reuse_test(
        reuse_dest_type,
        math_op,
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        math_fidelity,
        tile_dimensions,
        input_dimensions,
        output_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
        per_face_handoff=per_face_handoff,
        dest_reuse_unpack_a=True,
    )
