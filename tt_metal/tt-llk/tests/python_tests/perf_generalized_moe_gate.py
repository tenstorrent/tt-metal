# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

# Import under a non-test name. pytest collects every test* binding in this
# module, and these functional cases share recorded keys with the perf wrappers.
import test_generalized_moe_gate as moe_gate
from conftest import skip_for_quasar
from helpers.llk_params import ApproximationMode, DestSync, MathOperation
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES

pytestmark = skip_for_quasar


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(
    topk=[4, 6, 8],
    softmax=[False, True],
    approx=lambda topk: (
        [ApproximationMode.No, ApproximationMode.Yes]
        if topk == 8
        else [ApproximationMode.No]
    ),
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate(
    perf_report, topk, softmax, approx, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate,
        topk,
        softmax,
        approx,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(grouped=[False, True], topk=lambda grouped: [8] if grouped else [4, 8])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_sigmoid(
    perf_report, grouped, topk, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_sigmoid,
        grouped,
        topk,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(seed=[128], approx=[ApproximationMode.No, ApproximationMode.Yes])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_grouped(
    perf_report, seed, approx, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_grouped,
        seed,
        approx,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    sigmoid=[False, True],
    scaling=[(False, 0.0), (True, 0.0), (True, 0.5), (True, 1.0), (True, 1.75)],
    approx=[ApproximationMode.No, ApproximationMode.Yes],
    dest_sync=[DestSync.Half, DestSync.Full],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_grouped_single_face(
    perf_report, sigmoid, scaling, approx, dest_sync, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_grouped_single_face,
        sigmoid,
        scaling,
        approx,
        dest_sync,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    # Half + 4 faces is the main sweep's topk=8, softmax false point.
    dest_sync=[DestSync.Half, DestSync.Full],
    num_faces=lambda dest_sync: [1] if dest_sync == DestSync.Half else [1, 4],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_shipping_config(
    perf_report, dest_sync, num_faces, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_shipping_config,
        dest_sync,
        num_faces,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_normalization(
    perf_report, run_types, loop_factor, is_perf
):
    # (EPS, SCALE) matches the main sweep. softmax true at (0, 1) matches softmax_logits.
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_normalization,
        (0.0, 1.0),
        False,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_softmax_logits(
    perf_report, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_softmax_logits,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(store=[(0, 2), (4, 6)], idx_offset=[0, 256, 1792])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_produce_run(
    perf_report, store, idx_offset, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_produce_run,
        store,
        idx_offset,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    reload=[False, True],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_binary(
    perf_report, math_op, reload, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_binary,
        math_op,
        reload,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_binary_acc_to_dest(
    perf_report, math_op, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_binary_acc_to_dest,
        math_op,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(num_faces=[1, 2])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_binary_num_faces(
    perf_report, num_faces, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_binary_num_faces,
        num_faces,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(src_dst=[(0, 4), (4, 8), (8, 12), (12, 0)], srcb=[16, 20, 24, 28])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_copy4rows(
    perf_report, src_dst, srcb, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_copy4rows,
        src_dst,
        srcb,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    what=[
        (
            moe_gate.MODE_MOVE,
            moe_gate.MOVE_COPY4ROWS,
        ),  # a MOP runner: exercises the DEST-offset setup
        (
            moe_gate.MODE_RUN,
            moe_gate.RUN_PLACE_FIELD,
        ),  # SFPU only, so a difference here is not MOP-specific
        (
            moe_gate.MODE_GATE,
            0,
        ),  # the whole pipeline: MOPs, merges and the normalize interleaved
    ]
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_dest_sections(
    perf_report, what, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_dest_sections,
        what,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_copy4rows_back_to_back(
    perf_report, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_copy4rows_back_to_back,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(knobs=[(0, 8), (4, 0), (4, 8)])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_step1_hi(
    perf_report, knobs, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_step1_hi,
        knobs,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_step1_matches_step1_hi(
    perf_report, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_step1_matches_step1_hi,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_step0(perf_report, run_types, loop_factor, is_perf):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_step0,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(output_tiles=[2, 3])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_step2(
    perf_report, output_tiles, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_step2,
        output_tiles,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_merge4_top8(
    perf_report, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_merge4_top8,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    placement=[(0, 2, 4, 6), (0, 2, 8, 10), (4, 6, 0, 2), (0, 4, 8, 12)],
    after_mop=lambda placement: [False, True] if placement == (0, 2, 4, 6) else [False],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_relocate_run(
    perf_report, placement, after_mop, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_relocate_run,
        placement,
        after_mop,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    field=[0, 1, 2],
    src=lambda field: [(0, 4)] if field else [(0, 4), (8, 12)],
    dst=lambda field: [(0, 2)] if field else [(0, 2), (4, 6)],
    after_mop=lambda field, src, dst: (
        [False, True] if (field, src, dst) == (0, (0, 4), (0, 2)) else [False]
    ),
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_place_field(
    perf_report, field, src, dst, after_mop, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_place_field,
        field,
        src,
        dst,
        after_mop,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(store=[(8, 10), (12, 14)])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_combine(
    perf_report, store, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_combine,
        store,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_combine_relocated(
    perf_report, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_combine_relocated,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(topk=[4, 6, 8], softmax=[False, True])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_combine_finalize(
    perf_report, topk, softmax, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_combine_finalize,
        topk,
        softmax,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_generalized_moe_gate_idx_offset(
    perf_report, run_types, loop_factor, is_perf
):
    moe_gate.invoke_with_perf(
        moe_gate.test_generalized_moe_gate_idx_offset,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
