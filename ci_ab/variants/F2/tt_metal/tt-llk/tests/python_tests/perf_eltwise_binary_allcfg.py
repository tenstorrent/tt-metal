# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Scratch, not for merge: the eltwise binary configurations the checked-in perf modules do not reach (DestSync full,
# mixed A/B formats, dest reuse with fp32 DEST or full sync) and, where the tree has them, the hand-off and call forms.
# The same file runs on main and on the head; main lacks the form arguments, so its id list keeps only form "pf".

import inspect

import pytest
import test_eltwise_binary as T
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
    MathOperation,
    Transpose,
)
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES

LF = 32
RUN = inspect.signature(T._run_eltwise_binary_test).parameters
REUSE = inspect.signature(T._run_eltwise_binary_dest_reuse_test).parameters
F16B, F16, BFP8, BFP4, F32 = (
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Bfp8_b,
    DataFormat.Bfp4_b,
    DataFormat.Float32,
)
CORE = [
    InputOutputFormat(F16B, F16B),
    InputOutputFormat(F16, F16),
    InputOutputFormat(BFP8, BFP8),
    InputOutputFormat(F16B, BFP8),
    InputOutputFormat(F32, F32),
]
OPS = [MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul]
BCASTS = [BroadcastType.None_, BroadcastType.Row, BroadcastType.Column, BroadcastType.Scalar]
# form: hand-off (pf per face, pt per tile), ubN = _llk_unpack_AB_block_ mode N, mb = _llk_math_eltwise_binary_block_
FORMS = {
    "pf": (True, 0, False),
    "pt": (False, 0, False),
    "pf_ub1": (True, 1, False),
    "pt_ub1": (False, 1, False),
    "pf_ub2": (True, 2, False),
    "pt_ub2": (False, 2, False),
    "pt_mb": (False, 0, True),
    "pt_ub1_mb": (False, 1, True),
    "pt_ub2_mb": (False, 2, True),
}


def core_formats(dest_acc):
    return [f for f in T._get_valid_formats(dest_acc) if f in CORE]


def dims1(dest_acc, dest_sync, formats, tile_dimensions):
    return T.get_eltwise_binary_perf_input_dimensions(dest_acc, dest_sync, formats, tile_dimensions)[:1]


def form_kwargs(form, tile_dimensions, broadcast_type):
    pf, ub, mb = FORMS[form]
    if form == "pf":
        return {"per_face_handoff": True} if "per_face_handoff" in RUN else {}
    if "per_face_handoff" not in RUN:
        pytest.skip("form needs the head's hand-off arguments")
    return {"per_face_handoff": pf, "unpack_ab_block": ub, "math_block": mb}


def _run(dest_acc, dest_sync, formats, bcast, math_op, fid, tile, dims, acc, **kw):
    T._run_eltwise_binary_test(
        dest_acc, dest_sync, False, formats, bcast, math_op, fid, Transpose.No, dims, tile, acc,
        run_types=ALL_PERF_RUN_TYPES, loop_factor=LF, is_perf=True, perf_report=kw.pop("perf_report"), **kw
    )


@pytest.mark.perf
@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half, DestSync.Full],
    formats=core_formats,
    broadcast_type=BCASTS,
    math_op=OPS,
    math_fidelity=lambda formats, math_op: T._get_valid_math_fidelity(formats, math_op),
    tile_dimensions=[[32, 32], [16, 16], [32, 16], [1, 32]],
    input_dimensions=dims1,
    acc_to_dest=[False, True],
    form=list(FORMS),
)
def test_ac_binary(perf_report, dest_acc, dest_sync, formats, broadcast_type, math_op, math_fidelity, tile_dimensions,
                   input_dimensions, acc_to_dest, form):
    if broadcast_type in (BroadcastType.Column, BroadcastType.Row) and tile_dimensions == [32, 16]:
        pytest.skip("no column or row broadcast on 32x16")
    kw = form_kwargs(form, tile_dimensions, broadcast_type)
    _run(dest_acc, dest_sync, formats, broadcast_type, math_op, math_fidelity, tile_dimensions, input_dimensions,
         acc_to_dest, perf_report=perf_report, **kw)


# pair name: A x B to output (the format repr in a test id omits B)
MIXED = {
    "a16b_b8_o16b": InputOutputFormat(F16B, F16B, BFP8),
    "a8_b16b_o16b": InputOutputFormat(BFP8, F16B, F16B),
    "a16b_b8_o8": InputOutputFormat(F16B, BFP8, BFP8),
    "a8_b16b_o8": InputOutputFormat(BFP8, BFP8, F16B),
    "a16b_b4_o16b": InputOutputFormat(F16B, F16B, BFP4),
    "a32_b16b_o16b": InputOutputFormat(F32, F16B, F16B),
    "a16b_b32_o32": InputOutputFormat(F16B, F32, F32),
}


@pytest.mark.perf
@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    pair=list(MIXED),
    formats=lambda pair: [MIXED[pair]],
    broadcast_type=BCASTS,
    math_op=[MathOperation.Elwadd, MathOperation.Elwmul],
    math_fidelity=lambda formats, math_op: T._get_valid_math_fidelity(formats, math_op),
    tile_dimensions=[[32, 32]],
    input_dimensions=dims1,
    form=["pf", "pt"],
)
def test_ac_mixed(perf_report, dest_acc, dest_sync, pair, formats, broadcast_type, math_op, math_fidelity, tile_dimensions,
                  input_dimensions, form):
    kw = form_kwargs(form, tile_dimensions, broadcast_type)
    _run(dest_acc, dest_sync, formats, broadcast_type, math_op, math_fidelity, tile_dimensions, input_dimensions,
         False, perf_report=perf_report, **kw)


@pytest.mark.perf
@parametrize(
    reuse_dest_type=[EltwiseBinaryReuseDestType.DEST_TO_SRCA, EltwiseBinaryReuseDestType.DEST_TO_SRCB],
    math_op=T.DEST_REUSE_MATH_OPS,
    formats=T.get_dest_reuse_formats,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half, DestSync.Full],
    math_fidelity=lambda formats, math_op: T._get_valid_math_fidelity(formats, math_op),
    tile_dimensions=[[32, 32], [16, 16]],
    input_dimensions=lambda dest_acc, dest_sync, formats, tile_dimensions: T.get_dest_reuse_perf_input_dimensions(
        dest_acc, dest_sync, formats, tile_dimensions)[:1],
    output_dimensions=lambda dest_acc, dest_sync, formats, tile_dimensions, input_dimensions: (
        T.get_dest_reuse_perf_output_dimensions(dest_acc, dest_sync, formats, tile_dimensions, input_dimensions)[:1]),
    broadcast_type=[BroadcastType.None_, BroadcastType.Row],
    form=["pf", "pt", "pf_ua", "pt_ua"],
)
def test_ac_reuse(perf_report, reuse_dest_type, math_op, formats, dest_acc, dest_sync, math_fidelity, tile_dimensions,
                  input_dimensions, output_dimensions, broadcast_type, form):
    kw = {}
    if "broadcast_type" in REUSE:
        kw = {"per_face_handoff": form.startswith("pf"), "broadcast_type": broadcast_type,
              "dest_reuse_unpack_a": form.endswith("ua")}
    elif form not in ("pf", "pf_ua") or broadcast_type != BroadcastType.None_:
        pytest.skip("form or broadcast needs the head's dest reuse arguments")
    elif form == "pf_ua":
        if "dest_reuse_unpack_a" not in REUSE:
            pytest.skip("the dest-reuse unpack needs the harness port")
        kw = {"dest_reuse_unpack_a": True}
    if broadcast_type == BroadcastType.Row and form.endswith("ua"):
        pytest.skip("the dest-reuse unpack has no broadcast")
    if broadcast_type == BroadcastType.Row and (reuse_dest_type != EltwiseBinaryReuseDestType.DEST_TO_SRCA
                                                or tile_dimensions != [32, 32]):
        pytest.skip("row broadcast dest reuse: DEST_TO_SRCA on 32x32 only")
    T._run_eltwise_binary_dest_reuse_test(
        reuse_dest_type, math_op, formats, dest_acc, dest_sync, False, math_fidelity, tile_dimensions,
        input_dimensions, output_dimensions, run_types=ALL_PERF_RUN_TYPES, loop_factor=LF, is_perf=True,
        perf_report=perf_report, **kw)
