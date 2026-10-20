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
    MathFidelity,
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


# binary_ng's block section formats (a, b, c in bf16 or bfp8_b): per face (main's form), per-tile hand-off with the block
# unpack (LoFi in production) and with the block math too (above LoFi); multiplies at LoFi and HiFi4 for every mix.
BLOCKMIX = {f"a{a}_b{b}_o{c}": InputOutputFormat(fa, fc, fb)
            for a, fa in (("16b", F16B), ("8", BFP8)) for b, fb in (("16b", F16B), ("8", BFP8))
            for c, fc in (("16b", F16B), ("8", BFP8))}


@pytest.mark.perf
@parametrize(
    pair=list(BLOCKMIX),
    formats=lambda pair: [BLOCKMIX[pair]],
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    math_fidelity=lambda math_op: [MathFidelity.LoFi] + ([MathFidelity.HiFi4] if math_op == MathOperation.Elwmul else []),
    form=["pf", "pt_ub1", "pt_ub1_mb", "pt_ub2", "pt_ub2_mb"],
)
def test_ac_blockmix(perf_report, pair, formats, math_op, math_fidelity, form):
    tile = [32, 32]
    dims = T.get_eltwise_binary_perf_input_dimensions(DestAccumulation.No, DestSync.Half, formats, tile)[:1][0]
    kw = form_kwargs(form, tile, BroadcastType.None_)
    _run(DestAccumulation.No, DestSync.Half, formats, BroadcastType.None_, math_op, math_fidelity, tile, dims, False,
         perf_report=perf_report, **kw)


# Step 4, broadcast block forms: per face (main's form), the block unpack with B advancing (ub1, a row broadcast) or
# held (ub2, a column or scalar broadcast) under the per-face hand-off (add, sub and LoFi multiplies in production) and,
# for a HiFi4 multiply, with the per-tile hand-off and the block math (kernels opted into the broadcast hand-off).
BCASTMIX = {f"a{a}_b{b}_o{c}": InputOutputFormat(fa, fc, fb)
            for a, fa, b, fb, c, fc in (("16b", F16B, "16b", F16B, "16b", F16B), ("8", BFP8, "16b", F16B, "16b", F16B),
                                        ("16b", F16B, "16b", F16B, "8", BFP8), ("8", BFP8, "16b", F16B, "8", BFP8),
                                        ("16b", F16B, "8", BFP8, "16b", F16B))}
BCASTMIX["a32_b32_o32"] = InputOutputFormat(F32, F32, F32)


@pytest.mark.perf
@parametrize(
    pair=list(BCASTMIX),
    formats=lambda pair: [BCASTMIX[pair]],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    broadcast_type=[BroadcastType.Row, BroadcastType.Column, BroadcastType.Scalar],
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    math_fidelity=lambda math_op: [MathFidelity.LoFi] + ([MathFidelity.HiFi4] if math_op == MathOperation.Elwmul else []),
    form=["pf", "pf_ub1", "pf_ub2", "pt_ub1_mb", "pt_ub2_mb"],
)
def test_ac_bcastblock(perf_report, pair, formats, dest_acc, broadcast_type, math_op, math_fidelity, form):
    if form.endswith("_mb") and math_fidelity == MathFidelity.LoFi:
        pytest.skip("LoFi broadcasts keep the per-face hand-off")
    tile = [32, 32]
    dims = T.get_eltwise_binary_perf_input_dimensions(dest_acc, DestSync.Half, formats, tile)[:1][0]
    kw = form_kwargs(form, tile, broadcast_type)
    _run(dest_acc, DestSync.Half, formats, broadcast_type, math_op, math_fidelity, tile, dims, False,
         perf_report=perf_report, **kw)


# Step 4 with one DEST section (dst_full_sync_en): per face (main's form); add and subtract with the per-face hand-off and the
# block unpack; a multiply in the kernels opted into the broadcast hand-off takes the per-tile one there at every fidelity.
@pytest.mark.perf
@parametrize(
    pair=list(BCASTMIX),
    formats=lambda pair: [BCASTMIX[pair]],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    broadcast_type=[BroadcastType.Row, BroadcastType.Column, BroadcastType.Scalar],
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    math_fidelity=lambda math_op: [MathFidelity.LoFi] + ([MathFidelity.HiFi4] if math_op == MathOperation.Elwmul else []),
    form=["pf", "pf_ub1", "pf_ub2", "pt_ub1", "pt_ub2", "pt_ub1_mb", "pt_ub2_mb"],
)
def test_ac_bcastblock_full(perf_report, pair, formats, dest_acc, broadcast_type, math_op, math_fidelity, form):
    mul = math_op == MathOperation.Elwmul
    if form != "pf" and (form.startswith("pt") != mul or form.endswith("_mb") != (math_fidelity != MathFidelity.LoFi)):
        pytest.skip("not a form production selects with one DEST section")
    tile = [32, 32]
    dims = T.get_eltwise_binary_perf_input_dimensions(dest_acc, DestSync.Full, formats, tile)[:1][0]
    kw = form_kwargs(form, tile, broadcast_type)
    _run(dest_acc, DestSync.Full, formats, broadcast_type, math_op, math_fidelity, tile, dims, False,
         perf_report=perf_report, **kw)


# Step 4 attribution: the head's per-tile broadcast multiply at LoFi with one DEST section (no block), against main's per face.
@pytest.mark.perf
@parametrize(
    pair=list(BCASTMIX),
    formats=lambda pair: [BCASTMIX[pair]],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    broadcast_type=[BroadcastType.Row, BroadcastType.Column, BroadcastType.Scalar],
    form=["pf", "pt"],
)
def test_ac_bcast_pt_full(perf_report, pair, formats, dest_acc, broadcast_type, form):
    tile = [32, 32]
    dims = T.get_eltwise_binary_perf_input_dimensions(dest_acc, DestSync.Full, formats, tile)[:1][0]
    kw = form_kwargs(form, tile, broadcast_type)
    _run(dest_acc, DestSync.Full, formats, broadcast_type, MathOperation.Elwmul, MathFidelity.LoFi, tile, dims, False,
         perf_report=perf_report, **kw)


# Step 4 gate proof: kernel.cpp and bcast.h keep the per-face hand-off for a LoFi column multiply with fp32 DEST at full sync
# (PER_TILE_HANDOFF_COL_LOSS) when a kernel buffer is Bfp8_b; pg is that gate (pf there, pt elsewhere), run against main's pf.
@pytest.mark.perf
@parametrize(
    pair=list(BCASTMIX),
    formats=lambda pair: [BCASTMIX[pair]],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    broadcast_type=[BroadcastType.Row, BroadcastType.Column, BroadcastType.Scalar],
    form=["pf", "pg"],
)
def test_ac_bcast_pg_full(perf_report, pair, formats, dest_acc, broadcast_type, form):
    tile = [32, 32]
    dims = T.get_eltwise_binary_perf_input_dimensions(dest_acc, DestSync.Full, formats, tile)[:1][0]
    if form == "pg":
        has_bfp8 = "8" in pair.replace("16b", "")
        form = "pf" if dest_acc == DestAccumulation.Yes and broadcast_type == BroadcastType.Column and has_bfp8 else "pt"
    kw = form_kwargs(form, tile, broadcast_type)
    _run(dest_acc, DestSync.Full, formats, broadcast_type, MathOperation.Elwmul, MathFidelity.LoFi, tile, dims, False,
         perf_report=perf_report, **kw)
