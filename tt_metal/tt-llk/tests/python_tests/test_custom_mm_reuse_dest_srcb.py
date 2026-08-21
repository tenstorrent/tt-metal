# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Test for the Blackhole-only experimental LLK custom_mm_reuse_dest_srcb
# (PR #53297): the second matmul of a fused chain whose in0 operand is moved out
# of DEST (where a preceding custom_mm left its output) into SrcB instead of
# being unpacked from L1.  Only the weights (in1) are unpacked, into SrcA.
#
# The reuse LLK is meaningless without a DEST-resident in0 in exactly the layout
# custom_mm<dense_packing> produces, so the C++ driver runs the whole documented
# chain:  producer custom_mm<dense_packing>  ->  consumer custom_mm_reuse_dest_srcb.
#
# Golden (derived in custom_mm_reuse_dest_srcb_test.cpp from the headers):
#     in0    = A0 @ B0              (producer output, [8, 32*REUSE_KT])
#     golden = in0 @ B1  =  (A0 @ B0) @ B1        ([8, 32*REUSE_NT])
# a plain fp32 chained matmul, narrowed and tilized to the consumer accumulator
# tile layout.  Only the first IN0_TILE_R_DIM (=8) rows of each output tile are
# DEFINED; rows 8..31 are undefined and are NOT asserted.
#
# xfail: this is a faithful, correct-by-construction transcription of the header
# pipeline, but the fused two-op chain (DEST bank sharing + producer/consumer
# cross-thread sync) cannot be validated at runtime here (no Blackhole card).
# The bar met is a clean Blackhole compile + a golden that mirrors the header
# math.  Marked non-strict xfail so a real Blackhole run surfaces as XPASS rather
# than silently passing/failing.

import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CUSTOM_MM_REUSE_CFG,
    NUM_FACES,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# Fixed chain geometry.  in0 tile height is the full-height 8 case (MOVD2B copies
# all 8 rows, no zero-pad subtlety); a single output tile (nt=1) reduced over 2
# K-tiles, with a 2-tile-deep producer.
IN0_TILE_R_DIM = 8
PRODUCER_KT = 2  # producer inner dim in tiles (its kt_dim)
REUSE_KT = 2  # consumer inner dim in tiles == producer ct_dim == # DEST in0 tiles
REUSE_NT = 1  # consumer output width in tiles

# LoFi only for both custom_mm and the reuse LLK.
MATH_FIDELITY = MathFidelity.LoFi

# 16-bit float operands keep the chained matmul well inside format tolerance.
REUSE_FORMATS = input_output_formats([DataFormat.Float16_b])


def _tile(t, dims, fmt, face_r_dim=16):
    return tilize_block(t, dimensions=dims, stimuli_format=fmt, face_r_dim=face_r_dim)


@blackhole_only
@pytest.mark.xfail(
    reason="Fails on real BH: the fused custom_mm -> custom_mm_reuse_dest_srcb chain "
    "(DEST bank sharing + producer/consumer cross-thread sync) needs the compute-kernel "
    "framework flow-control this bare LLK driver does not reproduce. Non-strict so a real "
    "fix surfaces as XPASS. Clean assert-fail (not a wedge), so xfail is safe here.",
    strict=False,
)
# NOTE: a single-axis @parametrize(formats=...) passes formats as a 1-tuple
# (pytest does not unpack a comma-less argname), so we add the dest_acc axis.
# With >1 axis the framework unpacks the value tuple and `formats` is the object.
@parametrize(
    formats=REUSE_FORMATS,
    dest_acc=[DestAccumulation.No],
)
def test_custom_mm_reuse_dest_srcb(formats, dest_acc):
    torch_format = format_dict[formats.output_format]
    in_fmt = formats.input_format

    # Producer contracting dim (its K) in tiles; kept at PRODUCER_KT.
    m = IN0_TILE_R_DIM
    kp = PRODUCER_KT * 32  # producer contraction width in datums
    kt = REUSE_KT * 32  # producer output width == consumer contraction width
    nt = REUSE_NT * 32  # consumer output width

    torch.manual_seed(0)
    # in0 producer input A0 : [m, kp]   (in0 tile shape [8, 32])
    A0 = torch.rand((m, kp), dtype=torch.float32)
    # producer weights B0   : [kp, kt]  (32x32 tiles)
    B0 = torch.rand((kp, kt), dtype=torch.float32)
    # consumer weights B1   : [kt, nt]  (32x32 tiles)
    B1 = torch.rand((kt, nt), dtype=torch.float32)

    # Golden: chained matmul (A0 @ B0) @ B1, evaluated in fp32.
    in0 = A0 @ B0  # [m, kt]  == producer output that lands in DEST
    golden_full = in0 @ B1  # [m, nt] == consumer output (defined rows only)

    # The output tile is 32x32; only the first m(=8) rows are defined.  Pad the
    # golden up to a full tile with zeros for the undefined rows; the python
    # comparison below asserts ONLY the defined lanes.
    golden_tile = torch.zeros((32, nt), dtype=torch.float32)
    golden_tile[:m, :] = golden_full
    golden_tiled = _tile(
        golden_tile.flatten(), dims=[32, nt], fmt=formats.output_format
    ).to(torch_format)

    # Tilize the three inputs.  in0 (A0) is an 8-row partial-height operand, but
    # tilize_block lays data into full 32-row tiles (face_r_dim only sets the face
    # row count, not the tile height).  Pad A0's rows up to 32 with zeros before
    # tilizing; the extra rows are undefined on-device and are never asserted.
    A0_pad = torch.zeros((32, kp), dtype=torch.float32)
    A0_pad[:m, :] = A0
    A0_t = _tile(A0_pad.flatten(), dims=[32, kp], fmt=in_fmt, face_r_dim=IN0_TILE_R_DIM)
    B0_t = _tile(B0.flatten(), dims=[kp, kt], fmt=in_fmt)
    B1_t = _tile(B1.flatten(), dims=[kt, nt], fmt=in_fmt)

    configuration = TestConfig(
        "sources/custom_mm_reuse_dest_srcb_test.cpp",
        formats,
        templates=[
            CUSTOM_MM_REUSE_CFG(
                in0_tile_r_dim=IN0_TILE_R_DIM,
                producer_kt=PRODUCER_KT,
                reuse_kt=REUSE_KT,
                reuse_nt=REUSE_NT,
            ),
        ],
        runtimes=[
            NUM_FACES(num_faces=4, num_faces_A=4, num_faces_B=2),
        ],
        variant_stimuli=StimuliConfig(
            A0_t.flatten(),
            in_fmt,
            B0_t.flatten(),
            in_fmt,
            formats.output_format,
            tile_count_A=REUSE_KT,  # in0 spans REUSE_KT tiles along K
            tile_count_B=PRODUCER_KT * REUSE_KT,
            tile_count_res=REUSE_NT,
            buffer_C=B1_t.flatten(),
            stimuli_C_format=in_fmt,
            tile_count_C=REUSE_KT * REUSE_NT,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Assert ONLY the defined lanes.  The result holds REUSE_NT tiles; within each
    # 32x32 tile only the first IN0_TILE_R_DIM rows are defined.  The tilized
    # golden has zeros in the undefined rows, so restrict both to the defined
    # region before comparing (face-major: face0 rows 0..7, face1 rows 0..7).
    defined = _defined_lane_mask(REUSE_NT, IN0_TILE_R_DIM)
    assert len(res_tensor) == len(golden_tiled)
    assert passed_test(
        golden_tiled[defined], res_tensor[defined], formats.output_format
    ), "Assert against golden failed (defined lanes)"


def _defined_lane_mask(num_tiles, r_dim):
    """Indices of the DEFINED datums in a tilized [32, 32] result.

    Face-major tile layout: 4 faces of 16x16, laid out [f0, f1, f2, f3] where
    f0/f1 cover rows 0..15 (cols 0..15 / 16..31) and f2/f3 rows 16..31.  in0 is
    r_dim rows tall, so only the first r_dim rows of faces f0 and f1 are defined.
    """
    mask = []
    face_elems = 16 * 16
    for t in range(num_tiles):
        base = t * 32 * 32
        for face in range(4):
            fbase = base + face * face_elems
            for row in range(16):
                for col in range(16):
                    idx = fbase + row * 16 + col
                    # faces 0,1 hold output rows 0..15; defined when row < r_dim.
                    defined = face in (0, 1) and row < r_dim
                    if defined:
                        mask.append(idx)
    return mask
