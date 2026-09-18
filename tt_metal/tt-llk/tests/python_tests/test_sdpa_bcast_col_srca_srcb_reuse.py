# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ADVANCE TEST: covers experimental LLK sdpa_bcast_col_srca_srcb_reuse (tt-metal#47554 / tt-blaze#1971), promoted
# into tt_llk_blackhole/llk_lib/experimental/ on main by #53295. The compute kernel includes the canonical headers;
# the demo-fork shadow tree this test was first written against no longer exists.
#
# sdpa_bcast_col_srca_srcb_reuse documented contract (llk_math_sdpa_bcast_col_srca_srcb_reuse.h banner, plus what the
# op was measured to do on p100a):
#   - This is the DEST-to-DEST variant of the softmax scale step. It reuses DEST for BOTH operands: the MOP body is
#     [MOVD2A, MOVD2A, ELWMUL], so SrcA is refilled from DEST[dst] before every ELWMUL and every addrmod has
#     .srca.incr == 0. Nothing unpacked into SrcA survives, so there is no SrcA operand stream and no
#     Elwmul(src_A, src_B_bcast) golden -- an unpacked buffer_A tile would simply be discarded.
#   - Its ELWMUL carries CLR_NONE and ACCUMULATES into DEST, and SrcA is a copy of DEST, so the op computes
#         out = X + X * bcast_col(P)   ==   X * (1 + bcast_col(P))
#     with X the tile seeded at DEST[dst] and P the column source seeded at DEST[isrc]. This is exactly why the demo's
#     SFPU side subtracts 1 from the scale it produces ("Without -1: bcast = prev * exp + prev", sdpa.h:207-210).
#   - Both DEST indices are RAW DEST ROW offsets, not tile indices, so the kernel keeps them 64 apart (one 32x32 dest
#     tile) and passes isrc != dst -- with isrc == dst the golden degenerates to X * (1 + bcast_col(X)).
#   - The preamble waits on STALLWAIT(WAIT_SFPU | SRCA_VLD | SRCB_VLD), so the unpacker must call the SrcA+SrcB
#     dummy-valid helper _llk_unpack_A_sdpa_set_srca_srcb_dummy_valid_(), not the SrcB-only one.
#
# Geometry: the MOP is two 8-row ELWMULs with dest.incr == 8 and srcb.incr == 0, i.e. 16 CONTIGUOUS dest rows with
# both halves reusing the same 8 per-row scales. That is the demo's tile -- an 8x32 logical tile packed into one
# 16x16 DEST face ("Each tile is 8x32, which is the same as a full 16x16 face", sdpa.h:317) with dest rows 0-7
# holding logical columns 0-15 and rows 8-15 holding columns 16-31. So the test drives a single 16x16 face
# (num_faces == 1 on the unpack/pack side) and builds the golden directly in that flat DEST-row order; the MATH mop
# still gets num_faces == 2, the only value its LLK_ASSERT permits, which is where the two 8-row chunks come from.
#
# This test deliberately does NOT share helpers/sdpa_bcast_utils.py with the srcb_reuse pair: same op family, but a
# different function of the inputs (DEST-to-DEST scale-by-(1+P) vs a two-operand-tile SrcA stream), so there is no
# common golden to factor out.
#
# Blackhole-only (@blackhole_only): the primitive headers live under the Blackhole experimental/ tree.

import torch
from conftest import blackhole_only
from helpers.device import BootMode
from helpers.format_config import DataFormat
from helpers.golden_generators import EltwiseBinaryGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import NUM_FACES, TILE_COUNT
from helpers.tile_constants import FACE_C_DIM, MAX_FACE_R_DIM
from helpers.utils import passed_test

# LoFi-only, bf16-natural path. Keep the grid tiny for the advance test.
SDPA_FORMATS = input_output_formats([DataFormat.Float16_b])

# One 16x16 DEST face per tile (see the geometry note above).
NUM_FACES_HOST = 1
TILE_DIMS = [MAX_FACE_R_DIM, FACE_C_DIM]
# The op's 8 logical rows; dest rows r and r + 8 share the scale from P dest row r.
LOGICAL_ROWS = 8


@blackhole_only
@parametrize(
    formats=SDPA_FORMATS,
)
def test_sdpa_bcast_col_srca_srcb_reuse(
    formats,
    boot_mode=BootMode.DEFAULT,
):
    # A single-axis @parametrize passes the value as a 1-tuple; unwrap it.
    if isinstance(formats, tuple):
        (formats,) = formats

    torch_format = format_dict[formats.output_format]
    tile_cnt = 1

    # buffer_A = X, the tile the op scales in place. buffer_B = P, the column source; only column 0 of each of its
    # first 8 dest rows is read (that is what MOVD2B latches into SrcB and SRCB_BCAST_COL fans across the row).
    # A single 16x16 face means no tilize step is needed: face-major and row-major coincide.
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=TILE_DIMS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=TILE_DIMS,
        tile_dimensions=TILE_DIMS,
    )

    # Golden, in the same flat DEST-row order the packer writes back. The op's ELWMUL carries CLR_NONE and
    # accumulates into DEST while SrcA is a copy of DEST, so what the hardware computes is
    #   out[d, c] = X[d, c] + X[d, c] * P[d % 8, 0]
    # rather than the algebraically equal X * (1 + P): only the PRODUCT goes through the FPU, so only the
    # product carries LoFi's mantissa truncation. EltwiseBinaryGolden models that truncation (SrcA masked to
    # 5 mantissa bits, SrcB to 7), which a raw torch multiply does not. The column value repeats every 8 dest
    # rows because srcb.incr == 0 across the MOP's two 8-row chunks.
    x = src_A.reshape(MAX_FACE_R_DIM, FACE_C_DIM).to(torch_format)
    p_col0 = src_B.reshape(MAX_FACE_R_DIM, FACE_C_DIM).to(torch_format)[
        :LOGICAL_ROWS, 0
    ]
    bcast_col = (
        p_col0.repeat(MAX_FACE_R_DIM // LOGICAL_ROWS)
        .reshape(MAX_FACE_R_DIM, 1)
        .expand(MAX_FACE_R_DIM, FACE_C_DIM)
    )
    # Both operands are already quantized (stimuli are generated in input_format), so input_format is left
    # unset -- the same call shape test_experimental_reconfig_escape.py uses.
    product = get_golden_generator(EltwiseBinaryGolden)(
        MathOperation.Elwmul,
        x.flatten(),
        bcast_col.flatten(),
        formats.output_format,
        MathFidelity.LoFi,
    )
    golden_tensor = x.flatten() + product

    configuration = TestConfig(
        "sources/sdpa_bcast_col_srca_srcb_reuse_test.cpp",
        formats,
        templates=[
            # NUM_FACES is a TEMPLATE parameter, not a runtime one: it emits `constexpr num_faces`, and the
            # SDPA addrmod config needs a compile-time face count (SETC16 "n" asm constraint). The mop's own
            # inner-loop count is 2, not NUM_FACES_HOST -- that one is a fixed property of the primitive and
            # is pinned in the .cpp (MATH_MOP_NUM_FACES).
            NUM_FACES(NUM_FACES_HOST, NUM_FACES_HOST, NUM_FACES_HOST),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt,
            num_faces=NUM_FACES_HOST,
            face_r_dim=MAX_FACE_R_DIM,
            tile_dimensions=TILE_DIMS,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=DestAccumulation.No,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
