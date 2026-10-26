# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Driver for the SDPA column-broadcast SrcB-reuse advance test.

Drives `sources/sdpa_bcast_col_srcb_reuse_test.cpp`, which exercises
`llk_math_sdpa_bcast_col_srcb_reuse.h` together with `llk_unpack_A_sdpa.h` -- neither primitive is
testable alone (the unpack side is init/mop-config plus a dummy-SrcB-valid helper with no execute of
its own), so one kernel and one golden cover both. `test_unpack_A_sdpa.py`, which used to share this
driver to pin the unpack side by name, is owned by #53361.

Kept as a module rather than inlined into the single test file so the measured op semantics below stay
next to the golden that encodes them. `test_sdpa_bcast_col_srca_srcb_reuse.py` deliberately does NOT
share it: that variant sources both operands from DEST rather than from SrcA, so its golden is a
different function of the inputs (see its banner).

Verified op semantics (measured on p100a, see run_sdpa_bcast_col_srcb_reuse below), for
N = num_tiles output tiles, t in [0, N):

    out[t] = A[t] * bcast_col(P1) + A[N + t] * bcast_col(P2)          (ELWMUL)
    out[t] = (A[t] op bcast_col(P1)) + (A[N + t] op bcast_col(P2))    (ELWADD / ELWSUB, op = + / -)

with everything an 8x32 tile (two 8x16 faces):
  - A[0..N) and A[N..2N)  the two operand groups streamed into SrcA, one llk_unpack_A call per
            group; the SDPA unpack MOP walks N consecutive L1 tiles per call and the math MOP
            writes them to DEST[t]
  - P1, P2  column sources seeded into DEST[0] / DEST[1]; the math preamble MOVD2Bs DEST rows 0-7
            into SrcB rows 0-7 and DEST rows 64-71 into SrcB rows 8-15
  - bcast_col(P) row r  = P[r, 0] fanned across all 32 columns (SRCB_BCAST_COL)
  - both passes accumulate into DEST for every op, so the kernel passes clear_dest=true to zero the
    column-source seeds still sitting in the output tiles (P1 in DEST[0]; P2 in DEST[1] when N == 2)
"""

import torch
from helpers.device import BootMode
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    IN_FACE_DIMS,
    MATH_OP,
    NUM_FACES,
    SDPA_BCAST_NUM_TILES,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# LoFi-only, bf16-natural path. Keep the grid tiny for the advance test.
SDPA_BCAST_FORMATS = input_output_formats([DataFormat.Float16_b])

# The op's tile: 8 rows x 32 cols == two 8x16 faces. num_faces == 2 is the only value
# sdpa_bcast_col_srcb_reuse_configure_mop permits, and 8 rows is what the MOP actually writes
# (one eltwise op per face, 8 dest rows each).
SDPA_NUM_FACES = 2
SDPA_FACE_R_DIM = 8
SDPA_TILE_DIMS = [SDPA_FACE_R_DIM, 32]
# Two operand groups and two column-source tiles per math call.
SDPA_TILE_PAIRS = 2
# MOP outer-loop counts to cover; 2 is what reaches the MOP's last_inner_loop slot.
SDPA_BCAST_TILE_COUNTS = [1, 2]


def _bcast_col_untilized(tilized_tile, fmt, generator):
    """Column-broadcast one tilized 8x32 tile, returned untilized (row-major 8x32)."""
    bcast_tilized = generator(
        BroadcastType.Column,
        tilized_tile.flatten(),
        fmt,
        num_faces=SDPA_NUM_FACES,
        tile_cnt=1,
        face_r_dim=SDPA_FACE_R_DIM,
        input_format=fmt,
    )
    return untilize_block(
        bcast_tilized,
        fmt,
        SDPA_TILE_DIMS,
        num_faces=SDPA_NUM_FACES,
        tile_dimensions=SDPA_TILE_DIMS,
        face_r_dim=SDPA_FACE_R_DIM,
    ).reshape(SDPA_FACE_R_DIM, 32)


def run_sdpa_bcast_col_srcb_reuse(
    cpp_source,
    formats,
    mathop=MathOperation.Elwmul,
    num_tiles=1,
    boot_mode=BootMode.DEFAULT,
):
    """Drive `cpp_source` (an SDPA bcast-col SrcB-reuse kernel) and assert against the golden.

    `mathop` selects the FPU op (Elwmul / Elwadd / Elwsub) the kernel instantiates; it reaches the
    kernel as ELTWISE_BINARY_OP through the MATH_OP template parameter. `num_tiles` is the MOP's
    outer-loop count (SDPA_BCAST_NUM_TILES template parameter): `num_tiles` output tiles, each fed
    by one operand tile per group.
    """
    assert mathop in (
        MathOperation.Elwmul,
        MathOperation.Elwadd,
        MathOperation.Elwsub,
    ), f"sdpa_bcast_col_srcb_reuse supports ELWMUL/ELWADD/ELWSUB only, got {mathop}"

    torch_format = format_dict[formats.output_format]
    rows = SDPA_FACE_R_DIM
    # 8x32 tiles stacked row-wise so tilize_block emits them in L1 order: two groups of num_tiles
    # operand tiles, and two column-source tiles.
    dimensions_A = [SDPA_TILE_PAIRS * num_tiles * rows, 32]
    dimensions_B = [SDPA_TILE_PAIRS * rows, 32]

    # buffer_A = the operand tiles (SrcA), group g at tiles [g * num_tiles, (g + 1) * num_tiles).
    # buffer_B = the two column sources; only column 0 of each row matters (that is the value
    # SRCB_BCAST_COL fans across the row), but a full tile is a valid seed and keeps the stimuli
    # generator on its normal path.
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dimensions_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dimensions_B,
        tile_dimensions=SDPA_TILE_DIMS,
    )

    tilized_A = tilize_block(
        src_A,
        dimensions_A,
        formats.input_format,
        num_faces=SDPA_NUM_FACES,
        tile_dimensions=SDPA_TILE_DIMS,
        face_r_dim=SDPA_FACE_R_DIM,
    )
    tilized_B = tilize_block(
        src_B,
        dimensions_B,
        formats.input_format,
        num_faces=SDPA_NUM_FACES,
        tile_dimensions=SDPA_TILE_DIMS,
        face_r_dim=SDPA_FACE_R_DIM,
    )

    # Golden, in untilized (row-major) space to match the readback below. The column broadcast is
    # computed in TILIZED space because that is where the per-face column-0 value lives, then
    # untilized; the two per-group terms (A[g * num_tiles + t] op bcast_col(P_g)) are then formed
    # by the eltwise golden and summed row-major (the MOP's two passes accumulate into DEST for
    # every op).
    #
    # EltwiseBinaryGolden rather than a raw torch op: for MUL the products come off the FPU at
    # LoFi, which truncates the SrcA/SrcB mantissas before multiplying, and the generator models
    # that fidelity masking (ADD/SUB are fidelity-independent). Both operands are already quantized
    # here (stimuli in input_format, the scale straight out of BroadcastGolden), so input_format is
    # left unset -- the same call shape test_eltwise_bcast_col_custom.py and
    # test_experimental_reconfig_escape.py use.
    per_tile = SDPA_NUM_FACES * SDPA_FACE_R_DIM * 16
    broadcast_golden = get_golden_generator(BroadcastGolden)
    eltwise_golden = get_golden_generator(EltwiseBinaryGolden)
    operands = src_A.reshape(SDPA_TILE_PAIRS * num_tiles * rows, 32).to(torch_format)

    golden_tensor = torch.zeros(num_tiles * rows * 32, dtype=torch_format)
    for g in range(SDPA_TILE_PAIRS):
        scale = _bcast_col_untilized(
            tilized_B.flatten()[g * per_tile : (g + 1) * per_tile],
            formats.input_format,
            broadcast_golden,
        ).to(torch_format)
        for t in range(num_tiles):
            src_tile = g * num_tiles + t
            golden_tensor[t * rows * 32 : (t + 1) * rows * 32] += eltwise_golden(
                mathop,
                operands[src_tile * rows : (src_tile + 1) * rows, :].flatten(),
                scale.flatten(),
                formats.output_format,
                MathFidelity.LoFi,
            )

    configuration = TestConfig(
        cpp_source,
        formats,
        templates=[
            # NUM_FACES is a TEMPLATE parameter here, not a runtime one: the MATH mop needs the face
            # count as a compile-time constant (its ADDR_MOD dest.incr lands in a SETC16 whose
            # immediate takes the "n" asm constraint), and it emits `constexpr num_faces`, which the
            # unpack/pack sides read just as happily as the math thread.
            NUM_FACES(SDPA_NUM_FACES, SDPA_NUM_FACES, SDPA_NUM_FACES),
            MATH_OP(mathop=mathop),
            SDPA_BCAST_NUM_TILES(sdpa_bcast_num_tiles=num_tiles),
        ],
        runtimes=[
            TILE_COUNT(num_tiles),
            IN_FACE_DIMS(
                in0_face_r_dim=SDPA_FACE_R_DIM, in1_face_r_dim=SDPA_FACE_R_DIM
            ),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=num_tiles,
            num_faces=SDPA_NUM_FACES,
            face_r_dim=SDPA_FACE_R_DIM,
            tile_dimensions=SDPA_TILE_DIMS,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=DestAccumulation.No,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    res_from_L1 = untilize_block(
        res_from_L1,
        formats.output_format,
        [num_tiles * rows, 32],
        num_faces=SDPA_NUM_FACES,
        tile_dimensions=SDPA_TILE_DIMS,
        face_r_dim=SDPA_FACE_R_DIM,
    ).flatten()

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
