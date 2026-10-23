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

Verified op semantics (measured on p100a, see run_sdpa_bcast_col_srcb_reuse below):

    out = A0 * bcast_col(P1) + A1 * bcast_col(P2)

with everything an 8x32 tile (two 8x16 faces):
  - A0, A1  operand tiles streamed into SrcA, one llk_unpack_A each
  - P1, P2  column sources seeded into DEST[0] / DEST[1]; the math preamble MOVD2Bs DEST rows 0-7
            into SrcB rows 0-7 and DEST rows 64-71 into SrcB rows 8-15
  - bcast_col(P) row r  = P[r, 0] fanned across all 32 columns (SRCB_BCAST_COL)
  - the MOP's ELWMULs accumulate into DEST, and the kernel passes clear_dest=true so the
    column-source seed still sitting in DEST[DST_INDEX] is zeroed after the MOVD2Bs latch it
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
from helpers.test_variant_parameters import IN_FACE_DIMS, NUM_FACES, TILE_COUNT
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# LoFi-only, bf16-natural path. Keep the grid tiny for the advance test.
SDPA_BCAST_FORMATS = input_output_formats([DataFormat.Float16_b])

# The op's tile: 8 rows x 32 cols == two 8x16 faces. num_faces == 2 is the only value
# sdpa_bcast_col_srcb_reuse_configure_mop permits, and 8 rows is what the MOP actually writes
# (one ELWMUL per face, 8 dest rows each).
SDPA_NUM_FACES = 2
SDPA_FACE_R_DIM = 8
SDPA_TILE_DIMS = [SDPA_FACE_R_DIM, 32]
# Two operand tiles and two column-source tiles per math call.
SDPA_TILE_PAIRS = 2


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


def run_sdpa_bcast_col_srcb_reuse(cpp_source, formats, boot_mode=BootMode.DEFAULT):
    """Drive `cpp_source` (an SDPA bcast-col SrcB-reuse kernel) and assert against the golden."""
    # A single-axis @parametrize passes the value as a 1-tuple; unwrap it.
    if isinstance(formats, tuple):
        (formats,) = formats

    torch_format = format_dict[formats.output_format]
    rows = SDPA_FACE_R_DIM
    # Two 8x32 tiles per operand, stacked row-wise so tilize_block emits them in order.
    dimensions = [SDPA_TILE_PAIRS * rows, 32]

    # buffer_A = the two operand tiles (SrcA). buffer_B = the two column sources; only column 0 of
    # each row matters (that is the value SRCB_BCAST_COL fans across the row), but a full tile is a
    # valid seed and keeps the stimuli generator on its normal path.
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dimensions,
        tile_dimensions=SDPA_TILE_DIMS,
    )

    tilized_A = tilize_block(
        src_A,
        dimensions,
        formats.input_format,
        num_faces=SDPA_NUM_FACES,
        tile_dimensions=SDPA_TILE_DIMS,
        face_r_dim=SDPA_FACE_R_DIM,
    )
    tilized_B = tilize_block(
        src_B,
        dimensions,
        formats.input_format,
        num_faces=SDPA_NUM_FACES,
        tile_dimensions=SDPA_TILE_DIMS,
        face_r_dim=SDPA_FACE_R_DIM,
    )

    # Golden, in untilized (row-major) space to match the readback below. The column broadcast is
    # computed in TILIZED space because that is where the per-face column-0 value lives, then
    # untilized; the two products are then formed by the eltwise golden and summed row-major (the
    # MOP's two ELWMULs accumulate into DEST).
    #
    # EltwiseBinaryGolden rather than a raw torch multiply: the products come off the FPU at LoFi,
    # which truncates the SrcA/SrcB mantissas before multiplying, and the generator models that
    # fidelity masking. Both operands are already quantized here (stimuli in input_format, the
    # scale straight out of BroadcastGolden), so input_format is left unset -- the same call shape
    # test_eltwise_bcast_col_custom.py and test_experimental_reconfig_escape.py use.
    per_tile = SDPA_NUM_FACES * SDPA_FACE_R_DIM * 16
    broadcast_golden = get_golden_generator(BroadcastGolden)
    eltwise_golden = get_golden_generator(EltwiseBinaryGolden)
    operands = src_A.reshape(SDPA_TILE_PAIRS * rows, 32).to(torch_format)

    golden_tensor = torch.zeros(rows * 32, dtype=torch_format)
    for t in range(SDPA_TILE_PAIRS):
        scale = _bcast_col_untilized(
            tilized_B.flatten()[t * per_tile : (t + 1) * per_tile],
            formats.input_format,
            broadcast_golden,
        ).to(torch_format)
        golden_tensor += eltwise_golden(
            MathOperation.Elwmul,
            operands[t * rows : (t + 1) * rows, :].flatten(),
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
        ],
        runtimes=[
            TILE_COUNT(1),
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
            tile_count_res=1,
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
        SDPA_TILE_DIMS,
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
