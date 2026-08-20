# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for the Blackhole-only experimental LLK `sdpa_bcast_col_srcb_reuse`
(item 5, PR #53295), driven by sources/sdpa_bcast_col_srcb_reuse_test.cpp.

Semantics (derived from the header, see the .cpp's top comment):
  The op computes, for the top two faces (num_faces == 2 is asserted by
  configure_mop):

      DEST[0] = A  <mathop>  bcast_col(B)

  where
    * A is the operand freshly unpacked from L1,
    * B is the operand seeded into DEST tile 0 (here via an A2D datacopy) and
      then moved into SrcB by the preamble's MOVD2B, and
    * bcast_col replicates column 0 of each SrcB face across all 16 columns.
      Each of the two faces broadcasts ITS OWN column 0: the kernel A2D-datacopies
      B into DEST tiles 0 and 1, the preamble MOVD2Bs those into SrcB faces 0 and 1,
      and p_elwise::SRCB_BCAST_COL then spreads each face's column-0 within that face.
      (This is NOT BroadcastGolden._broadcast_column(num_faces=2): that path is for a
      16x32 tiny tile -- one column domain -- and folds both faces onto face 0. Our
      operand is the top half of a full 32x32 tile, whose faces 0/1 are the adjacent
      16x16 blocks at cols 0-15 and 16-31, each with its own column 0. The per-face
      broadcast is confirmed against ttsim.)

  On Tensix: ELWADD = A + B, ELWSUB = A - B, ELWMUL = A * B.

Only the first 2 faces (512 elements) are defined by this op, so validation is
restricted to those lanes.
"""

import pytest
import torch
from conftest import skip_for_coverage, skip_for_wormhole
from helpers.device_io import read_from_device, write_to_device
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.pack import pack_bfp16
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    MATH_FIDELITY,
    MATH_OP,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block
from helpers.unpack import unpack_res_tiles
from helpers.utils import passed_test

# INTERMITTENT DEADLOCK — skipped on all backends. The golden here is fixed (per-face
# column-0 broadcast, validated on a ttsim run that completed), but the kernel does NOT
# reliably run to completion: on ttsim it frequently deadlocks (sim reports 0 KHz — no
# instructions retired — the signature of a sync/handshake deadlock, which ttsim models
# faithfully). This op reuses SrcB straight out of DEST and is the same family as
# sdpa_custom_mm_reuse_dest_srcb, which has a CONFIRMED unmatched-semaphore deadlock
# (consumes UNPACK_MATH_DONE with no SFPU producer). A deadlock wedges the Tensix and
# cascades timeouts across the whole suite, so it must not run on hardware until fixed.
# TODO: supply the producer half of the handshake (or confirm it's a ttsim sync artifact
# on a BH card) and un-skip. Likely a header handshake gap to flag to pmilenkovic.
pytestmark = pytest.mark.skip(
    reason="Intermittent sync deadlock (0 KHz on ttsim); would wedge the BH suite. "
    "Golden is fixed; needs the SrcB-reuse handshake resolved. Un-skip once fixed."
)

# The op processes exactly 2 faces (configure_mop LLK_ASSERTs num_faces == 2).
OP_NUM_FACES = 2
FACE_R_DIM = 16
FACE_C_DIM = 16
# Elements covered by the 2 defined faces of a single 32x32 tile.
DEFINED_ELEMENTS = OP_NUM_FACES * FACE_R_DIM * FACE_C_DIM  # 512


@skip_for_coverage
@skip_for_wormhole
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    mathop=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    dest_acc=[DestAccumulation.No],
    math_fidelity=[MathFidelity.LoFi],
    input_dimensions=[[32, 32]],
)
def test_sdpa_bcast_col_srcb_reuse(
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    input_dimensions,
):
    # Generate A stimuli; B is generated independently below so we control both.
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Tilize inputs for hardware layout.
    tilized_A = tilize_block(
        src_A, dimensions=input_dimensions, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_dimensions, stimuli_format=formats.input_format
    )

    # GOLDEN -------------------------------------------------------------------
    # 1. Column-broadcast B, PER FACE.
    #
    #    The kernel A2D-datacopies B into DEST tiles 0 and 1 (OP_NUM_FACES=2), the
    #    preamble MOVD2Bs those two DEST tiles into SrcB faces 0 and 1, and the MOP
    #    then applies p_elwise::SRCB_BCAST_COL -- which replicates *each SrcB face's
    #    own column 0* across that face's 16 columns. So face 0 broadcasts B's face-0
    #    column 0 and face 1 broadcasts B's face-1 column 0.
    #
    #    Note we CANNOT use BroadcastGolden(Column, num_faces=2) here: that path is
    #    for a 16x32 "tiny tile" (a single 16-row strip, one column domain) and folds
    #    both faces onto face 0's column (face_0_broadcast.repeat(2)). Our operand is
    #    the top half of a full 32x32 tile, whose faces 0 and 1 are the horizontally
    #    adjacent 16x16 blocks (cols 0-15 and 16-31) -- each with its own column 0.
    #    Confirmed on ttsim: face 1 of the result matches A_f1 + bcast_col(B_f1),
    #    NOT A_f1 + bcast_col(B_f0).
    tilized_B_flat = tilized_B.flatten()
    face_size = FACE_R_DIM * FACE_C_DIM  # 256
    src_B_broadcasted = torch.empty(DEFINED_ELEMENTS, dtype=tilized_B_flat.dtype)
    for f in range(OP_NUM_FACES):
        face = tilized_B_flat[f * face_size : (f + 1) * face_size].view(
            FACE_R_DIM, FACE_C_DIM
        )
        col0 = face[:, 0]  # one value per row = this face's column 0
        src_B_broadcasted[f * face_size : (f + 1) * face_size] = (
            col0.view(FACE_R_DIM, 1).repeat(1, FACE_C_DIM).flatten()
        )

    # 2. Elementwise op A <mathop> bcast_col(B), restricted to the 2 defined faces.
    binary_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_full = binary_golden(
        mathop,
        tilized_A.flatten()[:DEFINED_ELEMENTS],
        src_B_broadcasted[:DEFINED_ELEMENTS],
        formats.output_format,
        math_fidelity,
    )
    golden = golden_full[:DEFINED_ELEMENTS].to(format_dict[formats.output_format])

    # BUILD --------------------------------------------------------------------
    configuration = TestConfig(
        "sources/sdpa_bcast_col_srcb_reuse_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
        ],
        runtimes=[
            TILE_COUNT(1),
        ],
        variant_stimuli=None,  # stimuli written manually to fixed addresses
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    configuration.generate_variant_hash()
    if TestConfig.BUILD_MODE in [BuildMode.PRODUCE, BuildMode.DEFAULT]:
        configuration.build_elfs()

    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    # Addresses must match the Operand base addresses in the .cpp.
    BUFFER_A_ADDR = 0x1A000
    BUFFER_B_ADDR = 0x1A800
    BUFFER_RES_ADDR = 0x1B000

    write_to_device(
        TestConfig.TENSIX_LOCATION, BUFFER_A_ADDR, pack_bfp16(tilized_A.flatten())
    )
    write_to_device(
        TestConfig.TENSIX_LOCATION, BUFFER_B_ADDR, pack_bfp16(tilized_B.flatten())
    )

    configuration.run_elf_files()
    configuration.wait_for_tensix_operations_finished()

    # READ + VALIDATE (only the 2 defined faces) -------------------------------
    tile_size = 2048  # Float16_b 32x32 tile
    read_data = read_from_device(
        TestConfig.TENSIX_LOCATION, BUFFER_RES_ADDR, num_bytes=tile_size
    )
    # Extract only the 2 active faces from the full tile in L1.
    res_from_L1 = unpack_res_tiles(
        read_data,
        formats.output_format,
        tile_count=1,
        sfpu=False,
        num_faces=OP_NUM_FACES,
        face_r_dim=FACE_R_DIM,
    )
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])[
        :DEFINED_ELEMENTS
    ]

    assert passed_test(
        golden, res_tensor, formats.output_format
    ), "Assert against golden failed (defined 2-face lanes)"
