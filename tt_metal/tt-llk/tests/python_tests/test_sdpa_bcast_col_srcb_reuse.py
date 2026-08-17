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
    * bcast_col replicates column 0 of each SrcB face across all 16 columns
      (matching BroadcastGolden._broadcast_column with num_faces=2, where both
      output faces reuse face 0's column values).

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
    # 1. Column-broadcast B over 2 faces: both output faces reuse face 0's
    #    column values (this mirrors the MOVD2B-into-SrcB + SRCB_BCAST_COL path).
    broadcast_golden = get_golden_generator(BroadcastGolden)
    src_B_broadcasted = broadcast_golden(
        BroadcastType.Column,
        tilized_B.flatten(),
        formats.input_format,
        num_faces=OP_NUM_FACES,
        tile_cnt=1,
        face_r_dim=FACE_R_DIM,
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
