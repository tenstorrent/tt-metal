# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blackhole-only unit test for the experimental LLK
``_llk_math_sdpa_bcast_col_srca_srcb_reuse_`` (promoted by tt-metal #53295).

This op is a MATH-only, DEST-resident column-broadcast eltwise where BOTH source
registers are reused out of DEST:

    * SrcB is loaded from DEST tile ``isrc`` by the preamble (MOVD2B).
    * SrcA is loaded from DEST tile ``dst_index`` by the MOP start-op (MOVD2A).
    * The MOP body is ELWSUB with ``p_elwise::SRCB_BCAST_COL`` (SrcB never cleared),
      accumulated back into DEST at ``dst_index``.

So for one 16x32 (2-face) tile the op computes::

    DEST[dst_index] = A - bcast_col(B)

where ``A`` is the DEST tile at ``dst_index``, ``B`` is the DEST tile at ``isrc``, and
``bcast_col`` replicates each face's column-0 across all 16 columns of that face
(``BroadcastType.Column`` semantics).

The kernel A2D-datacopies operand A into DEST tile 0 and operand B into DEST tile 1
(both fed tilized, so the DEST face layout matches the hardware column-broadcast domain),
runs the reuse op with ``dst_index=0`` / ``isrc=1``, and packs DEST tile 0. The golden is
therefore identical to the sibling ``sub_bcast_col`` golden — ``EltwiseBinary(ELWSUB, A,
bcast_col(B))`` — the whole 512-datum tile is defined and validated at the output tolerance.

Blackhole-only: the op's MOP/preamble (MOVD2A/MOVD2B from DEST) exist only on BH; there is
no BH card in this environment, so the bar is a clean BH compile plus a golden faithfully
mirroring the header math. num_faces is fixed at 2 (the op asserts ``num_faces == 2``).
"""

import logging

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
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
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import MATH_FIDELITY
from helpers.tile_constants import get_tile_params
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

logger = logging.getLogger(__name__)

# BLOCKED: the promoted header llk_math_sdpa_bcast_col_srca_srcb_reuse.h (tt-metal #53295)
# does not compile under the tt-llk test build's -Werror: five dead symbols trip
# -Wunused-variable / -Wunused-parameter (addr_mod, innerloop, outerloop, high_fidelity,
# and the num_faces parameter of sdpa_bcast_col_srca_srcb_reuse_configure_addrmod).
# The demo build does not use those flags, so it slipped through. This test is otherwise
# complete and compiles clean once the dead symbols are removed from the header.
# TODO: remove this skip after pmilenkovic drops the unused symbols in #53295.
pytestmark = pytest.mark.skip(
    reason="Blocked on -Werror unused-var/param in promoted header "
    "llk_math_sdpa_bcast_col_srca_srcb_reuse.h (#53295); un-skip once fixed."
)

# 16x32 tiny tile: one face-row, 2 faces of 16x16 -> 512 datums. The op asserts num_faces == 2.
TILE_DIMENSIONS = [16, 32]
ELEMENTS_PER_TILE = TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1]


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    # SUB is fidelity-independent (single MOP pass), so a single fidelity level suffices.
    math_fidelity=[MathFidelity.LoFi],
    # bf16 DEST (No) exercises the op; the header path does not depend on fp32 DEST for SUB.
    dest_acc=[DestAccumulation.No],
)
def test_sdpa_bcast_col_srca_srcb_reuse(formats, math_fidelity, dest_acc):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "sdpa_bcast_col_srca_srcb_reuse is a Blackhole-only experimental LLK"
        )

    # 16x32 tiny tile geometry: face_r_dim=16, num_faces=2.
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(TILE_DIMENSIONS)
    num_faces = num_faces_r_dim * num_faces_c_dim
    assert num_faces == 2, "srca_srcb_reuse op requires a 2-face (16x32) tile"

    # Operand A (SrcA source) and operand B (column-broadcast SrcB source). One tile each.
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=TILE_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=TILE_DIMENSIONS,
        tile_dimensions=TILE_DIMENSIONS,
    )

    # Both operands are consumed from buffer_A (tile 0 = A, tile 1 = B) by the kernel; they
    # are fed tilized so their DEST face layout matches the hardware bcast-col domain.
    src_A_tilized = tilize_block(
        src_A,
        TILE_DIMENSIONS,
        formats.input_format,
        num_faces=num_faces,
        tile_dimensions=TILE_DIMENSIONS,
        face_r_dim=face_r_dim,
    ).flatten()
    src_B_tilized = tilize_block(
        src_B,
        TILE_DIMENSIONS,
        formats.input_format,
        num_faces=num_faces,
        tile_dimensions=TILE_DIMENSIONS,
        face_r_dim=face_r_dim,
    ).flatten()

    # Golden: column-broadcast B, then A - bcast_col(B). Broadcast is computed on the tilized
    # tile (face domain), then untilized back to a row-major [16, 32] operand for the eltwise.
    broadcast_golden = get_golden_generator(BroadcastGolden)
    src_B_broadcasted_tilized = broadcast_golden(
        BroadcastType.Column,
        src_B_tilized,
        formats.input_format,
        num_faces=num_faces,
        tile_cnt=tile_cnt_B,
        face_r_dim=face_r_dim,
    )
    src_B_golden = untilize_block(
        src_B_broadcasted_tilized,
        formats.input_format,
        TILE_DIMENSIONS,
        num_faces=num_faces,
        tile_dimensions=TILE_DIMENSIONS,
        face_r_dim=face_r_dim,
    ).flatten()

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        MathOperation.Elwsub,
        src_A,
        src_B_golden,
        formats.output_format,
        math_fidelity,
    )

    # buffer_A holds both operand tiles (A at index 0, B at index 1). buffer_B is unused by
    # the kernel; a single dummy tile keeps the framework's operand layout valid.
    stim_A = torch.cat([src_A_tilized, src_B_tilized])
    dummy_B = src_B_tilized

    configuration = TestConfig(
        "sources/sdpa_bcast_col_srca_srcb_reuse_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
        ],
        variant_stimuli=StimuliConfig(
            stim_A,
            formats.input_format,
            dummy_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=2,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=dest_acc,
    )
    res_from_L1 = configuration.run().result

    # The op packs a tiled 16x32 result; untilize to row-major for comparison.
    res_from_L1 = untilize_block(
        res_from_L1,
        formats.output_format,
        TILE_DIMENSIONS,
        num_faces=num_faces,
        tile_dimensions=TILE_DIMENSIONS,
        face_r_dim=face_r_dim,
    ).flatten()

    assert (
        len(res_from_L1) == ELEMENTS_PER_TILE
    ), f"Expected one {ELEMENTS_PER_TILE}-element output tile, got {len(res_from_L1)}"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # All 512 lanes of the 16x32 tile are defined by the op; validate the whole tile.
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "sdpa_bcast_col_srca_srcb_reuse diverged from A - bcast_col(B) golden"
