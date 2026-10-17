# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""A partial-face ELWADD/ELWSUB must leave DEST rows 8..15 of every face slot untouched.

The kernel unpacks and packs 16x32 tiles but runs the math with 8-row faces, so
the packed rows 8..15 of each face expose whatever the MOP did past the tile's
rows. Boot ZEROACCs all of DEST and each pack ZEROACCs the packed half, so a
row nothing wrote reads as zero: those rows must come back as zero, and the
rows the tile does have must hold the op result. The dest-reuse variants seed
DEST with the first tile and fold the remaining tiles through DEST_TO_SRCA /
DEST_TO_SRCB, exercising the dest-reuse MOP the same way.

The same partial-face MOP is built for every SrcB broadcast, so the standard
and DEST_TO_SRCA variants also run with ROW / COL / SCALAR broadcast of the
unpacked SrcB. Those cells hold SrcB constant within each tile, so the golden
does not depend on which SrcB row or column the FPU samples (that is
test_eltwise_binary_broadcast's job); they only check that rows 8..15 stay
untouched. DEST_TO_SRCB takes SrcB from DEST, so it runs without broadcast.
"""

import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BroadcastType,
    DestSync,
    EltwiseBinaryReuseDestType,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    DEST_SYNC,
    MATH_OP,
    REUSE_DEST_TYPE,
    TILE_COUNT,
)
from helpers.tile_constants import FACE_C_DIM, get_tile_params
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from test_eltwise_binary import _apply_dest_reuse_op

# The probe drives the Wormhole/Blackhole llk_unpack_AB.h / llk_math_eltwise_binary.h MOPs.
pytestmark = skip_for_quasar

TILE_DIMENSIONS = [16, 32]


def _broadcast_types(reuse_dest_type):
    """SrcB broadcast only applies where SrcB comes from the unpacker."""
    if reuse_dest_type == EltwiseBinaryReuseDestType.DEST_TO_SRCB:
        return [BroadcastType.None_]
    return [
        BroadcastType.None_,
        BroadcastType.Row,
        BroadcastType.Column,
        BroadcastType.Scalar,
    ]


@parametrize(
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub],
    reuse_dest_type=[
        EltwiseBinaryReuseDestType.NONE,
        EltwiseBinaryReuseDestType.DEST_TO_SRCA,
        EltwiseBinaryReuseDestType.DEST_TO_SRCB,
    ],
    broadcast_type=lambda reuse_dest_type: _broadcast_types(reuse_dest_type),
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float16], same=True),
)
def test_partial_face_rows_untouched(math_op, reuse_dest_type, broadcast_type, formats):
    input_dimensions = [64, 32]
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(TILE_DIMENSIONS)
    num_faces = num_faces_r_dim * num_faces_c_dim
    tile_elements = num_faces * face_r_dim * FACE_C_DIM
    # Rows of each face the 8-row-face math op owns; the rest of the 16-row face must stay untouched.
    math_face_r_dim = face_r_dim // 2

    src_A, tile_cnt, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=TILE_DIMENSIONS,
    )

    def tilize(tensor):
        return tilize_block(
            tensor,
            dimensions=input_dimensions,
            stimuli_format=formats.input_format,
            num_faces=num_faces,
            tile_dimensions=TILE_DIMENSIONS,
            face_r_dim=face_r_dim,
        ).flatten()

    src_A_tilized = tilize(src_A)
    src_B_tilized = tilize(src_B)
    if broadcast_type != BroadcastType.None_:
        # Every element of a tile equals its first one, so ROW / COL / SCALAR broadcast of the
        # tile is the tile itself whichever SrcB row or column the FPU samples.
        src_B_tiles = src_B_tilized.view(tile_cnt, tile_elements)
        src_B_tiles[:] = src_B_tiles[:, :1]

    torch_format = format_dict[formats.output_format]
    a = src_A_tilized.to(torch_format).view(tile_cnt, tile_elements)
    b = src_B_tilized.to(torch_format).view(tile_cnt, tile_elements)

    def apply_op(src_a, src_b):
        # Same op and operand order as test_eltwise_binary's dest-reuse golden. ELWADD/ELWSUB
        # have no fidelity phases, so the helper never touches the golden generator (None).
        return _apply_dest_reuse_op(
            math_op, MathFidelity.LoFi, formats, torch_format, None, src_a, src_b
        )

    if reuse_dest_type == EltwiseBinaryReuseDestType.NONE:
        # One output tile per input tile.
        golden = apply_op(a, b)
    else:
        # Seed DEST with tile 0, then fold the remaining tiles through DEST.
        dest = apply_op(a[0], b[0])
        for i in range(1, tile_cnt):
            if reuse_dest_type == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
                dest = apply_op(dest, b[i])
            else:
                dest = apply_op(a[i], dest)
        golden = dest.view(1, tile_elements)
    output_tiles = golden.shape[0]
    # Face-major layout: tile t, face f, row r starts at ((t * num_faces + f) * face_r_dim + r) * FACE_C_DIM.
    golden = golden.reshape(output_tiles * num_faces, face_r_dim, FACE_C_DIM).clone()
    owned_rows = torch.zeros(face_r_dim, dtype=torch.bool)
    owned_rows[:math_face_r_dim] = True
    golden[:, ~owned_rows, :] = 0
    golden = golden.flatten()

    configuration = TestConfig(
        "sources/eltwise_binary_partial_face_rows_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=math_op),
            BROADCAST_TYPE(broadcast_type),
            DEST_SYNC(DestSync.Half),
            REUSE_DEST_TYPE(reuse_dest_type=reuse_dest_type),
        ],
        runtimes=[TILE_COUNT(tile_cnt)],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B_tilized,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=output_tiles,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
        ),
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == output_tiles * tile_elements, "Result length mismatch"
    res = torch.tensor(res_from_L1, dtype=torch_format).view(
        output_tiles * num_faces, face_r_dim, FACE_C_DIM
    )

    untouched = res[:, ~owned_rows, :]
    assert torch.all(untouched == 0), (
        "A partial-face op wrote DEST rows beyond its face_r_dim: rows "
        f"{math_face_r_dim}..{face_r_dim - 1} of the packed faces are not zero "
        f"(max |value| {untouched.abs().max().item()})"
    )
    assert passed_test(
        golden, res.flatten(), formats.output_format
    ), "Assert against golden failed"
