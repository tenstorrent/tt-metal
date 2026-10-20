# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TensorShape coverage for the math unary datacopy (tt-llk#1033, step 2).

Runs unpack_A -> _llk_math_eltwise_unary_datacopy_ -> pack on one tile per
shape, with the math side given the real TensorShape (unpack_A_test.cpp builds
it from the face grid). Covers the plain copy (A2D), the SrcB broadcasts (B2D)
and the 32-bit unpack-to-dest path (Float32 with dest_acc=Yes), with an exact
compare. This is the face-grid broadcast test for _llk_unpack_A_ as well
(tt-metal#47328); test_unpack_A.py covers the transpose of faces.

Shapes the LLK blocks with LLK_ASSERT are not listed here; see
_llk_math_eltwise_unary_datacopy_init_ for the supported set.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import BroadcastGolden, get_golden_generator
from helpers.llk_params import BroadcastType, DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_generator import generate_stimuli
from helpers.tile_constants import get_tile_params
from helpers.tile_shape import construct_tile_shape
from helpers.unpack_A_config import unpack_A_tile_config
from helpers.utils import passed_test

DATACOPY_TILE_SHAPES = [
    [1, 32],
    [2, 32],
    [4, 32],
    [8, 32],
    [16, 32],
    [32, 16],
    [16, 16],
    [32, 32],
]


@parametrize(
    tile_dimensions=DATACOPY_TILE_SHAPES,
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    broadcast_type=[
        BroadcastType.None_,
        BroadcastType.Scalar,
        BroadcastType.Row,
        BroadcastType.Column,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_eltwise_unary_datacopy_tensor_shape(
    tile_dimensions, formats, broadcast_type, dest_acc
):
    if dest_acc == DestAccumulation.No and formats.input_format.is_32_bit():
        pytest.skip("32-bit formats require dest accumulation")

    # Column broadcast through SrcB follows the face grid; the unpack-to-dest (32-bit) column
    # broadcast is still 32x32-only and _llk_unpack_A_init_ asserts it (tt-metal#47328).
    # Row and scalar unpack to dest run on every shape.
    if (
        broadcast_type == BroadcastType.Column
        and tile_dimensions != [32, 32]
        and formats.input_format.is_32_bit()
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip("Unpack-to-dest column broadcast is 32x32-only")

    # Same known Wormhole issue as test_bcast.py::test_unpack_bcast.
    if (
        get_chip_architecture() == ChipArchitecture.WORMHOLE
        and broadcast_type == BroadcastType.Row
        and dest_acc == DestAccumulation.Yes
        and formats.input_format == DataFormat.Float16_b
    ):
        pytest.skip("Row broadcast with dest_acc=Yes broken on Wormhole for Float16_b")

    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim
    input_dimensions = list(tile_dimensions)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    if broadcast_type != BroadcastType.None_:
        generate_broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_tensor = generate_broadcast_golden(
            broadcast_type,
            src_A,
            formats.output_format,
            num_faces=num_faces,
            tile_cnt=tile_cnt_A,
            face_r_dim=face_r_dim,
            num_faces_r_dim=num_faces_r_dim,
            num_faces_c_dim=num_faces_c_dim,
        )
    else:
        golden_tensor = src_A.to(format_dict[formats.output_format])

    configuration = unpack_A_tile_config(
        formats,
        tile_dimensions,
        src_A,
        tile_cnt_A,
        src_B,
        tile_cnt_B,
        dest_acc=dest_acc,
        broadcast_type=broadcast_type,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=construct_tile_shape(tile_dimensions),
    )

    # Datacopy/bcast should be bit-exact for float formats (no compute loss)
    if formats.input_format in (DataFormat.Float32, DataFormat.Float16_b):
        assert torch.equal(
            golden_tensor, res_tensor
        ), "Datacopy/bcast should be exact for float formats"
