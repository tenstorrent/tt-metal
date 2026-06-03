# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    BroadcastType,
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
    StochasticRounding,
    Transpose,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    BROADCAST_TYPE,
    DISABLE_SRC_ZERO_FLAG,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PARTIAL_FACE,
    REUSE_DEST_TYPE,
    STOCHASTIC_ROUNDING,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.tile_constants import get_tile_params
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

supported_formats = [
    DataFormat.Int32,
    DataFormat.UInt32,
    DataFormat.UInt16,
    DataFormat.Float32,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
]

# Filter out BH-unsupported formats at import time.
if get_chip_architecture() == ChipArchitecture.BLACKHOLE:
    supported_formats = [
        f
        for f in supported_formats
        if f
        not in (
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.UInt32,
            DataFormat.UInt16,
        )
    ]


def _valid_dest_acc(formats):
    """32-bit formats require dest accumulation; exclude DestAccumulation.No for them."""
    if formats.input_format.is_32_bit():
        return [DestAccumulation.Yes]
    return [DestAccumulation.Yes, DestAccumulation.No]


def _valid_bcast_types(tile_dimensions, formats, dest_acc):
    """Filter broadcast types based on tile geometry and known HW constraints."""
    all_types = [
        BroadcastType.None_,
        BroadcastType.Column,
        BroadcastType.Row,
        BroadcastType.Scalar,
    ]

    result = []
    for bt in all_types:
        # TODO: pgardner - Column broadcast for tiny tiles needs kernel support
        if tile_dimensions != [32, 32] and bt == BroadcastType.Column:
            continue

        # TODO: pgardner - Bfp8_b requires minimum 16 exponents per face
        if tile_dimensions[0] < 16 and formats.input_format == DataFormat.Bfp8_b:
            continue

        # TODO: pgardner - known WH issue with row broadcast + dest accumulation
        if (
            TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE
            and bt == BroadcastType.Row
            and dest_acc == DestAccumulation.Yes
            and formats.input_format in (DataFormat.Float16_b, DataFormat.Bfp8_b)
        ):
            continue

        result.append(bt)

    return result


@dataclass(frozen=True, repr=False)
class BcastConfig:
    tile_dimensions: tuple
    formats: object
    dest_acc: DestAccumulation
    broadcast_type: BroadcastType

    def __repr__(self):
        f = self.formats
        return (
            f"{f.input_format.name}->{f.output_format.name}"
            f"-{self.dest_acc.name}-{self.broadcast_type.name}"
            f"-tile{self.tile_dimensions[0]}x{self.tile_dimensions[1]}"
        )


def _sweep_bcast():
    tile_dims_list = [[32, 32]]
    fmt_list = input_output_formats(supported_formats, same=True)
    combos = []
    for da in [DestAccumulation.Yes, DestAccumulation.No]:
        for bt in [
            BroadcastType.None_,
            BroadcastType.Column,
            BroadcastType.Row,
            BroadcastType.Scalar,
        ]:
            for td in tile_dims_list:
                for fmt in fmt_list:
                    if da not in _valid_dest_acc(fmt):
                        continue
                    if bt not in _valid_bcast_types(td, fmt, da):
                        continue
                    combos.append(
                        BcastConfig(
                            tile_dimensions=tuple(td),
                            formats=fmt,
                            dest_acc=da,
                            broadcast_type=bt,
                        )
                    )
    return combos


ALL_BCAST_CONFIGS = _sweep_bcast()


@pytest.mark.parametrize("config", ALL_BCAST_CONFIGS)
def test_unpack_bcast(config: BcastConfig):
    tile_dimensions = list(config.tile_dimensions)
    formats = config.formats
    broadcast_type = config.broadcast_type
    dest_acc = config.dest_acc
    # --- Tile geometry ---------------------------------------------------
    # get_tile_params returns (face_r_dim, num_faces_r_dim, num_faces_c_dim).
    # For tiny tiles (e.g. [4,32]): face_r_dim=4, num_faces=2.
    # For full tiles ([32,32]):     face_r_dim=16, num_faces=4.
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim
    input_dimensions = list(tile_dimensions)

    tile_cnt_A = (input_dimensions[0] // tile_dimensions[0]) * (
        input_dimensions[1] // tile_dimensions[1]
    )
    tile_cnt_B = tile_cnt_A

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        DestAccumulation.No,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    # --- Kernel configuration --------------------------------------------
    stimuli = StimuliConfig(
        None,
        formats.input_format,
        None,
        formats.input_format,
        formats.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
        tile_count_res=tile_cnt_A,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
        tile_dimensions=tile_dimensions,
        use_dense_tile_dimensions=True,
    )

    configuration = TestConfig(
        "sources/unpack_A_test.cpp",
        formats,
        templates=[
            STOCHASTIC_ROUNDING(StochasticRounding.No),
            BROADCAST_TYPE(broadcast_type),
            ACC_TO_DEST(False),
            REUSE_DEST_TYPE(EltwiseBinaryReuseDestType.NONE),
            PARTIAL_FACE(
                partial_a=False,
                partial_face_pack=False,
                partial_b=False,
                partial_face_math=False,
            ),
            DISABLE_SRC_ZERO_FLAG(False),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_BLOCKS(num_blocks),
        ],
        variant_stimuli=stimuli,
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit()
        and dest_acc == DestAccumulation.Yes,
    )

    configuration.prepare()
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    # --- Stimuli generation ----------------------------------------------
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    # --- Golden model ----------------------------------------------------
    if broadcast_type != BroadcastType.None_:
        generate_broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_tensor = generate_broadcast_golden(
            broadcast_type,
            src_A,
            formats.output_format,
            num_faces=num_faces,
            tile_cnt=tile_cnt_A,
            face_r_dim=face_r_dim,
        )
    else:
        golden_tensor = src_A.to(format_dict[formats.output_format])

    stimuli.set_buffers(src_A, src_B)

    res_from_L1 = configuration.run().result

    # --- Assertions ------------------------------------------------------
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    # Pretty red/green diff output via passed_test (tolerance-based)
    tile_shape = construct_tile_shape(tile_dimensions)
    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    )

    # Datacopy/bcast should be bit-exact for float formats (no compute loss)
    if formats.input_format in (DataFormat.Float32, DataFormat.Float16_b):
        assert torch.equal(
            golden_tensor, res_tensor
        ), "Datacopy/bcast should be exact for float formats"
