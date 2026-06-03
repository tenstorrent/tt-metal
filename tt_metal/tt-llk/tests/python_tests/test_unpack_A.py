# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIMENSIONS,
    BroadcastGolden,
    DataCopyGolden,
    TransposeGolden,
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
    INPUT_DIMENSIONS,
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
from helpers.utils import passed_test


@dataclass(frozen=True, repr=False)
class UnpackConfig:
    testname: str
    formats: InputOutputFormat
    broadcast_type: BroadcastType
    disable_src_zero: bool
    acc_to_dest: bool
    stochastic_rnd: StochasticRounding
    reuse_dest: EltwiseBinaryReuseDestType
    transpose_of_faces: Transpose
    within_face_16x16_transpose: Transpose
    num_faces: int
    face_r_dim: int
    input_dimensions: tuple

    def __repr__(self):
        f = self.formats
        return (
            f"in_{f.input_format.name}-out_{f.output_format.name}"
            f"-bcast_{self.broadcast_type.name}"
            f"-dsz_{self.disable_src_zero}-atd_{self.acc_to_dest}"
            f"-sr_{self.stochastic_rnd.name}-rd_{self.reuse_dest.name}"
            f"-tr_{self.transpose_of_faces.name}-wft_{self.within_face_16x16_transpose.name}"
            f"-nf_{self.num_faces}-fr_{self.face_r_dim}"
            f"-{self.input_dimensions[0]}x{self.input_dimensions[1]}"
        )


# SUPPORTED FORMATS FOR TEST
supported_formats = [
    DataFormat.Float32,
    DataFormat.Float16,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
]

# Define parameter lists
broadcast_types = [
    BroadcastType.None_,
]
dest_acc = [DestAccumulation.Yes, DestAccumulation.No]
disable_src_zero_flags = [False, True]
acc_to_dest_flags = [False, True]
stochastic_rnd = [
    StochasticRounding.No,
    StochasticRounding.Fpu,
    StochasticRounding.Pack,
    StochasticRounding.All,
]
reuse_dest_types = [
    EltwiseBinaryReuseDestType.NONE,
    EltwiseBinaryReuseDestType.DEST_TO_SRCA,
    EltwiseBinaryReuseDestType.DEST_TO_SRCB,
]
transpose_of_faces_values = [Transpose.No, Transpose.Yes]
within_face_16x16_transpose_values = [Transpose.No, Transpose.Yes]
num_faces_values = [1, 2, 4]
face_r_dim_values = [1, 2, 4, 8, 16]

input_dimensions = [[r, 32] for r in face_r_dim_values] + [[256, 256]]
# Use only cross_test_formats as it already includes same-format combinations
test_formats = input_output_formats(supported_formats, False)


# Build all parameter combinations with template-affecting params in outer loops
# and runtime-only params in inner loops. This ensures variants sharing the same
# compiled ELF are consecutive, minimizing device ELF reloads.
#
# Template params (affect compile hash): broadcast_type, disable_src_zero,
#   acc_to_dest, stochastic_rnd, reuse_dest, face_r_dim (via PARTIAL_FACE)
# Runtime params (vary fastest): transpose_of_faces, within_face_16x16_transpose,
#   num_faces, input_dimensions
# formats affects hash only via unpack_to_dest (32-bit + acc_to_dest), placed
# between template and runtime params.

all_params = []
for bt in broadcast_types:
    for dsz in disable_src_zero_flags:
        for atd in acc_to_dest_flags:
            for sr in stochastic_rnd:
                for rd in reuse_dest_types:
                    for frd in face_r_dim_values:
                        for fmt in test_formats:
                            for tof in transpose_of_faces_values:
                                for wft in within_face_16x16_transpose_values:
                                    for nf in num_faces_values:
                                        for idims in input_dimensions:
                                            all_params.append(
                                                UnpackConfig(
                                                    testname="sources/unpack_A_test.cpp",
                                                    formats=fmt,
                                                    broadcast_type=bt,
                                                    disable_src_zero=dsz,
                                                    acc_to_dest=atd,
                                                    stochastic_rnd=sr,
                                                    reuse_dest=rd,
                                                    transpose_of_faces=tof,
                                                    within_face_16x16_transpose=wft,
                                                    num_faces=nf,
                                                    face_r_dim=frd,
                                                    input_dimensions=tuple(idims),
                                                )
                                            )


def filter_params_with_constraints(all_params):
    """Filter valid parameter combinations based on hardware constraints"""

    arch = TestConfig.CHIP_ARCH
    is_wormhole = arch == ChipArchitecture.WORMHOLE
    valid_params = []

    for params in all_params:
        # Fast checks first: simple integer/enum comparisons
        # For partial faces (face_r_dim < 16), require num_faces = 2
        if params.face_r_dim < 16:
            if params.num_faces != 2:
                continue
            # Block Bfp8_b input/output for partial faces
            if (
                params.formats.input_format == DataFormat.Bfp8_b
                or params.formats.output_format == DataFormat.Bfp8_b
            ):
                continue

        if params.face_r_dim < 16 and params.input_dimensions != (
            params.face_r_dim,
            32,
        ):
            continue

        # Full face requires 32x32-multiple dimensions to avoid zero tiles
        if params.face_r_dim == 16 and (
            params.input_dimensions[0] % TILE_DIMENSIONS[0] != 0
            or params.input_dimensions[1] % TILE_DIMENSIONS[1] != 0
        ):
            continue

        # User constraint: transpose_of_faces and within_face_16x16_transpose are mutually inclusive
        if params.transpose_of_faces != params.within_face_16x16_transpose:
            continue

        if params.transpose_of_faces == Transpose.Yes and params.num_faces == 2:
            continue

        # Block transpose operations for face_r_dim < 16
        if params.transpose_of_faces == Transpose.Yes and params.face_r_dim < 16:
            continue

        # BROADCAST + ACC_TO_DEST: ALL COMBINATIONS BROKEN (BLOCK ENTIRELY)
        # Check this early as it eliminates many combinations
        if params.broadcast_type != BroadcastType.None_ and params.acc_to_dest:
            continue

        # Broadcast type checks (fast enum comparisons)
        broadcast_none = params.broadcast_type == BroadcastType.None_

        # COL broadcast requires 4 faces
        if params.broadcast_type == BroadcastType.Column and (
            params.num_faces != 4 or params.face_r_dim < 16
        ):
            continue

        # ROW broadcast constraint: Requires 4 faces for proper row broadcast
        if params.broadcast_type == BroadcastType.Row:
            if params.num_faces != 4:
                continue
            # Block Wormhole Row broadcast with outlier format combinations
            if (
                is_wormhole
                and params.formats.input_format
                in (DataFormat.Float16_b, DataFormat.Bfp8_b)
                and params.formats.output_format == DataFormat.Float16
            ):
                continue

        # SCALAR broadcast + acc_to_dest not allowed (already checked above, but explicit)
        if params.broadcast_type == BroadcastType.Scalar and params.acc_to_dest:
            continue

        # Broadcast incompatible with DEST_TO_SRCB/SRCA
        if not broadcast_none:
            if params.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCB:
                continue
            if params.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
                continue

        # Reuse dest checks
        reuse_none = params.reuse_dest == EltwiseBinaryReuseDestType.NONE

        # Exclude acc_to_dest=True for simple datacopy operations
        if (
            params.acc_to_dest
            and params.transpose_of_faces == Transpose.No
            and broadcast_none
            and reuse_none
        ):
            continue

        # Hardware constraint: unpack_to_dest can only be true if acc_to_dest is false
        # But unpack_to_dest = is_32_bit() and acc_to_dest, so we must block
        # any case where is_32_bit() and acc_to_dest are both true
        if params.formats.input_format.is_32_bit() and params.acc_to_dest:
            # This would result in unpack_to_dest=True, but hardware requires acc_to_dest=False
            # when unpack_to_dest=True. Block this combination.
            continue

        # Format-specific checks (most expensive, do last)
        # Block Bfp8_b output with stochastic rounding (Pack or All)
        if params.formats.output_format == DataFormat.Bfp8_b:
            if params.stochastic_rnd in (
                StochasticRounding.Pack,
                StochasticRounding.All,
            ):
                continue

        # Block Float16/Float16_b transpose combinations that produce garbage values on CI runners
        if (
            params.formats.input_format in (DataFormat.Float16_b, DataFormat.Float16)
            and broadcast_none
            and params.acc_to_dest
            and (
                reuse_none
                or params.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA
            )
            and params.transpose_of_faces == Transpose.Yes
            and params.within_face_16x16_transpose == Transpose.Yes
        ):
            continue

        full_tiles = (
            params.input_dimensions[0] * params.input_dimensions[1]
        ) >= ELEMENTS_PER_TILE

        # Block Bfp8_b transpose with acc_to_dest + DEST_TO_SRCA (hardware mismatch) (see #1348 issue in tt-llk).
        if (
            params.formats.input_format == DataFormat.Bfp8_b
            and params.acc_to_dest
            and params.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA
            and params.transpose_of_faces == Transpose.Yes
            and params.within_face_16x16_transpose == Transpose.Yes
            and full_tiles
        ):
            continue

        # Block Bfp8_b transpose with acc_to_dest + reuse NONE (hardware mismatch) (see #1348 issue in tt-llk).
        if (
            params.formats.input_format == DataFormat.Bfp8_b
            and params.acc_to_dest
            and params.reuse_dest == EltwiseBinaryReuseDestType.NONE
            and params.transpose_of_faces == Transpose.Yes
            and params.within_face_16x16_transpose == Transpose.Yes
            and full_tiles
        ):
            continue

        # All constraints passed, add to valid params
        valid_params.append(params)

    return valid_params


# Apply constraint filtering
all_params = filter_params_with_constraints(all_params)

# When tests are randomised, they fail in various ways: https://github.com/tenstorrent/tt-llk/issues/1108
_skip_blackhole = get_chip_architecture() == ChipArchitecture.BLACKHOLE
_filtered_params = [] if _skip_blackhole else all_params


@pytest.mark.parametrize(
    "config",
    _filtered_params,
)
def test_unpack_comprehensive(config: UnpackConfig):
    testname = config.testname
    formats = config.formats
    broadcast_type = config.broadcast_type
    disable_src_zero = config.disable_src_zero
    acc_to_dest = config.acc_to_dest
    stochastic_rnd = config.stochastic_rnd
    reuse_dest = config.reuse_dest
    transpose_of_faces = config.transpose_of_faces
    within_face_16x16_transpose = config.within_face_16x16_transpose
    num_faces = config.num_faces
    face_r_dim = config.face_r_dim
    input_dimensions = list(config.input_dimensions)

    partial_face = face_r_dim < 16

    if partial_face:
        tile_cnt_A = 1
    else:
        tile_cnt_A = (input_dimensions[0] // TILE_DIMENSIONS[0]) * (
            input_dimensions[1] // TILE_DIMENSIONS[1]
        )

    raw_dimensions = [
        (
            input_dimensions[0]
            if input_dimensions[0] >= TILE_DIMENSIONS[0]
            else TILE_DIMENSIONS[0]
        ),
        input_dimensions[1],
    ]
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        DestAccumulation.Yes if acc_to_dest else DestAccumulation.No,
        formats,
        raw_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    # Construct StimuliConfig with layout info but no tensor data yet.
    stimuli = StimuliConfig(
        None,
        formats.input_format,
        None,
        formats.input_format,
        formats.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_A,
        tile_count_res=tile_cnt_A,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
    )
    configuration = TestConfig(
        testname,
        formats,
        templates=[
            STOCHASTIC_ROUNDING(stochastic_rnd),
            BROADCAST_TYPE(broadcast_type),
            ACC_TO_DEST(acc_to_dest),
            REUSE_DEST_TYPE(reuse_dest),
            PARTIAL_FACE(
                partial_a=partial_face,
                partial_face_pack=partial_face,
                partial_b=partial_face,
                partial_face_math=partial_face,
            ),
            DISABLE_SRC_ZERO_FLAG(disable_src_zero),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(transpose_of_faces),
            UNPACK_TRANS_WITHIN_FACE(within_face_16x16_transpose),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_BLOCKS(num_blocks),
            INPUT_DIMENSIONS(
                raw_dimensions[0] // TILE_DIMENSIONS[0],
                raw_dimensions[1] // TILE_DIMENSIONS[1],
            ),
        ],
        variant_stimuli=stimuli,
        dest_acc=(DestAccumulation.Yes if acc_to_dest else DestAccumulation.No),
        unpack_to_dest=(formats.input_format.is_32_bit() and acc_to_dest),
    )

    configuration.prepare()
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    # Phase 2: generate stimuli + golden (consumer / default mode only)
    src_A, _, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        face_r_dim=face_r_dim,
        num_faces=num_faces,
    )

    if broadcast_type in (
        BroadcastType.Scalar,
        BroadcastType.Column,
        BroadcastType.Row,
    ):
        generate_broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_tensor = generate_broadcast_golden(
            broadcast_type,
            src_A,
            formats.output_format,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_cnt=tile_cnt_A,
        )
    elif transpose_of_faces == Transpose.Yes:
        transpose_golden = get_golden_generator(TransposeGolden)
        if tile_cnt_A > 1:
            tiles = torch.tensor(src_A, dtype=format_dict[formats.input_format]).view(
                tile_cnt_A, ELEMENTS_PER_TILE
            )
            processed_tiles = []
            for tile in tiles:
                temp_tensor = transpose_golden.transpose_within_faces(
                    tile, formats.output_format, input_dimensions, num_faces
                )
                processed_tiles.append(
                    transpose_golden.transpose_faces(
                        temp_tensor, formats.output_format, input_dimensions, num_faces
                    )
                )
            golden_tensor = torch.cat(processed_tiles)
        else:
            temp_tensor = transpose_golden.transpose_within_faces(
                src_A, formats.output_format, input_dimensions, num_faces
            )
            golden_tensor = transpose_golden.transpose_faces(
                temp_tensor, formats.output_format, input_dimensions, num_faces
            )
    else:
        if reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA and acc_to_dest:
            if face_r_dim < 16:
                generate_golden = get_golden_generator(DataCopyGolden)
                golden_tensor = generate_golden(
                    src_A,
                    formats.output_format,
                    num_faces,
                    input_dimensions,
                    face_r_dim,
                )
            else:
                input_tensor = torch.tensor(
                    src_A, dtype=format_dict[formats.input_format]
                )
                face_size = face_r_dim * 16

                def _dest_to_srca_tile(tile_tensor: torch.Tensor) -> torch.Tensor:
                    if num_faces == 1:
                        input_face = tile_tensor[:face_size].to(
                            format_dict[formats.output_format]
                        )
                        half_face = face_size // 2
                        first_half = input_face[:half_face]
                        return torch.cat([first_half, first_half])

                    result = torch.zeros(
                        face_size * num_faces, dtype=format_dict[formats.output_format]
                    )
                    for face_idx in range(num_faces):
                        face_start = face_idx * face_size
                        face_end = face_start + face_size
                        input_face = tile_tensor[face_start:face_end].to(
                            format_dict[formats.output_format]
                        )
                        half_face = face_size // 2
                        first_half = input_face[:half_face]
                        face_output = torch.cat([first_half, first_half])
                        result[face_start:face_end] = face_output
                    return result

                if tile_cnt_A > 1:
                    tiles = input_tensor.view(tile_cnt_A, ELEMENTS_PER_TILE)
                    golden_tensor = torch.cat(
                        [_dest_to_srca_tile(tile) for tile in tiles]
                    )
                else:
                    golden_tensor = _dest_to_srca_tile(input_tensor)
        else:
            generate_golden = get_golden_generator(DataCopyGolden)
            golden_tensor = generate_golden(
                src_A, formats.output_format, num_faces, input_dimensions, face_r_dim
            )

    # Attach real tensor data before execution
    stimuli.set_buffers(src_A, src_B)

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
