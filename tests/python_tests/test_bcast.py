# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    DataCopyGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    EltwiseBinaryReuseDestType,
    StochasticRounding,
    Transpose,
    format_dict,
)
from helpers.param_config import generate_params, input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    BROADCAST_TYPE,
    DISABLE_SRC_ZERO_FLAG,
    NUM_FACES,
    PARTIAL_FACE,
    REUSE_DEST_TYPE,
    STOCHASTIC_ROUNDING,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.utils import passed_test

# SUPPORTED FORMATS FOR TEST
supported_formats = [
    DataFormat.Int32,
    DataFormat.UInt32,
    DataFormat.UInt16,
    DataFormat.Float32,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
]

# Define parameter lists
broadcast_types = [
    BroadcastType.None_,
    BroadcastType.Column,
    BroadcastType.Row,
    BroadcastType.Scalar,
]
dest_acc = [DestAccumulation.Yes, DestAccumulation.No]
disable_src_zero_flags = [False]
acc_to_dest_flags = [False]
stochastic_rnd = [
    StochasticRounding.No,
]
reuse_dest_types = [
    EltwiseBinaryReuseDestType.NONE,
]
transpose_of_faces_values = [Transpose.No]
within_face_16x16_transpose_values = [Transpose.No]
num_faces_values = [4]
face_r_dim_values = [16]


# Use only cross_test_formats as it already includes same-format combinations
test_formats = input_output_formats(supported_formats, False)


# Generate unpack_A specific parameter combinations using itertools.product
unpack_A_param_combinations = list(
    product(
        broadcast_types,
        disable_src_zero_flags,
        acc_to_dest_flags,
        dest_acc,
        stochastic_rnd,
        reuse_dest_types,
        transpose_of_faces_values,
        within_face_16x16_transpose_values,
        num_faces_values,
        face_r_dim_values,
    )
)

# Create unified parameter combinations
# This combines the power of generate_params with unpack_A specific parameters
all_params = []
# Use generate_params for base parameter structure (like datacopy test)
# Convert itertools.product to list
base_params = list(
    generate_params(testnames=["sources/unpack_A_test.cpp"], formats=test_formats)
)

# Extend base params with unpack_A specific parameters
for base_param in base_params:
    # base_param = (testname, format_config) - new format from main branch
    base_testname = base_param[0]
    formats = base_param[1]

    for unpack_params in unpack_A_param_combinations:
        broadcast_type = unpack_params[0]
        disable_src_zero = unpack_params[1]
        acc_to_dest = unpack_params[2]
        dest_acc = unpack_params[3]
        stochastic_rnd = unpack_params[4]
        reuse_dest = unpack_params[5]
        transpose_of_faces = unpack_params[6]
        within_face_16x16_transpose = unpack_params[7]
        num_faces = unpack_params[8]
        face_r_dim = unpack_params[9]

        # Create complete parameter tuple matching test signature
        combined_params = (
            base_testname,  # testname
            formats,  # formats
            broadcast_type,  # broadcast_type
            disable_src_zero,  # disable_src_zero
            acc_to_dest,  # acc_to_dest
            dest_acc,  # dest_acc
            stochastic_rnd,  # stochastic_rnd
            reuse_dest,  # reuse_dest
            transpose_of_faces,  # transpose_of_faces
            within_face_16x16_transpose,  # within_face_16x16_transpose
            num_faces,  # num_faces
            face_r_dim,  # face_r_dim
        )
        all_params.append(combined_params)


def filter_params_with_constraints(all_params):
    """Filter valid parameter combinations based on hardware constraints"""
    valid_params = []
    for params in all_params:
        # Extract parameters from tuple
        (
            testname,
            formats,
            broadcast_type,
            disable_src_zero,
            acc_to_dest,
            dest_acc,
            stochastic_rnd,
            reuse_dest,
            transpose_of_faces,
            within_face_16x16_transpose,
            num_faces,
            face_r_dim,
        ) = params

        if formats.input_format != formats.output_format:
            continue

        # BROADCAST + ACC_TO_DEST: ALL COMBINATIONS BROKEN (BLOCK ENTIRELY)
        # Check this early as it eliminates many combinations
        if broadcast_type != BroadcastType.None_ and acc_to_dest:
            continue

        if dest_acc == DestAccumulation.No and (
            formats.input_format == DataFormat.Float32
            or formats.input_format == DataFormat.Int32
            or formats.input_format == DataFormat.UInt32
        ):
            continue

        valid_params.append(params)

    return valid_params


# Apply constraint filtering
all_params = filter_params_with_constraints(all_params)


def create_simple_ids(all_params):
    """Create comprehensive but readable IDs for unpack_A tests"""
    ids = []
    for i, params in enumerate(all_params):
        # params = (testname, formats, broadcast_type, disable_src_zero,
        #           acc_to_dest, stoch_rnd_type, reuse_dest, transpose_of_faces,
        #           within_face_16x16_transpose, num_faces, face_r_dim)

        testname = params[0]
        formats = params[1]
        broadcast_type = params[2]
        disable_src_zero = params[3]
        acc_to_dest = params[4]
        dest_acc = params[5]
        stochastic_rnd = params[6]
        reuse_dest = params[7]
        transpose_of_faces = params[8]
        within_face_16x16_transpose = params[9]
        num_faces = params[10]
        face_r_dim = params[11]

        # Create a comprehensive but readable ID
        id_parts = [
            f"in_{formats.input_format.name}",
            f"out_{formats.output_format.name}",
            f"bcast_{broadcast_type.name}",
            f"disable_src_zero_{disable_src_zero}",
            f"acc_to_dest_{acc_to_dest}",
            f"dest_acc_{dest_acc}",
            f"stoch_rnd_{stochastic_rnd.name}",
            f"reuse_dest_{reuse_dest.name}",
            f"transpose_faces_{transpose_of_faces.name}",
            f"within_face_transpose_{within_face_16x16_transpose.name}",
            f"num_faces_{num_faces}",
            f"face_r_dim_{face_r_dim}",
        ]

        id_str = "-".join(id_parts)
        ids.append(id_str)

    return ids


param_ids = create_simple_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, broadcast_type, disable_src_zero, acc_to_dest, dest_acc, "
    "stochastic_rnd, reuse_dest, transpose_of_faces, "
    "within_face_16x16_transpose, num_faces, face_r_dim",
    all_params,
    ids=param_ids,
)
def test_unpack_bcast(
    testname,
    formats,
    broadcast_type,
    disable_src_zero,
    acc_to_dest,
    dest_acc,
    stochastic_rnd,
    reuse_dest,
    transpose_of_faces,
    within_face_16x16_transpose,
    num_faces,
    face_r_dim,
    workers_tensix_coordinates,
):
    # Note: All constraint validation has been done during parameter generation
    # No need for pytest.skip() calls - invalid combinations have been filtered out

    # Configure input dimensions based on face_r_dim
    # For partial faces (face_r_dim < 16), use [face_r_dim x 32] input tensors
    if face_r_dim < 16:
        input_dimensions = [face_r_dim, 32]  # [1x32], [2x32], [4x32], [8x32]
        partial_face = True
    else:
        input_dimensions = [32, 32]
        partial_face = False

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        face_r_dim=face_r_dim,
        num_faces=num_faces,
    )

    # generate golden tensor with proper broadcast and transpose handling
    # PRIORITY: Broadcast types take precedence over transpose operations
    if broadcast_type in (
        BroadcastType.Scalar,
        BroadcastType.Column,
        BroadcastType.Row,
    ):
        # Broadcast: replicate values according to broadcast type
        generate_broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_tensor = generate_broadcast_golden(
            broadcast_type,
            src_A,
            formats.output_format,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
        )
    elif transpose_of_faces == Transpose.Yes:
        # Both transpose flags are ALWAYS on together (mutually inclusive constraint)
        transpose_golden = get_golden_generator(TransposeGolden)
        # First apply within-face transpose, then face transpose
        temp_tensor = transpose_golden.transpose_within_faces(
            src_A, formats.output_format, input_dimensions, num_faces
        )
        golden_tensor = transpose_golden.transpose_faces(
            temp_tensor, formats.output_format, input_dimensions, num_faces
        )
    else:
        # No transpose - handle based on reuse_dest behavior
        if reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA and acc_to_dest:
            # DEST_TO_SRCA: destination registers get moved to srcA for reuse
            # This creates a feedback loop where processed data gets reused as source

            # For partial faces, DEST_TO_SRCA behavior may be different or unsupported
            if face_r_dim < 16:
                # For partial faces, fall back to regular data copy
                # DEST_TO_SRCA duplication logic may not apply to partial faces
                generate_golden = get_golden_generator(DataCopyGolden)
                golden_tensor = generate_golden(
                    src_A,
                    formats.output_format,
                    num_faces,
                    input_dimensions,
                    face_r_dim,
                )
            else:
                # Full faces: apply DEST_TO_SRCA duplication logic
                input_tensor = torch.tensor(
                    src_A, dtype=format_dict[formats.input_format]
                )
                face_size = face_r_dim * 16  # face_r_dim x 16 face

                if num_faces == 1:
                    # Single face with DEST_TO_SRCA + acc_to_dest:
                    # Hardware processes first half of face, then reuses/duplicates for second half
                    # DEST_TO_SRCA causes the first face_size/2 elements to be processed, then repeated
                    input_face = input_tensor[:face_size].to(
                        format_dict[formats.output_format]
                    )
                    half_face = face_size // 2
                    first_half = input_face[
                        :half_face
                    ]  # First half of variable-sized face
                    # Duplicate first half for second half due to DEST_TO_SRCA register reuse
                    golden_tensor = torch.cat([first_half, first_half])
                else:
                    # Multiple faces: DEST_TO_SRCA applies duplication pattern within each face
                    # Each face behaves like single face - first half duplicated for second half
                    result = torch.zeros(
                        face_size * num_faces, dtype=format_dict[formats.output_format]
                    )

                    for face_idx in range(num_faces):
                        face_start = face_idx * face_size
                        face_end = face_start + face_size
                        input_face = input_tensor[face_start:face_end].to(
                            format_dict[formats.output_format]
                        )

                        # Apply same duplication pattern as single face within each face
                        half_face = face_size // 2
                        first_half = input_face[
                            :half_face
                        ]  # First half of variable-sized face
                        face_output = torch.cat(
                            [first_half, first_half]
                        )  # Duplicate first half
                        result[face_start:face_end] = face_output

                    golden_tensor = result
        else:
            # Regular data copy for other reuse types or no acc_to_dest
            generate_golden = get_golden_generator(DataCopyGolden)
            golden_tensor = generate_golden(
                src_A, formats.output_format, num_faces, input_dimensions, face_r_dim
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
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
        ),
        dest_acc=(DestAccumulation.Yes if acc_to_dest else DestAccumulation.No),
        unpack_to_dest=(formats.input_format.is_32_bit()),
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates)

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
