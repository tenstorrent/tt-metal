# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Int4/UInt4 L1 -> Int8/UInt8 SrcA unpack, the on-the-fly int4 weight dequant
# step of a wint4_aint8 linear. Int8 -> Int8 rides the same kernel as a control.

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    DataCopyGolden,
    PackGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    PackerReluType,
    PerfRunType,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    RELU_CONFIG,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

# Value range the unpacker can see for each L1 format. Int4 is sign-magnitude
# S1.3, so -8 has no encoding; Int8 is sign-magnitude too, so -128 is excluded.
INT_RANGES = {
    DataFormat.Int4: (-7, 7),
    DataFormat.UInt4: (0, 15),
    DataFormat.Int8: (-127, 127),
}

# (input, output, dest_acc). Int32 output needs a 32-bit Dest.
INT4_UNPACK_COMBINATIONS = [
    (InputOutputFormat(DataFormat.Int8, DataFormat.Int8), DestAccumulation.No),
    (InputOutputFormat(DataFormat.Int4, DataFormat.Int8), DestAccumulation.No),
    (InputOutputFormat(DataFormat.Int4, DataFormat.Int8), DestAccumulation.Yes),
    (InputOutputFormat(DataFormat.Int4, DataFormat.Int32), DestAccumulation.Yes),
    (InputOutputFormat(DataFormat.UInt4, DataFormat.UInt8), DestAccumulation.No),
    (InputOutputFormat(DataFormat.UInt4, DataFormat.Int32), DestAccumulation.Yes),
]


def generate_int_stimuli(fmt: DataFormat, input_dimensions, seed: int):
    # Every representable value appears in each tile; the shuffle keeps any
    # nibble-order or face-order mistake from lining up with a pattern.
    lo, hi = INT_RANGES[fmt]
    num_elements = input_dimensions[0] * input_dimensions[1]
    values = torch.arange(lo, hi + 1, dtype=torch.int32).repeat(
        num_elements // (hi - lo + 1) + 1
    )[:num_elements]
    generator = torch.Generator().manual_seed(seed)
    values = values[torch.randperm(num_elements, generator=generator)]
    return values.to(format_dict[fmt])


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=INT4_UNPACK_COMBINATIONS,
    dest_sync=[DestSync.Half],
    input_dimensions=[[32, 32], [64, 64]],
)
def test_unpack_int4_quasar(formats_dest_acc, dest_sync, input_dimensions):
    formats, dest_acc = formats_dest_acc
    tile_dimensions = [32, 32]
    tile_shape = construct_tile_shape(tile_dimensions)
    num_faces = tile_shape.total_num_faces()
    tile_cnt = (input_dimensions[0] // tile_dimensions[0]) * (
        input_dimensions[1] // tile_dimensions[1]
    )

    src_A = generate_int_stimuli(formats.input_format, input_dimensions, seed=1)
    src_B = generate_int_stimuli(formats.input_format, input_dimensions, seed=2)

    golden_tensor = get_golden_generator(DataCopyGolden)(
        src_A,
        formats.output_format,
        num_faces=num_faces,
        face_r_dim=tile_shape.face_r_dim,
        input_dimensions=input_dimensions,
        input_format=formats.input_format,
        tile_shape=tile_shape,
    )

    relu_config = PackGolden.generate_relu_config(
        PackerReluType.NoRelu,
        relu_threshold=0.0,
        intermediate_format=formats.output_format,
    )

    test_config_kwargs = dict(
        test_name="sources/quasar/pack_quasar_test.cpp",
        formats=formats,
        templates=[
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            TEST_FACE_DIMS(tile_shape.face_r_dim),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt),
            RELU_CONFIG(relu_config),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            LOOP_FACTOR(1),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
            num_faces=num_faces,
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )
    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=[PerfRunType.L1_TO_L1],
        test_config_kwargs=test_config_kwargs,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    ), "Assert against golden failed"
