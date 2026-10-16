# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""add_one (x + 1) on Quasar, KT demo: a hand-written pytest, no fuser.

Two stacked tiles through the A2D datacopy into Dest, then the SFPU, then the packer."""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TilizeGolden,
    UnarySFPUGolden,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    DestSync,
    MathOperation,
    PerfRunType,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli, resolve_intervals
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    PERF_RUN_TYPE,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import passed_test

INPUT_DIMENSIONS = [64, 32]

# Random ranges, the points -2..2 (as zero-width intervals) and a large-magnitude band on both signs.
ADD_ONE_INTERVALS = resolve_intervals(
    [
        (-10100.0, -10000.0),
        (-10.0, -2.01),
        (-1.99, -1.01),
        (-0.99, -0.01),
        (0.01, 0.99),
        (1.01, 1.99),
        (2.01, 10.0),
        (10000.0, 10100.0),
        (-2.0, -2.0),
        (-1.0, -1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
    ]
)

FP16_FORMATS = [DataFormat.Float16_b, DataFormat.Float16]
ALL_FORMATS = [DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32]


def _add_one_formats_dest_acc():
    """Valid (formats, dest_acc) pairs. 16-bit Dest takes no Float32 and cannot pack
    Float16_b to Float16, so it outputs Float16_b; 32-bit Dest takes any in/out pair."""
    combos = [
        (InputOutputFormat(fmt, DataFormat.Float16_b), DestAccumulation.No)
        for fmt in FP16_FORMATS
    ]
    combos += [
        (InputOutputFormat(in_fmt, out_fmt), DestAccumulation.Yes)
        for in_fmt in ALL_FORMATS
        for out_fmt in ALL_FORMATS
    ]
    return combos


@pytest.mark.quasar
@parametrize(formats_dest_acc=_add_one_formats_dest_acc())
def test_sfpu_add_one_quasar(formats_dest_acc):
    # parametrize() wraps a tuple that holds an InputOutputFormat in a 1-tuple.
    formats, dest_acc = formats_dest_acc[0]
    torch.manual_seed(0)

    spec = StimuliSpec.uniform(intervals=ADD_ONE_INTERVALS)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=INPUT_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=INPUT_DIMENSIONS,
        spec_A=spec,
        spec_B=spec,
    )

    golden = get_golden_generator(UnarySFPUGolden)(
        MathOperation.AddOne,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        INPUT_DIMENSIONS,
    )

    # The device reads tiles from L1, so tilize the row-major stimuli on the way in.
    device_src_A = get_golden_generator(TilizeGolden)(
        src_A, INPUT_DIMENSIONS, formats.input_format
    )

    configuration = TestConfig(
        "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.AddOne),
            APPROX_MODE(ApproximationMode.No),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpA),
            DEST_SYNC(DestSync.Half),
            # The shared C++ source reads both, even for an op that is not a typecast or a perf run.
            TYPECAST_FORMATS(),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(MAX_NUM_FACES),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            LOOP_FACTOR(1),
        ],
        variant_stimuli=StimuliConfig(
            device_src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=MAX_NUM_FACES,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden), "Result and golden lengths differ"

    # ... and untilize the packed result back to row-major before comparing.
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = get_golden_generator(UntilizeGolden)(
        res_tensor, formats.output_format, INPUT_DIMENSIONS
    )

    assert passed_test(
        golden.flatten(), res_tensor.flatten(), formats.output_format
    ), "Assert against golden failed"
