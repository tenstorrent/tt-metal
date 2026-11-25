# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TilizeGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    PerfRunType,
    format_dict,
    format_tile_sizes,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    INPUT_DIMENSIONS,
    LOOP_FACTOR,
    NUM_FACES,
    PERF_RUN_TYPE,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tile_constants import get_tile_params
from helpers.tilize_untilize import tilize_block
from helpers.utils import assert_exact_widening, fp32_bits_as_int32, passed_test


def generate_input_dimensions(max_size: int) -> list[tuple[int, int]]:
    """
    Generates a list of tuples representing width and height in tiles for input tensors,
    up to the specified maximum size in tiles.
    For tilize tensor width is important so all widths from 1 to max_size are generated.
    In the interest of reducing the number of test cases, instead of generating all possible heights
    critical subsets of valid heights are generated (three smallest, three largest and three middle heights).
    Parameters:
    max_size (int): Maximum number of tiles the resulting tensor can have.
    Returns:
    List[tuple[int, int]]: A list of tuples representing the width and height of the input tensor in tiles.
    """
    dimensions = []
    for width in range(1, max_size + 1):
        max_height = max_size // width
        heights = [
            1,
            2,
            3,
            (max_height // 2) - 1,
            max_height // 2,
            (max_height // 2) + 1,
            max_height - 2,
            max_height - 1,
            max_height,
        ]
        heights = [h for h in heights if h > 0 and h <= max_height]
        heights = list(set(heights))
        for height in heights:
            dimensions.append((height, width))
    return dimensions


@skip_for_blackhole
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float32, DataFormat.Float16_b, DataFormat.Bfp8_b]
    ),
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    dimensions=generate_input_dimensions(25),
)
def test_fast_tilize(
    formats,
    dest_acc,
    dimensions,
):
    if formats.input == DataFormat.Bfp8_b:
        pytest.skip("Bfp8_b input format is not supported for fast tilize")

    golden_tensor, res_tensor = _run_fast_tilize(formats, dest_acc, dimensions)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
    if formats.input == DataFormat.Float16_b and formats.output == DataFormat.Float32:
        assert_exact_widening(golden_tensor, res_tensor)


# Quick subset of the Float16_b -> Float32 cases of the test_fast_tilize sweep.
# The widening is exact, so compare bit for bit: the Float32 tolerance of
# passed_test hides garbage in the low 16 bits of each datum (the packer reading
# 16-bit Dest with a Float32 input format under fp32_dest_acc_en).
@skip_for_blackhole
@parametrize(
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    dimensions=[(1, 1), (1, 2), (2, 3), (1, 5), (3, 8)],
)
def test_fast_tilize_bf16_to_fp32_exact(
    dest_acc,
    dimensions,
):
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32)
    golden_tensor, res_tensor = _run_fast_tilize(formats, dest_acc, dimensions)

    assert_exact_widening(golden_tensor, res_tensor)


def _run_fast_tilize(formats, dest_acc, dimensions):
    input_height, input_width = dimensions

    input_dimensions = [input_height * 32, input_width * 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(TilizeGolden)
    golden_tensor = generate_golden(src_A, input_dimensions, formats.output)

    configuration = TestConfig(
        "sources/fast_tilize_test.cpp",
        formats,
        templates=[PERF_RUN_TYPE(PerfRunType.L1_TO_L1)],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_cnt_A),
            LOOP_FACTOR(1),
            NUM_FACES(4),
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
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output])

    return golden_tensor, res_tensor


# Fast tilize -> uninit -> a normal pack, in one kernel, under fp32_dest_acc_en.
# Phase 1 fast-tilizes Float16_b to Float32. Phase 2 unpacks Float32 tiles straight
# to Dest and packs them with no pack init or reconfig in between, so it uses
# exactly the packer state _llk_pack_fast_tilize_uninit_ restores (input format,
# Dstacc, strides, Dest read control, tile header). Its data fills all 32 bits of
# each datum, so a packer left at the Float16_b input that fast tilize switches to
# would lose the low 16 bits. Covers 32x32 tiles (4 faces) and 16x32 tiles
# (2 faces); every case fits in one Dest bank.
@skip_for_blackhole
@pytest.mark.parametrize(
    "tile_dimensions, width",
    [
        ([32, 32], 1),
        ([32, 32], 2),
        ([32, 32], 3),
        ([32, 32], 4),
        ([16, 32], 2),
        ([16, 32], 4),
    ],
)
def test_fast_tilize_uninit_then_normal_pack(tile_dimensions, width):
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32)
    phase2_format = DataFormat.Float32

    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim
    tiny = num_faces == 2
    input_dimensions = [tile_dimensions[0], width * tile_dimensions[1]]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=phase2_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )
    assert tile_cnt_A == width and tile_cnt_B == width

    fast_tilize_golden = (
        tilize_block(
            src_A,
            input_dimensions,
            formats.output,
            num_faces=num_faces,
            tile_dimensions=tile_dimensions,
        )
        .flatten()
        .to(format_dict[formats.output])
    )
    # phase 2 copies the Float32 tiles of buffer B as they are
    normal_pack_golden = src_B.flatten().to(format_dict[formats.output])
    # every datum has bits set in its low 16 bits, or the check could not tell a
    # Float16_b packer input from a Float32 one
    low_halves = fp32_bits_as_int32(normal_pack_golden) & 0xFFFF
    assert bool((low_halves != 0).all())

    stimuli_kwargs = {}
    if tiny:
        # one result slot holds two dense 16x32 tiles, as in the tiny-tile fast tilize test
        stimuli_kwargs = dict(
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
            operand_res_tile_size=format_tile_sizes[formats.output_format],
        )

    configuration = TestConfig(
        "sources/fast_tilize_uninit_pack_test.cpp",
        formats,
        templates=[],
        runtimes=[
            INPUT_DIMENSIONS(
                full_rt_dim=1, full_ct_dim=width, block_ct_dim=width, block_rt_dim=1
            ),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            phase2_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=2 * tile_cnt_A,
            **stimuli_kwargs,
        ),
        dest_acc=DestAccumulation.Yes,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    half = len(fast_tilize_golden)
    assert len(res_from_L1) == 2 * half
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output])

    assert_exact_widening(fast_tilize_golden, res_tensor[:half])
    # Float32 -> Float32 is exact too
    assert_exact_widening(normal_pack_golden, res_tensor[half:])
