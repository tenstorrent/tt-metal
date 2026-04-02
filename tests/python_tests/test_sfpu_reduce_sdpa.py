# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from conftest import skip_for_coverage
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    MathOperation,
    ReducePool,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_OP,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test


# Has a compilation error on coverage, https://github.com/tenstorrent/tt-llk/issues/884
@skip_for_coverage
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b],  # Only Float16_b is supported for SDPA reduce
        same=True,
    ),
    dest_acc=[DestAccumulation.No],
    mathop=[MathOperation.ReduceColumn],
    reduce_pool=[ReducePool.Max],  # Only MAX is supported for SDPA reduce
    input_dimensions=[
        [128, 64],  # 4x2 subblock
    ],
)
def test_sfpu_reduce_sdpa(
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    input_dimensions,
    workers_tensix_coordinates,
):

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    src_A = tilize_block(src_A, input_dimensions).flatten()

    # GOLDEN GENERATION
    # *******************************************************

    # Undo tilization so src_A is standard [32, 32]
    src_A_untilized = untilize_block(src_A, formats.input_format, input_dimensions)

    # Take max along the height (dim=0) for each column
    col_max = torch.max(src_A_untilized, dim=0).values

    # Construct golden tensor: first row is column max, others are zero
    golden_tensor = torch.zeros_like(src_A_untilized)
    golden_tensor[0, :] = col_max

    # *******************************************************

    configuration = TestConfig(
        "sources/sfpu_reduce_sdpa_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop, pool_type=reduce_pool),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A)],
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
        unpack_to_dest=False,  # Must be False since math kernel does A2D copy
        dest_acc=dest_acc,
    )
    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, input_dimensions)

    # Check only the first row for correctness, not full tensors
    assert passed_test(golden_tensor[0], res_tensor[0], formats.output_format)
