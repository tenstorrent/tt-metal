# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
from conftest import skip_for_coverage
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathOperation,
    PerfRunType,
    ReducePool,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test


@dataclass
class ReduceImplTemplate(TemplateParameter):
    # Field name = the CSV column header this param would emit; must be
    # globally unique across parameter classes (FM-F1 contract).
    reduce_impl: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t REDUCE_IMPL = {self.reduce_impl}u;"


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
        [512, 64],  # four independent 4x2 subblocks
    ],
    reduce_impl=[0, 1],  # handwritten replay, generated SFPI math
)
def test_sfpu_reduce_sdpa(
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    input_dimensions,
    reduce_impl,
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

    # Each destination block is an independent 4x2 SDPA reduction subblock.
    golden_tensor = torch.zeros_like(src_A_untilized)
    subblock_rows = 4 * 32
    for row in range(0, input_dimensions[0], subblock_rows):
        golden_tensor[row, :] = torch.max(
            src_A_untilized[row : row + subblock_rows, :], dim=0
        ).values

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )

    # *******************************************************

    configuration = TestConfig(
        "sources/sfpu_reduce_sdpa_test.cpp",
        formats,
        templates=[
            generate_input_dim(
                input_dimensions, input_dimensions, block_ct_dim=2, block_rt_dim=4
            ),
            MATH_OP(mathop=mathop, pool_type=reduce_pool),
            ReduceImplTemplate(reduce_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
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
        unpack_to_dest=False,  # Must be False since math kernel does A2D copy
        dest_acc=dest_acc,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, input_dimensions)

    for row in range(0, input_dimensions[0], subblock_rows):
        assert passed_test(golden_tensor[row], res_tensor[row], formats.output_format)


@pytest.mark.parametrize(
    "reduce_impl,label", [(0, "handwritten_replay"), (1, "generated_sfpi")]
)
def test_sfpu_reduce_sdpa_device_profile(perf_report, reduce_impl, label):
    """Measure the same Reduce-SDPA body with device profiler timestamps."""
    input_dimensions = [512, 64]
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    src_A = tilize_block(src_A, input_dimensions).flatten()
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, DestAccumulation.No, formats, input_dimensions, TILE_DIMENSIONS
    )
    configuration = PerfConfig(
        "sources/sfpu_reduce_sdpa_test.cpp",
        formats,
        # The custom Reduce-SDPA LLK deliberately executes the SFPU body on
        # TRISC2, so its scoped marker belongs to the PACK_ISOLATE report.
        run_types=[PerfRunType.PACK_ISOLATE],
        templates=[
            generate_input_dim(
                input_dimensions, input_dimensions, block_ct_dim=2, block_rt_dim=4
            ),
            MATH_OP(mathop=MathOperation.ReduceColumn, pool_type=ReducePool.Max),
            ReduceImplTemplate(reduce_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
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
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report, run_count=1)
    rows = perf_report.frame()
    rows = rows[rows["marker"] == "REDUCE_SDPA_BODY"]
    assert len(rows) >= 1, rows.to_string(index=False)
    cycles = float(rows["mean(PACK_ISOLATE)"].sum())
    assert cycles > 0
    print(f"REDUCE_SDPA_DEVICE_PROFILE impl={label} body_cycles={cycles:.2f}")
