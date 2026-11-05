# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from conftest import skip_for_coverage
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    DestAccumulation,
    MathOperation,
    PerfRunType,
    ReducePool,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    MATH_OP,
    REDUCE_POOL_TYPE,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# The SDPA reduce supports only Float16_b and MAX column reduction.
FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
DEST_ACC = DestAccumulation.No
MATHOP = MathOperation.ReduceColumn
REDUCE_POOL = ReducePool.Max
INPUT_DIMENSIONS = [512, 64]  # four independent 4x2 subblocks
BLOCK_CT_DIM = 2
BLOCK_RT_DIM = 4

# all_negative catches a max seed that is not really -inf.
STIMULI = {
    "default": None,
    "all_negative": StimuliSpec.uniform(low=-2.0, high=-0.5),
}


# Has a compilation error on coverage, https://github.com/tenstorrent/tt-llk/issues/884
@skip_for_coverage
# mathop stays a parameter so `--op ReduceColumn` still selects these cases.
@parametrize(stimuli=list(STIMULI), mathop=[MATHOP])
def test_sfpu_reduce_sdpa(stimuli, mathop):
    _run_sfpu_reduce_sdpa(
        FORMATS,
        DEST_ACC,
        mathop,
        REDUCE_POOL,
        INPUT_DIMENSIONS,
        stimuli=stimuli,
    )


def _run_sfpu_reduce_sdpa(
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    input_dimensions,
    *,
    stimuli=None,
    loop_factor=1,
    is_perf=False,
    perf_report=None,
    run_types=None,
):
    if run_types is None:
        run_types = [PerfRunType.PACK_ISOLATE] if is_perf else [PerfRunType.L1_TO_L1]

    test_config_kwargs = {
        "test_name": "sources/sfpu_reduce_sdpa_test.cpp",
        "formats": formats,
        "unpack_to_dest": False,  # Math performs the A2D copy.
        "dest_acc": dest_acc,
    }

    if is_perf:
        tile_count = (input_dimensions[0] // TILE_DIMENSIONS[0]) * (
            input_dimensions[1] // TILE_DIMENSIONS[1]
        )
        # Keep this name literal so schema v4 can derive the perf compile parameters.
        templates = [
            MATH_OP(mathop=mathop),
            REDUCE_POOL_TYPE(reduce_pool),
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                block_ct_dim=BLOCK_CT_DIM,
                block_rt_dim=BLOCK_RT_DIM,
            ),
        ]
        test_config_kwargs.update(
            {
                "templates": templates,
                "runtimes": [
                    TILE_COUNT(tile_count),
                    LOOP_FACTOR(loop_factor),
                ],
                "variant_stimuli": StimuliConfig(
                    None,
                    formats.input_format,
                    None,
                    formats.input_format,
                    formats.output_format,
                    tile_count_A=tile_count,
                    tile_count_B=tile_count,
                    tile_count_res=tile_count,
                ),
            }
        )
        configuration = create_test_or_perf_config(
            is_perf=True,
            run_types=run_types,
            test_config_kwargs=test_config_kwargs,
        )
        configuration.run(perf_report)
        return

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=STIMULI[stimuli],
    )

    src_A = tilize_block(src_A, input_dimensions).flatten()

    # GOLDEN GENERATION
    # *******************************************************

    # Undo tilization so src_A is standard [32, 32]
    src_A_untilized = untilize_block(src_A, formats.input_format, input_dimensions)

    # Each destination block is an independent 4x2 SDPA reduction subblock.
    golden_tensor = torch.zeros_like(src_A_untilized)
    subblock_rows = BLOCK_RT_DIM * TILE_DIMENSIONS[0]
    for row in range(0, input_dimensions[0], subblock_rows):
        golden_tensor[row, :] = torch.max(
            src_A_untilized[row : row + subblock_rows, :], dim=0
        ).values

    # *******************************************************

    functional_compile_params = [
        generate_input_dim(
            input_dimensions,
            input_dimensions,
            block_ct_dim=BLOCK_CT_DIM,
            block_rt_dim=BLOCK_RT_DIM,
        ),
        MATH_OP(mathop=mathop, pool_type=reduce_pool),
    ]
    functional_runtime_params = [
        TILE_COUNT(tile_cnt_A),
        LOOP_FACTOR(loop_factor),
    ]
    test_config_kwargs.update(
        {
            "templates": functional_compile_params,
            "runtimes": functional_runtime_params,
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_cnt_A,
                tile_count_B=tile_cnt_B,
                tile_count_res=tile_cnt_A,
            ),
        }
    )
    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, input_dimensions)

    for row in range(0, input_dimensions[0], subblock_rows):
        assert passed_test(golden_tensor[row], res_tensor[row], formats.output_format)
