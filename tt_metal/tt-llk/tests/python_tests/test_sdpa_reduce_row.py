# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ADVANCE TEST: covers demo-fork experimental LLK sdpa_reduce_row (tt-metal#47554 / tt-blaze#1971), pending promotion.
# Include path (shadow -I) repoint on promotion. Primitive differs from tt-blaze only in FPU<->SFPU signalling cadence
# (orthogonal to this numerical golden).
#
# sdpa_reduce_row documented contract (ckernel_sfpu_sdpa_reduce_row.h primitive + compute_kernel_api/sdpa.h caller in
# models/demos/deepseek_v3_b1/kernel_includes/):
#   - A row-wise MAX reduce, run on the SFPU, over `block_width` consecutive 8x32 tiles held in DEST. Each 8x32 tile is
#     a full 16x16 face's worth of lanes (8 rows x 32 cols). For each of the 8 rows the op maxes across all
#     block_width*32 columns and writes the row-max into column 0 of the destination tile.
#   - Float16_b only (the primitive static_asserts format == Float16_b).
#   - Only ReducePool.Max is exercised here (the reduce-max / running-max step of flash attention). The sum variant
#     shares the same replay/epilogue structure.
#   - skip_signalling is pinned true in the C++ so the isolated PACK-thread kernel does not deadlock on the FPU<->SFPU
#     semaphore handshake; that handshake (and its DEMO-vs-tt-blaze cadence delta) is orthogonal to this numerical
#     golden.
#
# This advance test exercises the MAX instantiation, block_width == 1, on a single 8x32 tile carried inside a 32x32
# DEST tile.
#
# Blackhole-only. Deliverable here is compile-green (compile-producer). On-device numerical verification is pending
# Blackhole hardware/CI; this host is Wormhole.

import torch
from conftest import skip_for_coverage
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    ReduceBlockMaxRowGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathOperation,
    ReducePool,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# A single 32x32 tile carrying one 8x32 SFPU reduce span. block_width == 1.
TILE_DIM = 32


# Has a compilation error on coverage (shared with the analog sfpu_reduce_sdpa path,
# https://github.com/tenstorrent/tt-llk/issues/884).
@skip_for_coverage
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b],  # Only Float16_b is supported for SDPA reduce
        same=True,
    ),
    dest_acc=[DestAccumulation.No],
    mathop=[MathOperation.ReduceRow],
    reduce_pool=[ReducePool.Max],  # Only MAX is exercised for the SDPA row reduce
    input_dimensions=[
        [TILE_DIM, TILE_DIM],  # single 32x32 tile, one 8x32 reduce span
    ],
)
def test_sdpa_reduce_row(
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    input_dimensions,
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
    # Undo tilization so src_A is standard [32, 32], then take the per-row max across the full tile width into column
    # 0 (ReduceBlockMaxRowGolden), the SDPA row-reduce contract. PROVISIONAL until Blackhole on-device confirms the
    # exact DEST/lane addressing of the 8x32 reduce span; the golden validates the shared (non-signalling) numeric
    # path only.
    src_A_untilized = untilize_block(src_A, formats.input_format, input_dimensions)

    generate_golden = get_golden_generator(ReduceBlockMaxRowGolden)
    golden_tensor = generate_golden(
        src_A_untilized.flatten(),
        1,  # block_ct_dim == 1 (single 32-col span)
        formats.output_format,
        input_dimensions,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )

    # *******************************************************

    configuration = TestConfig(
        "sources/sdpa_reduce_row_test.cpp",
        formats,
        templates=[
            generate_input_dim(
                input_dimensions, input_dimensions, block_ct_dim=1, block_rt_dim=1
            ),
            MATH_OP(mathop=mathop, pool_type=reduce_pool),
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

    golden_rowmajor = golden_tensor.reshape(input_dimensions)

    # Only column [0] holds the row-max (the SFPSHFT2 within-row epilogue leaves other lanes unspecified), so validate
    # column [0]. Only the first 8 rows carry the 8x32 reduce span.
    assert passed_test(
        golden_rowmajor[:8, 0], res_tensor[:8, 0], formats.output_format
    ), "Assert against golden failed"
