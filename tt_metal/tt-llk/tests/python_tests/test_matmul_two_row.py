# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# The two-row scheme (reuse_dir 4, Blackhole) forced on every block shape up to the DEST capacity, odd row counts included (an odd last
# row of in0 is held alone), at both DEST syncs and k steps of 1 and 3. Run with --speed-of-light for the compile-time counts.

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import generate_tile_dims, matmul_pack_form
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATMUL_FORM_KEYS,
    MATMUL_PACK_BLOCK,
    MATMUL_REUSE_DIR,
    NUM_FACES,
    PERF_RUN_TYPE,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# Float32 operands take a 32-bit DEST: 4 tiles at half sync, 8 at full
TWO_ROW_BLOCKS = [
    (sync, rt, ct, kt)
    for sync, cap in ((DestSync.Half, 4), (DestSync.Full, 8))
    for rt in (1, 2, 3, 4, 5)
    for ct in (1, 2, 3, 5)
    for kt in (1, 3)
    if rt * ct <= cap
]


@skip_for_wormhole
@pytest.mark.nightly
@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
    ],
    block=TWO_ROW_BLOCKS,
)
def test_matmul_two_row(formats, block):
    dest_sync, rt, ct, kt = block
    dest_acc = DestAccumulation.Yes
    math_fidelity = MathFidelity.LoFi
    input_A_dimensions = [rt * 32, kt * 32]
    input_B_dimensions = [kt * 32, ct * 32]

    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=spec,
        spec_B=spec,
    )
    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))
    golden_tensor = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )
    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=formats.input_format
    )

    configuration = TestConfig(
        "sources/matmul_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(),
            MATMUL_REUSE_DIR(4),
            # a forced direction runs main's tile form (the unknown-k key)
            MATMUL_FORM_KEYS(0, formats.output_format, True),
            MATMUL_PACK_BLOCK(
                matmul_pack_form(
                    formats,
                    dest_sync,
                    math_fidelity,
                    matmul_dims.rt_dim,
                    matmul_dims.ct_dim,
                    matmul_dims.kt_dim,
                )
            ),
        ],
        runtimes=[
            NUM_FACES(),
            TILE_COUNT(matmul_dims.output_tile_cnt),
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
            LOOP_FACTOR(1),
            UNPACK_TRANS_FACES(Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=matmul_dims.output_tile_cnt,
        ),
        dest_acc=dest_acc,
    )
    res_tensor = torch.tensor(
        configuration.run().result,
        dtype=format_dict[formats.output_format],
    )
    assert len(res_tensor) == len(golden_tensor)
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
