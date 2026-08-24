# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import DestAccumulation, PerfRunType, VectorMode, format_dict
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    SAMPLING_LEGACY_COMPAT,
    SAMPLING_OP,
    SFPU_UNARY_SCALAR,
    VECTOR_MODE,
)
from helpers.tilize_untilize import tilize_block
from test_sfpu_sampling import (
    CLAMP_MAX,
    FORMATS,
    MUL_SCALAR,
    NUM_TILES,
    OUT_BACKGROUND,
    _bf16_row_values,
    _column_uniform_tile,
    _f32_bits,
)


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    op=["recip_scalar", "add", "le"],
)
def test_perf_sfpu_sampling(perf_report, formats, dest_acc, op):
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")

    torch.manual_seed(0)
    torch_format = format_dict[formats.input_format]
    in0_rows = _bf16_row_values()
    in1_rows = _bf16_row_values()
    in0_tile = _column_uniform_tile(in0_rows, torch_format)
    in1_tile = _column_uniform_tile(in1_rows, torch_format)
    background_tile = torch.full(
        (TILE_DIM, TILE_DIM), OUT_BACKGROUND, dtype=torch_format
    )
    src_A = torch.cat(
        [
            tilize_block(
                t.flatten(), [TILE_DIM, TILE_DIM], stimuli_format=formats.input_format
            ).flatten()
            for t in (in0_tile, in1_tile, background_tile)
        ]
    )
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)
    scalar_bits = _f32_bits(CLAMP_MAX if op == "clamp_max_scalar" else MUL_SCALAR)

    configuration = PerfConfig(
        "sources/sfpu_sampling_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            SAMPLING_OP(sampling_op=op),
            SAMPLING_LEGACY_COMPAT(legacy_compat=True),
            SFPU_UNARY_SCALAR(value_bits=scalar_bits),
            VECTOR_MODE(vector_mode=VectorMode.None_),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=NUM_TILES,
            tile_count_B=1,
            tile_count_res=NUM_TILES,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
    )
    configuration.run(perf_report)
