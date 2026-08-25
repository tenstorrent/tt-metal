# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import DestAccumulation, PerfRunType, format_dict
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import SOFTMAX_K, TILE_COUNT
from helpers.tilize_untilize import tilize_block
from test_sfpu_softmax_k import FACE_DIM, FORMATS, _build_input_tile


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    k=[FACE_DIM],
)
def test_perf_sfpu_softmax_k(perf_report, formats, dest_acc, k):
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")

    torch.manual_seed(0)
    torch_format = format_dict[formats.input_format]
    input_tile, _logits = _build_input_tile(k, torch_format)
    src_A = tilize_block(
        input_tile.flatten(), [TILE_DIM, TILE_DIM], stimuli_format=formats.input_format
    ).flatten()
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)

    configuration = PerfConfig(
        "sources/sfpu_softmax_k_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[SOFTMAX_K(softmax_k=k)],
        runtimes=[TILE_COUNT(1)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
    )
    configuration.run(perf_report)
