# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import InputOutputFormat
from helpers.llk_params import DestSync, PerfRunType, SdpaOp, format_dict
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.tilize_untilize import tilize_block
from test_sfpu_sdpa import TILE_DIMENSIONS, Precision, Variant, _stimulus, _templates


@pytest.mark.perf
@parametrize(
    dest_sync=[DestSync.Half, DestSync.Full],
    precision=list(Precision),
    op=[SdpaOp.ExpAccurate, SdpaOp.RecipIter, SdpaOp.Softplus],
)
def test_perf_sfpu_sdpa(perf_report, dest_sync, precision, op):
    variant = Variant(op=op, dest_sync=dest_sync)
    formats = InputOutputFormat(precision.data_format, precision.data_format)
    torch_format = format_dict[formats.input_format]
    src_A = _stimulus(variant).to(torch_format)
    src_B = torch.zeros_like(src_A)
    src_A_tilized = tilize_block(
        src_A, TILE_DIMENSIONS, stimuli_format=formats.input_format
    ).flatten()

    configuration = PerfConfig(
        "sources/sfpu_sdpa_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=_templates(variant),
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=precision.dest_acc,
        unpack_to_dest=precision.unpack_to_dest,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    configuration.run(perf_report)
