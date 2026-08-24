# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestSync,
    PerfRunType,
    SdpaFwOp,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DEST_SYNC,
    SDPA_EXP_SCALE,
    SDPA_FW_OP,
)
from helpers.tilize_untilize import tilize_block
from test_sfpu_sdpa_fw import (
    EXP_SCALE_BF16_VALUES,
    TILE_DIMENSIONS,
    Precision,
    _stimulus,
)


@pytest.mark.perf
@parametrize(
    dest_sync=[DestSync.Half, DestSync.Full],
    precision=list(Precision),
    op=[SdpaFwOp.Recip, SdpaFwOp.Exp],
)
def test_perf_sfpu_sdpa_fw(perf_report, dest_sync, precision, op):
    exp_scale_bf16 = EXP_SCALE_BF16_VALUES[1]
    formats = InputOutputFormat(precision.data_format, precision.data_format)
    torch_format = format_dict[formats.input_format]
    src_A = _stimulus(op, exp_scale_bf16).to(torch_format)
    src_B = torch.zeros_like(src_A)
    src_A_tilized = tilize_block(
        src_A, TILE_DIMENSIONS, stimuli_format=formats.input_format
    ).flatten()

    configuration = PerfConfig(
        "sources/sfpu_sdpa_fw_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            SDPA_FW_OP(op),
            APPROX_MODE(ApproximationMode.No),
            DEST_SYNC(dest_sync),
            SDPA_EXP_SCALE(scale_bf16=exp_scale_bf16),
        ],
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
