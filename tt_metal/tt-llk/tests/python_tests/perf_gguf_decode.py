# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MATH_ISOLATE perf of the GGUF decoders, the DEST-fed matmul and the page copy (Blackhole)."""

import pytest
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.gguf_records import FORMATS
from helpers.llk_params import DestAccumulation, MathFidelity, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    GGUF_PERF,
    GGUF_PERF_STEPS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    TILE_COUNT,
)

pytestmark = blackhole_only

DECODE_STEPS = ("decode", "params", "body")
TILE_CNT = 8  # every tile index of a group of up to 8 K tiles


@pytest.mark.perf
@parametrize(
    step=list(GGUF_PERF_STEPS),
    fmt=lambda step: list(FORMATS) if step in DECODE_STEPS else [None],
    in0_rows=lambda step: [8, 16, 32] if step == "matmul" else [32],
    fidelity=lambda step: (
        [MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4]
        if step == "matmul"
        else [MathFidelity.LoFi]
    ),
    page_faces=lambda step: [1, 2, 4] if step == "copy" else [4],
    loop_factor=[16],
)
def test_perf_gguf_decode(
    perf_report, step, fmt, in0_rows, fidelity, page_faces, loop_factor
):
    """Cycles per tile: mean(MATH_ISOLATE) of the TILE_LOOP row in the .post.csv; steps as GGUF_STEP in
    sources/gguf_decode_mm_perf.cpp (decode / params / body per format, matmul per in0 rows and fidelity, copy of a 1,
    2 or 4 KB page)."""
    configuration = PerfConfig(
        "sources/gguf_decode_mm_perf.cpp",
        InputOutputFormat(DataFormat.UInt32, DataFormat.Float32),
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            GGUF_PERF(
                gguf_step=step,
                gguf_fmt=FORMATS[fmt].id if fmt else 0,
                gguf_in0_rows=in0_rows,
                gguf_page_faces=page_faces,
            ),
            MATH_FIDELITY(fidelity),
            LOOP_FACTOR(loop_factor),
            TILE_COUNT(TILE_CNT),
        ],
        variant_stimuli=StimuliConfig(
            None,
            DataFormat.UInt32,
            None,
            DataFormat.Float16_b,
            DataFormat.Float32,
            tile_count_A=TILE_CNT,
            tile_count_B=TILE_CNT,
            tile_count_res=1,
        ),
        unpack_to_dest=True,
        disable_format_inference=True,
        dest_acc=DestAccumulation.Yes,
    )
    configuration.run(perf_report)
