# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Minimal Quasar SFPU perf test measuring 100 NOPs."""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import PerfRunType
from helpers.perf import PerfConfig


@pytest.mark.perf
@pytest.mark.quasar
def test_perf_sfpu_nop_quasar(perf_report):
    formats = InputOutputFormat(
        input_format=DataFormat.Float16_b,
        output_format=DataFormat.Float16_b,
    )

    configuration = PerfConfig(
        "sources/quasar/sfpu_nop_perf.cpp",
        formats,
        run_types=[
            PerfRunType.MATH_ISOLATE,
        ],
    )

    configuration.run(perf_report)
