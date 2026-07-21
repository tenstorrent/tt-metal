# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig

ITERATIONS = 500 # you can bump this if you wish

@pytest.mark.parametrize("worker_slot", range(15))
def test_exalens_repro_minimal(worker_slot):
    # input data prep
    formats = input_output_formats([DataFormat.Int32])[0]
    input_dimensions = [32, 96]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = TestConfig(
        "sources/risc_compute_test.cpp",
        formats,
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
    )

    configuration.run()

    for i in range(ITERATIONS):
        # we reset LAST_LOADED_ELFS so the harness reruns exalens' load_elf every iteration.
        TestConfig.LAST_LOADED_ELFS = Path()
        try:
            configuration.run()
        except Exception:
            print(
                f"worker_slot={worker_slot} failed (iteration {i})",
                flush=True,
            )
            raise
