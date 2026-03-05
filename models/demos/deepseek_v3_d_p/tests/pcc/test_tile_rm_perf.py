# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_model_device_perf_test


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            "pytest models/demos/deepseek_v3_d_p/tests/pcc/test_tile_rm.py::test_tilize[1chip-4Kx7K]",
            15_000_000,  # tilize 636us + untilize 14.35ms (round-trip verification)
            "deepseek_v3_d_p",
            "tilize_4k_7k",
            1,
            1,
            0.015,
            "4096x7168 bfloat16 tilize",
        ),
        (
            "pytest models/demos/deepseek_v3_d_p/tests/pcc/test_tile_rm.py::test_untilize[1chip-4Kx7K]",
            14_500_000,  # untilize ~14.35ms
            "deepseek_v3_d_p",
            "untilize_4k_7k",
            1,
            1,
            0.015,
            "4096x7168 bfloat16 untilize",
        ),
    ],
    ids=["tilize_4k_7k", "untilize_4k_7k"],
)
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_tile_rm(
    command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments
):
    run_model_device_perf_test(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
    )
