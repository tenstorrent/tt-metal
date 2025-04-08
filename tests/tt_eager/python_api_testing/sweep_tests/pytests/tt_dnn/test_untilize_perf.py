# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial
from loguru import logger


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import ttnn
import os
from pathlib import Path
import csv

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG, rm

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def get_profiler_data(perf_scope):
    # Import profiler log file and run perf related statistic calculation
    setup = device_post_proc_config.perf_analysis()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    data = []

    # Add UNTILIZE-BLOCK/OP zone average duration per trisc core
    data.append(
        deviceData["devices"][0]["cores"]["DEVICE"]["analysis"][f"trisc0_untilize_{perf_scope}_duration"]["stats"][
            "Average"
        ]
    )
    data.append(
        deviceData["devices"][0]["cores"]["DEVICE"]["analysis"][f"trisc1_untilize_{perf_scope}_duration"]["stats"][
            "Average"
        ]
    )
    data.append(
        deviceData["devices"][0]["cores"]["DEVICE"]["analysis"][f"trisc2_untilize_{perf_scope}_duration"]["stats"][
            "Average"
        ]
    )

    return data


shapes = [[[1, 1, 10 * 32, 5 * 32]]]


@pytest.mark.parametrize(
    "input_shapes",
    # (([[1, 1, 10*32, 5*32]], [[3, 1, 320, 384]], [[1, 1, 128, 1856]])),
    shapes,
)
@pytest.mark.parametrize(
    "untilize_args",
    (
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.TILE_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        },
    ),
)
def test_untilize_test(input_shapes, untilize_args, device, function_level_defaults):
    datagen_func = [generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_arange), torch.bfloat16, True)]

    # Compare only shape in case of performance testing,
    comparison_func = comparison_funcs.comp_shape

    # Enable llk perf testing
    os.environ["TT_ENABLE_LLK_PERF"] = "1"

    # Select OP or BLOCK level for perf measurement
    perf_scope = "op"
    if perf_scope == "block":
        os.environ["TT_LLK_PERF_BLOCK"] = "1"

    for perf in ["op", "op_no_dm", "unpack", "pack"]:
        # Set log csv file name, file will be used to store perf data
        ENVS = dict(os.environ)
        TT_METAL_HOME = Path(ENVS["TT_METAL_HOME"])
        log_file = TT_METAL_HOME / "generated" / f"untilize_llk_perf_{perf}.csv"

        # Set env variable used to select perf measurement thread target(unpack, math, pack)
        if perf in ["unpack", "pack", "math"]:
            os.environ[f"TT_LLK_PERF_{perf.upper()}"] = "1"

        # Set env variable to disable DM
        if perf != "op":
            os.environ[f"TT_LLK_PERF_NO_DM"] = "1"

        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            csv_header = [
                "rt_dim",
                "ct_dim",
                f"unpack_{perf_scope}_cycles",
                f"math_{perf_scope}_cycles",
                f"pack_{perf_scope}_cycles",
                "unpack_cycles_per_tile",
                "math_cycles_per_tile",
                "pack_cycles_per_tile",
            ]
            writer.writerow(csv_header)

            # Run tilize test for different rt and ct dims
            for rt_dim in range(1, 21, 1):
                for ct_dim in range(1, 21, 1):
                    input_shapes = [[1, 1, rt_dim * 32, ct_dim * 32]]

                    # Clean profiler log file
                    ttnn.DumpDeviceProfiler(device)
                    rm(profiler_log_path)

                    for i in range(10):
                        run_single_pytorch_test(
                            "untilize",
                            input_shapes,
                            datagen_func,
                            comparison_func,
                            device,
                            untilize_args,
                        )

                    # Process profiler log file and extract tilize data
                    ttnn.DumpDeviceProfiler(device)
                    rt_div = rt_dim if perf_scope == "op" else 1
                    profiler_data = get_profiler_data(perf_scope)
                    csv_data = [
                        rt_dim,
                        ct_dim,
                        f"{profiler_data[0]:.2f}",
                        f"{profiler_data[1]:.2f}",
                        f"{profiler_data[2]:.2f}",
                        f"{profiler_data[0] / ct_dim / rt_div:.2f}",
                        f"{profiler_data[1] / ct_dim / rt_div:.2f}",
                        f"{profiler_data[2] / ct_dim / rt_div:.2f}",
                    ]
                    writer.writerow(csv_data)
        # Unset env variable used to select perf measurement thread target(unpack, math, pack)
        if perf in ["unpack", "pack", "math"]:
            os.environ.pop(f"TT_LLK_PERF_{perf.upper()}", None)

        # Unset env variable to disable DM
        if perf != "op":
            os.environ.pop(f"TT_LLK_PERF_NO_DM")
