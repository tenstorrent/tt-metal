# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import os
from pathlib import Path
import csv
import ttnn
import numpy as np

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG, rm

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def get_op_duration(device_data, op_name):
    # TODO provide more bullet proof solution, currently assumes there is only 1 device and 1 core
    core_name = list(device_data["devices"][0]["cores"].keys())[0]

    # Get event timeseries for Unpacker and Packer cores (TRISC_0 and TRISC_2)
    trisc0_data = device_data["devices"][0]["cores"][core_name]["riscs"]["TRISC_0"]["timeseries"]
    trisc2_data = device_data["devices"][0]["cores"][core_name]["riscs"]["TRISC_2"]["timeseries"]

    # TODO provide safe guards, OP existis, there is same number of OP occurences etc... Add error handling
    # Get Unpacker op_name zone start time series
    trisc0_op_start = [i for i in trisc0_data if (i[0]["zone_name"] == op_name and i[0]["type"] == "ZONE_START")]

    # Get Packer op_name zone end time series
    trisc2_op_end = [i for i in trisc2_data if (i[0]["zone_name"] == op_name and i[0]["type"] == "ZONE_END")]

    # Get op duration for each OP occurence in timeseries, assuming everything is in place
    op_duration = []
    for i in range(len(trisc0_op_start)):
        op_duration.append(trisc2_op_end[i][1] - trisc0_op_start[i][1])

    return op_duration


def get_profiler_data(perf_scope, op_name, op_duration=False):
    # Import profiler log file and run perf related statistic calculation
    setup = device_post_proc_config.perf_analysis()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    data = []

    # Add UNTILIZE-BLOCK/OP zone average duration per trisc core
    data.append(
        deviceData["devices"][0]["cores"]["DEVICE"]["analysis"][f"trisc0_{op_name}_{perf_scope}_duration"]["stats"][
            "Average"
        ]
    )
    data.append(
        deviceData["devices"][0]["cores"]["DEVICE"]["analysis"][f"trisc1_{op_name}_{perf_scope}_duration"]["stats"][
            "Average"
        ]
    )
    data.append(
        deviceData["devices"][0]["cores"]["DEVICE"]["analysis"][f"trisc2_{op_name}_{perf_scope}_duration"]["stats"][
            "Average"
        ]
    )

    op_name_profiler = op_name + "-op"
    op_name_profiler = op_name_profiler.upper()
    if op_duration:
        data.append(np.mean(get_op_duration(deviceData, op_name_profiler)))

    return data


def test_untilize(device):
    # Enable llk perf testing
    os.environ["TT_ENABLE_LLK_PERF"] = "1"

    perf_scope = "op"  # can be on 'block' level
    if perf_scope == "block":
        os.environ["TT_LLK_PERF_BLOCK"] = "1"

    for perf in ["op", "op_no_dm", "unpack", "pack"]:
        # Set log csv file name, file will be used to store perf data
        ENVS = dict(os.environ)
        TT_METAL_HOME = Path(ENVS["TT_METAL_HOME"])
        log_file = TT_METAL_HOME / "generated" / f"untilize_llk_perf_{perf}_cpp.csv"

        # Set env variable used to select perf measurement thread target(unpack, math, pack)
        if perf in ["unpack", "pack", "math"]:
            os.environ[f"TT_LLK_PERF_{perf.upper()}"] = "1"

        # Set env variable to disable DM
        if perf != "op":
            os.environ[f"TT_LLK_PERF_NO_DM"] = "1"

        with open(log_file, mode="w", newline="") as file:
            if perf in ["op", "op_no_dm"] and perf_scope == "op":
                get_op_duration = True
            else:
                get_op_duration = False
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
            if get_op_duration:
                csv_header.append("op_duration_cycles")
                csv_header.append("op_duration_cycles_per_tile")
            writer.writerow(csv_header)

            # Run tilize test for different rt and ct dims
            for rt_dim in range(1, 19, 1):
                for ct_dim in range(1, 19, 1):
                    os.environ["RT_DIM"] = str(rt_dim)
                    os.environ["CT_DIM"] = str(ct_dim)

                    # Clean profiler log file
                    # ttnn.DumpDeviceProfiler(device)
                    rm(profiler_log_path)

                    # Put os system call to execute cpp test
                    if ct_dim < 9:
                        # Run pack untilize for ct_dim up to 8
                        os.system("./build/test/tt_metal/unit_tests_llk --gtest_filter=*TensixComputePackUntilizePerf")
                    else:
                        # Run unpack untilize for ct_dim greater then 8
                        os.system(
                            "./build/test/tt_metal/unit_tests_llk --gtest_filter=*TensixComputeUnpackUntilizePerf"
                        )

                    # Process profiler log file and extract tilize data
                    # ttnn.DumpDeviceProfiler(device)
                    rt_div = rt_dim if perf_scope == "op" else 1
                    profiler_data = get_profiler_data(perf_scope, "untilize", get_op_duration)
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
                    if get_op_duration:
                        csv_data.append(f"{profiler_data[3]:.2f}")
                        csv_data.append(f"{profiler_data[3] / ct_dim / rt_div:.2f}")
                    writer.writerow(csv_data)

        # Unset env variable used to select perf measurement thread target(unpack, math, pack)
        if perf in ["unpack", "pack", "math"]:
            os.environ.pop(f"TT_LLK_PERF_{perf.upper()}", None)

        # Unset env variable to disable DM
        if perf != "op":
            os.environ.pop(f"TT_LLK_PERF_NO_DM")
