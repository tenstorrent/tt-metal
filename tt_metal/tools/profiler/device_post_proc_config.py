# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_metal.tools.profiler.merge_meta_class import MergeMetaclass
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG


class default_setup(metaclass=MergeMetaclass):
    riscs = [
        "BRISC",
        "NCRISC",
        "TRISC_0",
        "TRISC_1",
        "TRISC_2",
        "ERISC",
    ]

    riscTypes = [
        "BRISC",
        "NCRISC",
        "TRISC",
        "ERISC",
    ]

    timerAnalysis = {
        "FW_START->FW_END": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "ANY", "zoneName": [f"{risc}-FW" for risc in riscTypes]},
            "end": {"core": "ANY", "risc": "ANY", "zoneName": [f"{risc}-FW" for risc in riscTypes]},
        },
        "KERNEL_START->KERNEL_END": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "ANY", "zoneName": [f"{risc}-KERNEL" for risc in riscTypes]},
            "end": {"core": "ANY", "risc": "ANY", "zoneName": [f"{risc}-KERNEL" for risc in riscTypes]},
        },
        "BR_KERNEL_START->BR_KERNEL_END": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "BRISC", "zoneName": "BRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "BRISC", "zoneName": "BRISC-KERNEL"},
        },
        "NC_KERNEL_START->NC_KERNEL_END": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "NCRISC", "zoneName": "NCRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "NCRISC", "zoneName": "NCRISC-KERNEL"},
        },
        "T0_KERNEL_START->T0_KERNEL_END": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "TRISC_0", "zoneName": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_0", "zoneName": "TRISC-KERNEL"},
        },
        "T1_KERNEL_START->T1_KERNEL_END": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "TRISC_1", "zoneName": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_1", "zoneName": "TRISC-KERNEL"},
        },
        "T2_KERNEL_START->T2_KERNEL_END": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "TRISC_2", "zoneName": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_2", "zoneName": "TRISC-KERNEL"},
        },
        "ER_KERNEL_START->ER_KERNEL_END": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "ERISC", "zoneName": "ERISC-KERNEL"},
            "end": {"core": "ANY", "risc": "ERISC", "zoneName": "ERISC-KERNEL"},
        },
    }

    riscsData = {
        "BRISC": {"color": "light:g"},
        "NCRISC": {"color": "light:r"},
        "TRISC_0": {"color": "light:gray"},
        "TRISC_1": {"color": "light:gray"},
        "TRISC_2": {"color": "light:gray"},
        "TENSIX": {"color": "light:b"},
    }

    displayStats = ["Count", "Average", "Max", "Median", "Min", "Sum", "Range"]

    cycleRange = None
    # Example
    # cycleRange = (34.676e9, 60e9)

    intrestingCores = None
    # Example
    # intrestingCores = [(0, 0), (0, 9), (6, 9)]

    # ignoreMarkers = None
    # Example
    ignoreMarkers = [65535]

    outputFolder = f"output/device"
    deviceInputLog = f"{PROFILER_LOGS_DIR}/{PROFILER_DEVICE_SIDE_LOG}"
    deviceRearranged = "device_rearranged_timestamps.csv"
    deviceAnalysisData = "device_analysis_data.json"
    deviceChromeTracing = "device_chrome_tracing.json"
    devicePerfHTML = "timeline.html"
    deviceStatsTXT = "device_stats.txt"
    deviceTarball = "device_perf_results.tgz"


class test_multi_op(default_setup):
    timerAnalysis = {
        "BRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "BRISC", "zoneName": "BRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "BRISC", "zoneName": "BRISC-KERNEL"},
        },
    }


class test_custom_cycle_count(default_setup):
    timerAnalysis = {
        "BRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "BRISC", "zoneName": "BRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "BRISC", "zoneName": "BRISC-KERNEL"},
        },
        "NCRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "NCRISC", "zoneName": "NCRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "NCRISC", "zoneName": "NCRISC-KERNEL"},
        },
        "TRISC_0 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_0", "zoneName": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_0", "zoneName": "TRISC-KERNEL"},
        },
        "TRISC_1 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_1", "zoneName": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_1", "zoneName": "TRISC-KERNEL"},
        },
        "TRISC_2 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_2", "zoneName": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_2", "zoneName": "TRISC-KERNEL"},
        },
    }


class test_full_buffer(default_setup):
    timerAnalysis = {
        "Marker Repeat": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ANY", "zoneName": "TEST-FULL"},
            "end": {"risc": "ANY", "zoneName": "TEST-FULL"},
        }
    }


class test_noc(default_setup):
    timerAnalysis = {
        "NoC For Loop": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 5},
            "end": {"risc": "NCRISC", "timerID": 6},
        }
    }
