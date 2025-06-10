# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config


def test_ttnop_time():
    os.system(
        f"TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/tensix/test_gathering"
    )
    setup = device_post_proc_config.default_setup()
    setup.timerAnalysis = {
        "latency_trisc_0": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "TRISC_0", "zone_name": "TEST-FULL"},
            "end": {"risc": "TRISC_0", "zone_name": "TEST-FULL"},
        },
        "latency_trisc_1": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "TRISC_1", "zone_name": "TEST-FULL"},
            "end": {"risc": "TRISC_1", "zone_name": "TEST-FULL"},
        },
        "latency_trisc_2": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "TRISC_2", "zone_name": "TEST-FULL"},
            "end": {"risc": "TRISC_2", "zone_name": "TEST-FULL"},
        },
    }
    stats = import_log_run_stats(setup)
    analysis = stats["devices"][0]["cores"]["DEVICE"]["analysis"]
    for stat_name in setup.timerAnalysis:
        device_stats = analysis[stat_name]["stats"]
        print(stat_name, device_stats)
