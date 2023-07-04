# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path
import pandas as pd

from tt_metal.tools.profiler.common import PROFILER_OUTPUT_DIR, PROFILER_SCRIPTS_ROOT


def post_process_ops_log(output_logs_subdir, columns):
    runDate = sorted(os.listdir(PROFILER_OUTPUT_DIR / output_logs_subdir))[-1]
    df = pd.read_csv(PROFILER_OUTPUT_DIR / output_logs_subdir / runDate / f"ops_perf_results_{runDate}.csv")
    results = {}
    for col in columns:
        df_filtered = df[df[col] != "-"]
        results[col] = df_filtered[col].astype(float).sum()
    return results


def run_device_profiler(command, output_logs_subdir):
    output_logs_dir = PROFILER_OUTPUT_DIR / output_logs_subdir
    profiler_cmd = f"python -m tracy -p -r -o {output_logs_dir} -t 5000 -m {command}"
    subprocess.run([profiler_cmd], shell=True, check=True)


def get_samples_per_s(time_ns, num_samples):
    ns_to_s = 1e-9
    return 1 / (time_ns * ns_to_s) * num_samples
