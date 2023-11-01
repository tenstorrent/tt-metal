# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import shutil
from pathlib import Path
import pandas as pd

root_dir = Path(os.environ["TT_METAL_HOME"])


def rm(path):
    if not os.path.exists(path):
        return
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    else:
        shutil.rmtree(path)


def clear_logs(output_logs_subdir):
    rm(root_dir / "tt_metal/tools/profiler/logs/")
    rm(root_dir / "tt_metal/tools/profiler/output/")
    rm(root_dir / ".profiler/" / output_logs_subdir)


def post_process_ops_log(output_logs_subdir, columns):
    runDate = sorted(os.listdir(root_dir / ".profiler" / output_logs_subdir / "ops_device/"))[-1]
    df = pd.read_csv(
        root_dir / ".profiler" / output_logs_subdir / "ops_device" / runDate / f"ops_perf_results_{runDate}.csv"
    )
    results = {}
    for col in columns:
        results[col] = df[col].sum()
    return results


def run_device_profiler(command, output_logs_subdir):
    output_logs_dir = root_dir / ".profiler" / output_logs_subdir
    profiler_cmd = root_dir / f'tt_metal/tools/profiler/profile_this.py -d -o {output_logs_dir} -c "{command}"'
    subprocess.run([profiler_cmd], shell=True, check=True)


def get_samples_per_s(time_ns, num_samples):
    ns_to_s = 1e-9
    return 1 / (time_ns * ns_to_s) * num_samples
