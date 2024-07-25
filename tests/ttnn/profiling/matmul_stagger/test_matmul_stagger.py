# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import subprocess

import pandas as pd

from models.utility_functions import is_grayskull
from tests.ttnn.profiling.matmul_stagger.test_run_matmul_with_and_without_stagger import MATMUL_VARIANTS
from tt_metal.tools.profiler.common import PROFILER_OUTPUT_DIR


# for all matmul variants affected by stagger, profile op with stagger enabled and disabled,
# and make sure the execution time of the op is longer with stagger enabled
@pytest.mark.skipif(is_grayskull(), reason="Stagger only supported on wormhole")
def test_matmul_stagger():
    duration_column = "DEVICE FW DURATION [ns]"
    output_logs_dir = PROFILER_OUTPUT_DIR / "test_matmul_stagger"
    for matmul_variant in MATMUL_VARIANTS:
        command = f"pytest tests/ttnn/profiling/matmul_stagger/test_run_matmul_with_and_without_stagger.py -k {matmul_variant}"
        profiler_cmd = f"python -m tracy -p -r  -o {output_logs_dir} -t 120 -m {command}"
        subprocess.run([profiler_cmd], shell=True, check=True)

        runDate = sorted(os.listdir(output_logs_dir))[-1]
        df = pd.read_csv(output_logs_dir / runDate / f"ops_perf_results_{runDate}.csv")

        time_no_stagger = df.loc[0, duration_column]
        time_with_stagger = df.loc[1, duration_column]

        assert (
            time_no_stagger < time_with_stagger
        ), f"There should be a visible perf drop when running {matmul_variant} with stagger"
