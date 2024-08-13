# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config


def fits_in_l1(num_channels, sample_size):
    return num_channels * sample_size <= 150 * 1024


@pytest.mark.skip("skip test due to CI error shows test not compelte")
@pytest.mark.parametrize("sample_counts", [(1024,)])  # , 8, 16, 64, 256],
@pytest.mark.parametrize(
    "sample_sizes",
    [(4096,)],
)  # , 1024, 2048, 4096],
@pytest.mark.parametrize("channel_counts", [(4,)])
def test_bidirectional_erisc_bandwidth(sample_counts, sample_sizes, channel_counts):
    test_string_name = f"test_ethernet_send_data_microbenchmark - \
            sample_counts: {sample_counts}, \
                sample_sizes: {sample_sizes}, \
                    channel_counts: {channel_counts}"
    print(f"{test_string_name}")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    if not fits_in_l1(channel_counts[0], sample_sizes[0]):
        pytest.skip("Data does not fit in L1. Skipping")

    sample_counts_str = " ".join([str(s) for s in sample_counts])
    sample_sizes_str = " ".join([str(s) for s in sample_sizes])
    channel_counts_str = " ".join([str(s) for s in channel_counts])

    cmd = f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_bidirectional_bandwidth_no_edm \
                {len(sample_counts)} {sample_counts_str} \
                    {len(sample_sizes)} {sample_sizes_str} \
                        {len(channel_counts)} {channel_counts_str}"
    print(f"Running command: {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        print("Error in running the test")
        assert False

    return True
