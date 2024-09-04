# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config


@pytest.mark.parametrize("sample_counts", [(1024,)])
@pytest.mark.parametrize("page_sizes", [(16,)])
@pytest.mark.parametrize("channel_counts", [(1,)])
@pytest.mark.parametrize("hop_counts", [(8,)])
def test_multichip_hop_latency(sample_counts, page_sizes, channel_counts, hop_counts):
    test_string_name = f"test_ethernet_send_data_microbenchmark - \
            sample_counts: {sample_counts}, \
                page_sizes: {page_sizes}, \
                    channel_counts: {channel_counts}, \
                        hop_counts: {hop_counts}"
    print(f"{test_string_name}")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    sample_counts_str = " ".join([str(s) for s in sample_counts])
    page_sizes_str = " ".join([str(s) for s in page_sizes])
    channel_counts_str = " ".join([str(s) for s in channel_counts])
    hop_counts_str = " ".join([str(s) for s in hop_counts])

    rc = os.system(
        f"TT_METAL_DEVICE_PROFILER=1 \
            ./build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_hop_latencies_no_edm \
                {len(sample_counts)} {sample_counts_str} \
                    {len(page_sizes)} {page_sizes_str} \
                        {len(channel_counts)} {channel_counts_str} \
                            {len(hop_counts)} {hop_counts_str}"
    )
    if rc != 0:
        print("Error in running the test")
        assert False

    return True
