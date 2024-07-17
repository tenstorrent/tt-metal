# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_possetupt_proc_config


def collect_sender_timeseries_for_zone(device_data, zone_name):
    times = []
    time_open = 0
    zone_open = False
    for entry in device_data["cores"]["DEVICE"]["riscs"]["ERISC"]["timeseries"][1:]:
        if zone_name == entry[0]["zone_name"]:
            if not zone_open:
                assert entry[0]["zone_phase"] == "begin"
                time_open = int(entry[1])
                zone_open = True
            else:
                assert entry[0]["zone_phase"] == "end"
                assert int(entry[1]) >= (time_open)
                times.append(int(entry[1]) - int(time_open))
                zone_open = False

    return times


def collect_sender_loop_iteration_time_results(profiler_log_path):
    setup = device_possetupt_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    devices_data = import_log_run_stats(setup)

    sender_loop_iter_zone_durations = collect_sender_timeseries_for_zone(devices_data["devices"][0], "SENDER-LOOP-ITER")
    sender_send_payloads_zone_durations = collect_sender_timeseries_for_zone(
        devices_data["devices"][0], "SEND-PAYLOADS-PHASE"
    )
    sender_wait_acks_zone_durations = collect_sender_timeseries_for_zone(devices_data["devices"][0], "WAIT-ACKS-PHASE")
    receiver_ping_replies_zone_durations = collect_sender_timeseries_for_zone(
        devices_data["devices"][1], "PING-REPLIES"
    )
    receiver_loop_iter_zone_durations = collect_sender_timeseries_for_zone(
        devices_data["devices"][1], "RECEIVER-LOOP-ITER"
    )
    csv = "SENDER-LOOP-ITER,SEND-PAYLOADS-PHASE,WAIT-ACKS-PHASE,PING-REPLIES,RECEIVER-LOOP-ITER\n"

    return (
        sender_loop_iter_zone_durations,
        sender_send_payloads_zone_durations,
        sender_wait_acks_zone_durations,
        receiver_ping_replies_zone_durations,
        receiver_loop_iter_zone_durations,
    )


@pytest.mark.parametrize("sample_counts", [(20,)])
@pytest.mark.parametrize(
    "sample_sizes",
    [
        (
            16,
            1024,
        )
    ],
)  # , 1024, 2048, 4096],
@pytest.mark.parametrize(
    "channel_counts",
    [(1,)],
)
def test_ethernet_link_roundtrip_ping_latency(sample_counts, sample_sizes, channel_counts):
    test_string_name = f"test_ethernet_send_data_microbenchmark - \
            sample_counts: {sample_counts}, \
                sample_sizes: {sample_sizes}, \
                    channel_counts: {channel_counts}"
    print(f"{test_string_name}")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    sample_counts_str = " ".join([str(s) for s in sample_counts])
    sample_sizes_str = " ".join([str(s) for s in sample_sizes])
    channel_counts_str = " ".join([str(s) for s in channel_counts])

    rc = os.system(
        f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_link_ping_latency_no_edm \
                {len(sample_counts)} {sample_counts_str} \
                    {len(sample_sizes)} {sample_sizes_str} \
                        {len(channel_counts)} {channel_counts_str} \
            "
    )
    if rc != 0:
        print("Error in running the test")
        assert False

    (
        sender_loop_iter_zone_durations,
        sender_send_payloads_zone_durations,
        sender_wait_acks_zone_durations,
        receiver_ping_replies_zone_durations,
        receiver_loop_iter_zone_durations,
    ) = collect_sender_loop_iteration_time_results(
        f"{os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv"
    )

    for n_samples in sample_counts:
        csv = "sample_size,SENDER-LOOP-ITER,SEND-PAYLOADS-PHASE,WAIT-ACKS-PHASE,PING-REPLIES,RECEIVER-LOOP-ITER\n"
        for i, size in enumerate(sample_sizes):
            csv += f"{size},"
            average_sender_loop_iter = (
                sum(sender_loop_iter_zone_durations[n_samples * i : n_samples * (i + 1)]) / n_samples
            )
            average_sender_send_payloads = (
                sum(sender_send_payloads_zone_durations[n_samples * i : n_samples * (i + 1)]) / n_samples
            )
            average_sender_wait_acks = (
                sum(sender_wait_acks_zone_durations[n_samples * i : n_samples * (i + 1)]) / n_samples
            )
            average_receiver_ping_replies = (
                sum(receiver_ping_replies_zone_durations[n_samples * i : n_samples * (i + 1)]) / n_samples
            )
            average_receiver_loop_iter = (
                sum(receiver_loop_iter_zone_durations[n_samples * i : n_samples * (i + 1)]) / n_samples
            )

            csv += f"{average_sender_loop_iter},{average_sender_send_payloads},{average_sender_wait_acks},{average_receiver_ping_replies},{average_receiver_loop_iter}\n"

    print(f"{csv}")

    return True


@pytest.mark.skip("Only run manually for now due to long stats collection time")
@pytest.mark.parametrize("sample_counts", [(1024,)])
@pytest.mark.parametrize(
    "sample_sizes",
    [
        (
            16,
            1024,
            2048,
            4096,
            8192,
            16384,
        )
    ],
)
@pytest.mark.parametrize(
    "channel_counts",
    [(1,)],
)
def test_ethernet_link_roundtrip_ping_latency_sweep(sample_counts, sample_sizes, channel_counts):
    test_string_name = f"test_ethernet_send_data_microbenchmark - \
            sample_counts: {sample_counts}, \
                sample_sizes: {sample_sizes}, \
                    channel_counts: {channel_counts}"
    print(f"{test_string_name}")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    sample_counts_str = " ".join([str(s) for s in sample_counts])
    sample_sizes_str = " ".join([str(s) for s in sample_sizes])
    channel_counts_str = " ".join([str(s) for s in channel_counts])

    rc = os.system(
        f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_link_ping_latency_no_edm \
                {len(sample_counts)} {sample_counts_str} \
                    {len(sample_sizes)} {sample_sizes_str} \
                        {len(channel_counts)} {channel_counts_str} \
            "
    )
    if rc != 0:
        print("Error in running the test")
        assert False

    (
        sender_loop_iter_zone_durations,
        sender_send_payloads_zone_durations,
        sender_wait_acks_zone_durations,
        receiver_ping_replies_zone_durations,
        receiver_loop_iter_zone_durations,
    ) = collect_sender_loop_iteration_time_results(
        f"{os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv"
    )

    for n_samples in sample_counts:
        csv = "sample_size,SENDER-LOOP-ITER,SEND-PAYLOADS-PHASE,WAIT-ACKS-PHASE,PING-REPLIES,RECEIVER-LOOP-ITER\n"
        for i, size in enumerate(sample_sizes):
            csv += f"{size},"
            average_sender_loop_iter = (
                sum(sender_loop_iter_zone_durations[n_samples * i : n_samples * (i + 1)]) / n_samples
            )
            average_sender_send_payloads = (
                sum(sender_send_payloads_zone_durations[n_samples * i : n_samples * (i + 1)]) / n_samples
            )
            average_sender_wait_acks = (
                sum(sender_wait_acks_zone_durations[n_samples * i : n_samples * (i + 1)]) / n_samples
            )
            average_receiver_ping_replies = (
                sum(receiver_ping_replies_zone_durations[n_samples * i : n_samples * (i + 1)]) / n_samples
            )
            average_receiver_loop_iter = (
                sum(receiver_loop_iter_zone_durations[n_samples * i : n_samples * (i + 1)]) / n_samples
            )

            csv += f"{average_sender_loop_iter},{average_sender_send_payloads},{average_sender_wait_acks},{average_receiver_ping_replies},{average_receiver_loop_iter}\n"

    print(f"{csv}")

    return True
