# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import pprint
from loguru import logger
import numpy as np
import pandas as pd
import csv
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

results_per_sender_link = {}


def append_to_csv(file_path, data=[], header=None, write_header=False, add_newline=False):
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or write_header:
            assert header is not None
            writer.writerow(header)
        writer.writerows([data])
        if add_newline:
            csvfile.write("\n")


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


def get_arch():
    setup = device_post_proc_config.default_setup()
    setup = device_post_proc_config.default_setup()
    deviceData = import_log_run_stats(setup)
    arch = deviceData["deviceInfo"]["arch"]
    return arch


def process_profile_results(packet_size, num_packets, channel_count, benchmark_type, test_latency, num_iterations):
    freq = get_device_freq() / 1000.0
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    main_test_body_string = "MAIN-TEST-BODY"
    devices_data = import_log_run_stats(setup)

    arch = get_arch()

    if arch == "wormhole_b0":
        link_stats_fname = PROFILER_LOGS_DIR / "eth_link_stats.csv"
        df = pd.read_csv(link_stats_fname)

    results = []
    for device_id in devices_data["devices"]:
        for core, core_data in devices_data["devices"][device_id]["cores"].items():
            if core == "DEVICE":
                continue
            timed_data = core_data["riscs"]["ERISC"]["timeseries"]
            sender_chip = sender_eth = receiver_chip = receiver_eth = None

            starts = [0] * num_iterations
            ends = [0] * num_iterations
            link_stats = [[]] * num_iterations
            for metadata, ts, ts_data in timed_data:
                if metadata["type"] == "TS_DATA":
                    # ts_data has sender - receiver link encoding
                    sender = (ts_data >> 32) & 0xFFFFFFFF
                    sender_chip = (sender >> 8) & 0xFF
                    sender_eth = sender & 0xFF
                    receiver = ts_data & 0xFFFFFFFF
                    receiver_chip = (receiver >> 8) & 0xFF
                    receiver_eth = receiver & 0xFF

                if metadata["zone_name"] == main_test_body_string:
                    run_host_id = metadata["run_host_id"]
                    if metadata["type"] == "ZONE_START":
                        starts[run_host_id] = ts
                    if metadata["type"] == "ZONE_END":
                        ends[run_host_id] = ts

                        if arch == "wormhole_b0":
                            link_stat_row = df.loc[
                                (df["Iteration"] == run_host_id)
                                & (df["Sender Device ID"] == sender_chip)
                                & (df["Sender Eth"] == sender_eth)
                            ]
                            for _, row in link_stat_row.iterrows():
                                s_retrain_count = row["S Retrain Count"]
                                s_crc_errs = row["S CRC Errs"]
                                s_pcs_faults = row["S PCS Faults"]
                                s_total_corr = row["S Total Corr"]
                                s_total_uncorr = row["S Total Uncorr"]
                                s_pcs_retrains = row["S Retrain by PCS"]
                                s_crc_retrains = row["S Retrain by CRC"]
                                r_retrain_count = row["S Retrain Count"]
                                r_crc_errs = row["R CRC Errs"]
                                r_pcs_faults = row["R PCS Faults"]
                                r_total_corr = row["R Total Corr"]
                                r_total_uncorr = row["R Total Uncorr"]
                                r_pcs_retrains = row["R Retrain by PCS"]
                                r_crc_retrains = row["R Retrain by CRC"]
                            link_stats[run_host_id] = [
                                s_retrain_count,
                                s_crc_errs,
                                s_pcs_faults,
                                s_total_corr,
                                s_total_uncorr,
                                s_pcs_retrains,
                                s_crc_retrains,
                                r_retrain_count,
                                r_crc_errs,
                                r_pcs_faults,
                                r_total_corr,
                                r_total_uncorr,
                                r_pcs_retrains,
                                r_crc_retrains,
                            ]
                        else:
                            link_stats[run_host_id] = []

            assert sender_chip != None

            main_loop_cycles = [end - start for end, start in zip(ends, starts)]
            if test_latency:
                results = [main_loop_cycle / freq for main_loop_cycle in main_loop_cycles]
            else:
                results = [
                    packet_size / (main_loop_cycle / freq / num_packets / channel_count)
                    for main_loop_cycle in main_loop_cycles
                ]

            # (sender_device_id, sender_eth): ( (receiver_chip_id, receiver_eth), benchmark, num_packets, packet_size, measurements[], link_stats[] )
            if (sender_chip, sender_eth) not in results_per_sender_link:
                results_per_sender_link[(sender_chip, sender_eth)] = (
                    (receiver_chip, receiver_eth),
                    benchmark_type,
                    num_packets,
                    packet_size,
                    results,
                    link_stats,
                )
            else:
                assert False

    return np.mean(results)


def write_results_to_csv(file_name, test_latency):
    if test_latency == 1:
        header = [
            "Benchmark ID",
            "Summary Statistics",
            "Sender Device ID",
            "Sender Eth Channel",
            "Receiver Device ID",
            "Receiver Eth Channel",
            "Num Packets",
            "Packet Size (B)",
            "Latency (ns)",
            "Sender Retrains",
            "Sender CRC Errs",
            "Sender PCS Faults",
            "Sender Total Corr CW",
            "Sender Total Uncorr CW",
            "Sender Retrains Triggered by PCS Faults",
            "Sender Retrains Triggered by CRC Errs",
            "Receiver Retrains",
            "Receiver CRC Errs",
            "Receiver PCS Faults",
            "Receiver Total Corr CW",
            "Receiver Total Uncorr CW",
            "Receiver Retrains Triggered by PCS Faults",
            "Receiver Retrains Triggered by CRC Errs",
            "Min",
            "Max",
            "Mean",
            "StdDev",
        ]
        test_description = [
            "Sender eth core sends sample size byte packet to receiver. Receiver sends same sized packet back. Measurement taken on sender and start before sending packet and stop when returned from receiver"
        ]
    else:
        header = [
            "Benchmark ID",
            "Summary Statistics",
            "Sender Device ID",
            "Sender Eth Channel",
            "Receiver Device ID",
            "Receiver Eth Channel",
            "Num Packets",
            "Packet Size (B)",
            "BW (GB/s)",
            "Sender Retrains",
            "Sender CRC Errs",
            "Sender PCS Faults",
            "Sender Total Corr CW",
            "Sender Total Uncorr CW",
            "Sender Retrains Triggered by PCS Faults",
            "Sender Retrains Triggered by CRC Errs",
            "Receiver Retrains",
            "Receiver CRC Errs",
            "Receiver PCS Faults",
            "Receiver Total Corr CW",
            "Receiver Total Uncorr CW",
            "Receiver Retrains Triggered by PCS Faults",
            "Receiver Retrains Triggered by CRC Errs",
            "Min",
            "Max",
            "Mean",
            "StdDev",
        ]
        test_description = [
            "Sender eth core sends multiple sample size byte packets to receiver. Receiver only sends acks back on packet receipt. Measurement taken on sender and start on first packet sent and stop after last ack from receiver."
        ]

    write_header = not os.path.exists(file_name)
    if write_header:
        append_to_csv(file_name, ["AICLK (MHz):", get_device_freq()], test_description, write_header)
        append_to_csv(file_name, add_newline=True)
        append_to_csv(file_name, header)

    mean = 0
    for sender_info, data_to_write in results_per_sender_link.items():
        receiver_info, benchmark_type, num_packets, packet_size, measurements, link_stats = data_to_write
        assert len(measurements) == len(link_stats)
        for measurement, link_stat in zip(measurements, link_stats):
            append_to_csv(
                file_name,
                [
                    benchmark_type,
                    0,
                    sender_info[0],
                    sender_info[1],
                    receiver_info[0],
                    receiver_info[1],
                    num_packets,
                    packet_size,
                    measurement,
                    *link_stat,
                ],
            )
        min_val = min(measurements)
        max_val = max(measurements)
        mean = np.mean(measurements)
        std_dev = np.std(measurements)
        summary_stats = [""] * len(header)
        summary_stats[0] = benchmark_type
        summary_stats[1] = 1
        summary_stats[2] = sender_info[0]
        summary_stats[3] = sender_info[1]
        summary_stats[4] = receiver_info[0]
        summary_stats[5] = receiver_info[1]
        summary_stats[6] = num_packets
        summary_stats[7] = packet_size
        summary_stats[-1] = std_dev
        summary_stats[-2] = mean
        summary_stats[-3] = max_val
        summary_stats[-4] = min_val
        append_to_csv(file_name, summary_stats)

    results_per_sender_link.clear()
    return mean
