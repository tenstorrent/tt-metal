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


def append_to_csv(file_path, data, header=None, write_header=False):
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or write_header:
            assert header is not None
            writer.writerow(header)
        writer.writerows([data])


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


def process_profile_results(sample_size, sample_count, channel_count, benchmark_type, test_latency, num_repititions):
    freq = get_device_freq() / 1000.0
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    main_test_body_string = "MAIN-TEST-BODY"
    devices_data = import_log_run_stats(setup)

    link_stats_fname = PROFILER_LOGS_DIR / "eth_link_stats.csv"
    df = pd.read_csv(link_stats_fname)

    for device_id in devices_data["devices"]:
        # pprint.pp(devices_data["devices"])

        for core, core_data in devices_data["devices"][device_id]["cores"].items():
            if core == "DEVICE":
                continue
            timed_data = core_data["riscs"]["ERISC"]["timeseries"]
            sender_chip = sender_core = receiver_chip = receiver_core = None
            # pprint.pp(timed_data)

            starts = [0] * num_repititions
            ends = [0] * num_repititions
            link_stats = [[]] * num_repititions
            for metadata, ts, ts_data in timed_data:
                # print(pprint.pp(metadata))
                if metadata["type"] == "TS_DATA":
                    # ts_data has sender - receiver link encoding
                    sender = (ts_data >> 32) & 0xFFFFFFFF
                    sender_chip = (sender >> 16) & 0xFF
                    sender_core = (((sender >> 8) & 0xFF), ((sender)) & 0xFF)
                    receiver = ts_data & 0xFFFFFFFF
                    receiver_chip = (receiver >> 16) & 0xFF
                    receiver_core = (((receiver >> 8) & 0xFF), ((receiver)) & 0xFF)

                if metadata["zone_name"] == main_test_body_string:
                    run_id = metadata["run_id"]
                    if metadata["type"] == "ZONE_START":
                        starts[run_id] = ts
                    if metadata["type"] == "ZONE_END":
                        ends[run_id] = ts

                        link_stat_row = df.loc[
                            (df["Iteration"] == run_id)
                            & (df["Sender Device ID"] == sender_chip)
                            & (df["Sender X"] == sender_core[0])
                            & (df["Sender Y"] == sender_core[1])
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
                        link_stats[run_id] = [
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

            assert sender_chip != None

            main_loop_cycles = [end - start for end, start in zip(ends, starts)]
            if test_latency:
                results = [main_loop_cycle / freq for main_loop_cycle in main_loop_cycles]
            else:
                results = [
                    sample_size / (main_loop_cycle / freq / sample_count / channel_count)
                    for main_loop_cycle in main_loop_cycles
                ]

            # (sender_device_id, sender_core): ( (receiver_chip_id, receiver_core), benchmark, sample_size, measurements[], link_stats[] )
            if (sender_chip, sender_core) not in results_per_sender_link:
                results_per_sender_link[(sender_chip, sender_core)] = (
                    (receiver_chip, receiver_core),
                    benchmark_type,
                    sample_size,
                    results,
                    link_stats,
                )
            else:
                assert False

    return np.mean(results)


def write_results_to_csv(file_name, test_latency):
    write_header = not os.path.exists(file_name)
    if test_latency == 1:
        header = [
            "Benchmark ID",
            "Sender Device ID",
            "Sender Virtual Coord",
            "Receiver Device ID",
            "Receiver Virtual Coord",
            "Sample Size (B)",
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
    else:
        header = [
            "Benchmark ID",
            "Sender Device ID",
            "Sender Virtual Coord",
            "Receiver Device ID",
            "Receiver Virtual Coord",
            "Sample Size (B)",
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

    for sender_info, data_to_write in results_per_sender_link.items():
        receiver_info, benchmark_type, sample_size, measurements, link_stats = data_to_write
        assert len(measurements) == len(link_stats)
        for measurement, link_stat in zip(measurements, link_stats):
            append_to_csv(
                file_name,
                [
                    benchmark_type,
                    sender_info[0],
                    sender_info[1],
                    receiver_info[0],
                    receiver_info[1],
                    sample_size,
                    measurement,
                    *link_stat,
                ],
                header,
                write_header,
            )
            if write_header:
                write_header = False
        min_val = min(measurements)
        max_val = max(measurements)
        mean = np.mean(measurements)
        std_dev = np.std(measurements)
        summary_stats = [""] * len(header)
        summary_stats[-1] = std_dev
        summary_stats[-2] = mean
        summary_stats[-3] = max_val
        summary_stats[-4] = min_val
        append_to_csv(file_name, summary_stats)

    results_per_sender_link.clear()
    return mean
