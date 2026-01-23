# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
import numpy as np
from tracy.process_device_log import import_log_run_stats
import tracy.device_post_proc_config as device_post_proc_config
from tracy.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def process_ping_latency_results(num_samples, sample_size, num_channels):
    """Process profiler results and compute latency statistics for all zones."""
    import os

    # Check if profiler log exists
    if not os.path.exists(profiler_log_path):
        logger.error(f"Profiler log not found at: {profiler_log_path}")
        return None

    logger.info(f"Processing profiler log from: {profiler_log_path}")

    try:
        setup = device_post_proc_config.default_setup()
        setup.deviceInputLog = profiler_log_path
        devices_data = import_log_run_stats(setup)

        freq = devices_data["deviceInfo"]["freq"] / 1000.0  # Convert to MHz -> ns conversion factor

        logger.info(f"Device frequency: {freq * 1000.0:.2f} MHz")
        logger.info(f"Number of devices in log: {len(devices_data['devices'])}")

        # Track all zones we want to report
        # MAIN-TEST-BODY: Total round-trip latency (sender perspective)
        # SENDER-LOOP-ITER: One complete ping-pong iteration on sender
        # SEND-PAYLOADS-PHASE: Time to send data packets
        # WAIT-ACKS-PHASE: Time waiting for receiver acknowledgments
        # RECEIVER-LOOP-ITER: One complete iteration on receiver
        # PING-REPLIES: Time to process and reply to pings
        zones_of_interest = [
            "MAIN-TEST-BODY",
            "SENDER-LOOP-ITER",
            "SEND-PAYLOADS-PHASE",
            "WAIT-ACKS-PHASE",
            "RECEIVER-LOOP-ITER",
            "PING-REPLIES",
        ]

        results_by_zone = {}
        sender_device = None
        sender_core = None
        receiver_device = None
        receiver_core = None

        for device_id in devices_data["devices"]:
            logger.info(f"Processing device {device_id}")
            for core, core_data in devices_data["devices"][device_id]["cores"].items():
                if core == "DEVICE":
                    continue

                if "ERISC" not in core_data["riscs"]:
                    continue

                timed_data = core_data["riscs"]["ERISC"]["timeseries"]

                # Collect timing data for all zones
                zone_timings = {zone: {"starts": [], "ends": []} for zone in zones_of_interest}

                for metadata, ts, ts_data in timed_data:
                    zone_name = metadata["zone_name"]
                    if zone_name in zones_of_interest:
                        if metadata["type"] == "ZONE_START":
                            zone_timings[zone_name]["starts"].append(ts)
                        if metadata["type"] == "ZONE_END":
                            zone_timings[zone_name]["ends"].append(ts)

                # Process each zone
                for zone_name, timings in zone_timings.items():
                    if timings["starts"] and timings["ends"]:
                        # Identify sender vs receiver
                        if zone_name.startswith("SENDER") or (
                            zone_name == "MAIN-TEST-BODY" and "SENDER" in str(zone_timings.keys())
                        ):
                            sender_device = device_id
                            sender_core = core
                        elif zone_name.startswith("RECEIVER") or zone_name == "PING-REPLIES":
                            receiver_device = device_id
                            receiver_core = core

                        # Calculate latencies
                        latencies = []
                        for start, end in zip(timings["starts"], timings["ends"]):
                            total_cycles = end - start
                            total_latency_ns = total_cycles / freq
                            # For per-sample latency, divide by num_samples
                            latency_per_sample_ns = total_latency_ns / num_samples
                            latencies.append(latency_per_sample_ns)

                        if latencies:
                            results_by_zone[f"{device_id}:{core}:{zone_name}"] = {
                                "zone": zone_name,
                                "device_id": device_id,
                                "core": core,
                                "min": np.min(latencies),
                                "max": np.max(latencies),
                                "mean": np.mean(latencies),
                                "std": np.std(latencies),
                                "count": len(latencies),
                            }

        if not results_by_zone:
            logger.warning("No timing data found in profiler log")
            return None

        # Map zone names to semantic meanings
        zone_semantics = {
            "MAIN-TEST-BODY": "Round-Trip Total",
            "SENDER-LOOP-ITER": "Sender: Full Iteration",
            "SEND-PAYLOADS-PHASE": "Sender: Send Data",
            "WAIT-ACKS-PHASE": "Sender: Wait for Ack",
            "RECEIVER-LOOP-ITER": "Receiver: Full Iteration",
            "PING-REPLIES": "Receiver: Process & Reply",
        }

        # Calculate derived metrics
        round_trip_time = None
        receiver_service_time = None

        for key, r in results_by_zone.items():
            if r["zone"] == "MAIN-TEST-BODY":
                round_trip_time = r["mean"]
            elif r["zone"] == "PING-REPLIES":
                receiver_service_time = r["mean"]

        # Print detailed results table
        print(f"\n{'='*120}")
        print(f"Ethernet Link Ping Latency - Detailed Breakdown")
        print(f"{'='*120}")
        print(f"Configuration: {num_samples} samples x {sample_size} bytes, {num_channels} channel(s)")
        print(f"Device Freq: {freq * 1000.0:.2f} MHz")
        if sender_device is not None:
            print(f"Sender: Device {sender_device}, Core {sender_core}")
        if receiver_device is not None:
            print(f"Receiver: Device {receiver_device}, Core {receiver_core}")
        print(f"\nNote: All times are per-ping latencies (ns)")
        print(f"{'='*120}")
        print(f"{'Measurement':<35} {'Device':<8} {'Core':<15} {'Mean':<12} {'Min':<12} {'Max':<12} {'Std Dev':<12}")
        print(f"{'-'*120}")

        # Sort by device and zone name for consistent ordering
        for key in sorted(results_by_zone.keys()):
            r = results_by_zone[key]
            semantic_name = zone_semantics.get(r["zone"], r["zone"])
            print(
                f"{semantic_name:<35} {r['device_id']:<8} {str(r['core']):<15} "
                f"{r['mean']:<12.2f} {r['min']:<12.2f} {r['max']:<12.2f} {r['std']:<12.2f}"
            )

        # Add derived metrics
        if round_trip_time is not None:
            print(f"{'-'*120}")
            print(f"{'Derived Metrics:':<35}")
            per_hop_simple = round_trip_time / 2.0
            print(
                f"{'  Per-Hop (RTT/2)':<35} {'-':<8} {'-':<15} "
                f"{per_hop_simple:<12.2f} {'-':<12} {'-':<12} {'-':<12}"
            )

            if receiver_service_time is not None:
                # Adjusted per-hop: (RTT - receiver_service_time) / 2
                # This gives the one-way link latency excluding receiver processing
                per_hop_adjusted = (round_trip_time - receiver_service_time) / 2.0
                print(
                    f"{'  Per-Hop Adjusted (RTT-RxSvc)/2':<35} {'-':<8} {'-':<15} "
                    f"{per_hop_adjusted:<12.2f} {'-':<12} {'-':<12} {'-':<12}"
                )
                print(
                    f"{'  Receiver Service Time':<35} {'-':<8} {'-':<15} "
                    f"{receiver_service_time:<12.2f} {'-':<12} {'-':<12} {'-':<12}"
                )

        print(f"{'='*120}\n")

        # Return summary with main test body results
        main_result = None
        for key, r in results_by_zone.items():
            if r["zone"] == "MAIN-TEST-BODY":
                main_result = {
                    "min": r["min"],
                    "max": r["max"],
                    "mean": r["mean"],
                    "std": r["std"],
                    "device_id": r["device_id"],
                    "core": r["core"],
                    "num_samples": num_samples,
                    "sample_size": sample_size,
                    "num_channels": num_channels,
                    "all_zones": results_by_zone,
                }
                break

        return (
            main_result
            if main_result
            else {
                "num_samples": num_samples,
                "sample_size": sample_size,
                "num_channels": num_channels,
                "all_zones": results_by_zone,
                "mean": list(results_by_zone.values())[0]["mean"] if results_by_zone else 0,
            }
        )

    except Exception as e:
        logger.error(f"Error processing profiler results: {e}")
        import traceback

        traceback.print_exc()
        return None


@pytest.mark.parametrize("sample_counts", [(1024,)])  # , 8, 16, 64, 256],
@pytest.mark.parametrize(
    "sample_sizes",
    [(16, 1024, 4096, 8192)],
)  # , 1024, 2048, 4096],
@pytest.mark.parametrize(
    "channel_counts",
    [(1,)],
)
def test_bidirectional_erisc_bandwidth(sample_counts, sample_sizes, channel_counts):
    test_string_name = f"test_ethernet_link_ping_latency - \
            sample_counts: {sample_counts}, \
                sample_sizes: {sample_sizes}, \
                    channel_counts: {channel_counts}"
    print(f"{test_string_name}")

    all_results = []

    # Run each configuration separately to get clean profiler logs
    for num_samples in sample_counts:
        for sample_size in sample_sizes:
            for num_channels in channel_counts:
                # Clear profiler log before each run
                os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

                print(f"\n{'='*60}")
                print(f"Running configuration: samples={num_samples}, size={sample_size}B, channels={num_channels}")
                print(f"{'='*60}")

                # Run the test for this specific configuration
                rc = os.system(
                    f"TT_METAL_DEVICE_PROFILER=1 \
                        {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_link_ping_latency_no_edm \
                            1 {num_samples} \
                                1 {sample_size} \
                                    1 {num_channels} \
                        "
                )
                if rc != 0:
                    print(
                        f"Error in running test for configuration: samples={num_samples}, size={sample_size}, channels={num_channels}"
                    )
                    assert False

                # Process results for this configuration
                results = process_ping_latency_results(num_samples, sample_size, num_channels)
                if results:
                    all_results.append(results)
                    # Optionally add assertions here based on expected latency ranges
                    assert results["mean"] > 0, "Mean latency should be positive"
                else:
                    print(
                        f"WARNING: No results found for configuration: samples={num_samples}, size={sample_size}, channels={num_channels}"
                    )

    if all_results:
        print(f"\n{'='*80}")
        print(f"Summary of All Configurations")
        print(f"{'='*80}")
        print(f"{'Samples':<10} {'Size (B)':<12} {'Channels':<10} {'Mean (ns)':<12} {'Std Dev (ns)':<15}")
        print(f"{'-'*80}")
        for result in all_results:
            print(
                f"{result['num_samples']:<10} {result['sample_size']:<12} {result['num_channels']:<10} "
                f"{result['mean']:<12.2f} {result['std']:<15.2f}"
            )
        print(f"{'='*80}\n")

    return True
