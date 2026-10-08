#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math

import pandas as pd

from tracy.perf_counter_analysis import compute_device_only_metrics


def test_device_only_metrics_use_request_counters():
    counter_values = {
        "SRCA_WRITE_REQ": 100,
        "SRCA_WRITE_NOT_BLOCKED_PORT": 80,
        "SRCA_WRITE_NOT_BLOCKED_OVR": 90,
        "SRCB_WRITE_REQ": 200,
        "SRCB_WRITE_NOT_BLOCKED_PORT": 150,
        "SRCB_WRITE_NOT_BLOCKED_OVR": 180,
        "PACKER0_DEST_READ_REQ": 50,
        "DEST_READ_GRANTED_0": 35,
        "PACKER_BUSY": 40,
        "UNPACK0_BUSY_THREAD0": 50,
        "UNPACK1_BUSY_THREAD0": 50,
    }
    perf_counter_df = pd.DataFrame(
        [
            {
                "run_host_id": 1,
                "trace_id_count": 1,
                "core_x": 0,
                "core_y": 0,
                "counter type": counter_name,
                "value": value,
                "ref cnt": 1,
            }
            for counter_name, value in counter_values.items()
        ]
    )

    metrics, _ = compute_device_only_metrics(perf_counter_df, "wormhole")

    assert math.isfinite(metrics["Packer Efficiency"]["avg"][(1, 1)])
    assert metrics["Packer Efficiency"]["avg"][(1, 1)] == 125.0
    assert metrics["Unpacker-to-Math Data Flow"]["avg"][(1, 1)] == 300.0
    assert metrics["SrcA Write Port Blocked Rate"]["avg"][(1, 1)] == 20.0
    assert metrics["SrcB Write Port Blocked Rate"]["avg"][(1, 1)] == 25.0
    assert metrics["SrcA Write Overwrite Blocked Rate"]["avg"][(1, 1)] == 10.0
    assert metrics["SrcB Write Overwrite Blocked Rate"]["avg"][(1, 1)] == 10.0
    assert metrics["SrcA Write Actual Efficiency"]["avg"][(1, 1)] == 80.0
    assert metrics["SrcB Write Actual Efficiency"]["avg"][(1, 1)] == 75.0
    assert metrics["Dest Read Backpressure"]["avg"][(1, 1)] == 30.0
